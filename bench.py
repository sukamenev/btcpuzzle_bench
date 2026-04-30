#!/usr/bin/env python3
"""bench.py — сравнение скоростей программ-перебирателей приватных ключей btc-puzzle.

Использование:
    python bench.py --prepare --device all   # clone+patch+build, без прогона
    python bench.py --device cpu             # бенчмарк CPU
    python bench.py --device cuda            # бенчмарк CUDA
    python bench.py --device amdgpu          # бенчмарк AMDGPU
    python bench.py --device all             # всё подряд
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# ===========================================================================
#  ┌─ КОНФИГУРАЦИЯ БЕНЧМАРКА (всё, что регулярно меняется — в этом блоке) ─┐
# ===========================================================================

# Число main-итераций (случайных ключей) на программу, по классу устройства.
# Env-override: BENCH_N (для всех) или BENCH_N_CPU / BENCH_N_CUDA / BENCH_N_AMDGPU.
NUM_ITERATIONS_PER_DEVICE = {
    "cpu":    int(os.environ.get("BENCH_N_CPU",    os.environ.get("BENCH_N", "100"))),
    "cuda":   int(os.environ.get("BENCH_N_CUDA",   os.environ.get("BENCH_N", "100"))),
    "amdgpu": int(os.environ.get("BENCH_N_AMDGPU", "4")),  # bitcrack всегда FAIL — 4 хватит для btcmole
}

# Число warmup-раундов перед main-измерениями (один и тот же набор кейсов
# для всех программ; CPU выводит на steady-state thermal).
WARMUP_ROUNDS = int(os.environ.get("BENCH_WARMUP", "2"))

# Битность интервала поиска по классу устройства.
# CPU=32 (2^32 ключей), CUDA=34 (4× больше — GPU быстро), AMDGPU=30 (4× меньше).
# Env-override: BENCH_BITS_CPU / BENCH_BITS_CUDA / BENCH_BITS_AMDGPU.
INTERVAL_BITS_PER_DEVICE = {
    "cpu":    int(os.environ.get("BENCH_BITS_CPU",    "33")),
    "cuda":   int(os.environ.get("BENCH_BITS_CUDA",   "35")),
    "amdgpu": int(os.environ.get("BENCH_BITS_AMDGPU", "30")),
}

# Таймаут на одну итерацию (сек). Если программа не нашла ключ — FAIL.
TIMEOUT_S = int(os.environ.get("BENCH_TIMEOUT", "300"))

# Seed для детерминированной генерации случайных ключей. Один и тот же seed
# даёт один и тот же набор тестовых ключей между прогонами/программами.
SEED = 42

# Эталоны для расчёта «× эталон» в отчёте — по классу устройства.
BASELINE = {
    "cpu":    "cyclone",
    "cuda":   "cudacyclone",
    "amdgpu": "bitcrack-opencl",
}

# Число CPU-потоков, передаётся всем CPU-программам как -t / +cpu:N.
# Env-override: BENCH_CPU_THREADS. По умолчанию = 90% от os.cpu_count() (10% headroom
# на system-task'и: graphics, ssh, bench-runner — чтобы не вносили thermal/scheduler-noise).
def _detect_cpu_threads() -> int:
    env = os.environ.get("BENCH_CPU_THREADS", "").strip()
    if env:
        return max(1, int(env))
    return max(1, int((os.cpu_count() or 1) * 0.9))


CPU_THREADS = _detect_cpu_threads()
print(f"[detect] CPU threads for benchmark: {CPU_THREADS} of {os.cpu_count()} available", flush=True)

# ===========================================================================
#  └─ конец блока конфигурации ─┘
# ===========================================================================


# Хелперы для per-device констант.
def bits_for_device(dev: str) -> int:
    return INTERVAL_BITS_PER_DEVICE[dev]


def range_for_device(dev: str) -> tuple[int, int]:
    bits = bits_for_device(dev)
    return (1 << bits, (1 << (bits + 1)) - 1)


def num_iterations_for_device(dev: str) -> int:
    return NUM_ITERATIONS_PER_DEVICE[dev]


# Пути файловой системы (стабильные, ≈ не меняются).
PROGRAMMING_DIR = Path.home() / "Kamenev" / "programming"
BENCH_DIR       = PROGRAMMING_DIR / "multiple_bench"
WORK_DIR        = BENCH_DIR / "work"
RESULTS_DIR     = BENCH_DIR / "results"
PATCHES_DIR     = BENCH_DIR / "patches"
LOGS_DIR        = BENCH_DIR / "logs"
PROGRESS_FILE   = BENCH_DIR / "PROGRESS.md"


# ---------------------------------------------------------------------------
#  Зависимости (pip-пакеты)
# ---------------------------------------------------------------------------
def ensure_deps() -> None:
    """ensure_deps проверяет нужные python-пакеты и доустанавливает недостающие."""
    needed = {"coincurve": "coincurve", "base58": "base58"}
    missing = []
    for mod, pkg in needed.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[deps] Устанавливаю pip-пакеты: {missing}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)


ensure_deps()
import base58       # noqa: E402
import coincurve    # noqa: E402


# ---------------------------------------------------------------------------
#  Определение возможностей CPU и GPU
# ---------------------------------------------------------------------------
def detect_cpu_avx() -> str:
    """detect_cpu_avx читает /proc/cpuinfo и возвращает 'avx512' / 'avx2' / 'none'."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    if "avx512f" in line:
                        return "avx512"
                    if "avx2" in line:
                        return "avx2"
                    return "none"
    except FileNotFoundError:
        pass
    return "none"


def detect_cuda_ccap() -> str:
    """detect_cuda_ccap возвращает compute capability первой видимой NVIDIA-карты,
    например '89'. Пустая строка, если nvidia-smi отсутствует или нет карт."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        ).strip().splitlines()
        if out:
            return out[0].strip().replace(".", "")
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return ""


def detect_cuda_dir() -> Optional[str]:
    """detect_cuda_dir ищет путь к CUDA toolkit. Стандартный /usr/local/cuda → /opt/cuda
    (Arch Linux convention) → ничего. Возвращает абсолютный путь, либо None."""
    for cand in ("/usr/local/cuda", "/opt/cuda"):
        if Path(cand, "include", "cuda.h").is_file() and \
           list(Path(cand, "lib64").glob("libcudart.so*")):
            return cand
    return None


# ---------------------------------------------------------------------------
#  Генерация тестовых кейсов (priv → P2PKH)
# ---------------------------------------------------------------------------
def privkey_to_p2pkh_and_h160(priv_int: int) -> tuple[str, str]:
    """privkey_to_p2pkh_and_h160 по приватному ключу (int) считает P2PKH-адрес
    (mainnet, compressed) и hash160 (hex). Возвращает (address, h160_hex)."""
    priv_bytes = priv_int.to_bytes(32, "big")
    pub = coincurve.PublicKey.from_secret(priv_bytes).format(compressed=True)
    sha = hashlib.sha256(pub).digest()
    rip = hashlib.new("ripemd160", sha).digest()
    payload = b"\x00" + rip
    chk = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    address = base58.b58encode(payload + chk).decode()
    return address, rip.hex()


@dataclass
class BenchCase:
    """BenchCase — один тестовый кейс: целевой приватный ключ + его P2PKH-адрес и hash160."""
    idx: int
    priv_int: int
    address: str
    hash160_hex: str

    @property
    def priv_hex64(self) -> str:
        return f"{self.priv_int:064x}"


def generate_cases(seed: int, n: int, dev: str) -> list[BenchCase]:
    """generate_cases генерирует n детерминированных кейсов внутри range устройства dev.
    Range зависит от dev (см. INTERVAL_BITS_PER_DEVICE). seed микшируется с bits,
    чтобы CPU/CUDA/AMDGPU имели разные наборы priv'ов даже при одинаковом seed."""
    bits = bits_for_device(dev)
    range_start, range_end = range_for_device(dev)
    rng = random.Random(seed ^ (bits << 8))
    cases = []
    for i in range(n):
        priv = rng.randint(range_start, range_end)
        addr, h160 = privkey_to_p2pkh_and_h160(priv)
        cases.append(BenchCase(i, priv, addr, h160))
    return cases


# ---------------------------------------------------------------------------
#  Реестр программ
# ---------------------------------------------------------------------------
@dataclass
class ProgramSpec:
    """ProgramSpec — описание одной программы для бенчмарка."""
    id: str                                                       # уникальный ID (имя папки в work/, ID в таблицах)
    label: str                                                    # отображаемое имя
    devices: set[str]                                             # {'cpu'} | {'cuda'} | {'amdgpu'}
    source_url: str                                               # git URL (upstream GitHub/GitLab), откуда клонируем
    shared_clone_with: Optional[str] = None                       # переиспользовать clone из work/<этот id> (один репо на несколько вариантов сборки, например btcmole)
    build_subdir: str = ""                                        # подпапка внутри work/<id>, в которой собирать
    build_cmds: object = field(default_factory=list)              # list[list[str]] ИЛИ Callable[[Path], list[list[str]]] — последовательность команд сборки
    bin_relpath: str = ""                                         # путь к бинарнику от work/<id>/
    cwd_relpath: str = ""                                         # cwd для запуска (от work/<id>/), по умолчанию = build_subdir
    make_argv: Optional[Callable] = None                          # (case, run_dir, device) -> argv
    pre_run: Optional[Callable] = None                            # (case, run_dir, device) -> None — подготовка input-файлов
    cleanup_globs: list[str] = field(default_factory=list)        # стереть эти файлы (glob от cwd) перед каждой итерацией
    extra_check_files: list[str] = field(default_factory=list)    # доп. файлы (от cwd), сканировать на предмет priv_hex
    speed_re: Optional[str] = None                                # regex Mkey/s (group 1 = число), для справки
    patch_file: Optional[str] = None                              # patches/<file>.patch для git apply
    timeout_s: int = TIMEOUT_S


# ---------------------------------------------------------------------------
#  make_argv хелперы для каждой программы
# ---------------------------------------------------------------------------
def _hex_range_lower(dev: str) -> str:
    s, e = range_for_device(dev)
    return f"{s:x}:{e:x}"


def _argv_cyclone(case: BenchCase, run_dir: Path, dev: str) -> list[str]:
    return ["./Cyclone", "-a", case.address, "-r", _hex_range_lower(dev),
            "-t", str(CPU_THREADS)]


def _argv_ecloop(case: BenchCase, run_dir: Path, dev: str) -> list[str]:
    return [
        "./ecloop", "add",
        "-f", "target.h160",
        "-r", _hex_range_lower(dev),
        "-t", str(CPU_THREADS),
        "-o", "found.txt",
    ]


def _pre_ecloop(case: BenchCase, run_dir: Path, dev: str) -> None:
    (run_dir / "target.h160").write_text(case.hash160_hex + "\n")


def _argv_keyhunt(case: BenchCase, run_dir: Path, dev: str) -> list[str]:
    # Sequential mode (без -R): потоки берут чанки по N_SEQUENTIAL_MAX из общего курсора
    # под mutex (см. keyhunt.cpp:2552-2563). Random mode (-R) даёт длинный хвост по времени
    # (геометрическое распределение): на нашем стенде регулярно вылетал в 97с/180с timeout.
    # -n 1048576 (1M-чанк) делит range на ~4096 чанков для 4.3B → 35 чанков на поток при 115t.
    # Без -n keyhunt берёт N_SEQUENTIAL_MAX=2^32 = весь range на один поток (3 Mkeys/s).
    # -b BITS эквивалентен range [2^(BITS-1), 2^BITS); -q глушит per-thread "Base key:" stdout.
    return [
        "./keyhunt",
        "-t", str(CPU_THREADS),
        "-m", "rmd160",
        "-f", "target.rmd",
        "-b", str(bits_for_device(dev) + 1),
        "-l", "compress",
        "-n", "1048576",
        "-q",
    ]


def _pre_keyhunt(case: BenchCase, run_dir: Path, dev: str) -> None:
    (run_dir / "target.rmd").write_text(case.hash160_hex + "\n")


def _argv_btcmole(case: BenchCase, run_dir: Path, dev: str) -> list[str]:
    # CPU-вариант: число диггеров = CPU_THREADS (то же что у других CPU-программ).
    # При CUDA/AMDGPU флаги CPU отключают, чтобы тестировать только GPU-класс.
    flags = {
        "cpu":    [f"+cpu:{CPU_THREADS}", "-cuda", "-amdgpu"],
        "cuda":   ["-cpu", "+cuda", "-amdgpu"],
        "amdgpu": ["-cpu", "-cuda", "+amdgpu"],
    }[dev]
    return [
        "./bm", "bf",
        "--address", case.address,
        "--range", _hex_range_lower(dev),
        *flags,
    ]


def _argv_cudacyclone(case: BenchCase, run_dir: Path, dev: str) -> list[str]:
    return [
        "./CUDACyclone",
        "--range", _hex_range_lower(dev),
        "--address", case.address,
    ]


def _argv_keyhunt_cuda(case: BenchCase, run_dir: Path, dev: str) -> list[str]:
    return [
        "./KeyHunt",
        "-g", "--gpui", "0",
        "-m", "ADDRESS",
        "--coin", "BTC",
        "--range", _hex_range_lower(dev),
        case.address,
    ]


def _argv_keykiller(case: BenchCase, run_dir: Path, dev: str) -> list[str]:
    # keykiller принимает только -r BITS
    return [
        "./kk",
        "-r", str(bits_for_device(dev) + 1),
        "-a", case.address,
    ]


def _argv_keyscanner(case: BenchCase, run_dir: Path, dev: str) -> list[str]:
    return [
        "./keyscanner",
        "-g", "--gpui", "0", "--gpux", "256,256",
        "-m", "address", "--coin", "BTC",
        "-r", str(bits_for_device(dev) + 1),
        "-o", "found.txt",
        "--range", _hex_range_lower(dev),
        case.address,
    ]


def _argv_bitcrack(case: BenchCase, run_dir: Path, dev: str) -> list[str]:
    binary = "./cuBitCrack" if dev == "cuda" else "./clBitCrack"
    return [
        binary,
        "--keyspace", _hex_range_lower(dev),
        "-o", "found.txt",
        case.address,
    ]


# ---------------------------------------------------------------------------
#  Сборщик реестра — параметризован под обнаруженные возможности железа
# ---------------------------------------------------------------------------
def build_registry() -> list[ProgramSpec]:
    """build_registry строит список программ с учётом текущего CPU/GPU."""
    avx = detect_cpu_avx()
    ccap = detect_cuda_ccap() or "75"     # дефолт SM_75, если nvidia-smi нет
    cuda_dir = detect_cuda_dir() or "/usr/local/cuda"
    print(f"[detect] AVX={avx}, CUDA toolkit={cuda_dir}, GPU compute_cap=sm_{ccap}", flush=True)

    # Cyclone: автовыбор AVX-512 / AVX-2 на этапе сборки
    cyclone_subdir = "Cyclone_avx512" if avx == "avx512" else "Cyclone_avx2"
    cyclone_label  = "Cyclone (AVX-512)" if avx == "avx512" else (
        "Cyclone (AVX-2)" if avx == "avx2" else "Cyclone (no-AVX)"
    )
    cyclone_avx2_build = [
        "g++", "-std=c++17", "-Ofast", "-funroll-loops", "-ftree-vectorize",
        "-fstrict-aliasing", "-fno-semantic-interposition",
        "-fvect-cost-model=unlimited", "-fno-trapping-math",
        "-fipa-ra", "-fipa-modref", "-flto", "-fassociative-math", "-fopenmp",
        "-mavx2", "-mbmi2", "-madx", "-o", "Cyclone",
        "Cyclone.cpp", "SECP256K1.cpp", "Int.cpp", "IntGroup.cpp",
        "IntMod.cpp", "Point.cpp", "ripemd160_avx2.cpp",
        "p2pkh_decoder.cpp", "sha256_avx2.cpp",
    ]
    cyclone_avx512_build = [
        "g++", "-std=c++17", "-Ofast", "-ffast-math", "-funroll-loops",
        "-ftree-vectorize", "-fstrict-aliasing", "-fno-semantic-interposition",
        "-fvect-cost-model=unlimited", "-fno-trapping-math",
        "-fipa-ra", "-mavx512f", "-mavx512vl", "-mavx512bw", "-mavx512dq",
        "-fipa-modref", "-flto", "-fassociative-math", "-fopenmp",
        "-mavx2", "-mbmi2", "-madx", "-o", "Cyclone",
        "Cyclone.cpp", "SECP256K1.cpp", "Int.cpp", "IntGroup.cpp",
        "IntMod.cpp", "Point.cpp", "ripemd160_avx2.cpp",
        "p2pkh_decoder.cpp", "sha256_avx2.cpp",
        "ripemd160_avx512.cpp", "sha256_avx512.cpp",
    ]
    cyclone_build = cyclone_avx512_build if avx == "avx512" else cyclone_avx2_build

    # btcmole: версия читается из bm_latest.txt после клонирования (например '0.7.4' → 'bm074'),
    # из zip берётся вариант под обнаруженный CPU AVX (g1_generic / g3_avx2 / g4_avx512),
    # бинарник кладётся под общее имя 'bm' — все запуски используют './bm'.
    btcmole_variant = {"avx512": "g4_avx512", "avx2": "g3_avx2"}.get(avx, "g1_generic")
    def btcmole_build_factory(zip_flavor: str):
        """zip_flavor: '' для CPU, '_cuda', '_amdgpu' — суффикс в имени zip."""
        def build(work_dir: Path) -> list[list[str]]:
            ver = (work_dir / "bm_latest.txt").read_text().strip()
            bm_id = "bm" + ver.replace(".", "")          # '0.7.4' → 'bm074'
            zip_name = f"linux/{bm_id}.linux_amd64{zip_flavor}.zip"
            return [
                ["unzip", "-o", "-q", zip_name, "-d", "."],
                ["cp", f"{btcmole_variant}/{bm_id}", "bm"],
                ["chmod", "+x", "bm"],
            ]
        return build

    BTCMOLE_URL = "https://github.com/keymole/btcmole"

    progs = [
        # ----------------- CPU -----------------
        ProgramSpec(
            id="cyclone",
            label=cyclone_label,
            devices={"cpu"},
            source_url="https://github.com/Dookoo2/Cyclone.git",
            build_subdir=cyclone_subdir,
            build_cmds=[cyclone_build],
            bin_relpath=f"{cyclone_subdir}/Cyclone",
            cwd_relpath=cyclone_subdir,
            make_argv=_argv_cyclone,
            cleanup_globs=["candidates.txt", "progress.txt"],
            speed_re=r"Mkeys/s\s*[:=]?\s*([\d.]+)",
        ),
        ProgramSpec(
            id="ecloop",
            label="ecloop",
            devices={"cpu"},
            source_url="https://github.com/vladkens/ecloop.git",
            build_cmds=[["make", "build"]],
            bin_relpath="ecloop",
            make_argv=_argv_ecloop,
            pre_run=_pre_ecloop,
            cleanup_globs=["found.txt", "target.h160"],
            extra_check_files=["found.txt"],
            speed_re=r"([\d.]+)\s*[MK]?keys?/s",
        ),
        ProgramSpec(
            id="keyhunt",
            label="keyhunt",
            devices={"cpu"},
            source_url="https://github.com/albertobsd/keyhunt.git",
            build_cmds=[["make"]],
            bin_relpath="keyhunt",
            make_argv=_argv_keyhunt,
            pre_run=_pre_keyhunt,
            cleanup_globs=["KEYFOUNDKEYFOUND.txt", "target.rmd"],
            extra_check_files=["KEYFOUNDKEYFOUND.txt"],
            speed_re=r"([\d.]+)\s*Mkeys/s",
        ),
        ProgramSpec(
            id="btcmole-cpu",
            label="btcmole (CPU)",
            devices={"cpu"},
            source_url=BTCMOLE_URL,
            build_cmds=btcmole_build_factory(""),
            bin_relpath="bm",
            make_argv=_argv_btcmole,
            cleanup_globs=["FOUND_KEY_*.txt", "*.state", "*.map"],
            extra_check_files=[],  # имя файла зависит от адреса; добавим динамически
        ),
        # ----------------- CUDA -----------------
        ProgramSpec(
            id="cudacyclone",
            label="CUDACyclone",
            devices={"cuda"},
            source_url="https://github.com/Dookoo2/CUDACyclone.git",
            build_cmds=[["make", "-j"]],
            bin_relpath="CUDACyclone",
            make_argv=_argv_cudacyclone,
            cleanup_globs=["found_keys.txt", "cyclone_tests_results.txt"],
            speed_re=r"([\d.]+)\s*[MGK]keys/s",
        ),
        ProgramSpec(
            id="keyhunt-cuda",
            label="KeyHunt-Cuda",
            devices={"cuda"},
            source_url="https://github.com/Qalander/KeyHunt-Cuda.git",
            build_subdir="KeyHunt-Cuda",
            build_cmds=[["make", "gpu=1", f"CCAP={ccap}", f"CUDA={cuda_dir}", "all", "-j"]],
            bin_relpath="KeyHunt-Cuda/KeyHunt",
            cwd_relpath="KeyHunt-Cuda",
            make_argv=_argv_keyhunt_cuda,
            cleanup_globs=["Found.txt"],
            extra_check_files=["Found.txt"],
            speed_re=r"([\d.]+)\s*Mk/s",
            patch_file="keyhunt-cuda.patch",
        ),
        ProgramSpec(
            id="keykiller-cuda",
            label="KeyKiller-Cuda",
            devices={"cuda"},
            source_url="https://gitlab.com/8891689/keykiller-cuda.git",
            build_cmds=[["make", f"CUDA={cuda_dir}", "all", "-j"]],
            bin_relpath="kk",
            make_argv=_argv_keykiller,
            cleanup_globs=["found.txt"],
            extra_check_files=["found.txt"],
            speed_re=r"([\d.]+)\s*Mkey/s",
            patch_file="keykiller-cuda.patch",
        ),
        ProgramSpec(
            id="keyscanner",
            label="KeyScanner",
            devices={"cuda"},
            source_url="https://github.com/graffitilogic/KeyScanner.git",
            # Upstream KeyScanner не содержит Linux Makefile (только msvc/) — копируем
            # самописанный Makefile из patches/ перед сборкой.
            build_cmds=[
                ["cp", str(PATCHES_DIR / "keyscanner_Makefile"), "Makefile"],
                ["make", f"CCAP={ccap}", f"CUDA={cuda_dir}", "-j"],
            ],
            bin_relpath="keyscanner",
            make_argv=_argv_keyscanner,
            cleanup_globs=["found.txt"],
            extra_check_files=["found.txt"],
            speed_re=r"([\d.]+)\s*[GMK]k/s",
            patch_file="keyscanner.patch",
        ),
        ProgramSpec(
            id="bitcrack-cuda",
            label="BitCrack (CUDA)",
            devices={"cuda"},
            source_url="https://github.com/brichard19/BitCrack.git",
            # Patch выставляет CUDA_HOME ?= /opt/cuda и -std=c++17. COMPUTE_CAP=
            # обязательно передавать под целевую карту, иначе "invalid device symbol"
            # на runtime (kernel скомпилирован под чужой sm_*).
            build_cmds=[["make", "BUILD_CUDA=1",
                         f"CUDA_HOME={cuda_dir}",
                         f"COMPUTE_CAP={ccap}",
                         "-j"]],
            bin_relpath="bin/cuBitCrack",
            cwd_relpath="bin",
            make_argv=_argv_bitcrack,
            cleanup_globs=["found.txt"],
            extra_check_files=["found.txt"],
            speed_re=r"([\d.]+)\s*M?Key/s",
            patch_file="bitcrack.patch",
        ),
        ProgramSpec(
            id="btcmole-cuda",
            label="btcmole (CUDA)",
            devices={"cuda"},
            source_url=BTCMOLE_URL,
            shared_clone_with="btcmole-cpu",
            build_cmds=btcmole_build_factory("_cuda"),
            bin_relpath="bm",
            make_argv=_argv_btcmole,
            cleanup_globs=["FOUND_KEY_*.txt", "*.state", "*.map"],
        ),
        # ----------------- AMDGPU -----------------
        ProgramSpec(
            id="bitcrack-opencl",
            label="BitCrack (OpenCL)",
            devices={"amdgpu"},
            source_url="https://github.com/brichard19/BitCrack.git",
            build_cmds=[["make", "BUILD_OPENCL=1",
                         f"CUDA_HOME={cuda_dir}",
                         f"COMPUTE_CAP={ccap}",
                         "-j"]],
            bin_relpath="bin/clBitCrack",
            cwd_relpath="bin",
            make_argv=_argv_bitcrack,
            cleanup_globs=["found.txt"],
            extra_check_files=["found.txt"],
            speed_re=r"([\d.]+)\s*M?Key/s",
            patch_file="bitcrack.patch",
        ),
        ProgramSpec(
            id="btcmole-amdgpu",
            label="btcmole (AMDGPU)",
            devices={"amdgpu"},
            source_url=BTCMOLE_URL,
            shared_clone_with="btcmole-cpu",
            build_cmds=btcmole_build_factory("_amdgpu"),
            bin_relpath="bm",
            make_argv=_argv_btcmole,
            cleanup_globs=["FOUND_KEY_*.txt", "*.state", "*.map"],
        ),
    ]
    return progs


# ---------------------------------------------------------------------------
#  Подготовка: clone / patch / build
# ---------------------------------------------------------------------------
def _run(cmd: list[str], cwd: Path, log_path: Path) -> int:
    """_run выполняет команду, выводит её и пишет stdout+stderr в log_path."""
    print(f"  $ {' '.join(cmd)}  (cwd={cwd})", flush=True)
    with open(log_path, "ab") as logf:
        logf.write(f"\n=== {cmd} (cwd={cwd}) ===\n".encode())
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode


def prepare_program(p: ProgramSpec, rebuild: bool = False) -> bool:
    """prepare_program клонирует, патчит и собирает программу. True при успехе."""
    work = WORK_DIR / p.id
    bin_path = work / p.bin_relpath
    log = LOGS_DIR / f"{p.id}_build.log"
    log.unlink(missing_ok=True)

    if bin_path.exists() and not rebuild:
        print(f"[{p.id}] бинарник уже собран: {bin_path}")
        return True

    # Клонирование (idempotent — если уже есть, пропускаем).
    # Если задан shared_clone_with — переиспользуем уже существующий clone
    # другой spec'и (например, все btcmole-* варианты делят один clone github
    # и отличаются только тем, какой zip распаковывают).
    if not work.exists():
        if p.shared_clone_with:
            shared = WORK_DIR / p.shared_clone_with
            if not shared.exists():
                print(f"[{p.id}] FAIL: shared_clone_with={p.shared_clone_with}, но {shared} ещё не существует. Готовь {p.shared_clone_with} ПЕРЕД {p.id}.")
                return False
            print(f"[{p.id}] reuse clone из {shared} -> {work}")
            rc = _run(["cp", "-r", str(shared), str(work)], cwd=WORK_DIR, log_path=log)
            if rc != 0:
                print(f"[{p.id}] FAIL: cp -r (см. {log})")
                return False
        else:
            print(f"[{p.id}] git clone {p.source_url} -> {work}")
            rc = _run(["git", "clone", "--depth", "1", p.source_url, str(work)], cwd=WORK_DIR, log_path=log)
            if rc != 0:
                print(f"[{p.id}] FAIL: git clone (см. {log})")
                return False

    # Патч (если есть)
    if p.patch_file:
        patch = PATCHES_DIR / p.patch_file
        if patch.exists():
            print(f"[{p.id}] git apply {patch}")
            rc = _run(["git", "apply", "--whitespace=nowarn", str(patch)], cwd=work, log_path=log)
            if rc != 0:
                print(f"[{p.id}] FAIL: git apply (см. {log})")
                return False

    # Сборка
    build_cwd = work / p.build_subdir if p.build_subdir else work
    cmds = p.build_cmds(build_cwd) if callable(p.build_cmds) else p.build_cmds
    for cmd in cmds:
        rc = _run(cmd, cwd=build_cwd, log_path=log)
        if rc != 0:
            print(f"[{p.id}] FAIL: build (см. {log})")
            return False

    if not bin_path.exists():
        print(f"[{p.id}] FAIL: бинарник {bin_path} не появился после сборки (см. {log})")
        return False

    print(f"[{p.id}] OK: {bin_path}")
    return True


# ---------------------------------------------------------------------------
#  Запуск одной итерации
# ---------------------------------------------------------------------------
HEX_RE = re.compile(r"[0-9A-Fa-f]{8,}")


def verify_found(case: BenchCase, texts: list[str]) -> bool:
    """verify_found ищет в списке текстов hex-токены и сравнивает их (как int)
    с ожидаемым priv_int. Возвращает True если ключ найден хоть где-то."""
    target = case.priv_int
    for t in texts:
        for tok in HEX_RE.findall(t):
            try:
                if int(tok, 16) == target:
                    return True
            except ValueError:
                pass
    return False


@dataclass
class IterResult:
    """IterResult — результат одной итерации одной программы."""
    case_idx: int
    elapsed_s: float
    ok: bool
    note: str = ""
    speed_mkeys: Optional[float] = None


def _has_priv(text: str, target: int) -> bool:
    """_has_priv ищет в тексте hex-токен, равный target по значению."""
    for tok in HEX_RE.findall(text):
        try:
            if int(tok, 16) == target:
                return True
        except ValueError:
            pass
    return False


def _btcmole_file(case: BenchCase) -> str:
    """_btcmole_file возвращает имя FOUND_KEY-файла, в который btcmole пишет результат."""
    return f"FOUND_KEY_{case.address}.txt"


def run_iteration(p: ProgramSpec, case: BenchCase, dev: str) -> IterResult:
    """run_iteration запускает программу на одном кейсе.
    Останавливает процесс как только в stdout/файлах появился ожидаемый ключ
    (некоторые программы не выходят сами после found — ждут остальные потоки).
    """
    work = WORK_DIR / p.id
    cwd = work / (p.cwd_relpath or p.build_subdir)
    log_path = LOGS_DIR / f"{p.id}_{dev}_{case.idx:02d}.log"

    # Очистка артефактов прошлой итерации
    for pat in p.cleanup_globs:
        for f in cwd.glob(pat):
            try: f.unlink()
            except OSError: pass

    if p.pre_run:
        p.pre_run(case, cwd, dev)

    argv = p.make_argv(case, cwd, dev)
    print(f"  [{p.id}/iter{case.idx}] {' '.join(argv)}", flush=True)

    log_f = open(log_path, "w", buffering=1)
    # NB: НЕ пишем expected priv в header — наш verify_found ищет hex-токены и
    # засчитал бы наш же header как «найденный ключ» (был баг с BitCrack).
    log_f.write(
        f"=== argv: {argv}\n=== cwd: {cwd}\n=== address: {case.address}\n\n"
    )

    # Список файлов, которые тоже сканируем на match (помимо stdout)
    extra_files = list(p.extra_check_files)
    if p.id.startswith("btcmole"):
        extra_files.append(_btcmole_file(case))

    target = case.priv_int
    found_event = threading.Event()
    recent: list[str] = []          # последние строки stdout (для match)
    RECENT_MAX = 200
    lock = threading.Lock()

    proc = subprocess.Popen(
        argv, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True, errors="replace",
    )

    def reader():
        """reader читает stdout процесса построчно, пишет в лог и сигналит found_event при match."""
        for line in proc.stdout:
            log_f.write(line)
            if _has_priv(line, target):
                found_event.set()
            else:
                with lock:
                    recent.append(line)
                    if len(recent) > RECENT_MAX:
                        del recent[: len(recent) - RECENT_MAX]
        # stdout закрылся — процесс завершается

    rt = threading.Thread(target=reader, daemon=True)
    rt.start()

    t0 = time.monotonic()
    deadline = t0 + p.timeout_s
    note = ""
    found = False
    POLL_S = 0.1

    while True:
        # 1) match в stdout (через found_event)
        if found_event.is_set():
            found = True
            break
        # 2) процесс сам завершился
        if proc.poll() is not None:
            # дочитываем хвост через reader (короткое ожидание)
            rt.join(timeout=2.0)
            with lock:
                if _has_priv("".join(recent), target):
                    found = True
            break
        # 3) match в output-файлах
        for fname in extra_files:
            fp = cwd / fname
            if fp.exists():
                try:
                    if _has_priv(fp.read_text(errors="replace"), target):
                        found = True
                        break
                except OSError:
                    pass
        if found:
            break
        # 4) timeout
        if time.monotonic() >= deadline:
            note = f"TIMEOUT>{p.timeout_s}s"
            break
        time.sleep(POLL_S)

    elapsed = time.monotonic() - t0

    # Грейсфул-стоп: TERM, потом KILL если не успел
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            try: proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired: pass
    rt.join(timeout=1.0)
    log_f.close()

    # Финальная sanity-проверка по логу + файлам (на случай, если match попал в файл уже после kill)
    texts: list[str] = []
    try: texts.append(log_path.read_text(errors="replace"))
    except OSError: pass
    for fname in extra_files:
        fp = cwd / fname
        if fp.exists():
            try: texts.append(fp.read_text(errors="replace"))
            except OSError: pass
    # При TIMEOUT сразу FAIL — даже если verify_found находит ключ post-mortem
    # (это значит программа не сообщила о нём в realtime, бенчмарк нечестный).
    if note.startswith("TIMEOUT"):
        final_ok = False
    else:
        final_ok = found or verify_found(case, texts)
        if not final_ok:
            note = "NOT_FOUND"

    speed = None
    if p.speed_re:
        for t in texts:
            for m in re.finditer(p.speed_re, t):
                try:
                    val = float(m.group(1))
                    speed = val if speed is None else max(speed, val)
                except ValueError:
                    pass

    return IterResult(case.idx, elapsed, ok=final_ok, note=note, speed_mkeys=speed)


# ---------------------------------------------------------------------------
#  Прогон одного устройства
# ---------------------------------------------------------------------------
@dataclass
class ProgramTotal:
    """ProgramTotal — итог по программе на одном устройстве."""
    spec: ProgramSpec
    iter_results: list[IterResult] = field(default_factory=list)

    @property
    def total_s(self) -> Optional[float]:
        oks = [r.elapsed_s for r in self.iter_results if r.ok]
        return sum(oks) if oks else None

    @property
    def n_ok(self) -> int:
        return sum(1 for r in self.iter_results if r.ok)

    @property
    def n_fail(self) -> int:
        return sum(1 for r in self.iter_results if not r.ok)


def run_device(dev: str, all_progs: list[ProgramSpec],
               cases_main: list[BenchCase], cases_warmup: list[BenchCase]) -> list[ProgramTotal]:
    """run_device прогоняет все программы устройства в interleaved-режиме:
    в каждом раунде поочерёдно (по алфавиту id) каждая программа делает одну итерацию.
    Это выравнивает thermal-условия — все программы получают равный набор «тёплых» и
    «горячих» прогонов. Перед main идут warmup-раунды (результаты не учитываются),
    чтобы CPU вышел в steady-state.
    """
    progs = sorted([p for p in all_progs if dev in p.devices], key=lambda p: p.id)
    progs = [p for p in progs if (WORK_DIR / p.id / p.bin_relpath).exists()]
    if not progs:
        return []

    print(f"\n=== {dev.upper()}: {len(progs)} программ × ({len(cases_warmup)} warmup + {len(cases_main)} main) ===")
    print(f"   порядок (алфавит): {', '.join(p.id for p in progs)}\n")

    # --- Warmup (не учитываем) ---
    if cases_warmup:
        print(f"--- Warmup ({len(cases_warmup)} раундов) ---")
        for case in cases_warmup:
            for p in progs:
                res = run_iteration(p, case, dev)
                mark = "OK " if res.ok else "FAIL"
                print(f"  [warmup r{case.idx} {p.id}]: {mark} {res.elapsed_s:7.2f}s  {res.note}")
        print()

    # --- Main (учитываем) ---
    totals: dict[str, ProgramTotal] = {p.id: ProgramTotal(spec=p) for p in progs}
    print(f"--- Main ({len(cases_main)} раундов) ---")
    for case in cases_main:
        for p in progs:
            res = run_iteration(p, case, dev)
            totals[p.id].iter_results.append(res)
            mark = "OK " if res.ok else "FAIL"
            print(f"  [main r{case.idx} {p.id}]: {mark} {res.elapsed_s:7.2f}s  {res.note}")
    return list(totals.values())


# ---------------------------------------------------------------------------
#  Markdown-отчёт
# ---------------------------------------------------------------------------
def render_report(dev: str, totals: list[ProgramTotal]) -> str:
    """render_report строит markdown-таблицу для одного устройства."""
    base_id = BASELINE.get(dev)
    base_total = next((t for t in totals if t.spec.id == base_id and t.total_s is not None), None)
    base_t = base_total.total_s if base_total else None

    bits = bits_for_device(dev)
    range_start, range_end = range_for_device(dev)
    lines = [
        f"# Бенчмарк {dev.upper()} — {dt.datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"- Range: `[{range_start:#x}, {range_end:#x}]` (size = 2^{bits})",
        f"- Итераций: {num_iterations_for_device(dev)}",
        f"- Эталон (Скорость = 1.00): **{base_id}**" + (f" — {base_t:.2f}s суммарно" if base_t else " — N/A"),
        "",
        "| Программа | OK / FAIL | Σ время (s) | Avg (s) | Скорость (× эталон) |",
        "|-----------|-----------|-------------|---------|---------------------|",
    ]

    # Сортировка: успешные сначала, по возрастанию total
    def sort_key(t: ProgramTotal):
        ts = t.total_s
        return (0 if ts is not None else 1, ts if ts is not None else 0)

    for t in sorted(totals, key=sort_key):
        if t.total_s is None:
            spd = "—"
            avg = "—"
            tot = "—"
        else:
            tot = f"{t.total_s:.2f}"
            avg = f"{t.total_s / max(t.n_ok, 1):.2f}"
            # Скорость относительно эталона: factor = t_base / t_program.
            # 1.00 = как у эталона; 1.20 = на 20% быстрее; 0.50 = вдвое медленнее.
            if base_t and t.total_s > 0:
                spd = f"{base_t / t.total_s:.2f}"
            else:
                spd = "—"
        lines.append(
            f"| {t.spec.label} | {t.n_ok} / {t.n_fail} | {tot} | {avg} | {spd} |"
        )

    # Заметки про FAIL
    fails = [(t.spec.label, r.case_idx, r.note)
             for t in totals for r in t.iter_results if not r.ok]
    if fails:
        lines += ["", "## Замечания (FAIL-итерации)", ""]
        for label, idx, note in fails:
            lines.append(f"- **{label}** iter {idx}: {note}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["cpu", "cuda", "amdgpu", "all"], default="all",
                    help="на каком классе устройств прогонять")
    ap.add_argument("--prepare", action="store_true",
                    help="только клонировать/собрать (без прогона)")
    ap.add_argument("--rebuild", action="store_true",
                    help="пересобрать даже если бинарник уже есть")
    ap.add_argument("--only", default="",
                    help="прогнать только указанные ID через запятую (cyclone,ecloop,...)")
    args = ap.parse_args()

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    all_progs = build_registry()
    if args.only:
        wanted = set(s.strip() for s in args.only.split(","))
        all_progs = [p for p in all_progs if p.id in wanted]

    devices = ["cpu", "cuda", "amdgpu"] if args.device == "all" else [args.device]
    progs_for_devices = [p for p in all_progs if p.devices & set(devices)]

    # Подготовка
    print(f"\n=== Подготовка ({len(progs_for_devices)} программ) ===\n")
    for p in progs_for_devices:
        prepare_program(p, rebuild=args.rebuild)

    if args.prepare:
        print("\n[--prepare] подготовка завершена, бенчмарк не запускается.")
        return

    # Кейсы — отдельный набор на каждое устройство (range зависит от dev).
    today = dt.date.today().isoformat()
    for dev in devices:
        bits = bits_for_device(dev)
        rs, re = range_for_device(dev)
        n_iter = num_iterations_for_device(dev)
        main_cases = generate_cases(SEED, n_iter, dev)
        warmup_cases = generate_cases(SEED ^ 0xDEAD, WARMUP_ROUNDS, dev)
        print(f"\n=== {dev.upper()}: range 2^{bits} = [{rs:#x}, {re:#x}], "
              f"{len(main_cases)} main + {len(warmup_cases)} warmup кейсов ===")
        for c in main_cases:
            print(f"  main  iter {c.idx}: priv={c.priv_hex64}  addr={c.address}")
        for c in warmup_cases:
            print(f"  warm  iter {c.idx}: priv={c.priv_hex64}  addr={c.address}")
        totals = run_device(dev, all_progs, main_cases, warmup_cases)
        if not totals:
            continue
        report = render_report(dev, totals)
        out = RESULTS_DIR / f"{today}_{dev}.md"
        out.write_text(report)
        print(f"\n=== Отчёт {dev.upper()} → {out} ===\n{report}")


if __name__ == "__main__":
    main()
