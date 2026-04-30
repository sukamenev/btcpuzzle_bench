# Relative-speed benchmark for BTC puzzle solvers (puzzle 71 and beyond)

Сравнительный бенчмарк программ для перебора приватных ключей биткойн-головоломок (puzzle 1..160). Все участники запускаются на **одной и той же задаче** (один и тот же диапазон, один и тот же случайный ключ-цель), время фиксируется от старта процесса до выхода с найденным ключом — никаких внутренних счётчиков `Mkey/s`, которым нельзя верить.

Подробное описание методики и таблицы результатов — в статье на Хабре: https://habr.com/ru/companies/ruvds/articles/1029952/

## Что измеряется

- **Wall-clock** от запуска процесса до того момента, когда программа сама напечатала найденный приватный ключ в нужном формате.
- 100 итераций на каждую программу — стандартная ошибка среднего ≈ σ/√100 ≈ σ/10.
- Один и тот же `seed = 42` фиксирует набор тестовых ключей между прогонами, чтобы результаты были воспроизводимыми.

## Какие программы

| Класс  | Программы                                                                  |
|--------|----------------------------------------------------------------------------|
| CPU    | Cyclone (AVX-2), ecloop, KeyHunt, btcmole-cpu                              |
| CUDA   | CUDACyclone, KeyHunt-Cuda, KeyKiller-Cuda, KeyScanner, BitCrack (CUDA), btcmole-cuda |
| AMDGPU | BitCrack (OpenCL), btcmole-amdgpu                                          |

URL'ы исходников — в `bench.py` (поле `source_url` для каждой записи `ProgramSpec`).

## Установка

```bash
git clone https://github.com/inetstar/btcpuzzle_bench
cd btcpuzzle_bench

# Виртуальное окружение (опционально, если в системе нет нужных пакетов)
python3 -m venv .venv && source .venv/bin/activate

# Зависимости (coincurve + base58)
pip install -r requirements.txt

# Подготовка: клонирование всех репозиториев, применение патчей, сборка бинарников
./bench.py --prepare --device all
```

`--prepare` выполняет `git clone --depth 1 <url>` в `work/<id>/`, накатывает соответствующий патч из `patches/<id>.patch` (если есть), запускает `make` или иной build-step. Логи каждого шага кладутся в `logs/<id>_prepare.log`. На свежем CUDA/ROCm большая часть программ собирается без вмешательства, но 4 из обзора **требуют патчей** — лежат в `patches/`.

## Запуск

```bash
./bench.py --device cpu             # бенчмарк CPU
./bench.py --device cuda            # бенчмарк CUDA
./bench.py --device amdgpu          # бенчмарк AMDGPU
./bench.py --device all             # всё подряд

# Запуск только определённых программ
./bench.py --device cuda --only cudacyclone,btcmole-cuda

# Принудительная пересборка перед прогоном
./bench.py --device cuda --rebuild --only btcmole-cuda
```

Результат — Markdown-отчёт в `results/<YYYY-MM-DD>_<device>.md` с таблицей: программа / OK-FAIL / суммарное время / среднее / скорость относительно эталона.

## Конфигурация

Все важные константы — в верхнем блоке `bench.py` между двумя комментариями `===`:

| Константа                        | Что регулирует |
|----------------------------------|----------------|
| `NUM_ITERATIONS_PER_DEVICE`      | число прогонов на программу для CPU/CUDA/AMDGPU (по умолчанию 100/100/4) |
| `WARMUP_ROUNDS`                  | сколько прогонов на разогрев перед основными замерами |
| `INTERVAL_BITS_PER_DEVICE`       | размер диапазона поиска в битах (CPU=33, CUDA=35, AMDGPU=30) |
| `BENCH_TIMEOUT`                  | тайм-аут на одну итерацию, сек |
| `SEED`                           | seed PRNG'а для генерации ключей-целей |
| `BASELINE`                       | программа-эталон (×1.00 в отчёте) для каждого класса устройств |
| `BENCH_CPU_THREADS`              | сколько потоков отдавать CPU-программам (по умолчанию ~90% от `os.cpu_count()`) |

Любая из них переопределяется одноимённой переменной окружения с префиксом `BENCH_`:

```bash
BENCH_N_CUDA=20 BENCH_TIMEOUT=300 ./bench.py --device cuda
```

## Стенд (на котором сняты результаты в `results/`)

- **CPU**: AMD EPYC 7C13 (64 ядра / 128 потоков, фиксированный TDP без турбо-буста, base-clock 2.45 GHz)
- **GPU**: NVIDIA CMP 90HX (sm_86, 50 SM, 200 W power-limit, 1530 MHz фиксированно)
- **AMDGPU**: AMD Radeon R9 Fury (gfx803, 56 CU)
- **OS**: Linux (Calculate Linux)

## Патчи

Часть программ из обзора без правок не собирается на свежем GCC/CUDA/ROCm. В `patches/` лежат минимальные правки (Makefile-фиксы, замены deprecated CUDA API, исправления include'ов):

- `bitcrack.patch` — fix сборки на современном NVCC
- `keyhunt-cuda.patch` — поддержка свежих NVCC + sm_86/89
- `keykiller-cuda.patch` — учёт CPU-времени в счётчике (был только GPU-time, искажало внутренние Mkey/s) + правка вывода для парсинга
- `keyscanner.patch`, `keyscanner_Makefile` — Makefile полностью переписан, апстрим был сломан

## Структура репозитория

```
btcpuzzle_bench/
├── bench.py             — основной скрипт
├── README.md            — этот файл
├── patches/             — патчи под сборку для конкретных программ
├── work/                — клонированные исходники (создаётся скриптом)
├── logs/                — логи сборки и прогонов (создаётся скриптом)
└── results/             — Markdown-отчёты с таблицами
```

## Лицензия

MIT — см. `LICENSE`.

## Contributing

PR с новыми программами для перебора, патчами под свежие версии toolchain'а или результатами на другом железе — приветствуются. Новую программу добавлять в `register_all_programs()` с заполнением `id`, `source_url`, `make_argv`, `success_re` и т.д. (см. существующие записи как образец).
