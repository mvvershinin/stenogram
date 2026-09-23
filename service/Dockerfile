# syntax=docker/dockerfile:1
#
# Один образ — API и воркер запускаются из него разными командами (см. `command:`
# у каждого сервиса в docker-compose.yml). Собирать, публиковать и патчить их порознь
# незачем: оба ставят один и тот же пакет и стартуют за секунды.
#
# Контекст сборки — КОРЕНЬ репозитория (Dockerfile лежит в service/, но пакету
# stenogram_service нужен пакет stenogram рядом): docker-compose.yml собирает так
#
#     build: {context: .., dockerfile: service/Dockerfile}
#
# Модели НЕ вшиваются в образ — тома:
#   STENOGRAM_HOME        каталог ONNX-моделей разметки (пакет stenogram качает сам)
#   STENOGRAM_WHISPER_DIR download_root модели распознавания (настройка воркера)
# Обе точки — read-only для сервисов, см. docker-compose.yml и service/README.md.

FROM python:3.12-slim AS base

# ffmpeg — системным пакетом, а не через static-ffmpeg (зависимость stenogram): сервис
# обязан стартовать без единого сетевого запроса (docker-compose.yml держит сервисы
# с моделями в internal-сети без доступа наружу), а static-ffmpeg тянет статический
# бинарник с GitHub при первом вызове. apt делает это один раз при сборке образа —
# к рантайму сеть уже не нужна.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала только метаданные пакетов — слой с зависимостями кэшируется, пока не
# меняются pyproject.toml; исходники правятся значительно чаще зависимостей.
COPY pyproject.toml README.md ./
COPY service/pyproject.toml ./service/pyproject.toml
COPY stenogram ./stenogram
COPY service/stenogram_service ./service/stenogram_service

# Корень — обычная зависимость сервиса (имени «stenogram» на PyPI нет, ставится из
# исходников рядом). Без dev-экстры: ruff, pytest, httpx в образе не нужны.
RUN pip install --no-cache-dir . ./service

# Непривилегированный пользователь с фиксированным UID/GID 1000 — чтобы владелец
# именованных томов (/data), которые Docker засевает содержимым этого каталога из
# образа, был предсказуем. На классическом (не rootless) Docker это совпадает с
# типичным первым пользователем на однопользовательской машине. На rootless Docker —
# ЭТОТ хост именно такой — UID контейнера переписывается через user namespace
# (subuid), и «UID 1000 внутри» на хосте оказывается другим числом: это и есть
# причина `Permission denied` на файлах с host-модом 600/700 под смонтированным
# каталогом моделей (см. service/README.md, «Оговорки») — сам /data это не задевает,
# он в именованном томе и заводится под UID контейнера с нуля.
RUN groupadd -g 1000 stenogram && useradd -u 1000 -g stenogram -m -d /home/stenogram stenogram \
    && mkdir -p /data && chown stenogram:stenogram /data
USER stenogram
ENV HOME=/home/stenogram

# Умолчание — воркер; сервис api переопределяет command в docker-compose.yml.
CMD ["python", "-m", "stenogram_service.worker"]


# --- Видеокарта: тот же образ, плюс CUDA-библиотеки и путь к ним -----------------
#
# cuBLAS 12 и cuDNN 9 транзитивно с ctranslate2/faster-whisper не приходят (проверено
# здесь: `pip list` в базовом образе на `nvidia-*` пуст) — ставим явно, тем же путём,
# что на машине с картой (см. память проекта про GPU-машину: там это отдельные пакеты
# venv, а не то, что тянет faster-whisper сам). Второй RUN, отдельным слоем — CPU-стадия
# эти ~1.4 ГБ (581 МБ cuBLAS + 770 МБ cuDNN + 90 МБ nvidia-cuda-nvrtc, замер сборки)
# не платит. Отдельного base-образа nvidia/cuda всё равно не нужно: нужен только
# драйвер на ХОСТЕ и nvidia-container-toolkit (`docker run --gpus all` /
# `deploy.resources.reservations.devices` в docker-compose.yml) — libcuda.so.1
# приезжает из драйвера, а не из образа/pip.
#
# Без этого блока faster-whisper тихо не находит свои CUDA-библиотеки:
# `ctranslate2.get_cuda_device_count()` дёргает только драйвер и отвечает 1 даже без
# LD_LIBRARY_PATH на cuBLAS/cuDNN, а отказ приходит лишь на первом вызове энкодера —
# см. `cuda_preflight` в stenogram_service/worker.py, который эту гонку ловит на старте.
#
# Путь захардкожен под python:3.12-slim (`/usr/local/lib/python3.12/site-packages`) —
# это единственный python в образе, ставится этим же Dockerfile, так что путь
# детерминирован. Смена базового образа/версии Python требует поправить путь тут же.
#
# Собирается выбором стадии: `docker build --target cuda -f service/Dockerfile ..`;
# в docker-compose.yml — сервис worker-cuda (профиль gpu). НЕ ПРОВЕРЕНО НА ЖЕЛЕЗЕ:
# на этой машине видеокарты нет, стадия собрана и проверена только на компилируемость
# и на то, что worker.cuda_preflight падает понятным отказом без настоящей CUDA —
# честная приёмка на карте осталась в docs/ускорение-на-gpu.md как открытый пункт.
FROM base AS cuda
USER root
# Версии не пины `==`, а диапазоны major: cuBLAS/cuDNN здесь — рантайм-библиотеки
# под конкретную установленную CUDA/cuDNN на карте, а не то, что резолвится вместе
# с ctranslate2 (см. предупреждение выше) — жёсткий пин молчаливо бы устарел.
RUN pip install --no-cache-dir "nvidia-cublas-cu12>=12,<13" "nvidia-cudnn-cu12>=9,<10"
USER stenogram
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib
CMD ["python", "-m", "stenogram_service.worker"]
