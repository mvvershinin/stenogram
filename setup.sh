#!/usr/bin/env bash
# Разворачивает окружение: venv, зависимости, модели диаризации.
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"

echo "==> Создаю виртуальное окружение ./venv"
[ -d venv ] || "$PYTHON" -m venv venv

echo "==> Ставлю зависимости"
./venv/bin/pip install --upgrade pip --quiet
# Напрямую из pyproject.toml (единственный источник правды про версии), не через
# requirements.txt: строка "-e ." в requirements.txt резолвится pip относительно
# текущего каталога вызывающего, а не относительно самого файла, — путь на такое
# наткнулся ровно там, где это незаметно (см. .docs/plans/2026-08-25_публикация.md).
# Здесь безопасно всегда: скрипт уже сделал cd в свой каталог выше.
./venv/bin/pip install -e . --quiet

echo "==> Проверяю, что PyTorch не пришёл транзитивно (решение 0004)"
./venv/bin/pip show torch >/dev/null 2>&1 && { echo "PyTorch в venv — нарушено .docs/decisions/0004-без-pytorch.md" >&2; exit 1; } || true

echo "==> Проверяю ffmpeg"
./venv/bin/python -c "from stenogram.audio import find_ffmpeg; print('   ', *find_ffmpeg())"

echo "==> Скачиваю модели диаризации (~35 МБ, один раз)"
./venv/bin/python -c "from stenogram.models import ensure_all; print('   ', *ensure_all())"

cat <<'MSG'

Готово. Запуск:

  ./stenogram.sh запись.mp4
  ./stenogram.sh запись.mp4 --speakers 8 --names "Судья,Эксперт,Истец"

Модель распознавания (~1.6 ГБ) скачается при первом запуске.
MSG
