#!/usr/bin/env bash
# Разворачивает окружение: venv, зависимости, модели диаризации.
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"

echo "==> Создаю виртуальное окружение ./venv"
[ -d venv ] || "$PYTHON" -m venv venv

echo "==> Ставлю зависимости"
./venv/bin/pip install --upgrade pip --quiet
./venv/bin/pip install -r requirements.txt --quiet

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
