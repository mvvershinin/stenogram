#!/usr/bin/env bash
# Отправляет запись в stenogram-service и кладёт стенограмму рядом с ней:
#
#   service/transcribe.sh запись.mp3 ['{"language": "ru", "speakers": 4}']
#
# Второй аргумент — параметры задачи из POST /jobs. Адрес и ключи — в
# ~/.config/stenogram/remote.env: STENOGRAM_HOST, MINIO_ROOT_USER, MINIO_ROOT_PASSWORD.
set -euo pipefail

source "${STENOGRAM_REMOTE_ENV:-$HOME/.config/stenogram/remote.env}"

file=$(realpath "$1")
params=${2:-'{"language": "ru"}'}
api="http://$STENOGRAM_HOST:${STENOGRAM_API_PORT:-8000}"
s3="http://$STENOGRAM_HOST:${STENOGRAM_S3_PORT:-9000}"

s3curl() {
    curl -fsS --aws-sigv4 aws:amz:us-east-1:s3 --user "$MINIO_ROOT_USER:$MINIO_ROOT_PASSWORD" \
        -H 'x-amz-content-sha256: UNSIGNED-PAYLOAD' "$@"
}

if ! curl -fsS -m 30 -o /dev/null "$api/healthz"; then
    echo "Сервис на $STENOGRAM_HOST не отвечает: машина выключена или загружена в Windows" >&2
    exit 1
fi

# Ключ по содержимому: кириллица в адресе ломает подпись запроса, а повторная
# заливка того же файла ничего не меняет.
key="uploads/$(sha256sum "$file" | cut -d' ' -f1).${file##*.}"
echo "Заливаю $(basename "$file") ..."
s3curl -T "$file" "$s3/stenogram/$key"

job=$(jq -n --arg source "s3://stenogram/$key" --argjson params "$params" '$params + {source: $source}' |
    curl -fsS -H 'content-type: application/json' -d @- "$api/jobs" | jq -r .job_id)
echo "Задача $job"

last=""
while :; do
    manifest=$(curl -fsS "$api/jobs/$job")
    status=$(jq -r .status <<<"$manifest")
    case $status in
        done) break ;;
        failed)
            jq .error <<<"$manifest" >&2
            exit 1
            ;;
    esac
    line=$(jq -r '[.status, .stage // "", (if .fraction then "\(.fraction * 100 | floor)%" else "" end)] | join(" ")' <<<"$manifest")
    if [[ $line != "$last" ]]; then
        echo "  $line"
        last=$line
    fi
    sleep 5
done

stem=${file%.*}
s3curl -o "$stem.txt" "$s3/$(jq -r '.result.transcript | ltrimstr("s3://")' <<<"$manifest")"
s3curl -o "$stem.blocks.json" "$s3/$(jq -r '.result.blocks | ltrimstr("s3://")' <<<"$manifest")"
jq -c '{quality, timing, cache}' <<<"$manifest"
echo "Готово: $stem.txt"
