#!/usr/bin/env bash
# Отправляет запись в stenogram-service и кладёт стенограмму рядом с ней:
#
#   service/transcribe.sh запись.mp3 ['{"language": "ru", "speakers": 4}']
#
# Второй аргумент — параметры задачи из POST /jobs. Настройки — в ~/.config/stenogram/remote.env:
# STENOGRAM_API, STENOGRAM_S3_URL, STENOGRAM_S3_ACCESS_KEY, STENOGRAM_S3_SECRET_KEY и, если
# отличаются от умолчаний, STENOGRAM_S3_BUCKET и STENOGRAM_S3_REGION.
set -euo pipefail

source "${STENOGRAM_REMOTE_ENV:-$HOME/.config/stenogram/remote.env}"

file=$(realpath "$1")
params=${2:-'{"language": "ru"}'}
api=$STENOGRAM_API
s3=$STENOGRAM_S3_URL
bucket=${STENOGRAM_S3_BUCKET:-stenogram}
region=${STENOGRAM_S3_REGION:-us-east-1}

s3curl() {
    curl -fsS --aws-sigv4 "aws:amz:$region:s3" --user "$STENOGRAM_S3_ACCESS_KEY:$STENOGRAM_S3_SECRET_KEY" \
        -H 'x-amz-content-sha256: UNSIGNED-PAYLOAD' "$@"
}

if ! curl -fsS -m 30 -o /dev/null "$api/healthz"; then
    echo "Сервис $api не отвечает: машина выключена или сервис не запущен" >&2
    exit 1
fi

# Ключ по содержимому: кириллица в адресе ломает подпись запроса, а повторная
# заливка того же файла ничего не меняет.
key="uploads/$(sha256sum "$file" | cut -d' ' -f1).${file##*.}"
echo "Заливаю $(basename "$file") ..."
s3curl -T "$file" "$s3/$bucket/$key"

job=$(jq -n --arg source "s3://$bucket/$key" --argjson params "$params" '$params + {source: $source}' |
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
