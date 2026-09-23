"""Реплики stenogram → документ `blocks.json`, общий формат декодеров RAG-конвейера.

Наружу отдаётся он, а не схема `Utterance`: иначе единый контракт декодеров
разваливается на первой интеграции. Имена в текст блока не вшиваются — `speakers`
отдельной таблицей, и переименование участника не меняет ни текста, ни эмбеддингов.

`doc_id` — sha256 исходного файла, а не ключ результата: перезапуск того же файла
с другими `speakers` — новая версия того же документа, и в индексе она должна
заменить старую, а не лечь рядом дублем.
"""

from __future__ import annotations

from typing import Any

from stenogram.align import Utterance

SCHEMA = "blocks/1"


def build(utterances: list[Utterance], doc_id: str) -> dict[str, Any]:
    """Обезличенный документ — такой кладётся в кэш. Адрес и имена дописывает `for_job`."""
    return {
        "schema": SCHEMA,
        "doc_id": doc_id,
        "source_uri": None,
        "speakers": {},
        "blocks": [
            {
                "i": i,
                "text": u.text,
                "locator_kind": "timecode",
                "locator": {"start": round(float(u.start), 3), "end": round(float(u.end), 3), "speaker_id": int(u.speaker)},
            }
            for i, u in enumerate(utterances)
        ],
    }


def for_job(document: dict[str, Any], *, source_uri: str, names: dict[int, str] | None = None) -> dict[str, Any]:
    """Копия документа из кэша с адресом исходника и таблицей имён задачи.

    `names` — по номеру участника с нуля, как `speaker_id` в блоках."""
    return {**document, "source_uri": source_uri, "speakers": {str(k): v for k, v in sorted((names or {}).items())}}
