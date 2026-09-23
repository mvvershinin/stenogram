"""Utterance[] → blocks.json: формат, который сервис отдаёт наружу."""

from __future__ import annotations

import json

import numpy as np

from stenogram.align import Utterance
from stenogram.asr import Word
from stenogram_service import blocks

SHA = "ab" * 32


def _utterances() -> list[Utterance]:
    return [
        Utterance(0.5, 3.25, 0, "Добрый день, коллеги.", [Word(0.5, 0.9, " Добрый")]),
        Utterance(np.float64(3.8), np.float32(6.1), np.int64(1), "Начнём с первого вопроса."),
    ]


def test_build_maps_utterances_to_timecode_blocks():
    doc = blocks.build(_utterances(), SHA)

    assert doc == {
        "schema": "blocks/1",
        "doc_id": SHA,
        "source_uri": None,
        "speakers": {},
        "blocks": [
            {"i": 0, "text": "Добрый день, коллеги.", "locator_kind": "timecode", "locator": {"start": 0.5, "end": 3.25, "speaker_id": 0}},
            {"i": 1, "text": "Начнём с первого вопроса.", "locator_kind": "timecode", "locator": {"start": 3.8, "end": 6.1, "speaker_id": 1}},
        ],
    }
    # numpy-числа из пайплайна не ломают сериализацию и не превращаются в 6.099999904632568
    assert json.loads(json.dumps(doc)) == doc


def test_names_go_to_table_not_into_text():
    cached = blocks.build(_utterances(), SHA)

    doc = blocks.for_job(cached, source_uri="s3://records/x.mp4", names={1: "Иванов", 0: "Председатель"})

    assert doc["source_uri"] == "s3://records/x.mp4"
    assert doc["speakers"] == {"0": "Председатель", "1": "Иванов"}
    assert all("Иванов" not in b["text"] for b in doc["blocks"])
    assert doc["blocks"] == cached["blocks"]
    assert cached["source_uri"] is None and cached["speakers"] == {}  # копия из кэша не тронута


def test_empty_names_and_empty_recording():
    doc = blocks.for_job(blocks.build([], SHA), source_uri="s3://records/x.mp4")

    assert doc["speakers"] == {}
    assert doc["blocks"] == []
