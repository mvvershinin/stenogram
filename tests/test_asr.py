"""Тесты asr без настоящей модели: загрузка, переиспользование, прогресс.

faster_whisper подменяется в sys.modules — тест не тянет ни CTranslate2, ни модель.
Проверяется обвязка, а не распознавание:
  - уже загруженная модель используется как есть: воркер не должен перечитывать
    полтора гигабайта на каждую задачу;
  - local_files_only и download_root доходят до WhisperModel — иначе запретить
    поход в сеть на машине без интернета нельзя;
  - progress-функция получает те же строки, что печатаются в stdout, и долю пройденного.
"""

from __future__ import annotations

import sys
import types

import pytest

from stenogram import asr


class _FakeModel:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def transcribe(self, wav, **kwargs):
        items = [
            types.SimpleNamespace(start=float(i), end=float(i + 1), text=f" фраза {i}", words=[types.SimpleNamespace(start=float(i), end=i + 0.5, word=" фраза")])
            for i in range(50)
        ]
        info = types.SimpleNamespace(language="ru", language_probability=0.99, duration=50.0)
        return iter(items), info


@pytest.fixture
def fake_whisper(monkeypatch):
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = _FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return module


def test_load_model_passes_offline_switches(fake_whisper):
    model = asr.load_model("small", local_files_only=True, download_root="/opt/models")

    assert model.args == ("small",)
    assert model.kwargs["local_files_only"] is True
    assert model.kwargs["download_root"] == "/opt/models"


def test_transcribe_uses_given_model_without_loading(fake_whisper, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("модель передана — загружать заново нельзя")

    monkeypatch.setattr(asr, "load_model", boom)

    segments, language = asr.transcribe("audio.wav", model=_FakeModel(), progress=False)

    assert language == "ru"
    assert len(segments) == 50
    assert segments[0].text == "фраза 0"


def test_transcribe_loads_model_with_offline_switches(fake_whisper, monkeypatch):
    seen = {}

    def fake_load(*args):
        seen["args"] = args
        return _FakeModel()

    monkeypatch.setattr(asr, "load_model", fake_load)

    asr.transcribe("audio.wav", model_name="small", progress=False, local_files_only=True, download_root="/opt/models")

    assert seen["args"][0] == "small"
    assert seen["args"][-2:] == (True, "/opt/models")


def test_progress_function_gets_messages_and_fractions_instead_of_stdout(fake_whisper, capsys):
    got: list[tuple[str, float | None]] = []

    asr.transcribe("audio.wav", model=_FakeModel(), progress=lambda message, fraction: got.append((message, fraction)))

    assert capsys.readouterr().out == ""
    assert got[0][0].startswith("  язык: ru")
    fractions = [f for _, f in got[1:]]
    assert fractions == [25 / 50, 50 / 50]  # строка прогресса — на каждый 25-й сегмент


def test_progress_true_still_prints(fake_whisper, capsys):
    asr.transcribe("audio.wav", model_name="small")

    out = capsys.readouterr().out
    assert "загружаю модель small" in out
    assert "язык: ru" in out
