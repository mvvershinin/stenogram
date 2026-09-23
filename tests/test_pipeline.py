"""Тесты pipeline.run — порядка этапов, вынесенного из cli.main.

Что здесь держится:
  - повторный прогон из кэша (`reuse`) не трогает ни звук, ни распознавание, ни видео —
    ради этого кэш и существует: 32 минуты из 35 на часовой записи;
  - несовпадение tile_speakers с плитками обрывает прогон ДО распознавания;
  - упавший поиск плиток — не отказ, а откат на голос, и это видно в результате;
  - без progress функция молчит: библиотека не печатает в чужой stdout;
  - «плитки не искали» (None) и «искали и не нашли» ([]) различаются — от этого
    зависит, можно ли переразмечать по видео из кэша без исходника; упавший поиск
    хранит свою причину и в кэше;
  - плитки в кэше — знание о записи, а не приказ: повтор с выключенным видео
    размечает по голосу и плитки не теряет;
  - исключения колбэка progress всплывают наружу, а не выдаются за отказ видео.

Всё, что ходит в ffmpeg, ONNX и сеть, подменено; середина пайплайна настоящая.
"""

from __future__ import annotations

import dataclasses
import gc
import weakref
from pathlib import Path

import numpy as np
import pytest

from stenogram import asr, audio, diarization, pipeline, video
from stenogram.asr import Segment, Word

_TILES = [video.Tile(10, 0, 3, 20), video.Tile(50, 0, 3, 20)]


def _segments() -> list[Segment]:
    """Речь 0–20 с с одной настоящей паузой 9.2–10.8 с — туда притягивается граница плиток."""
    starts = [round(i * 0.4, 2) for i in range(23)] + [round(10.8 + i * 0.4, 2) for i in range(23)]
    words = [Word(start=t, end=round(t + 0.35, 2), text=" слово") for t in starts]
    return [Segment(start=0.0, end=words[-1].end, text="слово", words=words)]


def _labels() -> np.ndarray:
    """0–10 с подсвечена плитка 0, 10–20 с — плитка 1; шаг 0.2 с."""
    labels = np.zeros(100, dtype=int)
    labels[50:] = 1
    return labels


def _boom(*args, **kwargs):
    raise AssertionError("этот этап вызываться не должен")


@pytest.fixture
def stages(monkeypatch):
    """Подменяет внешние этапы; возвращает список вызванных, чтобы проверять, что пропущено."""
    called: list[str] = []

    def track(name, result):
        def fake(*args, **kwargs):
            called.append(name)
            return result

        return fake

    monkeypatch.setattr(audio, "extract_wav", track("extract_wav", None))
    monkeypatch.setattr(audio, "probe_duration", lambda path: 20.0)
    monkeypatch.setattr(audio, "load_wav", lambda path: np.zeros(16000 * 20, dtype=np.float32))
    monkeypatch.setattr(video, "has_video", lambda path: True)
    monkeypatch.setattr(video, "find_tiles", track("find_tiles", _TILES))
    monkeypatch.setattr(video, "active_speaker", track("active_speaker", (_labels(), 0.2)))
    monkeypatch.setattr(asr, "transcribe", track("transcribe", (_segments(), "ru")))
    monkeypatch.setattr(diarization, "segment", lambda audio, num_threads=4, progress=True: [diarization.Turn(i + 0.05, i + 0.95, 0) for i in range(20)])
    return called


def _run(tmp_path: Path, **kwargs) -> pipeline.Result:
    source = tmp_path / "запись.mp4"
    source.write_bytes(b"")
    return pipeline.run(source, tmp_path / "audio16k.wav", **kwargs)


def test_full_run_calls_every_stage_and_keeps_intermediate(tmp_path, stages):
    result = _run(tmp_path)

    assert stages == ["extract_wav", "find_tiles", "transcribe", "active_speaker"]
    assert result.intermediate.tiles == _TILES
    assert result.intermediate.language == "ru"
    assert result.intermediate.step == 0.2
    assert np.array_equal(result.intermediate.labels, _labels())
    assert {u.speaker for u in result.utterances} == {0, 1}
    assert result.video_error is None


def test_reuse_skips_audio_asr_and_video(tmp_path, stages, monkeypatch):
    """Повторный прогон с другим числом участников: звук, текст и метки подсветки — из кэша."""
    first = _run(tmp_path)
    for name in ("extract_wav", "find_tiles", "transcribe", "active_speaker"):
        monkeypatch.setattr({"extract_wav": audio, "find_tiles": video, "transcribe": asr, "active_speaker": video}[name], name, _boom)

    events: list[pipeline.Event] = []
    again = pipeline.run(None, tmp_path / "audio16k.wav", pipeline.Options(tile_speakers=[1, 1]), reuse=first.intermediate, progress=events.append)

    assert [u.text for u in again.utterances] == [u.text for u in first.utterances]
    assert again.intermediate.tiles == _TILES
    messages = [e.message for e in events]
    assert "[1/4] Звук взят из кэша" in messages
    assert "[3/4] Распознавание взято из кэша" in messages


def test_reuse_without_tiles_needs_source_for_video(tmp_path, stages, monkeypatch):
    """Первый прогон шёл без видео — плиток в кэше нет, и без исходника разметить по видео нечем.
    Молча откатиться на голос было бы хуже: разметка по голосу на порядок ненадёжнее."""
    monkeypatch.setattr(diarization, "diarize", lambda *a, **k: [diarization.Turn(0.0, 20.0, 0)])
    first = _run(tmp_path, options=pipeline.Options(video=False))
    assert first.intermediate.tiles is None

    with pytest.raises(ValueError, match="нужен исходный файл"):
        pipeline.run(None, tmp_path / "audio16k.wav", reuse=first.intermediate)


def test_tile_speakers_mismatch_stops_before_transcription(tmp_path, stages, monkeypatch):
    monkeypatch.setattr(asr, "transcribe", _boom)

    with pytest.raises(pipeline.TileSpeakersMismatch, match=r"передано — 3.*найдено — 2"):
        _run(tmp_path, options=pipeline.Options(tile_speakers=[1, 1, 1]))


def test_failed_tile_search_falls_back_to_voice_and_says_so(tmp_path, stages, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("видео нестандартное")

    monkeypatch.setattr(video, "find_tiles", broken)
    monkeypatch.setattr(video, "active_speaker", _boom)
    monkeypatch.setattr(diarization, "diarize", lambda *a, **k: [diarization.Turn(0.0, 20.0, 0)])
    events: list[pipeline.Event] = []

    result = _run(tmp_path, progress=events.append)

    assert result.video_error == "видео нестандартное"
    assert result.intermediate.tiles == []
    # [] без причины значило бы «подсветки нет»; с причиной — «поиск не удался», и
    # искать ли снова, решает вызывающий.
    assert result.intermediate.video_error == "видео нестандартное"
    failed = [e for e in events if e.error is not None]
    assert len(failed) == 1
    assert "останусь на звуке" in failed[0].message


def test_source_without_video_stream_counts_as_searched(tmp_path, stages, monkeypatch):
    monkeypatch.setattr(video, "has_video", lambda path: False)
    monkeypatch.setattr(diarization, "diarize", lambda *a, **k: [diarization.Turn(0.0, 20.0, 0)])

    result = _run(tmp_path)

    assert result.intermediate.tiles == []
    assert "find_tiles" not in stages


def test_without_progress_nothing_is_printed(tmp_path, stages, capsys):
    _run(tmp_path)

    assert capsys.readouterr().out == ""


def test_progress_events_carry_stage_and_module_messages(tmp_path, stages, monkeypatch):
    """Сообщения модулей идут в тот же колбэк, что и шаги, с этапом и долей пройденного."""

    def fake_transcribe(*args, progress=True, **kwargs):
        progress("  10.0 / 20.0 мин", 0.5)
        return _segments(), "ru"

    monkeypatch.setattr(asr, "transcribe", fake_transcribe)
    events: list[pipeline.Event] = []

    _run(tmp_path, progress=events.append)

    assert pipeline.Event("asr", "  10.0 / 20.0 мин", 0.5) in events
    assert [e.message for e in events if e.message.startswith("[")] == [
        "[1/4] Извлекаю аудио из запись.mp4 ...",
        "[2/4] Ищу подсветку активного говорящего в видео ...",
        "[3/4] Распознаю речь ...",
        "[4/4] Определяю участников ...",
    ]


def test_whisper_model_is_passed_to_transcribe(tmp_path, stages, monkeypatch):
    seen = {}

    def fake_transcribe(*args, model=None, **kwargs):
        seen["model"] = model
        return _segments(), "ru"

    monkeypatch.setattr(asr, "transcribe", fake_transcribe)
    held = object()

    _run(tmp_path, whisper_model=held)

    assert seen["model"] is held


def test_no_source_and_no_cache_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="Нужен исходный файл"):
        pipeline.run(None, tmp_path / "audio16k.wav")


def _voice_only(*args, **kwargs):
    """Разметка по голосу, которую видео дать не может: весь разговор — участник 0."""
    return [diarization.Turn(0.0, 20.0, 0)]


@pytest.mark.parametrize("off", [{"video": False}, {"diarization": False}])
def test_reuse_with_video_or_diarization_off_matches_fresh_run(tmp_path, stages, monkeypatch, off):
    """Плитки в кэше — то, что известно о записи, а не приказ размечать по ним.
    Повтор с выключенным видео (или разметкой) обязан дать то же, что свежий прогон
    с теми же настройками, и сохранить плитки с метками для следующего повтора."""
    monkeypatch.setattr(diarization, "diarize", _voice_only)
    first = _run(tmp_path)
    fresh = _run(tmp_path, options=pipeline.Options(**off))
    monkeypatch.setattr(video, "find_tiles", _boom)
    monkeypatch.setattr(video, "active_speaker", _boom)

    # Три числа на две плитки: плитки в этом прогоне не используются — и сверять не с чем.
    again = pipeline.run(None, tmp_path / "audio16k.wav", pipeline.Options(tile_speakers=[1, 1, 1], **off), reuse=first.intermediate)

    assert {u.speaker for u in first.utterances} == {0, 1}
    assert again.utterances == fresh.utterances
    assert again.turns == fresh.turns
    assert again.video_used is False
    assert first.video_used is True
    assert again.intermediate.tiles == _TILES
    assert np.array_equal(again.intermediate.labels, _labels())
    assert again.intermediate.step == 0.2


def test_failed_tile_search_is_kept_in_cache_with_its_error(tmp_path, stages, monkeypatch):
    """Упавший поиск — не то же, что «подсветки нет»: причина едет в кэше, и повтор
    отдаёт её вызывающему, а тот решает, искать ли снова."""

    def out_of_memory(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr(video, "find_tiles", out_of_memory)
    monkeypatch.setattr(diarization, "diarize", _voice_only)
    events: list[pipeline.Event] = []

    first = _run(tmp_path, progress=events.append)
    monkeypatch.setattr(video, "find_tiles", _boom)
    again = pipeline.run(None, tmp_path / "audio16k.wav", reuse=first.intermediate)

    assert "      не получилось (); останусь на звуке" in [e.message for e in events]  # строка CLI как была
    assert first.video_error == "MemoryError"  # str(MemoryError()) пуст
    assert first.intermediate.video_error == "MemoryError"
    assert again.video_error == "MemoryError"
    assert again.intermediate.video_error == "MemoryError"
    assert again.video_used is False

    # Вызывающий решил искать снова, и на этот раз вышло: старая причина больше не правда.
    monkeypatch.setattr(video, "find_tiles", lambda path, **kwargs: _TILES)
    retried = _run(tmp_path, reuse=dataclasses.replace(first.intermediate, tiles=None))

    assert retried.video_used is True
    assert retried.video_error is None
    assert retried.intermediate.video_error is None


def test_progress_callback_error_during_tile_search_is_not_a_video_failure(tmp_path, stages, monkeypatch):
    """Колбэк вызывающего упал (например, `stenogram … | head` закрыл трубу) — это не
    нестандартное видео, и откатываться на голос, продолжая работу, нельзя."""

    def reporting_find_tiles(path, progress=True, **kwargs):
        progress("      ищу плитки участников (1920x1080) ...", None)
        return _TILES

    def closed_pipe(event):
        events.append(event)
        if event.stage == "video" and "ищу плитки" in event.message:
            raise BrokenPipeError

    monkeypatch.setattr(video, "find_tiles", reporting_find_tiles)
    monkeypatch.setattr(asr, "transcribe", _boom)
    monkeypatch.setattr(diarization, "diarize", _boom)
    events: list[pipeline.Event] = []

    with pytest.raises(BrokenPipeError):
        _run(tmp_path, progress=closed_pipe)

    assert all(e.error is None for e in events)


def test_video_error_event_does_not_keep_find_tiles_locals_alive(tmp_path, stages, monkeypatch):
    """Долгоживущий воркер копит события; трейсбек в `Event.error` не должен держать
    кадры find_tiles — в них массив насыщенности до гигабайта на 1080p."""

    class Frames:
        pass

    alive: list[weakref.ref] = []

    def broken(*args, **kwargs):
        big = Frames()
        alive.append(weakref.ref(big))
        raise RuntimeError("видео нестандартное")

    monkeypatch.setattr(video, "find_tiles", broken)
    monkeypatch.setattr(diarization, "diarize", _voice_only)
    events: list[pipeline.Event] = []

    _run(tmp_path, progress=events.append)
    gc.collect()

    assert [e.error for e in events if e.error is not None]  # событие с ошибкой живо
    assert alive[0]() is None


@pytest.mark.parametrize(
    ("options", "match"),
    [
        (pipeline.Options(speakers=0), r"speakers.*не меньше 1.*0"),
        (pipeline.Options(max_speakers=0), r"max_speakers.*не меньше 1.*0"),
        (pipeline.Options(tile_speakers=[0, 1]), r"tile_speakers.*не меньше 1.*\[0, 1\]"),
    ],
)
def test_bad_speaker_numbers_are_rejected_before_any_work(tmp_path, stages, options, match):
    """CLI проверяет это сам, но сервис зовёт run напрямую: speakers=0 молча давал одного
    участника, а tile_speakers=[0, 1] падал уже после распознавания."""
    with pytest.raises(ValueError, match=match):
        _run(tmp_path, options=options)

    assert stages == []


def test_highlight_has_its_own_progress_stage(tmp_path, stages, monkeypatch):
    """Метки подсветки и сегментация — разные этапы: иначе доля внутри «diarization»
    доходит до 100% и откатывается на 0."""

    def reporting_active_speaker(path, tiles, progress=True, **kwargs):
        progress("      подсветка плитки 1/2", 0.5)
        return _labels(), 0.2

    monkeypatch.setattr(video, "active_speaker", reporting_active_speaker)
    events: list[pipeline.Event] = []

    _run(tmp_path, progress=events.append)

    assert pipeline.Event("highlight", "      подсветка плитки 1/2", 0.5) in events
