# Справочник API

Все модули пригодны для использования по отдельности. Времена везде — секунды от начала
записи (float), индексы участников — с нуля.

```python
from stenogram import align, asr, audio, diarization, models, pipeline, render, video
```

Параметр `progress` у функций: `True` — печатать ход работы, `False` — молчать, функция
`(сообщение, доля пройденного или None)` — получать сообщения вместо печати.

---

## pipeline

Весь пайплайн одной функцией, в том же порядке этапов, что у CLI. Встраивать stenogram
в свой код стоит через неё, а не повторяя вызовы модулей: порядок этапов и умолчания
стоят за замерами.

### `run(source, wav, options=None, *, reuse=None, whisper_model=None, progress=None) -> Result`

```python
from stenogram import pipeline

result = pipeline.run(Path("запись.mp4"), Path("/tmp/audio16k.wav"), pipeline.Options(language="ru", speakers=4))
result.utterances     # list[Utterance] — то, что рендерится в файлы
result.intermediate   # кэш для повторного прогона
```

Файлы не пишет: рендеры из `render` возвращают строки. `wav` — куда извлечь звук.
Без `progress` молчит, с ним получает `Event(stage, message, fraction, error)` — те же
строки, что печатает CLI; `stage` — `audio`, `video`, `asr`, `highlight` (метки подсветки),
`diarization`, `align`. Исключения из `progress` всплывают наружу. `whisper_model` —
модель от `asr.load_model`. Неверные `speakers`, `max_speakers`, `tile_speakers`
(меньше 1) — `ValueError` до начала работы.

**Повторный прогон с другими параметрами разметки** обходится без распознавания и без
чтения видео — звук берётся из `wav` как есть:

```python
again = pipeline.run(None, wav, pipeline.Options(tile_speakers=[3, 1]), reuse=result.intermediate)
```

Если в кэше нет плиток (первый прогон шёл без видео), а разметка по видео нужна, нужен
и `source`. Плитки в кэше используются, только если в этом прогоне включены и видео,
и разметка; иначе они просто переходят в новый `intermediate`. Кэш распознавания
вызывающий ключует сам: `run` не знает, с какой моделью и языком он получен. Поиск
плиток, упавший в первом прогоне, повторяется, если передать копию кэша с `tiles=None`
(`dataclasses.replace(result.intermediate, tiles=None)`). `TileSpeakersMismatch` (подкласс `ValueError`) — чисел в `tile_speakers`
не столько, сколько плиток найдено; бросается до распознавания.

### `Options` · `Result` · `Intermediate`

`Options` — параметры с умолчаниями CLI: `model`, `language`, `prompt`, `beam_size`,
`device`, `compute_type`, `threads` (0 — все ядра, кроме двух), `denoise`, `diarization`,
`video`, `speakers`, `max_speakers`, `tile_speakers`, `highlight_lag`.

`Result` — `utterances`, `turns`, `duration`, `intermediate`, `tile_counts` (сколько людей
пришлось на плитку), `video_error` (поиск плиток упал — в этом прогоне или в кэше, — и
нужная разметка по видео откатилась на голос), `video_used` (разметка в этом прогоне
шла по видео).

`Intermediate` — `segments`, `language`, `tiles` (`None` — не искали, `[]` — искали и не
нашли), `labels` и `step` — метки подсветки, `video_error` — причина, если `tiles == []`
из-за упавшего поиска.

---

## audio

Извлечение и загрузка звука.

### `SAMPLE_RATE = 16000`

Частота дискретизации, с которой работает весь пайплайн.

### `find_ffmpeg() -> tuple[str, str]`

Возвращает пути к `ffmpeg` и `ffprobe`. Сначала ищет системные, затем подключает
пакет `static-ffmpeg`. Бросает `RuntimeError`, если не нашёл ни того, ни другого.

### `probe_duration(path: Path) -> float`

Длительность файла в секундах.

### `extract_wav(src: Path, dst: Path, denoise: bool = False) -> Path`

Извлекает моно-дорожку 16 кГц PCM 16 бит. `denoise=True` добавляет
`highpass=f=80,afftdn=nf=-25,dynaudnorm=f=200:g=5`. Бросает `RuntimeError` с текстом
ошибки ffmpeg, если извлечь не удалось.

### `load_wav(path: Path) -> np.ndarray`

Читает WAV в `float32` в диапазоне [−1, 1]. Требует моно 16 кГц, иначе `ValueError`.
Час записи — около 242 МБ в памяти.

---

## video

Подсветка активного говорящего. Зависит от `find_ffmpeg` (`audio`) и, только внутри
`to_turns`, от класса `Turn` (`diarization`) — им упаковываются интервалы речи.

### `Tile`

```python
@dataclass
class Tile:
    x: int; y: int; width: int; height: int
    probe -> tuple[int, int, int, int]     # полоска кадра для съёма сигнала
```

### `has_video(path: Path) -> bool`

Есть ли в файле видеодорожка.

### `video_size(path: Path) -> tuple[int, int]`

Ширина и высота кадра.

### `find_tiles(path: Path, samples: int = 120, progress: bool | Callable = True) -> list[Tile]`

Ищет плитки участников по мерцающей рамке подсветки. Пустой список означает, что
подсветки нет — вызывающий код должен перейти на разметку по голосу.

`samples` — сколько кадров анализировать. Больше кадров — устойчивее к записям, где
кто-то говорит редко; дороже по времени.

### `active_speaker(path, tiles, fps=5.0, progress: bool | Callable = True) -> tuple[np.ndarray, float]`

Возвращает `(метки, шаг)`: массив индексов подсвеченной плитки по кадрам и шаг по времени
в секундах. `-1` означает, что подсветки нет ни на одной плитке.

Час видео обрабатывается примерно за 16 секунд: полоска каждой плитки усредняется
в один пиксель на кадр.

### `word_gaps(segments, min_gap: float = 0.15) -> list[tuple[float, float]]`

Паузы между словами — кандидаты на границу реплики. На вход — `list[Segment]` из `asr`.

### `to_turns(labels, step, word_gaps=None, lag=1.2, snap_window=2.0) -> list[Turn]`

Превращает покадровые метки в интервалы речи. Границы сдвигаются на `lag` назад
(подсветка отстаёт от речи) и притягиваются к ближайшей паузе в окне `±snap_window`.

Без `word_gaps` привязка не выполняется и границы режут фразы посередине — передавать
паузы настоятельно рекомендуется.

---

## asr

### `Word` / `Segment`

```python
@dataclass
class Word:
    start: float; end: float; text: str      # текст с ведущим пробелом

@dataclass
class Segment:
    start: float; end: float; text: str; words: list[Word]
```

### `transcribe(...) -> tuple[list[Segment], str]`

```python
transcribe(
    wav: Path,
    model_name: str = "large-v3-turbo",
    language: str | None = None,        # None — автоопределение
    beam_size: int = 5,
    compute_type: str = "int8",         # int8 для CPU, float16 для GPU
    device: str = "cpu",                # cpu | cuda | auto
    num_threads: int = 8,
    initial_prompt: str | None = None,  # тема, термины, фамилии
    progress: bool | Callable = True,
    *,
    model=None,                         # уже загруженная load_model модель
    local_files_only: bool = False,
    download_root: str | None = None,
) -> tuple[list[Segment], str]
```

Возвращает сегменты с потайминговкой по словам и код определённого языка. С `model`
параметры загрузки не используются.

### `load_model(model_name, device, compute_type, num_threads, local_files_only, download_root)`

Загружает `WhisperModel`. Долгоживущий процесс держит одну модель и передаёт её в
`transcribe(model=...)`. `local_files_only=True` делает отсутствие модели на диске ошибкой,
а не походом в сеть.

Внутри жёстко заданы `word_timestamps=True`, `condition_on_previous_text=False` и
VAD-фильтр — эти значения не параметризованы намеренно, см. [алгоритмы](алгоритмы.md).

---

## diarization

### `Turn`

```python
@dataclass
class Turn:
    start: float; end: float; speaker: int
    duration -> float
```

Общий формат разметки участников. Его принимает `align.assign` независимо от источника.

### `diarize(audio, num_speakers=None, max_speakers=12, num_threads=4, progress: bool | Callable = True) -> list[Turn]`

Разметка только по звуку: сегментация → отпечатки → кластеризация → сглаживание.
`num_speakers=None` включает автооценку по eigengap.

### `diarize_with_video(...) -> tuple[list[Turn], dict[int, int]]`

```python
diarize_with_video(
    audio: np.ndarray,
    video_turns: list[Turn],            # из video.to_turns
    num_speakers: int | None = None,    # общее число участников
    tile_speakers: list[int] | None = None,   # по плиткам из find_tiles, например [3, 1]
    num_threads: int = 4,
    progress: bool | Callable = True,
) -> tuple[list[Turn], dict[int, int]]
```

Гибридная разметка: видео задаёт плитки, звук делит людей внутри плитки. Второй элемент
результата — сколько участников пришлось на каждую плитку: словарь по **id плитки**, а не
список, потому что плитки не обязаны нумероваться подряд.

`tile_speakers` точнее, чем `num_speakers`: при указании только общего числа лишние люди
относятся к плитке с наибольшим временем речи.

#### По каким плиткам нумеруются числа

Числа `tile_speakers` идут **по плиткам, которые вернул `find_tiles`, слева направо**:
n-е число — n-я найденная в кадре плитка. Так можно, потому что метка плитки в `Turn` —
это и есть индекс в списке `find_tiles`: `active_speaker` отдаёт argmax по нему.

```python
tiles = video.find_tiles(src)          # [плитка слева, плитка в центре, плитка справа]
tile_speakers = [3, 1, 1]              #   трое,          один,           один
```

Отсюда два следствия, из-за которых `tile_speakers` **не** сверяется на равенство длин:

- **Плитка, найденная в кадре, но ни разу уверенно не подсвеченная, свой номер в списке
  всё равно занимает.** Её подсветка могла не дожить до `to_turns`: короткий фрагмент
  сдвигается на `lag` и притягивается к паузе, границы схлопываются, интервал исчезает.
  Число такой плитки просто не используется — пропускать его нельзя, иначе все следующие
  числа сдвинутся на чужие плитки.
- **Требуется лишь, чтобы в списке нашлось число для самой правой дожившей плитки**
  (`max(tiles) < len(tile_speakers)`). Лишние числа — не ошибка.

Если вы собираете `video_turns` не из `find_tiles`, а из своего источника, — нумеруйте
`speaker` подряд с нуля, и тогда `tile_speakers` читается как «по одному числу на плитку
в порядке следования».

### Отдельные шаги

```python
segment(audio, num_threads=4, progress=True) -> list[Turn]
    # границы реплик, метки говорящих черновые

embed(audio, turns, num_threads=4) -> tuple[np.ndarray, list[int]]
    # (нормированные отпечатки, индексы озвученных реплик)
    # реплики короче 0.4 с пропускаются

estimate_speakers(embeddings, max_speakers=12) -> int
    # оценка числа участников по максимальному разрыву в спектре лапласиана

cluster(embeddings, weights, num_speakers,
        gaps=None, stay=0.80, scale=10.0, iterations=6) -> np.ndarray
    # метки участников; gaps=None отключает сглаживание HMM
```

`weights` — длительности реплик. `gaps` — паузы перед каждой репликой; чем длиннее пауза,
тем дешевле смена говорящего.

---

## align

### `Utterance`

```python
@dataclass
class Utterance:
    start: float; end: float; speaker: int; text: str; words: list[Word]
```

### `assign(segments, turns, max_pause=2.0, min_utterance=0.35) -> list[Utterance]`

Размечает каждое слово участником и собирает слова в реплики.

- `max_pause` — пауза внутри реплики одного человека, после которой начинается новая.
  `pipeline.run` передаёт 3.0 (`pipeline.MAX_PAUSE`): на записях совещаний паузы длиннее.
- `min_utterance` — реплики короче отдаются соседям, чтобы убрать дребезг разметки.

С пустым `turns` возвращает сегменты распознавания как есть, все с участником 0.

### `speaker_stats(utterances) -> dict[int, dict]`

Сводка по участникам, отсортированная по убыванию времени речи:

```python
{0: {"seconds": 1092.4, "turns": 115, "words": 1865, "share": 0.34}, ...}
```

---

## render

### `hhmmss(seconds, millis=False, comma=False) -> str`

`00:12:34`, либо `00:12:34.567`, либо `00:12:34,567` для SRT.

### `speaker_name(index, names) -> str`

Имя из словаря `names` либо `«Говорящий N»`.

### Форматы

```python
to_text(utterances, names=None, timestamps=True) -> str
to_markdown(utterances, names=None, title="Стенограмма", meta=None) -> str
to_srt(utterances, names=None, max_chars=90) -> str
to_vtt(utterances, names=None, max_chars=90) -> str
to_json(utterances, names=None, meta=None) -> str
```

`WRITERS: dict[str, callable]` — реестр форматов. CLI берёт список допустимых значений
`--formats` отсюда, так что новый формат достаточно добавить в словарь.

### `write(utterances, stem, formats, names=None, meta=None) -> list[Path]`

Пишет файлы `stem.<формат>` и возвращает пути. Неизвестный формат — `ValueError`.

```python
render.write(utterances, Path("совещание"), ["txt", "srt"],
             names={0: "Иванов", 1: "Петров"},
             meta={"Источник": "совещание.mp4"})
```

---

## models

### `cache_dir() -> Path`

`$STENOGRAM_HOME` либо `~/.cache/stenogram`.

### `ensure_segmentation_model() -> Path` · `ensure_embedding_model() -> Path` · `ensure_all() -> tuple[Path, Path]`

Возвращают пути к моделям, скачивая при необходимости. Повторные вызовы бесплатны.

Чтобы подставить свою модель отпечатков, переопределите `models.EMBEDDING_URL` и
`models.EMBEDDING_FILE` до первого вызова. Требования: ONNX, вход 16 кГц моно.

---

## cli

### `main(argv: list[str] | None = None) -> int`

Точка входа. Возвращает код возврата процесса:

| Код | Когда |
|---:|---|
| 0 | успех, а также штатный обрыв конвейера (`\| head`) |
| 1 | ошибка: одна строка «Ошибка: ...» в stderr вместо трейсбека |
| 2 | argparse отверг аргументы (например, `--speakers 0.5`) |
| 130 | прерывание пользователем (Ctrl+C) |

Переменная окружения **`STENOGRAM_TRACEBACK=1`** отключает перевод исключений в одну
строку: исключение выходит наружу как есть, с полным трейсбеком. Нужна для баг-репортов —
без неё сообщение вроде `Ошибка: maximum recursion depth exceeded` не говорит ничего о
месте отказа. `KeyboardInterrupt` (код 130) и откат «видео не разобралось → работаю по
звуку» она не меняет: в последнем случае трейсбек печатается, но прогон продолжается.

### `parse_names(raw: str | None) -> dict[int, str] | None`

`"Аня,Борис"` → `{0: "Аня", 1: "Борис"}`; `"3=Глеб"` → `{2: "Глеб"}`.

### `parse_tile_speakers(raw: str | None) -> list[int] | None`

`"3,1"` → `[3, 1]`: по одному числу на каждую плитку из `find_tiles`, слева направо
(см. [`diarize_with_video`](#по-каким-плиткам-нумеруются-числа)). Значения меньше 1
отвергаются здесь же, на границе CLI. Длину сверяет `pipeline.run`: число плиток
известно только после шага `[2/4]`.

### `build_parser() -> argparse.ArgumentParser`

Готовый парсер аргументов — пригодится, если нужно встроить пайплайн в свою утилиту.

---

## Примеры

**Только распознавание, без участников**

```python
from pathlib import Path
from stenogram import asr, audio, render, align

wav = audio.extract_wav(Path("лекция.mp4"), Path("/tmp/a.wav"))
segments, lang = asr.transcribe(wav, language="ru", model_name="small")
render.write(align.assign(segments, []), Path("лекция"), ["txt"])
```

**Полный гибридный пайплайн**

Так `pipeline.run` устроен внутри; в своём коде вызывайте его, а не повторяйте шаги.

```python
from pathlib import Path
from stenogram import align, asr, audio, diarization, render, video

src = Path("дейли.mp4")
wav = audio.extract_wav(src, Path("/tmp/a.wav"))
segments, lang = asr.transcribe(wav, language="ru")

tiles = video.find_tiles(src)                            # len(tiles) == 2
labels, step = video.active_speaker(src, tiles)
vturns = video.to_turns(labels, step, video.word_gaps(segments))

# Числа — по плиткам из find_tiles, слева направо: одно число на каждую найденную
# плитку, включая ту, чья подсветка могла не дожить до vturns.
turns, counts = diarization.diarize_with_video(
    audio.load_wav(wav), vturns, tile_speakers=[3, 1])   # трое в переговорной, один удалённо

utterances = align.assign(segments, turns, max_pause=3.0)
render.write(utterances, Path("дейли"), ["txt", "md", "json"],
             names={3: "Глеб (удалённо)"})
```

**Только детектор активного говорящего**

```python
from pathlib import Path
from stenogram import video

src = Path("созвон.mp4")
tiles = video.find_tiles(src)
labels, step = video.active_speaker(src, tiles)

for turn in video.to_turns(labels, step):
    print(f"{turn.start:7.1f} – {turn.end:7.1f}  плитка {turn.speaker + 1}")
```

**Своя разметка участников из внешнего источника**

```python
from stenogram.diarization import Turn

turns = [Turn(start, end, speaker_id) for start, end, speaker_id in мой_лог_микрофонов]
utterances = align.assign(segments, turns)
```
