---
name: validate
description: "Проверка архитектуры: слои, направление зависимостей, контракты между модулями. Запускает architecture-validator."
user-invocable: true
allowed-tools: Agent, Read, Grep, Glob, Bash
---

# Проверка архитектуры

## Действия

1. Прочитай `docs/архитектура.md`
2. Прогони быстрые проверки
3. Запусти Agent: `model` sonnet, правила из `.claude/agents/07-architecture-validator.md`

## Быстрые проверки

```bash
# направление зависимостей
grep -n "^from \.\|^from stenogram\|^import stenogram" stenogram/*.py

# тяжёлые импорты на уровне модуля — нарушение
grep -n "^import sklearn\|^import sherpa\|^from faster_whisper\|^import faster_whisper" stenogram/*.py
```

## Что проверяется

Зависимости строго в одну сторону: `audio`, `models` — нижний слой; `video` → `audio`;
`asr` — ни от чего; `diarization` → `audio`, `models`; `align` → `asr`, `diarization`;
`render` → `align`; `cli` → все.

`asr.py` обязан оставаться полностью автономным.

Тяжёлые библиотеки импортируются **внутри функций**, иначе `--help` ждёт загрузки
ONNX Runtime.

Любой источник разметки отдаёт `list[Turn]` — на этом держится расширяемость.

Распознавание идёт до разметки участников: обратный порядок ломает привязку границ
к паузам и при этом не падает.
