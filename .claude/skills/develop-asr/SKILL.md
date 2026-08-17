---
name: develop-asr
description: "Распознавание речи: модели whisper, параметры декодирования, качество текста, языки. Запускает asr-expert."
user-invocable: true
allowed-tools: Agent, Read, Grep, Glob, Bash
---

# Распознавание речи

## Действия

1. Прочитай `docs/алгоритмы.md` раздел 2 и `.docs/decisions/0005-порядок-этапов.md`
2. Запусти Agent: `subagent_type` general-purpose, `model` opus,
   правила из `.claude/agents/01-asr-expert.md`

## Контекст

Модуль `stenogram/asr.py` — единственный, не зависящий ни от чего внутри проекта.
Так и должно остаться.

Три параметра неприкосновенны: `word_timestamps=True` (на нём держится вся разметка),
`condition_on_previous_text=False` (иначе зацикливание на длинных записях),
`vad_filter=True`.

Проверять на коротком фрагменте с `-m small`: минута записи проходит пайплайн примерно
за минуту. Полный прогон `large-v3-turbo` — 25 минут на час.
