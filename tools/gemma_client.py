"""Клиент к локальным моделям в Ollama: текст, зрение, эмбеддинги.

Лежит в `tools/`, а НЕ в пакете `stenogram` — и это осознанно. Стенограмма работает
офлайн, без сети и без языковых моделей; клиент к ним туда не входит и в установку
через pip не попадает (`packages = ["stenogram"]` в pyproject.toml). Это подсобный
инструмент для проработок — см. `docs/ускорение-на-gpu.md` и планы надстройки.

Зависимостей нет, только стандартная библиотека.

    from gemma_client import ask, ask_image, chat, embed, health

    ask("Объясни идемпотентность в двух предложениях")
    ask(длинный_текст, num_ctx=32768, num_predict=600)
    текст, м = ask(запрос, metrics=True)      # tok/s, время, обрезан ли ответ
    ask_image("Что на фото?", "снимок.jpg")
    embed("текст")                            # 1024 float, норма 1.0

Адрес хоста берётся из переменной окружения `STENOGRAM_LLM_HOST`, по умолчанию
локальный. Прописывать чей-то адрес в код не нужно:

    export STENOGRAM_LLM_HOST=http://адрес-машины:11434

## Чему в ответах нельзя верить

Замерено 09.09.2026: модель **ошибается в цифрах, артикулах и фамилиях**. На снимке
прочитала артикул AA16 как A16 и насчитала 5 коробок вместо 6; в пересказе стенограммы
исказила фамилию. Пересказ годится как навигация по записи, но ни одна цифра, дата,
сумма или фамилия из него не может идти в документ без сверки с исходным текстом.

## Контекст

`num_ctx` до 32768 держится целиком в видеопамяти на карте с 12 ГБ: замер показал рост
с 8.5 до 9.0 ГБ, скорость генерации не проседает. Значение 4096 — это умолчание Ollama,
подобранное по объёму видеопамяти, а не потолок. Час записи — около 15 600 токенов,
два часа — около 31 000, оба влезают одним куском, резать текст незачем.
"""

import base64
import json
import os
import time
import urllib.request

HOST = os.environ.get("STENOGRAM_LLM_HOST", "http://127.0.0.1:11434").rstrip("/")
# urlopen открывает и file://, а адрес приходит из окружения: схему проверяем один раз здесь.
if not HOST.startswith(("http://", "https://")):
    raise ValueError(f"STENOGRAM_LLM_HOST должен начинаться с http:// или https://, получено: {HOST!r}")
MODEL = "gemma4:12b"
EMBED_MODEL = "bge-m3"


def _post(path, payload, timeout=900):
    req = urllib.request.Request(  # noqa: S310 -- схема HOST проверена при импорте
        HOST + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())  # noqa: S310 -- схема HOST проверена при импорте


def _stats(d, wall):
    ev, ed = d.get("eval_count", 0), d.get("eval_duration", 1) / 1e9
    pv, pd = d.get("prompt_eval_count", 0), d.get("prompt_eval_duration", 1) / 1e9
    return {
        "prompt_tokens": pv,
        "prompt_tok_s": round(pv / pd, 1) if pd else 0,
        "eval_tokens": ev,
        "eval_tok_s": round(ev / ed, 1) if ed else 0,
        "load_s": round(d.get("load_duration", 0) / 1e9, 1),
        "wall_s": round(wall, 1),
        "truncated": d.get("done_reason") == "length",
    }


def ask(prompt, *, system=None, num_ctx=8192, num_predict=600, temperature=0.2, think=False, model=MODEL, metrics=False, timeout=900):
    """Один запрос к модели. Возвращает текст, либо (текст, метрики) при metrics=True.

    think=False обязателен: иначе весь бюджет num_predict уходит в размышление,
    а поле response приходит ПУСТЫМ. Ключ живёт в корне запроса, не в options.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": think,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }
    if system:
        payload["system"] = system

    t = time.time()
    d = _post("/api/generate", payload, timeout)
    text = d.get("response", "")
    return (text, _stats(d, time.time() - t)) if metrics else text


def ask_image(prompt, image_path, *, num_ctx=8192, num_predict=600, temperature=0.2, model=MODEL, metrics=False, timeout=900):
    """Запрос с картинкой. Точность на мелких деталях низкая — см. предупреждение выше."""
    with open(image_path, "rb") as f:
        img = base64.b64encode(f.read()).decode()
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img],
        "stream": False,
        "think": False,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }
    t = time.time()
    d = _post("/api/generate", payload, timeout)
    text = d.get("response", "")
    return (text, _stats(d, time.time() - t)) if metrics else text


def chat(messages, *, num_ctx=8192, num_predict=600, temperature=0.2, think=False, model=MODEL, timeout=900):
    """Многоходовой диалог. messages: [{"role": "user"|"assistant", "content": "..."}]"""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }
    return _post("/api/chat", payload, timeout)["message"]["content"]


def embed(texts, *, model=EMBED_MODEL, timeout=600):
    """Эмбеддинги. Строка или список строк. Возвращает список векторов по 1024 float.

    Векторы нормализованы (норма 1.0), поэтому косинус == скалярное произведение:
    в pgvector можно брать vector_ip_ops вместо vector_cosine_ops.
    """
    single = isinstance(texts, str)
    d = _post("/api/embed", {"model": model, "input": texts}, timeout)
    return d["embeddings"][0] if single else d["embeddings"]


def health():
    """Список моделей на хосте. Проверка, что сервис жив."""
    with urllib.request.urlopen(HOST + "/api/tags", timeout=15) as r:  # noqa: S310 -- схема HOST проверена при импорте
        return [m["name"] for m in json.load(r)["models"]]


if __name__ == "__main__":
    print("хост:", HOST)
    print("модели:", health())
    txt, m = ask("Ответь одним предложением: что такое идемпотентность?", metrics=True)
    print(txt.strip())
    print(m)
    v = embed("проверка")
    print(f"эмбеддинг: {len(v)} измерений, норма {sum(x * x for x in v) ** 0.5:.4f}")
