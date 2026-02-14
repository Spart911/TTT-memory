# LLM_TTT_E2E

Проект реализует адаптацию тест‑времени (TTT) и meta‑training для памяти модели, с вариантами:
- базовая TTT‑модель,
- TinyLoRA‑адаптеры для диалогового SFT,
- тесты “памяти” через pre/post‑loss.

## Состав

- `my_model.py` — модель + TTT/meta‑training.
- `ttt_decode.py` — TTT‑декодер (KV‑cache).
- `ttt_infer.py` — чат‑инференс базовой модели.
- `ttt_infer_adapter.py` — чат‑инференс с TinyLoRA адаптером.
- `train_tinylora_dialog.py` — SFT‑обучение TinyLoRA на диалогах.
- `ttt_memory_test.py` — простой тест TTT‑памяти (pre/post loss).
- `ttt_memory_test_adapter.py` — тест памяти с адаптером.
- `ttt_memory_test_full.py` — полный тест base vs adapter (grid).

## Установка

```bash
pip install torch transformers datasets tqdm
```

## Обучение TTT/meta (основная модель)

```bash
python my_model.py
```

Параметры настраиваются внутри `my_model.py`:
- `MODEL_SIZE` — выбор размера.
- `WINDOW`, `TTT_BATCH`, `TRAIN_BATCH`, `MICRO_BATCH`.

### Модель и обучение (my_model.py)

**Архитектура**
- Transformer с sliding‑window attention.
- QK‑norm включен (`USE_QK_NORM=True`).
- TTT‑блоки — последние 1/4 блоков.
- В TTT‑блоках две MLP: `mlp_ttt` (обновляемая) + `mlp_static` (safe storage).
- MLP hidden уменьшен глобально, чтобы компенсировать добавление второй MLP.

**TTT‑адаптация**
- Обновляются **только MLP** в последних 1/4 блоков.
- Inner‑loop: mini‑batch TTT с window `k` и batch `b` (условие `k ≥ b`).
- Градиенты вычисляются по fast‑весам (batched fast‑deltas).

**Meta‑training**
- В `forward_ttt_meta` делается split последовательности:
  - первая часть = TTT‑адаптация,
  - вторая часть = post‑TTT loss.
- Цель обучения — минимизировать post‑TTT loss (bi‑level).

**Функция ошибки**
- next‑token cross‑entropy.
- Для meta‑training используется post‑TTT loss.

## TinyLoRA диалоговое обучение

```bash
python train_tinylora_dialog.py --dataset OpenRL/daily_dialog
```

Файл сохраняет адаптеры в:
```
tinylora_dialog_adapter.pth
```

Параметры обучения (локально):
- Датасет: `OpenRL/daily_dialog`
- Использовано: 100% датасета (без сэмплирования)
- Адаптеры TinyLoRA поверх базовой модели

## Инференс

### Базовая модель
```bash
python ttt_infer.py
```


Пример генерации (без адаптера, чат):

```
You: Hello
Assistant: .	you cool.	. i am a of.	oh. you are you? i just got and. i am a lot not like, i do you like to get a living?
i like it is good, i am a bit
i am a lot you?
i. what do you do?
```

Качество: модель поддерживает базовый диалоговый формат, но ответы остаются шумными и с повторениями.
Это ожидаемо для маленькой модели и короткого контекста.


### С TinyLoRA адаптером
```bash
python ttt_infer_adapter.py --adapter-path tinylora_dialog_adapter.pth
```

Пример генерации (с адаптером, чат):

```
You: Hello
Assistant: do you have, to eat and a of. do
 do..
 is!. do for a. do,.. i the beach.. the you much?
 to make you to the way just get a... do you the best to make! my?! the beach i
```

Качество: модель поддерживает базовый диалоговый формат, но ответы остаются шумными и с повторениями.
Это ожидаемо для маленькой модели и короткого контекста.

## Тест памяти

### Базовый тест
```bash
python ttt_memory_test.py
```

### Тест с адаптером
```bash
python ttt_memory_test_adapter.py
```

### Полный тест (base vs adapter)
```bash
python ttt_memory_test_full.py --grid
```

## Примечания

- В инференсе по умолчанию используется history последних 5 реплик.
- Если повторения в тестах не влияют на результат — проверьте `MAX_LEN`, скорее всего контекст обрезается.
- При OOM уменьшайте `TRAIN_BATCH` и/или `MICRO_BATCH`.

## Результаты (локальные тесты)

TTT‑память измерялась через **pre/post‑loss** на synthetic prompts (см. `ttt_memory_test_full.py`).
Пример наблюдений (grid‑тесты):

- Base модель: улучшение post‑loss на ~0.33–0.56% (delta ~0.033–0.060).
- Модель + TinyLoRA: базовый loss ниже, но относительная TTT‑дельта меньше: ~0.21–0.31%.

Вывод: TinyLoRA улучшает базовый уровень, но не усиливает TTT‑эффект в процентах на коротком контексте.

## Лицензия

Добавьте при необходимости.
