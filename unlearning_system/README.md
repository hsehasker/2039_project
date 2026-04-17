# 2039 — Unlearning System

Каркас прототипа для исследования Machine Unlearning.
Проект «Механизмы цифрового забвения», МИЭМ НИУ ВШЭ, ТК 26.

## Установка

```bash
cd unlearning_system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Типовой сценарий (full cycle из ТЗ §6.5.3)

`Генерация → подготовка → обучение → удаление → переоценка → отчёт`

```bash
# 1. Сгенерировать датасет (соседний пакет)
cd ../synthetic_data_generator
python main.py generate --count 10000 --seed 42

# 2. Обучить SISA-ансамбль
cd ../unlearning_system
python cli.py train \
    --config configs/default.yaml \
    --data ../synthetic_data_generator/data/synthetic_dataset.csv

# 3. Выполнить unlearning (удалить по списку ID)
python cli.py unlearn \
    --config configs/default.yaml \
    --data ../synthetic_data_generator/data/synthetic_dataset.csv \
    --user-ids USR-000001 USR-000002 USR-000003 \
    --reason "user_request"

# 4. Удаление по фильтру (все без согласия)
python cli.py unlearn \
    --config configs/default.yaml \
    --data ../synthetic_data_generator/data/synthetic_dataset.csv \
    --filter "consent_given == False" \
    --reason "152-FZ art.14"

# 5. Показать накопленные метрики запуска
python cli.py report --run runs/<run_id>
```

## Структура

```
unlearning_system/
├── cli.py                  # точка входа
├── configs/default.yaml    # YAML-конфиг (ТЗ §6.3.2)
├── ml/
│   ├── preprocess.py       # FeaturePipeline (числ/кат/бул)
│   ├── shards.py           # ShardPlan + стабильный хеш маршрутизации
│   ├── base.py             # BaseLearner (абстракция)
│   ├── logreg.py           # PyTorch logistic regression
│   ├── rf.py               # RandomForest wrapper
│   ├── sisa.py             # ансамбль SISA (fit / retrain_shards)
│   ├── unlearning.py       # Unlearner: delete-by-ID / delete-by-filter
│   ├── metrics.py          # Accuracy, Precision, Recall, F1, AUC, MSE
│   ├── registry.py         # runs/<id>/ — manifest, артефакты, events
│   └── pipeline.py         # ExperimentPipeline (фасад)
├── docs/THEORY.md          # теория: обучение, unlearning, метрики
└── tests/test_smoke.py     # e2e + юнит-тесты
```

## Теоретическая база

См. [`docs/THEORY.md`](docs/THEORY.md). Коротко:

- **Обучение**: датасет шардируется по `user_id` стабильным хешом;
  каждая мини-модель обучается изолированно; предсказание — soft-voting.
- **Удаление**: запрос резолвится в индексы строк; затронутые шарды
  переобучаются с нуля без удалённых записей; остальные остаются
  бит-в-бит (exact unlearning).
- **Метрики**: классические (Accuracy/Precision/Recall/F1/AUC/MSE) +
  Δ-метрики + prediction disagreement + время/ресурсы.
- **Воспроизводимость**: все seed'ы и SHA-1 датасета фиксируются в
  `runs/<id>/manifest.yaml`.

## Тесты

```bash
python -m pytest tests/ -v
```
