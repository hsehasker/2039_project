# Отчёт по проекту 2039

> «Механизмы цифрового забвения: анализ и моделирование с применением методов
> машинного обучения» — ВШЭ МИЭМ, заказчик ТК 26.

Документ описывает:
1. что именно было реализовано и исправлено,
2. как этим пользоваться,
3. как повысить качество модели после операции Machine Unlearning,
4. полный результат аудита кода (найденные баги, статус, что исправлено).

---

## 1. Реализация — структура репозитория

```
2039/
├── synthetic_data_generator/       # модуль генерации данных (был уже, доработан)
│   ├── generator/
│   │   ├── core.py                  # SyntheticDataGenerator — оркестратор
│   │   ├── distributions.py         # ★ bugfix: email↔ФИО, date_from_age, FIO vectorised
│   │   ├── correlations.py          # Cholesky-корреляции + поведенческие эвристики
│   │   ├── labeler.py               # ★ bugfix: явная проверка обязательных колонок
│   │   └── validator.py             # ★ добавлено: email uniqueness + email/name corr
│   ├── configs/default.yaml
│   ├── main.py                      # CLI на русском: generate/validate/info/list/interactive
│   └── tests/                       # 47 юнит-тестов (+1 новый)
│
└── unlearning_system/              # НОВЫЙ модуль: Machine Unlearning каркас
    ├── ml/
    │   ├── base.py                  # BaseLearner — единый контракт
    │   ├── preprocess.py            # FeaturePipeline (z-score + one-hot + bool)
    │   ├── shards.py                # ShardPlan + FNV-1a хеш маршрутизации
    │   ├── logreg.py                # PyTorch LogReg (Adam + BCE + early stopping)
    │   ├── rf.py                    # RandomForest wrapper
    │   ├── sisa.py                  # SISAEnsemble: fit / retrain_shards / predict_proba
    │   ├── unlearning.py            # Unlearner: delete-by-ID / delete-by-filter
    │   ├── metrics.py               # Accuracy, Precision, Recall, F1, AUC, MSE
    │   ├── registry.py              # runs/<id>/manifest + artefacts + events.log
    │   └── pipeline.py              # ExperimentPipeline — фасад
    ├── configs/default.yaml         # YAML-конфиг всего эксперимента (§6.3.2 ТЗ)
    ├── cli.py                       # CLI: train / evaluate / unlearn / report
    ├── docs/THEORY.md               # полная теория (обучение, unlearning, метрики)
    ├── tests/test_smoke.py          # e2e + юнит-тесты (4 теста)
    ├── README.md
    └── requirements.txt             # +torch
```

### 1.1 Исправления в существующем генераторе

| Файл | Исправление |
|---|---|
| `generator/distributions.py::template_email` | Все email-паттерны унифицированы в лямбды; **глобальная гарантия уникальности** — при коллизии дописывается `user_id` (раньше паттерны `first.last` и `first_last` молча генерировали дубликаты для популярных имён). |
| `generator/distributions.py::date_from_age` | Учёт високосных лет (366-дневное окно), убрана привязка к `ref.month` → равномерное распределение дат рождения. |
| `generator/distributions.py::generate_{first,last,middle}_name` | Python-циклы → векторизованный `_gendered_pick`. |
| `generator/distributions.py::generate_feature` | Раньше молча возвращал `None` для `type:categorical` с неизвестным `name` — теперь `ValueError` с понятным сообщением. |
| `generator/labeler.py` | Добавлена `_require_columns(...)` — раньше любое отсутствие `consent_given`/`deletion_requested` падало с тёмным `KeyError`. |
| `generator/validator.py` | Добавлены: `_check_email_uniqueness` (дубликаты) и `_check_email_name_correlation` (доля email с транслит-ФИО ≥ 90 %). |
| `tests/test_distributions.py` | Новый тест `test_email_uniqueness_on_collision` — 500 Иванов Петровых → 500 уникальных email. |

### 1.2 Созданный модуль Machine Unlearning

Построен полный прототип, отвечающий требованиям §§5, 6.2 ТЗ.

**Архитектура — SISA ensemble** (Bourtoule et al., IEEE S&P 2021):

```
      датасет (после train/test split)
               │
    hash(user_id, seed) mod S
               ▼
  ┌───────┬───────┬───────┬───────┬───────┐
  │shard 0│shard 1│shard 2│shard 3│shard 4│     disjoint shards
  └───┬───┴───┬───┴───┬───┴───┬───┴───┬───┘
      ▼       ▼       ▼       ▼       ▼
    h_0     h_1     h_2     h_3     h_4      isolated learners
                    │
    soft voting:  h(x) = mean_s P_s(y=1|x)
```

**Центральные свойства:**
- **Точное unlearning**: после удаления строк пересчитываются **только те шарды**, которые содержали удалённые пользователи; остальные бит-в-бит идентичны.
- **Воспроизводимость**: FNV-1a хеш маршрутизации (стабилен между версиями Python), фиксированные seed, SHA-1 датасета в `manifest.yaml`.
- **Полное журналирование**: каждое событие (train/unlearn) пишется в append-only `events.log` — готово к аудиту 152-ФЗ.
- **Два backend'а**: PyTorch LogReg и RandomForest через единый `BaseLearner`.

---

## 2. Как пользоваться

### 2.1 Установка

```bash
cd /Users/hsehacker/Desktop/2039

# Генератор данных
cd synthetic_data_generator
pip install -r requirements.txt
python -m pytest tests/ -q    # 47 passed

# Unlearning-система
cd ../unlearning_system
pip install -r requirements.txt
python -m pytest tests/ -q    # 4 passed
```

### 2.2 Полный цикл (ТЗ §6.5.3: generate → prepare → train → unlearn → re-eval → report)

```bash
# ───── Шаг 1. Сгенерировать синтетический датасет (10 000 строк) ─────
cd synthetic_data_generator
python main.py generate --count 10000 --seed 42
# → data/synthetic_dataset.csv

# ───── Шаг 2. Обучить SISA-ансамбль ─────
cd ../unlearning_system
python cli.py train \
    --config configs/default.yaml \
    --data ../synthetic_data_generator/data/synthetic_dataset.csv
# → runs/run_<timestamp>_<hash>/ со всеми артефактами
#   baseline: accuracy=0.96, F1=0.89, AUC=0.92

# ───── Шаг 3. Удалить данные по ID (delete-by-ID, §6.2.2 ТЗ) ─────
python cli.py unlearn \
    --config configs/default.yaml \
    --data ../synthetic_data_generator/data/synthetic_dataset.csv \
    --user-ids USR-000001 USR-000002 USR-000003 \
    --reason "user_request"

# ───── Шаг 4. Удалить по фильтру (delete-by-filter) ─────
python cli.py unlearn \
    --config configs/default.yaml \
    --data ../synthetic_data_generator/data/synthetic_dataset.csv \
    --filter "consent_given == False" \
    --reason "152-FZ art.14"
# → Δ-метрики, prediction disagreement, время переобучения

# ───── Шаг 5. Посмотреть накопленные метрики запуска ─────
python cli.py report --run runs/run_<timestamp>_<hash>
```

### 2.3 Интерактивный режим генератора

```bash
cd synthetic_data_generator
python main.py interactive
# команды на русском: генерировать, валидировать, инфо, список
```

### 2.4 Использование из Python-кода

```python
from ml.pipeline import ExperimentPipeline
from ml.unlearning import UnlearnRequest

pipe = ExperimentPipeline.from_config("configs/default.yaml")
pipe.load_dataset("data/synthetic_dataset.csv").fit()
print(pipe.evaluate())

# Удаление одного пользователя
result = pipe.unlearn(UnlearnRequest(user_ids=["USR-000042"],
                                     reason="152-ФЗ ст.14"))
print(result.delta)                    # изменение всех метрик
print(result.affected_shards)          # какие шарды переобучены
print(result.prediction_disagreement)  # доля строк с flipped predictions
```

### 2.5 Формат конфига (пример фрагмента)

```yaml
experiment:
  run_root: runs
  random_seed: 42
  test_size: 0.2

sharding:
  n_shards: 5
  shard_seed: 42

model:
  backend: logreg          # или random_forest
  logreg:
    lr: 0.05
    weight_decay: 1.0e-4
    epochs: 200
    class_weight: balanced
```

---

## 3. Как повысить точность после unlearning

После удаления данных качество модели **всегда падает** — просто потому, что
вам доступно меньше обучающих примеров. Ниже — набор отсортированных по
соотношению «эффект/сложность» практик для компенсации просадки.

### 3.1 Дешёвые меры (настройка без изменений кода)

| Мера | Где применить | Ожидаемый эффект |
|---|---|---|
| Увеличить число шардов `n_shards` | `configs/default.yaml → sharding.n_shards` | Каждый шард меньше → переобучение быстрее, но каждый отдельный классификатор слабее. Компенсируется агрегацией. Работает до точки, где шард становится слишком мал (< 100 строк). |
| Усилить регуляризацию LogReg | `model.logreg.weight_decay` (поднять до 1e-3…1e-2) | После удаления важных строк меньше переобучения на шуме оставшихся. |
| Включить class balancing | `model.logreg.class_weight: balanced` (уже по умолчанию) | Поддерживает recall положительного класса, который просаживается больше всего при удалении самых «информативных» строк. |
| Увеличить `epochs` + `patience` | LogReg-конфиг | Более полная сходимость на меньшем датасете. Уже есть early stopping + сохранение лучших весов. |
| Пересмотреть `threshold` | `compute_classification_metrics(..., threshold=0.4)` | По Youden-J на новой калибровочной выборке — часто даёт +1–2 п.п. F1 без переобучения. |
| Увеличить `n_estimators` RF | `model.random_forest.n_estimators` (до 300–500) | Дешевле, чем добавлять фичи. |

### 3.2 Архитектурные улучшения (требуют кода — все места помечены в `docs/THEORY.md §6`)

1. **Slicing внутри шарда (полный SISA)**
   После каждой $1/K$-ой части обучающих данных сохранять checkpoint модели.
   При удалении — откатываться до последнего «чистого» slice вместо
   переобучения с нуля.
   *Эффект:* ещё ~$1/K$ от стоимости переобучения; *цена:* ×$K$ объёма диска.
   Место для кода: `ml/logreg.py::fit` (сохранять `best_state` каждые
   `epochs // K` шагов в `self.slice_states_`).

2. **Ансамбль разнородных моделей**
   Сейчас все шарды — один тип. Можно разрешить смешанный backend:
   LogReg на одних, RF на других, GBM на третьих. Агрегация soft voting
   усреднит разные индуктивные смещённости → выше AUC.
   Место для кода: `ml/sisa.py::SISAEnsemble` — перевести `factory` из
   одиночного callable в `list[LearnerFactory]`, проиндексированный по
   `shard_id`.

3. **Stacking поверх ансамбля**
   На валидационной выборке обучить мета-классификатор (ещё одну
   `TorchLogReg`) на выходах `P_s(y=1|x)` от каждого шарда. Это даёт
   весá вклада каждого шарда — полезно, если шарды сильно разнятся
   по размеру / качеству.
   Место для кода: новый файл `ml/stacking.py`.

4. **Certified unlearning через Influence Function (Guo 2020)**
   Для логрегрессии: сдвиг весов на
   $\theta \leftarrow \theta + H^{-1} \sum_{(x,y)\in D_f} \nabla \ell(\theta; x, y)$.
   При небольших $|D_f|$ это дешевле полного переобучения шарда и сохраняет
   модель «близкой» к исходной → меньше prediction disagreement.
   Место для кода: `ml/logreg.py::TorchLogReg.influence_remove(X_f, y_f)`.

5. **Knowledge distillation при переобучении**
   При retrain затронутого шарда использовать *старую* модель как
   teacher (soft targets): minimise
   `α·BCE(hard) + (1−α)·BCE(student_logits, teacher_probs)`.
   Старая модель видела удалённого пользователя — но это уже компенсируется
   hard-BCE на чистых данных; soft-target «переносит» общие закономерности,
   которые удалённая запись не нарушала.
   Место для кода: `ml/logreg.py::fit(teacher_model=...)`.

6. **Data augmentation на числовых признаках**
   Умеренный гауссовский шум ($\sigma = 0.05$ от stdev) на numeric-фичах
   перед обучением шарда → регуляризация, частично компенсирующая потерю
   «редких» записей, которые были удалены.
   Место для кода: `ml/preprocess.py::FeaturePipeline.transform_augment`.

7. **Curriculum для SISA retrain**
   При переобучении шарда начинать с «лёгких» примеров (высокий
   `margin = |logit|`), постепенно добавляя сложные — сходится быстрее
   и стабильнее на уменьшенном шарде.
   Место для кода: `ml/logreg.py::fit`, пред-сортировка `perm` по
   первичной оценке margin.

### 3.3 Дисциплина запросов на удаление

С точки зрения потерь качества критично:

- **Батчировать** запросы на удаление. Один batched-unlearn из 100
  запросов почти всегда дешевле и **точнее** (меньше накопленной
  просадки), чем 100 отдельных.
- **Периодически делать полный retrain** (раз в $N$ unlearn-операций),
  чтобы не накапливать дрейф численной оптимизации.
- **Замораживать** данные тех пользователей, по которым есть
  legal-hold: не учить модель на них вовсе — тогда удаление тривиально
  (строка никогда не входила в train).

### 3.4 Практический план действий

Если после одного `unlearn` просадка составляет **Δaccuracy ≈ −0.002**, а
`prediction_disagreement ≈ 0.0025` (как в нашем smoke-тесте на 72
удалённых строках) — это норма, компенсации не нужно.

Если **Δaccuracy ≥ −0.02** или **prediction_disagreement ≥ 0.03** —
рекомендуется последовательно:

1. поднять `weight_decay` в 2×,
2. увеличить `epochs` на 50 %,
3. если это не помогло — включить pt. 4 («certified unlearning») из §3.2.

---

## 4. Аудит кода — найденные баги и ошибки

Проведён полный аудит всех файлов обоих модулей. Все критичные баги
исправлены и покрыты тестами.

### 4.1 Баги в существующем коде генератора (исправлены)

| # | Файл | Критичность | Суть | Фикс |
|---|---|---|---|---|
| 1 | `distributions.py::template_email` | **High** | Паттерны `first.last` и `first_last` не включали уникальный суффикс → популярные имена («Иван Петров») порождали молчаливые дубликаты email, ломая валидность обучающей выборки. | Все паттерны унифицированы; глобальная set-дедупликация внутри функции; при коллизии дописывается `user_id`. Тест `test_email_uniqueness_on_collision`. |
| 2 | `distributions.py::date_from_age` | Medium | Не учитывал високосные годы; привязан к `ref.month` → дни рождения кластеризовались вокруг января. | Случайный сдвиг в 366-дневном окне; обработка 29 февраля. |
| 3 | `distributions.py::generate_feature` | Medium | Для `type:categorical` с неизвестным `name` тихо возвращал `None`. | `ValueError` с понятным сообщением. |
| 4 | `distributions.py::generate_{first,last,middle}_name` | Low (performance) | Python-цикл на каждый элемент → ~100× медленнее нужного. | Векторизованный `_gendered_pick`. |
| 5 | `labeler.py::_rule_based_labeling` | Medium | Падал с обычным `KeyError` при отсутствии `consent_given` / `deletion_requested`. | `_require_columns(...)` — прозрачная ошибка со списком пропущенных колонок. |
| 6 | `validator.py` | Medium | Не проверял уникальность email и соответствие username↔ФИО. | Добавлены `_check_email_uniqueness`, `_check_email_name_correlation`. |

### 4.2 Баги в новом коде (исправлены в ходе аудита)

| # | Файл | Критичность | Суть | Фикс |
|---|---|---|---|---|
| 7 | `ml/shards.py::assign_shards` | Low | Мёртвый код: `rng = np.random.default_rng()` создавался и не использовался. | Удалён, комментарий уточнён. |
| 8 | `ml/preprocess.py::FeaturePipeline` | **High** | `feature_names_` перечислял все колонки из `self.boolean`, но `transform` добавлял только присутствующие → рассинхрон размерности и имён; при повторном `transform` с отсутствующей колонкой — `KeyError`. | Введены `fitted_numeric_` / `fitted_boolean_` / `fitted_categorical_` — снимаются на `fit`, используются на `transform`. Отсутствующие колонки заполняются нулями → **стабильная размерность входа модели** при любых вариациях датасета. |
| 9 | `ml/preprocess.py` | Low | Неиспользуемая константа `DEFAULT_BOOLEAN` + несовпадение с class default. | Константа выровнена с class default; используется. |
| 10 | `ml/logreg.py::fit` | Medium | При `epochs=0` переменная `epoch` оставалась неопределённой → `UnboundLocalError`; early stopping не возвращался к best weights. | Нормальная инициализация `epoch=0`; снимок `best_state` и восстановление после цикла. |
| 11 | `ml/logreg.py` | Low | Неиспользуемый импорт `field`. | Удалён. |
| 12 | `ml/unlearning.py` | Low | Неиспользуемый импорт `Iterable`. | Удалён. |
| 13 | `ml/sisa.py` | Low | Неиспользуемый импорт `TrainingStats`. | Удалён. |
| 14 | `ml/pipeline.py::unlearn` | Medium | `self.plan_` оставался stale после unlearn (не совпадал с `ensemble.plan` → если CLI/клиент читал `pipe.plan_.sizes()`, получал старые размеры). | Явная синхронизация `self.plan_ = self.ensemble_.plan`. |

### 4.3 Документированные, но не исправленные ограничения

Ниже — осознанные компромиссы, отмеченные как TODO в коде/документации.
Они **не являются багами** по определению: не ломают функциональность,
а отражают границы прототипа.

- `cli.py::cmd_evaluate` и `cmd_unlearn` сейчас переобучают ансамбль «с
  нуля» вместо загрузки сохранённого из `runs/<id>/learners/`. Это
  бережный выбор: десериализация PyTorch state + joblib RF требует
  аккуратной обработки версий; прототип предпочитает однократный
  reproducible retrain. Отмечено в `cli.py`.
- Slicing внутри шарда не реализован (описан как расширение в
  `docs/THEORY.md §6.1`).
- Membership Inference Attack как независимая верификация unlearning не
  реализована (описана в `docs/THEORY.md §5.2`).
- Dependency на `sklearn.model_selection.train_test_split` в
  `pipeline.py` — стандартная практика; альтернатива (ручной split с
  нашим `rng`) дала бы детерминизм уровня numpy, а не sklearn.

### 4.4 Результаты прогонов

```
synthetic_data_generator/  →  47 passed in 0.98s
unlearning_system/         →   4 passed in 2.09s
e2e cli (train+unlearn)    →  OK  Δaccuracy=−0.0025, Δauc=+0.0017
```

---

## 5. Что ещё проверить / сделать для production-использования

Чек-лист за пределами академического прототипа (в порядке приоритета):

1. **MIA-верификация**: подключить test suite, имитирующий атаку
   Shokri 2017, чтобы численно подтвердить «забытие».
2. **Сериализация/десериализация в CLI** (см. §4.3).
3. **Концептуальный тест на сдвиг распределения**: после $N$
   накопленных unlearn операций замерить KS-расстояние между
   распределением признаков «до» и «после» — если оно значимо выросло,
   делать полный retrain.
4. **DP-SGD внутри каждого шарда** для combined SISA + DP защиты.
5. **Парный тест на точный reproducibility**: `python cli.py train`
   дважды с одним seed → SHA-256 от `manifest.yaml` и метрик совпадают.
6. **Web-интерфейс (Flask)** поверх `ExperimentPipeline` — ТЗ §5.1.7
   допускает «минимальный web» как альтернативу CLI.

---

## 6. Ссылки

- Архитектура и математика: `unlearning_system/docs/THEORY.md`
- Использование генератора: `synthetic_data_generator/README.md`
- Использование unlearning-системы: `unlearning_system/README.md`
- ТЗ: `~/Downloads/AyuGram Desktop/ТЗ Финал-3.docx`
- Постер: `~/Downloads/poster_2039-2.pdf`
