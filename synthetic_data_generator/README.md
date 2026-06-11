# Генератор синтетических данных

> Проект «Механизмы цифрового забвения» — HSE MIEM

Инструмент для генерации синтетических датасетов e-commerce профилей пользователей. Предназначен для исследования методов Machine Unlearning.

## 📋 Содержание

- [Требования](#требования)
- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Использование](#использование)
  - [Веб-интерфейс](#веб-интерфейс)
  - [Интерактивный режим](#интерактивный-режим)
  - [Командная строка](#командная-строка)
- [Конфигурация](#конфигурация)
- [Структура данных](#структура-данных)
- [Валидация](#валидация)
- [Структура проекта](#структура-проекта)
- [Тестирование](#тестирование)

---

## Требования

- Python 3.10+
- pip (менеджер пакетов Python)

## Установка

### Шаг 1: Клонирование репозитория

```bash
git clone https://github.com/Jkey198/synthetic_data_generator.git
cd synthetic_data_generator
```

### Шаг 2: Создание виртуального окружения (рекомендуется)

```bash
# Создание виртуального окружения
python3 -m venv venv

# Активация (Linux/macOS)
source venv/bin/activate

# Активация (Windows)
venv\Scripts\activate
```

### Шаг 3: Установка зависимостей

```bash
pip install -r requirements.txt
```

Список зависимостей:
- `numpy` — работа с массивами
- `pandas` — работа с данными
- `scipy` — статистические функции
- `scikit-learn` — машинное обучение
- `pyyaml` — работа с YAML конфигурацией
- `pytest` — тестирование
- `flask` — веб-интерфейс

---

## Быстрый старт

### Самый простой способ — интерактивный режим:

```bash
python main.py interactive
```

Это запустит программу в интерактивном режиме, где вы сможете вводить команды.

### Быстрая генерация датасета:

```bash
python main.py generate --count 10000 --seed 42
```

Создаст датасет из 10 000 записей с фиксированным зерном случайности.

---

## Использование

### Веб-интерфейс

Для удобной работы с генератором доступен веб-интерфейс:

```bash
python3 web_app.py
```

Откройте в браузере: http://127.0.0.1:5000

**Возможности веб-интерфейса:**
- Генерация датасетов с настройкой параметров (количество записей, seed, режим разметки)
- Предпросмотр первых 50 записей
- Статистика по классам (label=0 / label=1)
- Скачивание в форматах CSV, JSON, Parquet

---

### Интерактивный режим

Запустите программу в интерактивном режиме:

```bash
python main.py interactive
```

После запуска вы увидите приглашение `>>>`. Доступные команды:

| Команда | Сокращение | Описание |
|---------|------------|----------|
| `генерировать [N] [SEED]` | `gen`, `g`, `1` | Создать датасет |
| `валидировать [ФАЙЛ]` | `val`, `v`, `2` | Проверить датасет |
| `инфо [ФАЙЛ]` | `info`, `i`, `3` | Статистика датасета |
| `список` | `list`, `ls`, `4` | Показать все датасеты |
| `очистить` | `clear`, `cls` | Очистить экран |
| `помощь` | `help`, `h`, `?` | Показать справку |
| `выход` | `exit`, `quit`, `q` | Выйти из программы |

**Примеры команд в интерактивном режиме:**

```
>>> генерировать 5000 42
>>> gen 1000
>>> валидировать
>>> инфо
>>> список
>>> выход
```

**Для выхода:** введите `выход` или нажмите `Ctrl+C`

---

### Командная строка

#### Справка

```bash
python main.py --help
```

#### 1. Генерация датасета (`generate`)

Создаёт новый синтетический датасет.

```bash
python main.py generate [опции]
```

**Опции:**

| Опция | Сокращение | Описание | По умолчанию |
|-------|------------|----------|--------------|
| `--config` | `-c` | Путь к YAML-конфигу | `configs/default.yaml` |
| `--count` | `-n` | Количество записей | из конфига (10000) |
| `--seed` | `-s` | Зерно случайности | из конфига (42) |
| `--labeling-mode` | `-l` | Режим разметки: `rule` или `probabilistic` | из конфига |
| `--skip-validation` | — | Пропустить валидацию | — |

**Примеры:**

```bash
# Базовая генерация (параметры из конфига)
python main.py generate

# Генерация 5000 записей с зерном 123
python main.py generate --count 5000 --seed 123

# Генерация с вероятностной разметкой
python main.py generate --labeling-mode probabilistic

# Использование своего конфига
python main.py generate --config configs/my_config.yaml
```

**Вывод после генерации:**
- Параметры генерации
- Результаты валидации
- Путь к сохранённому файлу
- Краткая статистика

---

#### 2. Валидация датасета (`validate`)

Проверяет существующий датасет на соответствие конфигурации.

```bash
python main.py validate --input-file ФАЙЛ [опции]
```

**Опции:**

| Опция | Сокращение | Описание |
|-------|------------|----------|
| `--input-file` | `-i` | Путь к CSV/Parquet файлу (обязательно) |
| `--config` | `-c` | Путь к YAML-конфигу |
| `--json` | — | Вывести результаты в JSON |

**Примеры:**

```bash
# Валидация файла
python main.py validate --input-file data/synthetic_dataset.csv

# Валидация с выводом в JSON
python main.py validate -i data/synthetic_dataset.csv --json
```

**Проверки валидации:**
- ✓ Пропущенные значения (NaN)
- ✓ Уникальность user_id
- ✓ Распределения признаков
- ✓ Ограничения (clip)
- ✓ Корреляции между признаками
- ✓ Форматы дат
- ✓ Баланс классов

---

#### 3. Информация о датасете (`info`)

Выводит подробную статистику датасета.

```bash
python main.py info [опции]
```

**Опции:**

| Опция | Сокращение | Описание |
|-------|------------|----------|
| `--input-file` | `-i` | Путь к файлу (по умолчанию: последний в data/) |
| `--config` | `-c` | Путь к YAML-конфигу |

**Примеры:**

```bash
# Информация о последнем датасете
python main.py info

# Информация о конкретном файле
python main.py info --input-file data/my_dataset.csv
```

**Выводимая информация:**
- Метаданные файла (размер, дата изменения)
- Список столбцов с типами данных
- Баланс меток (label)
- Статистика числовых признаков
- Распределение категориальных признаков
- Статистика флаговых признаков

---

#### 4. Список датасетов (`list`)

Показывает все датасеты в папке `data/`.

```bash
python main.py list
```

Выводит:
- Название файла
- Размер в МБ
- Дата изменения
- Маркер последнего файла

---

## Конфигурация

Конфигурация задаётся в YAML файле. По умолчанию используется `configs/default.yaml`.

### Структура конфига

```yaml
generator:
  n_samples: 10000              # Количество записей
  random_seed: 42               # Зерно случайности
  output_format: csv            # Формат: csv или parquet
  output_path: data/synthetic_dataset.csv
  reference_date: "2025-01-01"  # Опорная дата для birth_date

features:
  # Определение признаков...

correlations:
  # Корреляции между признаками...

labeling:
  mode: rule                    # rule или probabilistic
  target_column: label

logging:
  level: INFO
  save_to_file: true
```

### Типы признаков

#### 1. Нормальное распределение
```yaml
age:
  distribution: normal
  mean: 35
  std: 10
  clip: [18, 75]    # Ограничение значений
```

#### 2. Логнормальное распределение
```yaml
purchase_amount:
  distribution: lognormal
  mean: 4.5
  std: 1.0
```

#### 3. Экспоненциальное распределение
```yaml
days_since_last_purchase:
  distribution: exponential
  lambda: 0.05
  clip: [0, 365]
```

#### 4. Категориальное распределение
```yaml
region:
  distribution: categorical
  categories:
    Moscow: 0.30
    SPb: 0.20
    Novosibirsk: 0.10
    Ekaterinburg: 0.10
    Other: 0.30
```

#### 5. Бернулли (булево)
```yaml
is_premium:
  distribution: bernoulli
  p: 0.15
```

#### 6. Специальные типы
```yaml
# Последовательный ID
user_id:
  type: sequential_id
  prefix: "USR"

# Email из шаблона
email:
  type: template_email
  domains: {mail.ru: 0.4, gmail.com: 0.3, yandex.ru: 0.3}

# Телефон
phone:
  type: template_phone

# Дата рождения из возраста
birth_date:
  type: derived_from_age

# Случайная дата в диапазоне
registration_date:
  type: random_date_range
  start: "2020-01-01"
  end: "2025-01-01"
```

### Корреляции

```yaml
correlations:
  purchase_amount_count: 0.5       # Сумма ↔ количество покупок
  purchase_amount_avg_check: 0.6   # Сумма ↔ средний чек
  days_since_purchase_count: -0.4  # Дней с покупки ↔ количество
```

### Режимы разметки

**Rule (детерминированный):**
```
label = 1, если deletion_requested = True ИЛИ consent_given = False
```

**Probabilistic (вероятностный):**
```
score = 2.0 × deletion + 1.5 × no_consent + 0.5 × inactive + 0.3 × zero_purchases
prob = sigmoid(score - 2.0)
label ~ Bernoulli(prob)
```

---

## Структура данных

### Генерируемые признаки

| Признак | Тип | Описание |
|---------|-----|----------|
| `user_id` | string | Уникальный идентификатор (USR-000001) |
| `last_name` | string | Фамилия (русские) |
| `first_name` | string | Имя (русские) |
| `middle_name` | string | Отчество (русские) |
| `birth_date` | date | Дата рождения (YYYY-MM-DD) |
| `email` | string | Email адрес |
| `phone` | string | Телефон (+7-9XX-XXX-XX-XX) |
| `age` | int | Возраст (18-75) |
| `region` | string | Регион (Moscow, SPb, и др.) |
| `purchase_amount` | float | Сумма покупок (руб.) |
| `purchase_count` | int | Количество покупок |
| `avg_check` | float | Средний чек (руб.) |
| `days_since_last_purchase` | int | Дней с последней покупки |
| `is_premium` | bool | Премиум-статус |
| `is_subscribed` | bool | Подписка на рассылку |
| `preferred_category` | string | Предпочитаемая категория |
| `device_type` | string | Тип устройства |
| `registration_date` | date | Дата регистрации |
| `consent_given` | bool | Согласие на обработку данных |
| `deletion_requested` | bool | Запрос на удаление данных |
| `label` | int | Метка (0 или 1) |

### Целевая переменная (label)

- `label = 1` — данные пользователя подлежат удалению
- `label = 0` — данные пользователя остаются

---

## Структура проекта

```
synthetic_data_generator/
├── main.py                 # Точка входа CLI
├── web_app.py              # Веб-интерфейс (Flask)
├── requirements.txt        # Зависимости
├── README.md              # Документация
│
├── configs/
│   └── default.yaml       # Конфигурация по умолчанию
│
├── generator/
│   ├── __init__.py
│   ├── core.py            # Основной генератор
│   ├── distributions.py   # Генерация признаков
│   ├── correlations.py    # Инъекция корреляций
│   ├── labeler.py         # Разметка данных
│   └── validator.py       # Валидация датасетов
│
├── templates/
│   └── index.html         # HTML шаблон веб-интерфейса
│
├── static/                # Статические файлы (CSS, JS)
│
├── data/                  # Сгенерированные датасеты
│   └── synthetic_dataset.csv
│
├── logs/                  # Логи выполнения
│   └── run_YYYYMMDDTHHMMSS.log
│
└── tests/                 # Тесты
    ├── __init__.py
    ├── test_distributions.py
    ├── test_correlations.py
    └── test_labeler.py
```

---

## Тестирование

### Запуск всех тестов

```bash
python -m pytest tests/ -v
```

### Запуск конкретного модуля

```bash
python -m pytest tests/test_distributions.py -v
```

### Запуск с покрытием (требуется pytest-cov)

```bash
pip install pytest-cov
python -m pytest tests/ --cov=generator --cov-report=html
```

---

## Примеры использования

### Пример 1: Генерация датасета для обучения

```bash
# Генерация обучающего датасета
python main.py generate --count 50000 --seed 42

# Проверка результата
python main.py info
```

### Пример 2: Генерация нескольких датасетов

```bash
# Датасет 1 с rule-разметкой
python main.py generate -n 10000 -s 1 -l rule

# Переименовать файл вручную, затем:
# Датасет 2 с probabilistic-разметкой  
python main.py generate -n 10000 -s 2 -l probabilistic
```

### Пример 3: Интерактивная работа

```bash
python main.py interactive
```

```
>>> gen 5000 42
>>> инфо
>>> val
>>> список
>>> выход
```

---

## Лицензия

MIT License

---

## Авторы

Проект разработан в рамках исследования «Механизмы цифрового забвения» — HSE MIEM.
