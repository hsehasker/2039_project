#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic data generator - CLI with Russian interface.

Supports four commands:
  generate   - create a new synthetic dataset
  validate   - verify an existing dataset
  info       - show dataset statistics
  list       - show all datasets in data/ folder
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from generator.core import SyntheticDataGenerator

# ─── Color output ────────────────────────────────────────────────────────────

class _COLOR:
    """ANSI codes for terminal color output."""
    RESET       = "\033[0m"
    BOLD        = "\033[1m"
    GREEN       = "\033[32m"
    YELLOW      = "\033[33m"
    RED         = "\033[31m"
    BLUE        = "\033[34m"
    LIGHT_BLUE  = "\033[36m"
    GREY        = "\033[90m"

def _color(text: str, *codes: str) -> str:
    """Wrap text with ANSI codes (only if output is a terminal)."""
    if sys.stdout.isatty():
        return "".join(codes) + text + _COLOR.RESET
    return text

def ok(text: str) -> str:
    return _color(f"✓ {text}", _COLOR.GREEN)

def warning(text: str) -> str:
    return _color(f"⚠ {text}", _COLOR.YELLOW)

def error(text: str) -> str:
    return _color(f"✗ {text}", _COLOR.RED)

def title(text: str) -> str:
    return _color(text, _COLOR.BOLD, _COLOR.LIGHT_BLUE)

def separator(width: int = 60) -> str:
    return _color("─" * width, _COLOR.GREY)


# ─── Logging setup ───────────────────────────────────────────────────────────

def _setup_logging(level: str = "INFO", save: bool = True) -> None:
    """Configure logging with file and terminal output."""
    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    time_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_file = log_dir / f"run_{time_label}.log"

    handlers: list[logging.Handler] = []

    # Console handler - only warnings and above
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING)
    handlers.append(console)

    if save:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        handlers=handlers,
    )

    if save:
        logging.getLogger().info("Лог-файл: %s", log_file)

    return log_file if save else None


# ─── Output formatting ───────────────────────────────────────────────────────

def _print_progress_bar(value: float, width: int = 20) -> str:
    """Draw a text progress bar for proportions."""
    filled = int(value * width)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    return _color(bar, _COLOR.GREEN)


def _format_number(number: float, precision: int = 2) -> str:
    """Format number with thousands separator."""
    if isinstance(number, int) or number == int(number):
        return f"{int(number):,}".replace(",", " ")
    return f"{number:,.{precision}f}".replace(",", " ")


def _check_status(passed: bool) -> str:
    """Return formatted status string based on check result."""
    return ok("OK") if passed else error("ОШИБКА")


# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_generate(args: argparse.Namespace) -> None:
    """Handle the 'generate' command."""
    config_path = Path(args.config)

    if not config_path.exists():
        print(error(f"Файл конфигурации не найден: {config_path}"))
        sys.exit(1)

    log_file_path = _setup_logging(save=True)

    print()
    print(title("╔══════════════════════════════════════════════════════╗"))
    print(title("║     Генератор синтетических данных — HSE MIEM        ║"))
    print(title("╚══════════════════════════════════════════════════════╝"))
    print()

    # Generation parameters
    print(title("Параметры генерации:"))
    print(separator())
    if args.count:
        print(f"  Количество записей : {_color(str(args.count), _COLOR.YELLOW)}")
    else:
        print(f"  Количество записей : {_color('из конфига', _COLOR.GREY)}")
    if args.seed is not None:
        print(f"  Зерно случайности  : {_color(str(args.seed), _COLOR.YELLOW)}")
    else:
        print(f"  Зерно случайности  : {_color('из конфига', _COLOR.GREY)}")

    labeling_mode_label = {
        "rule": "Детерминированный (правила)",
        "probabilistic": "Вероятностный (сигмоид)",
        None: "из конфига",
    }.get(args.labeling_mode, args.labeling_mode)
    print(f"  Режим разметки     : {_color(labeling_mode_label, _COLOR.YELLOW)}")
    print(f"  Конфиг             : {_color(str(config_path), _COLOR.GREY)}")
    if log_file_path:
        print(f"  Лог-файл           : {_color(str(log_file_path), _COLOR.GREY)}")
    print()

    # Generation
    print(title("Генерация..."))
    print(separator())

    try:
        generator = SyntheticDataGenerator(
            config_path=config_path,
            n_samples=args.count,
            seed=args.seed,
            labeling_mode=args.labeling_mode,
        )

        print(f"  {'Признаки':30s}", end="", flush=True)
        df = generator.generate()
        print(ok(f"сгенерировано {len(df.columns)} признаков, {_format_number(len(df))} записей"))

    except Exception as e:
        print()
        print(error(f"Ошибка генерации: {e}"))
        sys.exit(1)

    # Validation
    print()
    print(title("Валидация датасета:"))
    print(separator())

    validation_results = generator.validate(df)
    check_names = {
        "nan":               "Пропущенные значения",
        "user_id_uniqueness": "Уникальность user_id",
        "distributions":     "Распределения признаков",
        "clipping":          "Ограничения (clip)",
        "correlations":      "Корреляции",
        "date_formats":      "Форматы дат",
        "class_balance":     "Баланс классов",
    }
    for key, data in validation_results["checks"].items():
        check_name = check_names.get(key, key)
        passed = data.get("passed", True)
        status = _check_status(passed)
        print(f"  {check_name:35s} {status}")

    print()
    overall_passed = validation_results["passed"]
    if overall_passed:
        print(f"  Итог валидации: {ok('ПРОЙДЕНА')}")
    else:
        print(f"  Итог валидации: {error('ПРОВАЛЕНА')}")

    # Saving
    print()
    print(title("Сохранение:"))
    print(separator())

    try:
        output_path = generator.save(df)
        file_size = output_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        print(f"  Файл   : {_color(str(output_path), _COLOR.GREEN)}")
        print(f"  Размер : {file_size_mb:.2f} МБ")
    except Exception as e:
        print(error(f"Ошибка сохранения: {e}"))
        sys.exit(1)

    # Dataset statistics
    print()
    print(title("Краткая статистика:"))
    print(separator())

    info = generator.get_info(df)
    print(f"  Размерность: {info['shape'][0]:,} строк × {info['shape'][1]} столбцов".replace(",", " "))

    if "class_balance" in info:
        class_balance = info["class_balance"]
        positive_rate = class_balance["positive_rate"]
        print(f"  Баланс меток ({class_balance['target_column']}):")
        print(f"    label=1: {_format_number(class_balance['count_true'])} ({positive_rate:.1%})  {_print_progress_bar(positive_rate)}")
        print(f"    label=0: {_format_number(class_balance['count_false'])} ({1-positive_rate:.1%})  {_print_progress_bar(1 - positive_rate)}")

    # Numeric features
    key_features = ["age", "purchase_amount", "purchase_count", "avg_check", "days_since_last_purchase"]
    column_labels = {
        "age":                      "Возраст",
        "purchase_amount":          "Сумма покупок",
        "purchase_count":           "Кол-во покупок",
        "avg_check":                "Средний чек",
        "days_since_last_purchase": "Дней с покупки",
    }
    print()
    print(f"  {'Признак':<25} {'Среднее':>10} {'Ст.откл':>10} {'Мин':>10} {'Макс':>10}")
    print(f"  {separator(67)}")
    for col in key_features:
        if col in info["numeric_summary"]:
            stats = info["numeric_summary"][col]
            label = column_labels.get(col, col)
            print(
                f"  {label:<25}"
                f" {stats['mean']:>10.1f}"
                f" {stats['std']:>10.1f}"
                f" {stats['min']:>10.1f}"
                f" {stats['max']:>10.1f}"
            )

    # Categorical features
    if "region" in df.columns:
        print()
        print(f"  {'Регионы':}")
        region_counts = df["region"].value_counts(normalize=True)
        for region, share in region_counts.items():
            bar = _print_progress_bar(share, 15)
            print(f"    {region:<15} {share:5.1%}  {bar}")

    if "preferred_category" in df.columns:
        print()
        print(f"  {'Категории товаров':}")
        category_counts = df["preferred_category"].value_counts(normalize=True)
        for category, share in category_counts.items():
            bar = _print_progress_bar(share, 15)
            print(f"    {category:<20} {share:5.1%}  {bar}")

    print()
    print(separator())
    if overall_passed:
        print(ok("Генерация завершена успешно!"))
    else:
        print(warning("Генерация завершена с предупреждениями валидации."))
    print()


def cmd_validate(args: argparse.Namespace) -> None:
    """Handle the 'validate' command."""
    config_path = Path(args.config)
    _setup_logging()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(error(f"Файл не найден: {input_path}"))
        sys.exit(1)

    print()
    print(title("Валидация датасета"))
    print(separator())
    print(f"  Файл   : {_color(str(input_path), _COLOR.YELLOW)}")

    try:
        if input_path.suffix == ".csv":
            df = pd.read_csv(input_path)
        else:
            df = pd.read_parquet(input_path)
    except Exception as e:
        print(error(f"Ошибка чтения файла: {e}"))
        sys.exit(1)

    print(f"  Размер : {len(df):,} строк × {len(df.columns)} столбцов".replace(",", " "))
    print()

    generator = SyntheticDataGenerator(config_path=config_path)
    validation_results = generator.validate(df)

    if args.json:
        print(json.dumps(validation_results, indent=2, ensure_ascii=False, default=str))
    else:
        check_names = {
            "nan":               "Пропущенные значения",
            "user_id_uniqueness": "Уникальность user_id",
            "distributions":     "Распределения признаков",
            "clipping":          "Ограничения (clip)",
            "correlations":      "Корреляции",
            "date_formats":      "Форматы дат",
            "class_balance":     "Баланс классов",
        }

        for key, data in validation_results["checks"].items():
            check_name = check_names.get(key, key)
            passed = data.get("passed", True)
            status = _check_status(passed)
            print(f"  {check_name:35s} {status}")

            # Correlation details
            if key == "correlations" and "pairs" in data:
                pair_labels = {
                    "purchase_amount_count":     "сумма ↔ кол-во",
                    "purchase_amount_avg_check":  "сумма ↔ ср.чек",
                    "days_since_purchase_count":  "дней ↔ кол-во",
                }
                for pair_key, pair_data in data["pairs"].items():
                    pair_label = pair_labels.get(pair_key, pair_key)
                    target = pair_data["target"]
                    actual = pair_data["actual"]
                    icon = "✓" if pair_data["passed"] else "✗"
                    color = _COLOR.GREEN if pair_data["passed"] else _COLOR.RED
                    print(f"    {_color(icon, color)} {pair_label:<22} цель={target:+.2f}  факт={actual:+.2f}")

            # Date format details
            if key == "date_formats" and "columns" in data:
                for col, col_data in data["columns"].items():
                    invalid_count = col_data.get("invalid_count", 0)
                    if invalid_count > 0:
                        print(f"    ✗ {col}: {invalid_count} неверных значений")

        print()
        overall_passed = validation_results["passed"]
        if overall_passed:
            print(ok("Датасет прошёл валидацию!"))
        else:
            print(error("Датасет НЕ прошёл валидацию."))

    print()

    if not validation_results["passed"]:
        sys.exit(1)


def cmd_info(args: argparse.Namespace) -> None:
    """Handle the 'info' command."""
    config_path = Path(args.config)
    _setup_logging(save=False)

    if args.input_file:
        input_path = Path(args.input_file)
    else:
        data_dir = Path.cwd() / "data"
        all_files = sorted(
            list(data_dir.glob("*.csv")) + list(data_dir.glob("*.parquet")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not all_files:
            print(error("Не найдено датасетов в папке data/"))
            sys.exit(1)
        input_path = all_files[0]

    if not input_path.exists():
        print(error(f"Файл не найден: {input_path}"))
        sys.exit(1)

    try:
        if input_path.suffix == ".csv":
            df = pd.read_csv(input_path)
        else:
            df = pd.read_parquet(input_path)
    except Exception as e:
        print(error(f"Ошибка чтения файла: {e}"))
        sys.exit(1)

    generator = SyntheticDataGenerator(config_path=config_path)
    info = generator.get_info(df)

    print()
    print(title("╔══════════════════════════════════════════════════════╗"))
    print(title("║              Статистика датасета                     ║"))
    print(title("╚══════════════════════════════════════════════════════╝"))
    print()

    # Meta information
    print(title("Файл и размер:"))
    print(separator())
    file_size = input_path.stat().st_size / (1024 * 1024)
    modified_time = datetime.fromtimestamp(input_path.stat().st_mtime).strftime("%d.%m.%Y %H:%M")
    print(f"  Путь     : {_color(str(input_path), _COLOR.BLUE)}")
    print(f"  Формат   : {input_path.suffix.lstrip('.')}")
    print(f"  Размер   : {file_size:.2f} МБ")
    print(f"  Изменён  : {modified_time}")
    print(f"  Строк    : {_color(_format_number(info['shape'][0]), _COLOR.YELLOW)}")
    print(f"  Столбцов : {_color(str(info['shape'][1]), _COLOR.YELLOW)}")

    # Columns
    print()
    print(title("Столбцы:"))
    print(separator())
    dtype_labels = {
        "int64": "целое", "int32": "целое", "float64": "вещественное",
        "object": "строка", "bool": "булево",
    }
    column_list = list(zip(info["columns"], [info["dtypes"][c] for c in info["columns"]]))
    # Two-column output
    half = (len(column_list) + 1) // 2
    for i in range(half):
        left = column_list[i]
        left_type = dtype_labels.get(left[1], left[1])
        line = f"  {left[0]:<28} {_color(left_type, _COLOR.GREY):<18}"
        if i + half < len(column_list):
            right = column_list[i + half]
            right_type = dtype_labels.get(right[1], right[1])
            line += f"  {right[0]:<28} {_color(right_type, _COLOR.GREY)}"
        print(line)

    # Class balance
    if "class_balance" in info:
        class_balance = info["class_balance"]
        positive_rate = class_balance["positive_rate"]
        print()
        print(title(f"Баланс меток ({class_balance['target_column']}):"))
        print(separator())
        print(f"  label=1 (подлежат удалению) : {_format_number(class_balance['count_true'])} ({positive_rate:.2%})")
        print(f"  label=0 (остаются)          : {_format_number(class_balance['count_false'])} ({1-positive_rate:.2%})")
        print()
        print(f"  label=1  {_print_progress_bar(positive_rate, 40)}  {positive_rate:.1%}")
        print(f"  label=0  {_print_progress_bar(1 - positive_rate, 40)}  {1-positive_rate:.1%}")

    # Numeric features
    print()
    print(title("Числовые признаки:"))
    print(separator(70))
    print(f"  {'Признак':<30} {'Среднее':>10} {'Ст.откл':>10} {'Мин':>8} {'Макс':>10}")
    print(f"  {separator(66)}")
    column_labels = {
        "age":                      "Возраст",
        "purchase_amount":          "Сумма покупок (руб.)",
        "purchase_count":           "Кол-во покупок",
        "avg_check":                "Средний чек (руб.)",
        "days_since_last_purchase": "Дней с последней покупки",
        "label":                    "Метка (label)",
    }
    for col, stats in info["numeric_summary"].items():
        label = column_labels.get(col, col)
        print(
            f"  {label:<30}"
            f" {stats['mean']:>10.1f}"
            f" {stats['std']:>10.1f}"
            f" {stats['min']:>8.1f}"
            f" {stats['max']:>10.1f}"
        )

    # Categorical features
    categorical_columns = {
        "region": "Распределение по регионам",
        "device_type": "Типы устройств",
        "preferred_category": "Категории товаров",
    }
    for col, header in categorical_columns.items():
        if col in df.columns:
            print()
            print(title(f"{header}:"))
            print(separator())
            value_counts = df[col].value_counts()
            value_counts_norm = df[col].value_counts(normalize=True)
            for value in value_counts.index:
                share = value_counts_norm[value]
                count = value_counts[value]
                bar = _print_progress_bar(share, 20)
                print(f"  {value:<22} {count:>7,}  {share:5.1%}  {bar}".replace(",", " "))

    # Boolean flags
    flags = {
        "deletion_requested": ("Запрос на удаление", _COLOR.RED),
        "consent_given": ("Согласие на обработку", _COLOR.GREEN),
        "is_premium": ("Премиум-статус", _COLOR.YELLOW),
        "is_subscribed": ("Подписка", _COLOR.BLUE),
    }
    print()
    print(title("Флаговые признаки:"))
    print(separator())
    for col, (label, color) in flags.items():
        if col in df.columns:
            count_true = df[col].astype(bool).sum()
            rate_true = count_true / len(df)
            bar = _print_progress_bar(rate_true, 20)
            print(
                f"  {label:<30} {_color(f'{count_true:>6,}', color).replace(',', ' ')} "
                f" {rate_true:5.1%}  {bar}"
            )

    print()
    print(separator())
    print()


def cmd_list(args: argparse.Namespace) -> None:
    """Handle the 'list' command - show all datasets in data/ folder."""
    data_dir = Path.cwd() / "data"

    if not data_dir.exists():
        print(error("Папка data/ не найдена."))
        sys.exit(1)

    all_files = sorted(
        list(data_dir.glob("*.csv")) + list(data_dir.glob("*.parquet")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    print()
    print(title("Датасеты в папке data/:"))
    print(separator(70))

    if not all_files:
        print(warning("  Датасеты не найдены."))
        print()
        return

    for i, file_path in enumerate(all_files, 1):
        file_size = file_path.stat().st_size / (1024 * 1024)
        modified_time = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%d.%m.%Y %H:%M")
        marker = _color("►", _COLOR.GREEN) if i == 1 else " "
        note = _color(" (последний)", _COLOR.GREEN) if i == 1 else ""
        print(f"  {marker} {file_path.name:<40} {file_size:6.2f} МБ  {modified_time}{note}")

    print()


# ─── Interactive mode ────────────────────────────────────────────────────────

def cmd_interactive(args: argparse.Namespace) -> None:
    """Run interactive menu mode."""
    config_path = args.config
    
    print()
    print(title("╔══════════════════════════════════════════════════════╗"))
    print(title("║     Генератор синтетических данных — HSE MIEM        ║"))
    print(title("║            Интерактивный режим                       ║"))
    print(title("╚══════════════════════════════════════════════════════╝"))
    print()
    print(_color("  Введите 'помощь' для списка команд или 'выход' для завершения.", _COLOR.GREY))
    print()
    
    while True:
        try:
            print(separator(50))
            user_input = input(_color(">>> ", _COLOR.LIGHT_BLUE)).strip().lower()
            
            if not user_input:
                continue
            
            # Parse command and arguments
            parts = user_input.split()
            cmd = parts[0]
            cmd_args = parts[1:]
            
            if cmd in ("выход", "exit", "quit", "q"):
                print()
                print(ok("До свидания!"))
                print()
                break
            
            elif cmd in ("помощь", "help", "h", "?"):
                _print_interactive_help()
            
            elif cmd in ("генерировать", "gen", "g", "1"):
                _interactive_generate(config_path, cmd_args)
            
            elif cmd in ("валидировать", "val", "v", "2"):
                _interactive_validate(config_path, cmd_args)
            
            elif cmd in ("инфо", "info", "i", "3"):
                _interactive_info(config_path, cmd_args)
            
            elif cmd in ("список", "list", "ls", "4"):
                # Create a simple namespace for cmd_list
                list_args = argparse.Namespace()
                cmd_list(list_args)
            
            elif cmd in ("очистить", "clear", "cls"):
                print("\033[2J\033[H", end="")  # Clear terminal
            
            else:
                print(warning(f"Неизвестная команда: '{cmd}'. Введите 'помощь' для списка команд."))
        
        except KeyboardInterrupt:
            print()
            print()
            print(ok("Прервано пользователем. До свидания!"))
            print()
            break
        
        except EOFError:
            print()
            print(ok("До свидания!"))
            print()
            break
        
        except Exception as e:
            print(error(f"Ошибка: {e}"))


def _print_interactive_help() -> None:
    """Print help for interactive mode."""
    print()
    print(title("Доступные команды:"))
    print(separator(50))
    print(f"  {_color('генерировать', _COLOR.GREEN)} [N] [SEED]  — создать датасет")
    print(f"    или: gen, g, 1")
    print(f"    Примеры: генерировать 5000 42")
    print(f"             gen 1000")
    print()
    print(f"  {_color('валидировать', _COLOR.GREEN)} [ФАЙЛ]     — проверить датасет")
    print(f"    или: val, v, 2")
    print(f"    Примеры: валидировать data/my_data.csv")
    print(f"             val  (последний файл)")
    print()
    print(f"  {_color('инфо', _COLOR.GREEN)} [ФАЙЛ]             — статистика датасета")
    print(f"    или: info, i, 3")
    print()
    print(f"  {_color('список', _COLOR.GREEN)}                  — показать все датасеты")
    print(f"    или: list, ls, 4")
    print()
    print(f"  {_color('очистить', _COLOR.GREEN)}                — очистить экран")
    print(f"    или: clear, cls")
    print()
    print(f"  {_color('помощь', _COLOR.GREEN)}                  — показать эту справку")
    print(f"    или: help, h, ?")
    print()
    print(f"  {_color('выход', _COLOR.GREEN)}                   — завершить программу")
    print(f"    или: exit, quit, q")
    print()


def _interactive_generate(config_path: str, cmd_args: list) -> None:
    """Handle generate command in interactive mode."""
    # Parse optional arguments: count and seed
    count = None
    seed = None
    
    if len(cmd_args) >= 1:
        try:
            count = int(cmd_args[0])
        except ValueError:
            print(error(f"Неверное количество: '{cmd_args[0]}'. Ожидается число."))
            return
    
    if len(cmd_args) >= 2:
        try:
            seed = int(cmd_args[1])
        except ValueError:
            print(error(f"Неверное зерно: '{cmd_args[1]}'. Ожидается число."))
            return
    
    # Create namespace with arguments
    args = argparse.Namespace(
        config=config_path,
        count=count,
        seed=seed,
        labeling_mode=None,
        skip_validation=False,
    )
    
    try:
        cmd_generate(args)
    except SystemExit:
        pass  # Don't exit in interactive mode


def _interactive_validate(config_path: str, cmd_args: list) -> None:
    """Handle validate command in interactive mode."""
    # Determine input file
    if cmd_args:
        input_file = cmd_args[0]
    else:
        # Use the latest file
        data_dir = Path.cwd() / "data"
        all_files = sorted(
            list(data_dir.glob("*.csv")) + list(data_dir.glob("*.parquet")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not all_files:
            print(error("Не найдено датасетов в папке data/"))
            return
        input_file = str(all_files[0])
        print(_color(f"  Используется последний файл: {input_file}", _COLOR.GREY))
    
    args = argparse.Namespace(
        config=config_path,
        input_file=input_file,
        json=False,
    )
    
    try:
        cmd_validate(args)
    except SystemExit:
        pass


def _interactive_info(config_path: str, cmd_args: list) -> None:
    """Handle info command in interactive mode."""
    input_file = cmd_args[0] if cmd_args else None
    
    args = argparse.Namespace(
        config=config_path,
        input_file=input_file,
    )
    
    try:
        cmd_info(args)
    except SystemExit:
        pass


# ─── CLI Arguments ───────────────────────────────────────────────────────────

def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Генератор синтетических данных для исследования Machine Unlearning.\n"
            "Проект «Механизмы цифрового забвения» — HSE MIEM."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python main.py interactive                          — интерактивный режим\n"
            "  python main.py generate --count 5000 --seed 42\n"
            "  python main.py generate --labeling-mode probabilistic\n"
            "  python main.py validate --input-file data/synthetic_dataset.csv\n"
            "  python main.py info\n"
            "  python main.py list\n"
        ),
    )

    parser.add_argument(
        "--version", action="version", version="%(prog)s 2.0.0"
    )

    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")

    # ── generate ──────────────────────────────────────────────────────────────
    gen_parser = subparsers.add_parser(
        "generate",
        help="Создать новый синтетический датасет",
        description="Генерирует синтетический датасет e-commerce профилей.",
    )
    gen_parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        metavar="PATH",
        help="Путь к YAML-конфигу (по умолчанию: configs/default.yaml)",
    )
    gen_parser.add_argument(
        "--count", "-n",
        type=int,
        default=None,
        metavar="N",
        help="Количество записей для генерации",
    )
    gen_parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        metavar="SEED",
        help="Зерно генератора случайных чисел",
    )
    gen_parser.add_argument(
        "--labeling-mode", "-l",
        type=str,
        default=None,
        choices=["rule", "probabilistic"],
        dest="labeling_mode",
        metavar="{rule,probabilistic}",
        help="Режим разметки: rule (правила) или probabilistic (вероятностный)",
    )
    gen_parser.add_argument(
        "--skip-validation",
        action="store_true",
        dest="skip_validation",
        help="Пропустить валидацию после генерации",
    )

    # ── validate ──────────────────────────────────────────────────────────────
    val_parser = subparsers.add_parser(
        "validate",
        help="Проверить существующий датасет",
        description="Запускает статистическую валидацию датасета.",
    )
    val_parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        metavar="PATH",
        help="Путь к YAML-конфигу",
    )
    val_parser.add_argument(
        "--input-file", "-i",
        type=str,
        required=True,
        dest="input_file",
        metavar="FILE",
        help="Путь к CSV или Parquet файлу",
    )
    val_parser.add_argument(
        "--json",
        action="store_true",
        help="Вывести результаты в формате JSON",
    )

    # ── info ─────────────────────────────────────────────────────────────────
    info_parser = subparsers.add_parser(
        "info",
        help="Показать статистику по датасету",
        description="Выводит подробную статистику датасета.",
    )
    info_parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        metavar="PATH",
        help="Путь к YAML-конфигу",
    )
    info_parser.add_argument(
        "--input-file", "-i",
        type=str,
        default=None,
        dest="input_file",
        metavar="FILE",
        help="Путь к файлу (по умолчанию: последний в data/)",
    )

    # ── list ──────────────────────────────────────────────────────────────────
    subparsers.add_parser(
        "list",
        help="Показать все датасеты в папке data/",
        description="Выводит список сохранённых датасетов.",
    )

    # ── interactive ───────────────────────────────────────────────────────────
    int_parser = subparsers.add_parser(
        "interactive",
        help="Запустить интерактивный режим (меню)",
        description="Запускает интерактивный режим с меню команд.",
    )
    int_parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        metavar="PATH",
        help="Путь к YAML-конфигу (по умолчанию: configs/default.yaml)",
    )

    return parser


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print()
        sys.exit(0)

    command_map = {
        "generate":     cmd_generate,
        "validate":     cmd_validate,
        "info":         cmd_info,
        "list":         cmd_list,
        "interactive":  cmd_interactive,
    }

    command_map[args.command](args)


if __name__ == "__main__":
    main()
