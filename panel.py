#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Единая интерактивная CLI-панель проекта 2039
«Механизмы цифрового забвения» — HSE MIEM.

Объединяет в одном процессе полный цикл эксперимента:

    генерация   →   валидация   →   обучение   →   удаление   →   отчёт

Запуск::

    python panel.py                  # интерактивное меню
    python panel.py --no-banner      # без заставки

Интерактивное меню поддерживает ввод как цифровых номеров, так и
русских/английских слов-алиасов (см. команду `помощь`). Между
действиями панель сохраняет состояние: загруженный датасет,
обученный ансамбль, последний run_id и текущий YAML-конфиг.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ─── Пути проекта ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
GEN_DIR = PROJECT_ROOT / "synthetic_data_generator"
UNL_DIR = PROJECT_ROOT / "unlearning_system"

# Пускаем Python в обе под-системы
sys.path.insert(0, str(UNL_DIR))
sys.path.insert(0, str(GEN_DIR))

# ─── ANSI-цвета (совместимы с main.py генератора) ────────────────────────────

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    GREY = "\033[90m"


def _isatty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def col(text: str, *codes: str) -> str:
    if _isatty():
        return "".join(codes) + text + C.RESET
    return text


def ok(text: str) -> str:
    return col(f"✓ {text}", C.GREEN)


def warn(text: str) -> str:
    return col(f"⚠ {text}", C.YELLOW)


def err(text: str) -> str:
    return col(f"✗ {text}", C.RED)


def hdr(text: str) -> str:
    return col(text, C.BOLD, C.CYAN)


def sep(width: int = 64) -> str:
    return col("─" * width, C.GREY)


def kv(key: str, value: str, key_width: int = 22) -> str:
    return f"  {key:<{key_width}} {value}"


def prompt(text: str) -> str:
    return col(text, C.BOLD, C.BLUE)


def format_number(n: float | int, precision: int = 2) -> str:
    if isinstance(n, int) or n == int(n):
        return f"{int(n):,}".replace(",", " ")
    return f"{n:,.{precision}f}".replace(",", " ")


def clear_screen() -> None:
    print("\033[2J\033[H", end="")


# ─── Состояние панели ────────────────────────────────────────────────────────


@dataclass
class PanelState:
    """Общее состояние между командами одной сессии."""

    gen_config: Path = GEN_DIR / "configs" / "default.yaml"
    unl_config: Path = UNL_DIR / "configs" / "default.yaml"
    dataset_path: Optional[Path] = None       # последний активный CSV
    run_id: Optional[str] = None              # последний run в runs/
    run_dir: Optional[Path] = None
    pipeline: Any = None                      # ExperimentPipeline (lazy)
    baseline: Any = None                      # ClassificationReport
    events: list[dict] = field(default_factory=list)  # журнал сессии

    # ---- краткое отображение -------------------------------------------------

    def as_lines(self) -> list[str]:
        lines = [
            kv("Конфиг генератора :",
               col(str(self.gen_config.relative_to(PROJECT_ROOT)), C.GREY)),
            kv("Конфиг unlearning :",
               col(str(self.unl_config.relative_to(PROJECT_ROOT)), C.GREY)),
        ]
        if self.dataset_path is not None:
            rel = self.dataset_path
            try:
                rel = rel.relative_to(PROJECT_ROOT)
            except ValueError:
                pass
            lines.append(kv("Активный датасет   :", col(str(rel), C.YELLOW)))
        else:
            lines.append(kv("Активный датасет   :", col("—", C.GREY)))
        if self.pipeline is not None:
            trained = "обучен" if getattr(self.pipeline, "ensemble_", None) else "загружен"
            lines.append(kv("SISA-ансамбль      :",
                            col(trained, C.GREEN if trained == "обучен" else C.YELLOW)))
        else:
            lines.append(kv("SISA-ансамбль      :", col("не инициализирован", C.GREY)))
        if self.run_id is not None:
            lines.append(kv("Текущий run_id     :", col(self.run_id, C.MAGENTA)))
        else:
            lines.append(kv("Текущий run_id     :", col("—", C.GREY)))
        if self.baseline is not None:
            lines.append(kv(
                "Baseline F1 / AUC :",
                col(f"{self.baseline.f1:.3f} / {self.baseline.auc:.3f}", C.GREEN),
            ))
        return lines


# ─── Логирование ─────────────────────────────────────────────────────────────


def _setup_logging() -> None:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_file = log_dir / f"panel_{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


# ─── Вспомогательные диалоги ────────────────────────────────────────────────


def ask(text: str, default: Optional[str] = None) -> str:
    """Однострочный ввод с опциональным значением по умолчанию."""
    suffix = f" [{default}]" if default is not None else ""
    try:
        reply = input(f"{prompt(text)}{suffix}: ").strip()
    except EOFError:
        print()
        return default or ""
    if not reply and default is not None:
        return default
    return reply


def ask_int(text: str, default: Optional[int] = None) -> Optional[int]:
    raw = ask(text, str(default) if default is not None else None)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(err(f"Ожидалось целое число, получено: {raw!r}"))
        return ask_int(text, default)


def ask_bool(text: str, default: bool = False) -> bool:
    dflt = "Y/n" if default else "y/N"
    raw = ask(text + f" ({dflt})", "").lower()
    if not raw:
        return default
    return raw in ("y", "yes", "д", "да", "1", "true")


def pause(text: str = "Нажмите Enter чтобы продолжить…") -> None:
    try:
        input(col(text, C.GREY))
    except EOFError:
        pass


def find_latest_dataset() -> Optional[Path]:
    data_dir = GEN_DIR / "data"
    if not data_dir.exists():
        return None
    candidates = sorted(
        list(data_dir.glob("*.csv")) + list(data_dir.glob("*.parquet")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def list_runs() -> list[Path]:
    runs_dir = UNL_DIR / "runs"
    if not runs_dir.exists():
        return []
    return sorted(
        [p for p in runs_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


# ─── Действия: генерация / валидация / инфо ─────────────────────────────────


def action_generate(state: PanelState) -> None:
    """Генерация нового синтетического датасета (1)."""
    print()
    print(hdr("▸ Генерация синтетического датасета"))
    print(sep())

    if not state.gen_config.exists():
        print(err(f"Конфиг не найден: {state.gen_config}"))
        return

    count = ask_int("Количество записей", default=10000)
    seed = ask_int("Seed генератора", default=42)
    labeling_choice = ask(
        "Режим разметки: 1) rule  2) probabilistic  [enter = из конфига]",
        default="",
    ).strip()
    label_mode = {"1": "rule", "rule": "rule",
                  "2": "probabilistic", "probabilistic": "probabilistic"}.get(
                      labeling_choice.lower())

    # Работаем из подкаталога генератора — так logs/ и data/ лягут на своё место
    prev_cwd = os.getcwd()
    os.chdir(GEN_DIR)
    try:
        from generator.core import SyntheticDataGenerator  # type: ignore

        print(sep())
        print(col("  Генерация…", C.DIM))
        t0 = time.time()
        gen = SyntheticDataGenerator(
            config_path=state.gen_config,
            n_samples=count,
            seed=seed,
            labeling_mode=label_mode,
        )
        df = gen.generate()
        gen_time = time.time() - t0

        print(ok(f"Сгенерировано: {format_number(len(df))} строк "
                 f"× {len(df.columns)} признаков за {gen_time:.2f} с"))

        print()
        print(col("  Валидация…", C.DIM))
        vr = gen.validate(df)
        _print_validation(vr)

        output_path = Path(gen.save(df))
        # Генератор сохраняет путь относительно cwd=GEN_DIR. Нормализуем
        # в абсолютный, чтобы дальше не зависеть от os.chdir.
        if not output_path.is_absolute():
            output_path = GEN_DIR / output_path
        state.dataset_path = output_path.resolve()
        print()
        print(ok(f"Сохранено: {state.dataset_path}"))

        # Сбрасываем кэш пайплайна — датасет сменился
        state.pipeline = None
        state.baseline = None
        state.events.append({
            "action": "generate", "rows": len(df), "seed": seed,
            "labeling_mode": label_mode, "path": str(state.dataset_path),
            "ts": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        print(err(f"Ошибка генерации: {e}"))
        logging.exception("generate failed")
    finally:
        os.chdir(prev_cwd)


def _print_validation(vr: dict) -> None:
    check_names = {
        "nan": "Пропущенные значения",
        "user_id_uniqueness": "Уникальность user_id",
        "distributions": "Распределения признаков",
        "clipping": "Ограничения (clip)",
        "correlations": "Корреляции",
        "date_formats": "Форматы дат",
        "class_balance": "Баланс классов",
        "email_uniqueness": "Уникальность email",
        "email_name_correlation": "Соответствие email↔ФИО",
    }
    for key, data in vr.get("checks", {}).items():
        name = check_names.get(key, key)
        passed = data.get("passed", True)
        mark = ok("OK") if passed else err("FAIL")
        print(kv(f"{name:<30}", mark, key_width=0))
    print()
    passed = vr.get("passed", False)
    print(kv("Итог:", ok("валидация пройдена") if passed else err("валидация не пройдена")))


def action_validate(state: PanelState) -> None:
    """Валидация существующего датасета (2)."""
    print()
    print(hdr("▸ Валидация датасета"))
    print(sep())

    default = str(state.dataset_path) if state.dataset_path else ""
    if not default:
        latest = find_latest_dataset()
        if latest:
            default = str(latest)

    path_str = ask("Путь к файлу", default=default or None)
    if not path_str:
        print(err("Путь не задан."))
        return
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        # Пробуем: относительно cwd, потом относительно GEN_DIR
        if not path.exists():
            alt = GEN_DIR / path
            if alt.exists():
                path = alt

    if not path.exists():
        print(err(f"Файл не найден: {path}"))
        return

    prev_cwd = os.getcwd()
    os.chdir(GEN_DIR)
    try:
        import pandas as pd
        from generator.core import SyntheticDataGenerator  # type: ignore

        if path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)
        print(ok(f"Прочитано: {len(df):,} строк × {len(df.columns)} столбцов".replace(",", " ")))
        gen = SyntheticDataGenerator(config_path=state.gen_config)
        vr = gen.validate(df)
        _print_validation(vr)
    except Exception as e:
        print(err(f"Ошибка валидации: {e}"))
        logging.exception("validate failed")
    finally:
        os.chdir(prev_cwd)


def action_info(state: PanelState) -> None:
    """Краткая статистика по датасету (3)."""
    print()
    print(hdr("▸ Статистика датасета"))
    print(sep())

    path = state.dataset_path or find_latest_dataset()
    if not path:
        print(err("Нет активного датасета. Сначала сгенерируйте или выберите файл."))
        return
    user_in = ask("Путь к файлу", default=str(path))
    path = Path(user_in).expanduser()
    if not path.exists():
        alt = GEN_DIR / path
        if alt.exists():
            path = alt
    if not path.exists():
        print(err(f"Файл не найден: {path}"))
        return

    prev_cwd = os.getcwd()
    os.chdir(GEN_DIR)
    try:
        import pandas as pd
        from generator.core import SyntheticDataGenerator  # type: ignore

        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
        gen = SyntheticDataGenerator(config_path=state.gen_config)
        info = gen.get_info(df)

        print(kv("Файл :", col(str(path), C.BLUE)))
        print(kv("Размерность :", f"{info['shape'][0]:,} × {info['shape'][1]}".replace(",", " ")))
        if "class_balance" in info:
            cb = info["class_balance"]
            pr = cb["positive_rate"]
            print(kv("Баланс меток :",
                      f"label=1: {format_number(cb['count_true'])} ({pr:.1%})   "
                      f"label=0: {format_number(cb['count_false'])} ({1-pr:.1%})"))
        print()
        print(hdr("Числовые признаки"))
        print(sep())
        print(f"  {'Признак':<30} {'Среднее':>10} {'Ст.откл':>10} {'Мин':>8} {'Макс':>10}")
        for colname, s in info.get("numeric_summary", {}).items():
            print(f"  {colname:<30} {s['mean']:>10.2f} {s['std']:>10.2f} "
                  f"{s['min']:>8.2f} {s['max']:>10.2f}")
    except Exception as e:
        print(err(f"Ошибка: {e}"))
        logging.exception("info failed")
    finally:
        os.chdir(prev_cwd)


def action_list_datasets(state: PanelState) -> None:
    """Список датасетов в data/ (4)."""
    print()
    print(hdr("▸ Датасеты в synthetic_data_generator/data/"))
    print(sep())
    data_dir = GEN_DIR / "data"
    if not data_dir.exists():
        print(err("Папка data/ не найдена."))
        return
    files = sorted(
        list(data_dir.glob("*.csv")) + list(data_dir.glob("*.parquet")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        print(warn("Датасеты не найдены."))
        return
    active = state.dataset_path.resolve() if state.dataset_path else None
    for i, fp in enumerate(files, 1):
        size_mb = fp.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(fp.stat().st_mtime).strftime("%d.%m.%Y %H:%M")
        is_active = active is not None and fp.resolve() == active
        marker = col("►", C.GREEN) if is_active else " "
        note = col(" (активный)", C.GREEN) if is_active else ""
        print(f"  {i:>2}. {marker} {fp.name:<36} {size_mb:6.2f} МБ  {mtime}{note}")

    pick = ask("\nВыбрать № как активный [enter=оставить]", default="")
    if pick.isdigit():
        idx = int(pick)
        if 1 <= idx <= len(files):
            state.dataset_path = files[idx - 1].resolve()
            state.pipeline = None
            state.baseline = None
            print(ok(f"Активный датасет: {state.dataset_path}"))
        else:
            print(err("Номер вне диапазона."))


# ─── Действия: обучение / оценка / удаление ─────────────────────────────────


def _load_pipeline(state: PanelState):
    from ml.pipeline import ExperimentPipeline  # type: ignore
    pipe = ExperimentPipeline.from_config(state.unl_config)
    pipe.load_dataset(state.dataset_path)
    return pipe


def _fresh_registry(state: PanelState):
    from ml.registry import RunRegistry, file_sha1  # type: ignore
    runs_root = UNL_DIR / "runs"
    registry = RunRegistry(runs_root, run_id=None)
    state.run_id = registry.run_id
    state.run_dir = registry.dir
    return registry, file_sha1


def _print_report(title: str, report) -> None:
    print(kv(f"{title} accuracy :", f"{report.accuracy:.4f}"))
    print(kv(f"{title} precision :", f"{report.precision:.4f}"))
    print(kv(f"{title} recall :", f"{report.recall:.4f}"))
    print(kv(f"{title} F1 :", col(f"{report.f1:.4f}", C.GREEN)))
    print(kv(f"{title} AUC :", col(f"{report.auc:.4f}", C.GREEN)))
    print(kv(f"{title} MSE :", f"{report.mse:.4f}"))


def action_train(state: PanelState) -> None:
    """Обучение SISA-ансамбля на текущем датасете (5)."""
    print()
    print(hdr("▸ Обучение SISA-ансамбля"))
    print(sep())

    if state.dataset_path is None:
        latest = find_latest_dataset()
        if latest is None:
            print(err("Нет датасета. Сначала выполните генерацию (пункт 1)."))
            return
        use_latest = ask_bool(f"Использовать последний: {latest.name}?", default=True)
        if use_latest:
            state.dataset_path = latest.resolve()
        else:
            print(err("Обучение отменено."))
            return

    if not state.unl_config.exists():
        print(err(f"Конфиг не найден: {state.unl_config}"))
        return

    prev_cwd = os.getcwd()
    os.chdir(UNL_DIR)
    try:
        print(col("  Загрузка + предобработка…", C.DIM))
        t0 = time.time()
        pipe = _load_pipeline(state)
        print(ok(f"train={len(pipe.df_train_)} test={len(pipe.df_test_)} "
                 f"features={pipe.X_train_.shape[1]} shards={pipe.plan_.n_shards}"))

        print(col("  Обучение ансамбля…", C.DIM))
        pipe.fit()
        train_time = time.time() - t0
        state.pipeline = pipe

        report = pipe.evaluate()
        state.baseline = report

        registry, file_sha1 = _fresh_registry(state)
        registry.write_manifest({
            "run_id": registry.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_path": str(state.unl_config),
            "data_path": str(state.dataset_path),
            "data_sha1": file_sha1(state.dataset_path),
            "train_rows": int(len(pipe.df_train_)),
            "test_rows": int(len(pipe.df_test_)),
            "n_features": int(pipe.X_train_.shape[1]),
            "n_shards": pipe.plan_.n_shards,
            "shard_sizes": pipe.plan_.sizes(),
            "wall_time_train_sec": train_time,
            "backend": pipe.config["model"]["backend"],
        })
        registry.save_pipeline(pipe.pipeline_.state_dict())
        registry.save_plan(pipe.plan_.shard_of_row, pipe.plan_.n_shards)
        for sid, learner in enumerate(pipe.ensemble_.learners):
            if learner is None:
                continue
            registry.save_learner(sid, learner.__class__.__name__, learner.state_dict())
        registry.save_metrics("baseline", report.as_dict())
        registry.log_event("train", {
            "duration_sec": train_time,
            "shard_sizes": pipe.plan_.sizes(),
        })

        print()
        print(ok(f"Готово за {train_time:.2f} с, run_id = "
                 f"{col(registry.run_id, C.MAGENTA)}"))
        print(sep())
        _print_report("baseline", report)
        print(kv("Размеры шардов :", ", ".join(str(s) for s in pipe.plan_.sizes())))
        state.events.append({
            "action": "train", "run_id": registry.run_id,
            "duration_sec": train_time, "f1": report.f1, "auc": report.auc,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        print(err(f"Ошибка обучения: {e}"))
        logging.exception("train failed")
    finally:
        os.chdir(prev_cwd)


def action_evaluate(state: PanelState) -> None:
    """Переоценка текущего ансамбля на hold-out (6)."""
    print()
    print(hdr("▸ Оценка текущего ансамбля"))
    print(sep())
    if state.pipeline is None or getattr(state.pipeline, "ensemble_", None) is None:
        print(err("Ансамбль не обучен. Запустите пункт 5 (обучение)."))
        return
    try:
        report = state.pipeline.evaluate()
        _print_report("current", report)
        if state.baseline is not None:
            print()
            print(hdr("Δ от baseline"))
            for name in ("accuracy", "precision", "recall", "f1", "auc", "mse"):
                delta = getattr(report, name) - getattr(state.baseline, name)
                colorize = C.GREEN if delta >= 0 else C.RED
                print(kv(f"Δ {name} :", col(f"{delta:+.4f}", colorize)))
    except Exception as e:
        print(err(f"Ошибка оценки: {e}"))
        logging.exception("evaluate failed")


def _do_unlearn(state: PanelState, user_ids: Optional[list[str]],
                filter_query: Optional[str], reason: str) -> None:
    from ml.unlearning import UnlearnRequest  # type: ignore

    if state.pipeline is None or getattr(state.pipeline, "ensemble_", None) is None:
        print(col("  Ансамбль не обучен — запускаю обучение…", C.DIM))
        action_train(state)
        if state.pipeline is None or getattr(state.pipeline, "ensemble_", None) is None:
            return

    req = UnlearnRequest(user_ids=user_ids, filter_query=filter_query, reason=reason)
    if req.is_empty():
        print(err("Ничего не указано для удаления."))
        return

    prev_cwd = os.getcwd()
    os.chdir(UNL_DIR)
    try:
        print(col("  Выполняется разобучение…", C.DIM))
        t0 = time.time()
        result = state.pipeline.unlearn(req)
        total = time.time() - t0

        # Запись в текущий run_dir (если был train) — иначе создаём новый
        if state.run_dir is None:
            registry, _ = _fresh_registry(state)
        else:
            from ml.registry import RunRegistry  # type: ignore
            registry = RunRegistry(UNL_DIR / "runs", run_id=state.run_id)
        registry.save_metrics("before_unlearn", result.metrics_before.as_dict())
        registry.save_metrics("after_unlearn", result.metrics_after.as_dict())
        registry.save_metrics("delta", result.delta)
        registry.log_event("unlearn", {
            "user_ids": user_ids, "filter_query": filter_query,
            "reason": reason, "deleted_rows": result.deleted_rows,
            "affected_shards": result.affected_shards,
            "wall_time_sec": result.wall_time_sec,
            "prediction_disagreement": result.prediction_disagreement,
        })

        print(ok(f"Разобучено за {total:.3f} с"))
        print(sep())
        _print_report("before", result.metrics_before)
        print()
        _print_report("after ", result.metrics_after)
        print()
        print(hdr("Δ (after − before)"))
        for k, v in result.delta.items():
            colorize = C.GREEN if abs(v) < 0.01 else (C.YELLOW if abs(v) < 0.03 else C.RED)
            print(kv(f"Δ {k} :", col(f"{v:+.4f}", colorize)))
        print()
        print(kv("Удалено строк :", format_number(result.deleted_rows)))
        # result.affected_shards может быть set/list — сортируем для
        # стабильного вывода между запусками.
        shards_sorted = sorted(result.affected_shards)
        print(kv("Затронуто шардов :",
                  ", ".join(str(s) for s in shards_sorted) or "—"))
        print(kv("Prediction disagreement :",
                  f"{result.prediction_disagreement:.4f}"))
        print(kv("Retrain wall time :",
                  f"{result.retrain_report.total_wall_time_sec:.3f} с"))

        state.events.append({
            "action": "unlearn", "run_id": state.run_id,
            "deleted_rows": result.deleted_rows,
            "affected_shards": shards_sorted,
            "delta": result.delta,
            "wall_time_sec": result.wall_time_sec,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        print(err(f"Ошибка разобучения: {e}"))
        logging.exception("unlearn failed")
    finally:
        os.chdir(prev_cwd)


def action_unlearn_by_id(state: PanelState) -> None:
    """Удаление по списку user_id (7)."""
    print()
    print(hdr("▸ Разобучение по user_id"))
    print(sep())
    raw = ask("user_id (через пробел или запятую)", default="")
    if not raw.strip():
        print(err("Список идентификаторов пуст."))
        return
    ids = [x.strip() for x in raw.replace(",", " ").split() if x.strip()]
    reason = ask("Причина удаления", default="user_request")
    _do_unlearn(state, user_ids=ids, filter_query=None, reason=reason)


def action_unlearn_by_filter(state: PanelState) -> None:
    """Удаление по pandas .query()-выражению (8)."""
    print()
    print(hdr("▸ Разобучение по фильтру (pandas .query)"))
    print(sep())
    print(col("  Примеры фильтров:", C.GREY))
    print(col("    consent_given == False", C.GREY))
    print(col("    age < 25 and region == 'Moscow'", C.GREY))
    print(col("    purchase_count == 0 and days_since_last_purchase > 180", C.GREY))
    print()
    q = ask("Выражение", default="")
    if not q.strip():
        print(err("Пустое выражение."))
        return
    reason = ask("Причина удаления", default="152-FZ art.14")
    _do_unlearn(state, user_ids=None, filter_query=q, reason=reason)


# ─── Действия: run-реестр / конфиг / журнал ─────────────────────────────────


def action_list_runs(state: PanelState) -> None:
    """Список сохранённых run-ов (9)."""
    print()
    print(hdr("▸ Сохранённые запуски (runs/)"))
    print(sep())
    runs = list_runs()
    if not runs:
        print(warn("Запусков ещё нет."))
        return
    for i, r in enumerate(runs, 1):
        mtime = datetime.fromtimestamp(r.stat().st_mtime).strftime("%d.%m.%Y %H:%M")
        marker = col("►", C.GREEN) if state.run_id == r.name else " "
        note = col(" (активный)", C.GREEN) if state.run_id == r.name else ""
        # Попробуем достать backend и F1 из manifest/metrics
        extra = ""
        try:
            mf = r / "manifest.yaml"
            if mf.exists():
                import yaml
                with open(mf, encoding="utf-8") as f:
                    man = yaml.safe_load(f)
                extra = col(f"  backend={man.get('backend', '?')}, shards={man.get('n_shards', '?')}",
                             C.GREY)
        except Exception:
            pass
        print(f"  {marker} {r.name:<34} {mtime}{note}{extra}")


def action_show_report(state: PanelState) -> None:
    """Показать метрики выбранного run (10)."""
    print()
    print(hdr("▸ Метрики запуска"))
    print(sep())
    runs = list_runs()
    if not runs:
        print(warn("Запусков нет."))
        return
    default = state.run_id or runs[0].name
    name = ask("run_id", default=default)
    candidates = [r for r in runs if r.name == name]
    if not candidates:
        print(err(f"Не найден: {name}"))
        return
    run_dir = candidates[0]
    metrics = sorted((run_dir / "metrics").glob("*.json"))
    if not metrics:
        print(warn("Нет файлов метрик в этом run'е."))
        return
    for mf in metrics:
        with open(mf, encoding="utf-8") as f:
            payload = json.load(f)
        print(hdr(f"--- {mf.stem} ---"))
        for k, v in payload.items():
            if isinstance(v, float):
                print(kv(f"{k} :", f"{v:.4f}"))
            else:
                print(kv(f"{k} :", str(v)))
        print()

    # Показать события
    events_log = run_dir / "events.log"
    if events_log.exists():
        print(hdr("events.log (последние 5)"))
        print(sep())
        lines = events_log.read_text(encoding="utf-8").splitlines()[-5:]
        for ln in lines:
            print("  " + col(ln, C.GREY))


def action_switch_config(state: PanelState) -> None:
    """Смена YAML-конфига (11)."""
    print()
    print(hdr("▸ Конфигурация"))
    print(sep())
    for line in state.as_lines():
        print(line)
    print()
    which = ask("Что менять? 1) конфиг генератора  2) конфиг unlearning  [enter=отмена]",
                default="")
    if which == "1":
        p = ask("Путь к YAML", default=str(state.gen_config))
        new = Path(p).expanduser()
        if new.exists():
            state.gen_config = new
            print(ok(f"Обновлён gen_config: {new}"))
        else:
            print(err(f"Файл не найден: {new}"))
    elif which == "2":
        p = ask("Путь к YAML", default=str(state.unl_config))
        new = Path(p).expanduser()
        if new.exists():
            state.unl_config = new
            # Любой обученный ансамбль становится неконсистентным с новым
            # конфигом. Сбрасываем также run_id/run_dir — иначе следующий
            # unlearn попытался бы дописать метрики в run, созданный с
            # прежними гиперпараметрами.
            state.pipeline = None
            state.baseline = None
            state.run_id = None
            state.run_dir = None
            print(ok(f"Обновлён unl_config: {new}"))
            print(warn("Текущий ансамбль сброшен — переобучите (пункт 5)."))
        else:
            print(err(f"Файл не найден: {new}"))


def action_show_config(state: PanelState) -> None:
    """Показать активный unlearning-конфиг (12)."""
    print()
    print(hdr("▸ Содержимое unlearning-конфига"))
    print(sep())
    try:
        print(state.unl_config.read_text(encoding="utf-8"))
    except Exception as e:
        print(err(f"Не удалось прочитать: {e}"))


def action_show_session_journal(state: PanelState) -> None:
    """Журнал операций текущей сессии (13)."""
    print()
    print(hdr("▸ Журнал сессии"))
    print(sep())
    if not state.events:
        print(warn("Действий пока не было."))
        return
    for i, ev in enumerate(state.events, 1):
        ts = ev.get("ts", "")
        act = ev.get("action", "")
        summary = {k: v for k, v in ev.items() if k not in {"ts", "action"}}
        print(f"  {i:>2}. [{col(ts, C.GREY)}] {col(act, C.CYAN)}  "
              f"{col(json.dumps(summary, ensure_ascii=False, default=str), C.GREY)}")


# ─── Пайплайн "всё сразу" ───────────────────────────────────────────────────


def action_full_cycle(state: PanelState) -> None:
    """Запуск полного цикла в одном вызове (0)."""
    print()
    print(hdr("▸ Полный цикл: генерация → обучение → удаление → отчёт"))
    print(sep())
    if not ask_bool("Продолжить?", default=True):
        return
    action_generate(state)
    if state.dataset_path is None:
        return
    action_train(state)
    if state.pipeline is None:
        return
    # Демо-удаление: минимальный фильтр, который что-то удалит
    demo_filter = "age < 25 and region == 'Moscow'"
    print()
    print(hdr(f"Демо-удаление по фильтру: {demo_filter}"))
    _do_unlearn(state, user_ids=None, filter_query=demo_filter,
                reason="demo-full-cycle")
    print()
    action_show_report(state)


# ─── Меню ────────────────────────────────────────────────────────────────────


MENU_ITEMS = [
    ("1", ("генерировать", "gen", "g"),
     "Сгенерировать синтетический датасет", action_generate),
    ("2", ("валидировать", "val", "v"),
     "Проверить существующий датасет", action_validate),
    ("3", ("инфо", "info", "i"),
     "Статистика по датасету", action_info),
    ("4", ("датасеты", "ds", "list-data"),
     "Список и выбор датасетов", action_list_datasets),
    ("5", ("обучить", "train", "t"),
     "Обучить SISA-ансамбль", action_train),
    ("6", ("оценить", "eval", "e"),
     "Переоценить ансамбль на hold-out", action_evaluate),
    ("7", ("удалить-id", "uid", "unlearn-id"),
     "Разобучить по user_id", action_unlearn_by_id),
    ("8", ("удалить-ф", "uf", "unlearn-filter"),
     "Разобучить по фильтру", action_unlearn_by_filter),
    ("9", ("раны", "runs"),
     "Список запусков runs/", action_list_runs),
    ("10", ("отчёт", "отчет", "report", "r"),
     "Метрики выбранного run", action_show_report),
    ("11", ("конфиг", "config"),
     "Сменить YAML-конфиг", action_switch_config),
    ("12", ("показать-конфиг", "show-config"),
     "Показать unlearning-конфиг", action_show_config),
    ("13", ("журнал", "log", "j"),
     "Журнал операций сессии", action_show_session_journal),
    ("0", ("полный-цикл", "all", "a"),
     "Полный цикл генерация → обучение → удаление → отчёт", action_full_cycle),
]


def _find_action(keyword: str):
    key = keyword.lower().strip()
    for num, aliases, _desc, fn in MENU_ITEMS:
        if key == num or key in aliases:
            return fn
    return None


def print_banner() -> None:
    print()
    print(hdr("╔══════════════════════════════════════════════════════════════╗"))
    print(hdr("║   Панель проекта 2039 — Механизмы цифрового забвения         ║"))
    print(hdr("║   HSE MIEM • SISA Machine Unlearning • 152-ФЗ / GDPR         ║"))
    print(hdr("╚══════════════════════════════════════════════════════════════╝"))
    print()


def print_menu(state: PanelState) -> None:
    print()
    print(sep(72))
    print(hdr("Состояние сессии"))
    for line in state.as_lines():
        print(line)
    print(sep(72))
    print(hdr("Меню"))
    for num, aliases, desc, _ in MENU_ITEMS:
        alias_str = col("(" + " / ".join(aliases) + ")", C.GREY)
        # Сначала выравниваем по ширине, только затем красим —
        # иначе ANSI-коды попадают в подсчёт ширины format-spec.
        print(f"  {col(f'{num:>2}', C.YELLOW)}. {desc} {alias_str}")
    print(f"  {col(' h', C.YELLOW)}. Справка по командам "
          f"{col('(помощь / help / ?)', C.GREY)}")
    print(f"  {col(' c', C.YELLOW)}. Очистить экран "
          f"{col('(clear / cls)', C.GREY)}")
    print(f"  {col(' m', C.YELLOW)}. Показать меню заново "
          f"{col('(меню / menu)', C.GREY)}")
    print(f"  {col(' q', C.YELLOW)}. Выход "
          f"{col('(выход / exit / quit)', C.GREY)}")
    print(sep(72))


def print_help() -> None:
    print()
    print(hdr("Справка"))
    print(sep())
    print("  • Введите номер пункта (1..13, 0) или любое из слов-алиасов.")
    print("  • Команды устойчивы к отменам: Ctrl+C возвращает в меню.")
    print("  • Все артефакты сохраняются в unlearning_system/runs/<run_id>/.")
    print("  • Датасеты пишутся в synthetic_data_generator/data/.")
    print("  • Логи панели — в logs/panel_<timestamp>.log.")
    print()
    print(hdr("Типовой сценарий"))
    print("  1 → 5 → 8 → 10     (сгенерировать → обучить → удалить → отчёт)")
    print("  0                  (всё сразу в одну команду)")
    print()


# ─── Главный цикл ────────────────────────────────────────────────────────────


def run_interactive(state: PanelState, show_banner: bool) -> int:
    if show_banner:
        print_banner()
    print_menu(state)

    while True:
        try:
            raw = input(prompt("\n>>> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print(ok("Выход. До встречи!"))
            return 0
        if not raw:
            # пустой ввод — перерисовываем меню, чтобы пользователь
            # всегда видел контекст без необходимости помнить команды
            print_menu(state)
            continue
        lc = raw.lower()
        if lc in {"q", "quit", "exit", "выход", "й"}:
            print(ok("Выход. До встречи!"))
            return 0
        if lc in {"h", "help", "помощь", "?"}:
            print_help()
            print_menu(state)
            continue
        if lc in {"c", "clear", "cls", "очистить"}:
            clear_screen()
            print_banner()
            print_menu(state)
            continue
        if lc in {"m", "menu", "меню"}:
            print_menu(state)
            continue

        fn = _find_action(raw)
        if fn is None:
            print(err(f"Неизвестная команда: {raw!r}. Введите 'h' для справки."))
            print_menu(state)
            continue
        try:
            fn(state)
        except KeyboardInterrupt:
            print()
            print(warn("Прервано пользователем."))
        except Exception as e:  # защитный периметр — цикл не падает
            print(err(f"Непредвиденная ошибка: {e}"))
            logging.exception("action failed")
            if os.environ.get("PANEL_DEBUG"):
                traceback.print_exc()
        finally:
            # После КАЖДОГО действия заново показываем меню:
            # это требование UX, чтобы пользователь не нажимал `m`
            # вручную и всегда видел актуальное состояние сессии.
            print_menu(state)


# ─── argparse ───────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="panel",
        description="Единая интерактивная панель проекта 2039 "
                    "«Механизмы цифрового забвения» (HSE MIEM).",
    )
    p.add_argument("--no-banner", action="store_true",
                   help="Не выводить ASCII-заставку")
    p.add_argument("--gen-config", type=Path, default=None,
                   help="Путь к YAML генератора")
    p.add_argument("--unl-config", type=Path, default=None,
                   help="Путь к YAML unlearning-системы")
    p.add_argument("--dataset", type=Path, default=None,
                   help="Сразу выбрать активный CSV/Parquet")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    _setup_logging()
    args = build_parser().parse_args(argv)

    state = PanelState()
    if args.gen_config:
        state.gen_config = args.gen_config.expanduser().resolve()
    if args.unl_config:
        state.unl_config = args.unl_config.expanduser().resolve()
    if args.dataset:
        p = args.dataset.expanduser().resolve()
        if p.exists():
            state.dataset_path = p
        else:
            print(err(f"Датасет не найден: {p}"))

    return run_interactive(state, show_banner=not args.no_banner)


if __name__ == "__main__":
    sys.exit(main())
