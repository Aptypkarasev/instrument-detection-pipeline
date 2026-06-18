# -*- coding: utf-8 -*-
"""
Сравнение нескольких YOLO-детекторов на датасете marker.

Что делает:
  1. Обучает N конфигураций (разные модели И/ИЛИ гиперпараметры) на одном датасете.
  2. Оценивает каждую обученную модель на ТЕСТОВОЙ выборке (единые условия).
  3. Собирает сравнительный отчёт:
       - metrics_comparison.csv            — таблица метрик по всем моделям
       - compare_metrics_bars.png          — столбчатое сравнение mAP/P/R
       - compare_loss_curves.png           — кривые ошибок (train+val) всех моделей
       - compare_confusion_matrices.png    — матрицы ошибок (на тесте) сеткой
       - compare_speed_size.png            — точность vs размер/скорость модели
       - report.md                         — краткие выводы

Ультралитика сама пишет в папку каждого запуска results.csv (кривые ошибок),
confusion_matrix.png, PR-кривые и веса — мы их агрегируем в общие графики.

Запуск:
  python train_compare.py --data data.yaml --epochs 60 --imgsz 640 --batch 16
  Флаги --epochs/--imgsz/--batch управляют бюджетом обучения.
  --only v8n,v11n  — обучить только указанные конфигурации (через запятую).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# --------------------------------------------------------------- конфигурации сравнения
# Базовая идея: 2 оси сравнения за 4 запуска —
#   * версия архитектуры:  v8n  vs  v11n
#   * размер модели:       v8n  vs  v8s
#   * гиперпараметры:      v11n vs  v11n-tuned (другой оптимизатор/LR/аугментация)
CONFIGS = [
    {
        "name": "v8n",
        "model": "yolov8n.pt",
        "desc": "YOLOv8 nano, базовые гиперпараметры (SGD, lr0=0.01)",
        "hyp": {"optimizer": "SGD", "lr0": 0.01},
    },
    {
        "name": "v8s",
        "model": "yolov8s.pt",
        "desc": "YOLOv8 small — тот же набор, крупнее модель",
        "hyp": {"optimizer": "SGD", "lr0": 0.01},
    },
    {
        "name": "v11n",
        "model": "yolo11n.pt",
        "desc": "YOLO11 nano, базовые гиперпараметры (новее архитектура)",
        "hyp": {"optimizer": "SGD", "lr0": 0.01},
    },
    {
        "name": "v11n_tuned",
        "model": "yolo11n.pt",
        "desc": "YOLO11 nano с подобранными гиперпараметрами (AdamW, lr0=0.002, "
                "косинусный LR, усиленная аугментация)",
        "hyp": {"optimizer": "AdamW", "lr0": 0.002, "cos_lr": True,
                "mixup": 0.10, "copy_paste": 0.10, "hsv_h": 0.02, "degrees": 5.0},
    },
]


# --------------------------------------------------------------- обучение одной конфигурации

def train_one(cfg, args, device, project):
    print(f"\n{'=' * 70}\n[{cfg['name']}] {cfg['desc']}\n{'=' * 70}")
    model = YOLO(cfg["model"])
    # model.info() в части версий ultralytics возвращает None — считаем напрямую
    n_params = sum(p.numel() for p in model.model.parameters())
    try:
        gflops = get_flops(model.model, imgsz=args.imgsz)
    except Exception:
        gflops = None

    t0 = time.time()
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=project,
        name=cfg["name"],
        exist_ok=True,
        seed=0,
        deterministic=False,
        verbose=True,
        plots=True,
        patience=args.patience,
        workers=args.workers,
        **cfg["hyp"],
    )
    train_min = (time.time() - t0) / 60.0
    run_dir = Path(project) / cfg["name"]

    # --- оценка на ТЕСТОВОЙ выборке (единые условия для всех моделей)
    best = run_dir / "weights" / "best.pt"
    eval_model = YOLO(str(best))
    tm = eval_model.val(
        data=args.data, split="test", imgsz=args.imgsz, batch=args.batch,
        device=device, project=project, name=f"{cfg['name']}_test", exist_ok=True,
        plots=True, verbose=False,
    )
    speed_ms = sum(tm.speed.values()) if hasattr(tm, "speed") else None
    cm = tm.confusion_matrix.matrix if hasattr(tm, "confusion_matrix") else None

    return {
        "name": cfg["name"],
        "desc": cfg["desc"],
        "model_file": cfg["model"],
        "params_M": round(n_params / 1e6, 2) if n_params else None,
        "GFLOPs": round(gflops, 1) if gflops else None,
        "train_min": round(train_min, 1),
        "speed_ms_img": round(speed_ms, 1) if speed_ms else None,
        "test_mAP50": round(float(tm.box.map50), 4),
        "test_mAP50_95": round(float(tm.box.map), 4),
        "test_precision": round(float(tm.box.mp), 4),
        "test_recall": round(float(tm.box.mr), 4),
        "run_dir": str(run_dir),
        "cm": cm.tolist() if cm is not None else None,
    }


# --------------------------------------------------------------- агрегированные графики

def plot_metric_bars(rows, out):
    metrics = [("test_mAP50", "mAP@50"), ("test_mAP50_95", "mAP@50-95"),
               ("test_precision", "Precision"), ("test_recall", "Recall")]
    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (key, label) in enumerate(metrics):
        vals = [r[key] for r in rows]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("значение метрики")
    ax.set_title("Сравнение моделей на тестовой выборке")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_loss_curves(rows, out):
    """Кривые ошибок (train и val) для всех моделей на одном поле."""
    loss_keys = [("train/box_loss", "box (train)"), ("val/box_loss", "box (val)"),
                 ("train/cls_loss", "cls (train)"), ("val/cls_loss", "cls (val)")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (col, title) in zip(axes.flat, loss_keys):
        for r in rows:
            csv_path = Path(r["run_dir"]) / "results.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            if col in df.columns:
                ax.plot(df["epoch"], df[col], label=r["name"], linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("эпоха")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Кривые ошибок по эпохам", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_map_curves(rows, out):
    """Рост mAP@50 и mAP@50-95 на валидации по эпохам."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, title in [(axes[0], "metrics/mAP50(B)", "mAP@50 (val)"),
                           (axes[1], "metrics/mAP50-95(B)", "mAP@50-95 (val)")]:
        for r in rows:
            csv_path = Path(r["run_dir"]) / "results.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            if col in df.columns:
                ax.plot(df["epoch"], df[col], label=r["name"], linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("эпоха")
        ax.set_ylabel("mAP")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Рост качества на валидации", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_confusion_matrices(rows, out, class_names=("marker", "background")):
    """Сетка нормированных матриц ошибок (по тесту)."""
    n = len(rows)
    cols = min(n, 2)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(6 * cols, 5 * rows_n))
    axes = np.atleast_1d(axes).flatten()
    for ax, r in zip(axes, rows):
        cm = np.array(r["cm"], dtype=float) if r["cm"] else None
        if cm is None:
            ax.axis("off")
            continue
        cmn = cm / (cm.sum(axis=0, keepdims=True) + 1e-9)  # нормировка по столбцам (как в ultralytics)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{r['name']}  (mAP@50={r['test_mAP50']:.2f})")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Истинный класс")
        ax.set_ylabel("Предсказание")
        for i in range(cmn.shape[0]):
            for j in range(cmn.shape[1]):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black")
    for ax in axes[len(rows):]:
        ax.axis("off")
    fig.suptitle("Матрицы ошибок на тестовой выборке (нормированные)", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_speed_size(rows, out):
    """Точность против размера и скорости — выбор компромисса."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, xkey, xlabel in [(axes[0], "params_M", "Параметры, млн"),
                             (axes[1], "speed_ms_img", "Время на изображение, мс")]:
        for r in rows:
            if r[xkey] is None:
                continue
            ax.scatter(r[xkey], r["test_mAP50"], s=90)
            ax.annotate(r["name"], (r[xkey], r["test_mAP50"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mAP@50 (тест)")
        ax.grid(alpha=0.3)
    fig.suptitle("Точность vs размер и скорость", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130)
    plt.close(fig)


def write_report(rows, out, args, device):
    best = max(rows, key=lambda r: r["test_mAP50_95"])
    lines = [
        "# Сравнение YOLO-моделей для детекции метки `marker`",
        "",
        f"Датасет: train 1990 / val 426 / test 428 · 1 класс · "
        f"обучение {args.epochs} эпох, imgsz {args.imgsz}, batch {args.batch}, "
        f"устройство `{device}`.",
        "",
        "## Метрики на тестовой выборке",
        "",
        "| Модель | Параметры, млн | GFLOPs | mAP@50 | mAP@50-95 | Precision | Recall | Время обуч., мин | мс/изобр. |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['params_M']} | {r['GFLOPs']} | {r['test_mAP50']:.3f} | "
            f"{r['test_mAP50_95']:.3f} | {r['test_precision']:.3f} | {r['test_recall']:.3f} | "
            f"{r['train_min']} | {r['speed_ms_img']} |")
    lines += [
        "",
        "## Конфигурации",
        "",
    ]
    for r in rows:
        lines.append(f"- **{r['name']}** — {r['desc']}")
    lines += [
        "",
        "## Вывод",
        "",
        f"Лучшая по mAP@50-95 — **{best['name']}** ({best['test_mAP50_95']:.3f}). "
        "Графики: `compare_metrics_bars.png` (метрики), `compare_loss_curves.png` "
        "и `compare_map_curves.png` (обучение), `compare_confusion_matrices.png` "
        "(ошибки на тесте), `compare_speed_size.png` (компромисс точность/размер).",
        "",
        "Полные артефакты каждого запуска (веса, PR-кривые, примеры предсказаний) — "
        "в подпапках `runs/<имя>/`.",
    ]
    Path(out).write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Сравнение YOLO-моделей")
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--out", default="comparison")
    ap.add_argument("--only", default="", help="обучить только указанные конфиги, через запятую")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    dev_name = (torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != "cpu"
                else "CPU")
    print(f"Устройство: {device} ({dev_name}) | torch {torch.__version__}")

    configs = CONFIGS
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        configs = [c for c in CONFIGS if c["name"] in want]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    project = str(Path(args.project).resolve())

    rows = []
    for cfg in configs:
        try:
            rows.append(train_one(cfg, args, device, project))
        except Exception as e:
            print(f"[{cfg['name']}] ОШИБКА: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    if not rows:
        print("Ни одна модель не обучилась — нечего сравнивать.")
        return

    # сохраняем таблицу метрик
    keys = ["name", "model_file", "params_M", "GFLOPs", "train_min", "speed_ms_img",
            "test_mAP50", "test_mAP50_95", "test_precision", "test_recall"]
    with open(out_root / "metrics_comparison.csv", "w", newline="", encoding="utf-8-sig") as fp:
        wr = csv.DictWriter(fp, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in keys})

    plot_metric_bars(rows, out_root / "compare_metrics_bars.png")
    plot_loss_curves(rows, out_root / "compare_loss_curves.png")
    plot_map_curves(rows, out_root / "compare_map_curves.png")
    plot_confusion_matrices(rows, out_root / "compare_confusion_matrices.png")
    plot_speed_size(rows, out_root / "compare_speed_size.png")
    write_report(rows, out_root / "report.md", args, device)

    print(f"\nГотово. Сравнение: {out_root.resolve()}")
    print("Таблица:")
    for r in rows:
        print(f"  {r['name']:12s} mAP50={r['test_mAP50']:.3f} "
              f"mAP50-95={r['test_mAP50_95']:.3f} P={r['test_precision']:.3f} "
              f"R={r['test_recall']:.3f} | {r['train_min']} мин")


if __name__ == "__main__":
    main()
