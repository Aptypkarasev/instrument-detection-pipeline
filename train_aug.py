# -*- coding: utf-8 -*-
"""
Эксперимент с аугментацией для YOLOv8n (детекция метки marker).

Сравниваются две модели на одном датасете и одинаковом бюджете обучения:
  * baseline — без фотометрической аугментации (контроль);
  * aug      — аугментации, не меняющие разметку:
                 HSV-сдвиги (оттенок/насыщенность/яркость),
                 Blur / MedianBlur / GaussianBlur,
                 CLAHE (адаптивный контраст), ToGray, яркость/контраст, gamma, JPEG-сжатие,
                 Cutout / RandomErasing (стирание прямоугольника).

Идея: на чистом тесте обе модели почти равны (задача лёгкая), но аугментация
повышает УСТОЙЧИВОСТЬ к деградации изображения. Поэтому, кроме чистого теста,
скрипт строит «деградированный» тест (размытие+шум+яркость+JPEG, разметка та же)
и проверяет обе модели на нём, а также прогоняет их по реальным «плохим» фото.

Что на выходе (results/detection/):
  metrics.csv                  — метрики на чистом и деградированном тесте
  compare_metrics.png          — столбцы метрик (чистый vs деградированный)
  compare_loss_curves.png      — кривые ошибок по эпохам
  compare_confusion.png        — матрицы ошибок (чистый тест)
  robustness_drop.png          — падение mAP при деградации (baseline vs aug)
  bad_photos_robustness.csv    — детекции/уверенность на реальных «плохих» фото
  report.md

Запуск:
  python train_aug.py --data data.yaml --epochs 10 --imgsz 640 --batch 16
"""
import argparse
import csv
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ============================================================ МОНКИПАТЧ АУГМЕНТАЦИИ
# Ultralytics по умолчанию включает Blur/CLAHE через albumentations лишь с p=0.01
# (практически выключено). Подменяем сборку трансформов: при STRONG=on — рабочие
# вероятности, при STRONG=off — албументации полностью выключены (чистый baseline).
import ultralytics.data.augment as _aug

_ORIG_ALB_INIT = _aug.Albumentations.__init__
_STRONG = {"on": False}


def _strong_transforms():
    import albumentations as A
    # Фотометрические трансформы рамки не двигают. CoarseDropout (Cutout) —
    # «пространственный», но боксы не сдвигает; ultralytics корректно проводит его
    # через bbox-процессор albumentations (рамки остаются валидными).
    # ВАЖНО: erasing/RandomErasing в ultralytics работает ТОЛЬКО для классификации,
    # поэтому Cutout для детекции делаем именно через CoarseDropout.
    return [
        A.Blur(p=0.30),
        A.MedianBlur(p=0.15),
        A.GaussianBlur(p=0.25),
        A.CLAHE(p=0.30),
        A.ToGray(p=0.05),
        A.RandomBrightnessContrast(p=0.30),
        A.RandomGamma(p=0.20),
        A.ImageCompression(quality_range=(60, 100), p=0.20),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.05, 0.18),
                        hole_width_range=(0.05, 0.18), fill=0, p=0.30),  # Cutout
    ]


def _patched_alb_init(self, p=1.0, transforms=None):
    if not _STRONG["on"]:
        # baseline: албументации выключены целиком
        self.p = p
        self.transform = None
        self.contains_spatial = False
        return
    if transforms is None:
        transforms = _strong_transforms()
    _ORIG_ALB_INIT(self, p=p, transforms=transforms)


_aug.Albumentations.__init__ = _patched_alb_init


# ============================================================ КОНФИГУРАЦИИ
# Различаются ТОЛЬКО фотометрической аугментацией. Геометрия (mosaic, flip) общая.
CONFIGS = [
    {
        "name": "baseline",
        "desc": "YOLOv8n без фотометрической аугментации (контроль)",
        "strong": False,
        "hyp": {"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0},
    },
    {
        "name": "aug",
        "desc": "YOLOv8n с аугментацией: HSV + Blur/CLAHE/ToGray + яркость/контраст/gamma/JPEG + Cutout (CoarseDropout)",
        "strong": True,
        "hyp": {"hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4},
    },
]


# ============================================================ деградация изображения

def degrade(bgr, rng):
    """Сильная реалистичная деградация (как у «плохой» камеры): размытие + шум +
    сдвиг яркости/контраста + JPEG-артефакты. Геометрию НЕ меняем — разметка валидна.
    Имитирует условия, в которых видна разница между обычной и аугментированной моделью."""
    img = bgr.copy()
    k = int(rng.choice([7, 9, 11, 13]))
    img = cv2.GaussianBlur(img, (k, k), 0)
    # сдвиг яркости/контраста
    alpha = rng.uniform(0.45, 1.6)   # контраст
    beta = rng.uniform(-70, 70)      # яркость
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    # гауссов шум
    noise = rng.normal(0, rng.uniform(10, 30), img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # JPEG-артефакты
    q = int(rng.integers(12, 35))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


def build_degraded_test(data_yaml, out_dir, seed=0):
    """Создаёт деградированную копию тестовой выборки с теми же метками
    и data-yaml, указывающим на неё. Возвращает путь к новому data.yaml."""
    import yaml
    cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    test_images = root / cfg.get("test", "images/test")
    test_labels = Path(str(test_images).replace("images", "labels"))

    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)  # чистим, чтобы не осталось стейл-файлов от прошлого набора
    di, dl = out_dir / "images", out_dir / "labels"
    di.mkdir(parents=True, exist_ok=True)
    dl.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    n = 0
    for img_path in sorted(test_images.iterdir()):
        if img_path.suffix.lower() not in exts:
            continue
        data = np.fromfile(str(img_path), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        deg = degrade(bgr, rng)
        ok, enc = cv2.imencode(".jpg", deg)
        if ok:
            enc.tofile(str(di / (img_path.stem + ".jpg")))
        lbl = test_labels / (img_path.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, dl / (img_path.stem + ".txt"))
        n += 1
    # data.yaml для деградированного теста
    new_yaml = out_dir / "data_degraded.yaml"
    new_cfg = {"path": str(out_dir).replace("\\", "/"), "train": "images",
               "val": "images", "test": "images", "nc": cfg["nc"], "names": cfg["names"]}
    new_yaml.write_text(yaml.safe_dump(new_cfg, allow_unicode=True), encoding="utf-8")
    print(f"Деградированный тест: {n} изображений -> {out_dir}")
    return str(new_yaml)


# ============================================================ обучение и оценка

def train_one(cfg, args, device, project):
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    print(f"\n{'=' * 70}\n[{cfg['name']}] {cfg['desc']}\n{'=' * 70}")
    _STRONG["on"] = cfg["strong"]
    model = YOLO("yolov8n.pt")
    n_params = sum(p.numel() for p in model.model.parameters())
    try:
        gflops = get_flops(model.model, imgsz=args.imgsz)
    except Exception:
        gflops = None

    t0 = time.time()
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=device, project=project, name=cfg["name"], exist_ok=True,
                seed=0, deterministic=False, verbose=True, plots=True,
                patience=args.epochs, workers=args.workers, **cfg["hyp"])
    train_min = (time.time() - t0) / 60.0
    run_dir = Path(project) / cfg["name"]
    return {"name": cfg["name"], "desc": cfg["desc"], "strong": cfg["strong"],
            "params_M": round(n_params / 1e6, 2),
            "GFLOPs": round(gflops, 1) if gflops else None,
            "train_min": round(train_min, 1), "run_dir": str(run_dir),
            "best": str(run_dir / "weights" / "best.pt")}


def eval_split(best, data_yaml, args, device, project, name):
    from ultralytics import YOLO
    m = YOLO(best)
    tm = m.val(data=data_yaml, split="test", imgsz=args.imgsz, batch=args.batch,
               device=device, workers=0, project=project, name=name, exist_ok=True,
               plots=True, verbose=False)
    cm = tm.confusion_matrix.matrix if hasattr(tm, "confusion_matrix") else None
    return {"mAP50": round(float(tm.box.map50), 4), "mAP50_95": round(float(tm.box.map), 4),
            "precision": round(float(tm.box.mp), 4), "recall": round(float(tm.box.mr), 4),
            "speed_ms": round(sum(tm.speed.values()), 1) if hasattr(tm, "speed") else None,
            "cm": cm.tolist() if cm is not None else None}


def eval_bad_photos(rows, bad_dir, args, device):
    """Прогон обеих моделей по реальным «плохим» фото (без разметки):
    доля кадров с детекцией и средняя уверенность — мера устойчивости."""
    from ultralytics import YOLO
    bad = sorted([p for p in Path(bad_dir).iterdir()
                  if p.suffix.lower() in {".bmp", ".jpg", ".jpeg", ".png"}])
    out = []
    for r in rows:
        m = YOLO(r["best"])
        n_det, confs = 0, []
        for p in bad:
            res = m.predict(str(p), imgsz=args.imgsz, conf=0.25, device=device, verbose=False)[0]
            if len(res.boxes):
                n_det += 1
                confs.append(float(res.boxes.conf.max()))
        out.append({"model": r["name"], "n_images": len(bad), "n_detected": n_det,
                    "detect_rate": round(n_det / max(1, len(bad)), 3),
                    "mean_conf": round(float(np.mean(confs)), 3) if confs else 0.0})
        print(f"  [{r['name']}] плохие фото: детекций {n_det}/{len(bad)}, "
              f"ср.conf {out[-1]['mean_conf']}")
    return out


# ============================================================ графики

def plot_metrics(rows, out):
    labels = ["mAP@50", "mAP@50-95", "Precision", "Recall"]
    keys = ["mAP50", "mAP50_95", "precision", "recall"]
    x = np.arange(len(labels))
    w = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, split, title in [(axes[0], "clean", "Чистый тест"),
                             (axes[1], "degraded", "Деградированный тест")]:
        for i, r in enumerate(rows):
            vals = [r[split][k] for k in keys]
            bars = ax.bar(x + (i - 0.5) * w, vals, w, label=r["name"])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.08); ax.set_title(title); ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.suptitle("Метрики: baseline vs aug", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130); plt.close(fig)


def plot_robustness(rows, out):
    fig, ax = plt.subplots(figsize=(8, 6))
    names = [r["name"] for r in rows]
    clean = [r["clean"]["mAP50_95"] for r in rows]
    deg = [r["degraded"]["mAP50_95"] for r in rows]
    x = np.arange(len(names)); w = 0.35
    ax.bar(x - w / 2, clean, w, label="чистый тест")
    ax.bar(x + w / 2, deg, w, label="деградированный")
    for i, r in enumerate(rows):
        drop = r["clean"]["mAP50_95"] - r["degraded"]["mAP50_95"]
        ax.text(i, max(clean[i], deg[i]) + 0.02, f"падение\n-{drop:.3f}",
                ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("mAP@50-95"); ax.set_ylim(0, 1.0)
    ax.set_title("Устойчивость к деградации изображения")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_loss_curves(rows, out):
    keys = [("train/box_loss", "box (train)"), ("val/box_loss", "box (val)"),
            ("metrics/mAP50(B)", "mAP@50 (val)"), ("metrics/mAP50-95(B)", "mAP@50-95 (val)")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (col, title) in zip(axes.flat, keys):
        for r in rows:
            csv_path = Path(r["run_dir"]) / "results.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path); df.columns = [c.strip() for c in df.columns]
            if col in df.columns:
                ax.plot(df["epoch"], df[col], label=r["name"], linewidth=2)
        ax.set_title(title); ax.set_xlabel("эпоха"); ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Кривые обучения", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out, dpi=130); plt.close(fig)


def plot_confusion(rows, out, names=("marker", "background")):
    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 5))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, rows):
        cm = np.array(r["clean"]["cm"], dtype=float)
        cmn = cm / (cm.sum(axis=0, keepdims=True) + 1e-9)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{r['name']} (mAP@50={r['clean']['mAP50']:.2f})")
        ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)
        ax.set_xlabel("Истинный класс"); ax.set_ylabel("Предсказание")
        for i in range(cmn.shape[0]):
            for j in range(cmn.shape[1]):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black")
    fig.suptitle("Матрицы ошибок на чистом тесте", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(out, dpi=130); plt.close(fig)


def write_report(rows, bad_rows, out):
    lines = ["# Аугментация YOLOv8n: baseline vs aug", "",
             "## Метрики на тестовой выборке", "",
             "| Модель | mAP@50 (чист.) | mAP@50-95 (чист.) | mAP@50 (дегр.) | mAP@50-95 (дегр.) | Падение mAP@50-95 |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        drop = r["clean"]["mAP50_95"] - r["degraded"]["mAP50_95"]
        lines.append(f"| {r['name']} | {r['clean']['mAP50']:.3f} | {r['clean']['mAP50_95']:.3f} | "
                     f"{r['degraded']['mAP50']:.3f} | {r['degraded']['mAP50_95']:.3f} | -{drop:.3f} |")
    lines += ["", "## Устойчивость на реальных «плохих» фото", "",
              "| Модель | Кадров | С детекцией | Доля | Ср. уверенность |", "|---|---|---|---|---|"]
    for b in bad_rows:
        lines.append(f"| {b['model']} | {b['n_images']} | {b['n_detected']} | "
                     f"{b['detect_rate']} | {b['mean_conf']} |")
    base = next(r for r in rows if r["name"] == "baseline")
    aug = next(r for r in rows if r["name"] == "aug")
    d_base = base["clean"]["mAP50_95"] - base["degraded"]["mAP50_95"]
    d_aug = aug["clean"]["mAP50_95"] - aug["degraded"]["mAP50_95"]
    lines += ["", "## Вывод", "",
              f"На чистом тесте модели сопоставимы (задача лёгкая). Ключевой эффект — "
              f"устойчивость к деградации: при искусственной порче изображения mAP@50-95 "
              f"падает у baseline на {d_base:.3f}, у aug — на {d_aug:.3f}.",
              "", "Аугментации применялись без изменения разметки: HSV-сдвиги, Blur/MedianBlur/"
              "GaussianBlur, CLAHE, ToGray, яркость/контраст, gamma, JPEG-сжатие, Cutout (erasing). "
              "Интерпретация обеих моделей — в `results/interpretation/` (см. отдельный отчёт)."]
    Path(out).write_text("\n".join(lines), encoding="utf-8")


# ============================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--bad-photos", default=None, help="папка с «плохими» фото")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--out", default="results/detection")
    ap.add_argument("--eval-only", action="store_true",
                    help="не обучать заново, использовать готовые веса runs/<name>/weights/best.pt")
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device} | torch {torch.__version__}")

    project = str(Path(args.project).resolve())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # 1) обучение (или подхват готовых весов при --eval-only)
    if args.eval_only:
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import get_flops
        prev = {}
        if (out / "metrics.csv").exists():  # восстановить время обучения из прошлого прогона
            for d in csv.DictReader(open(out / "metrics.csv", encoding="utf-8-sig")):
                prev[d["model"]] = d.get("train_min")
        rows = []
        for cfg in CONFIGS:
            best = Path(project) / cfg["name"] / "weights" / "best.pt"
            m = YOLO(str(best))
            n_params = sum(p.numel() for p in m.model.parameters())
            try:
                gflops = get_flops(m.model, imgsz=args.imgsz)
            except Exception:
                gflops = None
            rows.append({"name": cfg["name"], "desc": cfg["desc"], "strong": cfg["strong"],
                         "params_M": round(n_params / 1e6, 2),
                         "GFLOPs": round(gflops, 1) if gflops else None,
                         "train_min": prev.get(cfg["name"]),
                         "run_dir": str(Path(project) / cfg["name"]), "best": str(best)})
        print("Режим --eval-only: используются готовые веса.")
    else:
        rows = [train_one(cfg, args, device, project) for cfg in CONFIGS]

    # 2) деградированный тест (одна копия, общая для обеих моделей)
    deg_yaml = build_degraded_test(args.data, Path(project).parent / "_degraded_test")

    # 3) оценка на чистом и деградированном тесте
    for r in rows:
        r["clean"] = eval_split(r["best"], args.data, args, device, project, f"{r['name']}_clean")
        r["degraded"] = eval_split(r["best"], deg_yaml, args, device, project, f"{r['name']}_degraded")
        print(f"  [{r['name']}] чистый mAP50-95={r['clean']['mAP50_95']} | "
              f"деградир. mAP50-95={r['degraded']['mAP50_95']}")

    # 4) реальные «плохие» фото
    bad_rows = []
    if args.bad_photos and Path(args.bad_photos).exists():
        bad_rows = eval_bad_photos(rows, args.bad_photos, args, device)

    # 5) графики + таблицы + отчёт
    plot_metrics(rows, out / "compare_metrics.png")
    plot_robustness(rows, out / "robustness_drop.png")
    plot_loss_curves(rows, out / "compare_loss_curves.png")
    plot_confusion(rows, out / "compare_confusion.png")
    with open(out / "metrics.csv", "w", newline="", encoding="utf-8-sig") as fp:
        wr = csv.writer(fp)
        wr.writerow(["model", "params_M", "GFLOPs", "train_min",
                     "clean_mAP50", "clean_mAP50_95", "clean_P", "clean_R",
                     "deg_mAP50", "deg_mAP50_95", "deg_P", "deg_R"])
        for r in rows:
            wr.writerow([r["name"], r["params_M"], r["GFLOPs"], r["train_min"],
                         r["clean"]["mAP50"], r["clean"]["mAP50_95"], r["clean"]["precision"], r["clean"]["recall"],
                         r["degraded"]["mAP50"], r["degraded"]["mAP50_95"], r["degraded"]["precision"], r["degraded"]["recall"]])
    if bad_rows:
        with open(out / "bad_photos_robustness.csv", "w", newline="", encoding="utf-8-sig") as fp:
            wr = csv.DictWriter(fp, fieldnames=list(bad_rows[0].keys()))
            wr.writeheader(); wr.writerows(bad_rows)
    write_report(rows, bad_rows, out / "report.md")
    print(f"\nГотово. Детекционное сравнение: {out.resolve()}")


if __name__ == "__main__":
    main()
