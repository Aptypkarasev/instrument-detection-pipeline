# -*- coding: utf-8 -*-
"""
Сравнение интерпретации (Grad-CAM) двух моделей — baseline и aug — на хороших
(чистый тест) и «плохих» (реальные кадры с камер) фото.

Для каждой модели и каждого изображения считаются:
  * faithfulness — насколько падает уверенность при маскировании горячей зоны Grad-CAM
                   против маскирования случайной зоны (честность карты);
  * pointing game — попал ли пик Grad-CAM внутрь детектированного бокса (локализация);
  * energy in box — доля массы карты внутри боксов (мягкая локализация);
  * detect rate / conf — доля кадров с детекцией и уверенность.

На выходе (results/interpretation/):
  interp_metrics.csv               — сводные метрики (model x набор)
  compare_interp_metrics.png       — столбчатое сравнение метрик интерпретации
  <набор>/<кадр>__compare.png      — Grad-CAM baseline vs aug бок о бок

Запуск:
  python interpret_compare.py --baseline runs/baseline/weights/best.pt \
      --aug runs/aug/weights/best.pt --good <папка_теста> --bad bad_photos --out results/interpretation
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultralytics.utils.ops import scale_boxes
import yolo_interpret as yi

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def analyze_image(ex, bgr, imgsz):
    """Grad-CAM + метрики для одного изображения одной моделью."""
    x, lb = yi.preprocess(bgr, imgsz)
    gcam, _, det = ex.gradcam(x)
    gcam_o = yi.cam_to_original(gcam, lb)
    h0, w0 = bgr.shape[:2]
    boxes, confs = [], []
    if len(det):
        import torch
        boxes = scale_boxes((imgsz, imgsz), det[:, :4].clone(), (h0, w0)).cpu().numpy()
        confs = [float(c) for c in det[:, 4]]
    faith, _ = ex.faithfulness(bgr, gcam_o) if len(det) else (None, None)
    loc = yi.localization_metrics(gcam_o, boxes) if len(boxes) else {"pointing_game": None, "energy_in_box": None}
    overlay = yi.overlay_jet(bgr, gcam_o)
    overlay = yi.draw_boxes(overlay, boxes, confs, color=(255, 255, 255), thickness=2)
    return {"n_det": len(det), "confs": confs, "faith": faith, "loc": loc, "overlay": overlay}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--aug", required=True)
    ap.add_argument("--good", required=True, help="папка хороших (тестовых) изображений")
    ap.add_argument("--bad", required=True, help="папка «плохих» изображений")
    ap.add_argument("--out", default="results/interpretation")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n-good", type=int, default=10)
    ap.add_argument("--n-bad", type=int, default=15)
    ap.add_argument("--n-panels", type=int, default=6, help="сколько side-by-side панелей сохранять на набор")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = {"baseline": args.baseline, "aug": args.aug}
    explainers = {m: yi.YoloExplainer(w, imgsz=args.imgsz, device="cpu") for m, w in models.items()}

    sets = {
        "good_test": sorted([p for p in Path(args.good).iterdir() if p.suffix.lower() in IMG_EXTS])[:args.n_good],
        "bad_photos": sorted([p for p in Path(args.bad).iterdir() if p.suffix.lower() in IMG_EXTS])[:args.n_bad],
    }

    agg = []  # строки метрик
    for set_name, files in sets.items():
        set_dir = out / set_name
        set_dir.mkdir(exist_ok=True)
        # накопители по моделям
        acc = {m: {"drop_cam": [], "drop_random": [], "pg": [], "energy": [],
                   "n_det": 0, "confs": []} for m in models}
        for i, f in enumerate(files):
            bgr = yi.imread_safe(f)
            if bgr is None:
                continue
            res = {}
            for m, ex in explainers.items():
                r = analyze_image(ex, bgr, args.imgsz)
                res[m] = r
                a = acc[m]
                if r["n_det"]:
                    a["n_det"] += 1
                    a["confs"].extend(r["confs"])
                if r["faith"]:
                    a["drop_cam"].append(r["faith"]["drop_cam"])
                    a["drop_random"].append(r["faith"]["drop_random"])
                if r["loc"]["pointing_game"] is not None:
                    a["pg"].append(r["loc"]["pointing_game"])
                    a["energy"].append(r["loc"]["energy_in_box"])
            # side-by-side панель для первых n_panels
            if i < args.n_panels:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                for ax, m in zip(axes, ["baseline", "aug"]):
                    ax.imshow(cv2.cvtColor(res[m]["overlay"], cv2.COLOR_BGR2RGB))
                    fa = res[m]["faith"]
                    lo = res[m]["loc"]
                    sub = (f"{m}: детекций {res[m]['n_det']}"
                           + (f", conf {max(res[m]['confs']):.2f}" if res[m]['confs'] else "")
                           + (f"\nfaith drop {fa['drop_cam']:.2f} vs rnd {fa['drop_random']:.2f}" if fa else "")
                           + (f", pg {lo['pointing_game']:.0f}, energy {lo['energy_in_box']:.2f}"
                              if lo['pointing_game'] is not None else ""))
                    ax.set_title(sub, fontsize=11)
                    ax.axis("off")
                fig.suptitle(f"{set_name} · {f.name} · Grad-CAM: baseline vs aug", fontsize=13)
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                fig.savefig(set_dir / f"{f.stem}__compare.png", dpi=110)
                plt.close(fig)
            print(f"  [{set_name}] {i + 1}/{len(files)} {f.name}")

        for m in models:
            a = acc[m]
            n = len(files)
            agg.append({
                "model": m, "set": set_name, "n_images": n,
                "detect_rate": round(a["n_det"] / max(1, n), 3),
                "mean_conf": round(float(np.mean(a["confs"])), 3) if a["confs"] else 0.0,
                "faith_drop_cam": round(float(np.mean(a["drop_cam"])), 3) if a["drop_cam"] else None,
                "faith_drop_random": round(float(np.mean(a["drop_random"])), 3) if a["drop_random"] else None,
                "pointing_game_rate": round(float(np.mean(a["pg"])), 3) if a["pg"] else None,
                "energy_in_box": round(float(np.mean(a["energy"])), 3) if a["energy"] else None,
            })

    # CSV
    with open(out / "interp_metrics.csv", "w", newline="", encoding="utf-8-sig") as fp:
        wr = csv.DictWriter(fp, fieldnames=list(agg[0].keys()))
        wr.writeheader()
        wr.writerows(agg)

    # сводный график по метрикам интерпретации
    plot_interp_metrics(agg, out / "compare_interp_metrics.png")
    print(f"\nГотово. Сравнение интерпретации: {out.resolve()}")
    for r in agg:
        print(f"  {r['model']:9s} {r['set']:11s} detect={r['detect_rate']} conf={r['mean_conf']} "
              f"pg={r['pointing_game_rate']} energy={r['energy_in_box']} "
              f"faith={r['faith_drop_cam']} vs rnd={r['faith_drop_random']}")


def plot_interp_metrics(agg, out):
    sets = ["good_test", "bad_photos"]
    metrics = [("detect_rate", "Доля детекций"), ("pointing_game_rate", "Pointing game"),
               ("energy_in_box", "Energy in box"), ("faith_drop_cam", "Faithfulness (drop CAM)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
    for ax, (key, title) in zip(axes, metrics):
        x = np.arange(len(sets))
        w = 0.35
        for i, m in enumerate(["baseline", "aug"]):
            vals = []
            for s in sets:
                row = next((r for r in agg if r["model"] == m and r["set"] == s), None)
                vals.append(row[key] if row and row[key] is not None else 0)
            bars = ax.bar(x + (i - 0.5) * w, vals, w, label=m)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(sets)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.suptitle("Метрики интерпретации: baseline vs aug", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
