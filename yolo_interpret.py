# -*- coding: utf-8 -*-
"""
Интерпретация YOLO-детектора (класс 'marker').

Для каждого изображения строит:
  1. Детекции модели (+ ground truth, если есть разметка)
  2. EigenCAM         — карта внимания по активациям шеи сети (без градиентов)
  3. Grad-CAM         — карта по градиентам от суммарного скора всех детекций
  4. Grad-CAM по каждой детекции отдельно (что "зажгло" конкретный бокс)
  5. SmoothGrad       — пиксельная значимость (усреднённый градиент по зашумлённым копиям)
  6. Integrated Gradients — пиксельная значимость (интеграл градиента от размытого базлайна)
  7. Faithfulness-тест — маскируем горячую зону Grad-CAM и проверяем, насколько
     падает уверенность модели (vs маскирование случайной зоны того же размера).

Запуск:
  python yolo_interpret.py --model best.pt --source папка_с_изображениями --out results
  Флаг --fast пропускает медленные пиксельные методы (SmoothGrad / IG).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
try:  # ultralytics >= 8.4
    from ultralytics.utils.nms import non_max_suppression
except ImportError:  # старые версии
    from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.ops import scale_boxes, xywh2xyxy
from ultralytics.utils.metrics import box_iou

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def imread_safe(path):
    """cv2.imread на Windows не умеет пути вне системной кодировки (например,
    кириллица при западной локали) — читаем байты через numpy."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite_safe(path, img):
    """Запись через imencode+tofile по той же причине; возвращает успех."""
    ok, buf = cv2.imencode(Path(path).suffix, img)
    if ok:
        buf.tofile(str(path))
    return bool(ok)


# ----------------------------------------------------------------------------- утилиты

def normalize(x, lo_pct=0.0, hi_pct=100.0):
    """Минимакс-нормализация с обрезкой по перцентилям (устойчивость к выбросам)."""
    lo = np.percentile(x, lo_pct)
    hi = np.percentile(x, hi_pct)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-12)


def preprocess(bgr, imgsz):
    """BGR -> letterbox -> тензор (1,3,H,W) 0..1. Возвращает тензор и параметры letterbox."""
    h0, w0 = bgr.shape[:2]
    lb = LetterBox((imgsz, imgsz), auto=False, stride=32)
    im = lb(image=bgr)
    x = torch.from_numpy(im[..., ::-1].copy().transpose(2, 0, 1)).float()[None] / 255.0
    r = min(imgsz / h0, imgsz / w0)
    nw, nh = round(w0 * r), round(h0 * r)
    top = int(round((imgsz - nh) / 2 - 0.1))
    left = int(round((imgsz - nw) / 2 - 0.1))
    return x, (h0, w0, nh, nw, top, left)


def cam_to_original(cam, lb_params):
    """Карта (imgsz x imgsz) -> убрать поля letterbox -> размер исходного изображения."""
    h0, w0, nh, nw, top, left = lb_params
    cam = cam[top:top + nh, left:left + nw]
    return cv2.resize(cam, (w0, h0), interpolation=cv2.INTER_LINEAR)


def overlay_jet(bgr, cam, alpha=0.85, gamma=0.7):
    """Наложение JET с прозрачностью, пропорциональной теплу:
    фон остаётся естественным, горячие зоны подсвечиваются."""
    cam = np.clip(cam, 0, 1) ** gamma
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)
    w = (alpha * cam)[..., None]
    out = bgr.astype(np.float32) * (1 - w) + heat * w
    return out.astype(np.uint8)


def render_pixel_map(sal, cmap="hot"):
    """Пиксельная карта значимости на чёрном фоне (стиль Integrated Gradients)."""
    cm = plt.get_cmap(cmap)
    rgb = (cm(np.clip(sal, 0, 1))[..., :3] * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_boxes(bgr, boxes, confs, color=(60, 220, 60), thickness=3, prefix="marker"):
    out = bgr.copy()
    for b, c in zip(boxes, confs):
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{prefix} {c:.2f}" if c is not None else prefix
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return out


def load_gt_boxes(label_path, w0, h0):
    """YOLO-разметка (cls cx cy w h, нормированная) -> xyxy в пикселях."""
    boxes = []
    if label_path and label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            p = line.split()
            if len(p) >= 5:
                cx, cy, w, h = float(p[1]) * w0, float(p[2]) * h0, float(p[3]) * w0, float(p[4]) * h0
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return boxes


def localization_metrics(cam_orig, boxes):
    """Количественная локализация карты внимания относительно боксов.
      pointing_game — попал ли самый горячий пиксель Grad-CAM внутрь какого-либо бокса
                      (классическая метрика «pointing game» для объяснимости);
      energy_in_box — доля «массы» карты, лежащая внутри боксов (мягкая локализация).
    Чем выше обе — тем точнее модель «смотрит» на объект, а не на фон."""
    h, w = cam_orig.shape
    box_mask = np.zeros((h, w), dtype=bool)
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            box_mask[y1:y2, x1:x2] = True
    if not box_mask.any():
        return {"pointing_game": None, "energy_in_box": None}
    peak = np.unravel_index(int(np.argmax(cam_orig)), cam_orig.shape)
    pg = float(box_mask[peak])
    total = float(cam_orig.sum()) + 1e-12
    energy = float(cam_orig[box_mask].sum()) / total
    return {"pointing_game": pg, "energy_in_box": round(energy, 4)}


# ----------------------------------------------------------------------------- ядро

class YoloExplainer:
    def __init__(self, weights, imgsz=640, conf=0.25, device="cpu"):
        self.yolo = YOLO(weights)
        self.tm = self.yolo.model.to(device).eval()
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.names = self.yolo.names
        # Слои, выходы которых идут в Detect-голову (P3/P4/P5 шеи) — цели для CAM
        detect = self.tm.model[-1]
        self.layer_idxs = list(detect.f)
        self.layers = [self.tm.model[i] for i in self.layer_idxs]
        self._acts, self._grads = {}, {}
        for i, m in zip(self.layer_idxs, self.layers):
            m.register_forward_hook(self._make_fwd_hook(i))
            m.register_full_backward_hook(self._make_bwd_hook(i))

    def _make_fwd_hook(self, key):
        def hook(_m, _inp, out):
            self._acts[key] = out
        return hook

    def _make_bwd_hook(self, key):
        def hook(_m, _gin, gout):
            self._grads[key] = gout[0].detach()
        return hook

    def forward(self, x):
        out = self.tm(x)
        preds = out[0] if isinstance(out, (tuple, list)) else out
        return preds  # (1, 4+nc, N): xywh в коорд. входа + вероятности классов

    def detect(self, preds):
        """NMS -> (n,6): xyxy, conf, cls в координатах входа (letterbox).
        clone() обязателен: NMS в ultralytics конвертирует боксы xywh->xyxy
        НА МЕСТЕ, а detach() разделяет память с живым preds."""
        det = non_max_suppression(preds.detach().clone(), conf_thres=self.conf, iou_thres=0.45)[0]
        return det

    @staticmethod
    def _anchor_boxes_xyxy(preds, det_box_xyxy):
        """Боксы анкеров в xyxy. Голова Detect отдаёт xywh, но формат тензора к этому
        моменту может отличаться по версиям ultralytics (а NMS старых версий и вовсе
        мутировал его на месте) — поэтому не доверяем формату, а выбираем интерпретацию,
        при которой бокс детекции после NMS совпадает с одним из анкеров (IoU ~ 1)."""
        raw = preds[0, :4, :].T                                # (N,4)
        as_xyxy, as_xywh = raw, xywh2xyxy(raw)
        iou1 = box_iou(det_box_xyxy[None], as_xyxy.detach())[0].max()
        iou2 = box_iou(det_box_xyxy[None], as_xywh.detach())[0].max()
        return as_xyxy if iou1 >= iou2 else as_xywh

    def _det_score(self, preds, det_box_xyxy, det_cls, iou_thr=0.6):
        """Дифференцируемый скор детекции: max вероятности класса среди анкеров,
        чьи боксы пересекаются с финальной детекцией (IoU > порога)."""
        anchors_xyxy = self._anchor_boxes_xyxy(preds, det_box_xyxy)
        cls_prob = preds[0, 4 + int(det_cls), :]               # (N,)
        iou = box_iou(det_box_xyxy[None], anchors_xyxy.detach())[0]
        mask = iou > iou_thr
        if mask.sum() == 0:
            mask = iou >= iou.max()                            # ближайший анкер
        return (cls_prob * mask.float()).max()

    def _cam_from_grads(self):
        """Поэлементный Grad-CAM (HiResCAM-стиль): карта = ReLU(сумма_по_каналам(grad * act)).
        Для детекторов даёт более локализованные карты, чем классическое усреднение градиента.
        Слои объединяются взвешенно по их «энергии», чтобы слабый слой не вносил шум."""
        maps, weights = [], []
        for k in self.layer_idxs:
            act, grad = self._acts[k].detach(), self._grads.get(k)
            if grad is None:
                continue
            cam = torch.relu((grad * act).sum(dim=1))[0].cpu().numpy()
            cam = cv2.resize(cam, (self.imgsz, self.imgsz))
            maps.append(cam)
            weights.append(float(cam.max()))
        if not maps:
            return np.zeros((self.imgsz, self.imgsz), dtype=np.float32)
        w_max = max(weights) + 1e-12
        total = np.zeros((self.imgsz, self.imgsz), dtype=np.float32)
        for cam in maps:
            total += cam / w_max
        return normalize(total, 0, 99.9)

    def gradcam(self, x):
        """Возвращает: глобальная карта, [карты по детекциям], детекции (input-коорд.)."""
        self._acts.clear()
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            preds = self.forward(x)
        det = self.detect(preds)
        per_det, global_cam = [], None
        if len(det) == 0:
            # Нет детекций — объясняем максимальный анкерный скор (куда модель смотрит "почти")
            score = preds[0, 4:, :].max()
            self.tm.zero_grad(set_to_none=True)
            self._grads.clear()
            score.backward(retain_graph=False)
            global_cam = self._cam_from_grads()
            return global_cam, [], det
        # Карта по каждой детекции
        for j in range(len(det)):
            score = self._det_score(preds, det[j, :4], det[j, 5])
            self.tm.zero_grad(set_to_none=True)
            self._grads.clear()
            score.backward(retain_graph=True)
            per_det.append(self._cam_from_grads())
        # Глобальная карта = сумма скоров всех детекций
        total_score = sum(self._det_score(preds, det[j, :4], det[j, 5]) for j in range(len(det)))
        self.tm.zero_grad(set_to_none=True)
        self._grads.clear()
        total_score.backward(retain_graph=False)
        global_cam = self._cam_from_grads()
        return global_cam, per_det, det

    def eigencam(self, x):
        """EigenCAM: первая главная компонента активаций (без градиентов)."""
        self._acts.clear()
        with torch.no_grad():
            self.forward(x)
        total = np.zeros((self.imgsz, self.imgsz), dtype=np.float32)
        for k in self.layer_idxs:
            a = self._acts[k][0].detach().cpu().numpy()        # (C,H,W)
            C, H, W = a.shape
            m = a.reshape(C, -1).T                             # (HW, C)
            m = m - m.mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(m, full_matrices=False)
            proj = (m @ vt[0]).reshape(H, W)
            if abs(proj.min()) > abs(proj.max()):
                proj = -proj
            proj = np.maximum(proj, 0)
            proj = cv2.resize(proj, (self.imgsz, self.imgsz))
            if proj.max() > 0:
                total += proj / proj.max()
        return normalize(total)

    def _pixel_target(self, x):
        """Скалярная цель для пиксельных градиентов: сумма скоров уверенных анкеров."""
        preds = self.forward(x)
        cls_prob = preds[0, 4:, :]
        strong = cls_prob[cls_prob > self.conf]
        return strong.sum() if strong.numel() > 0 else cls_prob.max()

    def smoothgrad(self, x, n_samples=12, sigma=0.10):
        acc = torch.zeros_like(x)
        for _ in range(n_samples):
            noisy = (x + torch.randn_like(x) * sigma).clamp(0, 1).requires_grad_(True)
            with torch.enable_grad():
                target = self._pixel_target(noisy)
            self.tm.zero_grad(set_to_none=True)
            target.backward()
            acc += noisy.grad.abs()
        sal = acc[0].max(dim=0).values.cpu().numpy()
        return normalize(sal, 0, 99.0)

    def integrated_gradients(self, x, steps=24):
        """Базлайн — сильно размытое изображение (для детекции честнее чёрного)."""
        img = (x[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        base = cv2.GaussianBlur(img, (63, 63), 0)
        x0 = torch.from_numpy(base.transpose(2, 0, 1)).float()[None].to(x.device) / 255.0
        acc = torch.zeros_like(x)
        for s in range(1, steps + 1):
            xi = (x0 + (x - x0) * (s / steps)).requires_grad_(True)
            with torch.enable_grad():
                target = self._pixel_target(xi)
            self.tm.zero_grad(set_to_none=True)
            target.backward()
            acc += xi.grad
        attr = ((x - x0) * acc / steps)[0].abs().sum(dim=0).cpu().numpy()
        return normalize(attr, 0, 99.5)

    def faithfulness(self, bgr, cam_orig, frac=0.10):
        """Маскируем ровно top-`frac` пикселей CAM размытием -> падение уверенности.
        Контроль: маска той же площади, циклически сдвинутая (случайная зона)."""
        def max_conf(img):
            r = self.yolo.predict(img, conf=0.01, imgsz=self.imgsz,
                                  device=self.device, verbose=False)[0]
            return float(r.boxes.conf.max()) if len(r.boxes) else 0.0

        k = max(1, int(frac * cam_orig.size))
        idx = np.argpartition(cam_orig.ravel(), -k)[-k:]
        mask = np.zeros(cam_orig.size, dtype=bool)
        mask[idx] = True
        mask = mask.reshape(cam_orig.shape)
        blurred = cv2.GaussianBlur(bgr, (151, 151), 0)
        top_masked = np.where(mask[..., None], blurred, bgr)
        h, w = mask.shape
        rnd_mask = np.roll(np.roll(mask, h // 3, axis=0), w // 2, axis=1)
        rnd_masked = np.where(rnd_mask[..., None], blurred, bgr)
        c_orig = max_conf(bgr)
        c_top = max_conf(top_masked)
        c_rnd = max_conf(rnd_masked)
        return {"conf_orig": c_orig, "conf_cam_masked": c_top, "conf_random_masked": c_rnd,
                "drop_cam": c_orig - c_top, "drop_random": c_orig - c_rnd}, top_masked


# ----------------------------------------------------------------------------- отчёт

def make_panel(out_dir, stem, bgr, det_img, eigen_ov, gcam_ov, sg_img, ig_img,
               n_det, faith):
    """Композитная фигура 2x3."""
    fig, axes = plt.subplots(2, 3, figsize=(19.2, 10.8))
    items = [
        (det_img, f"Детекции модели: {n_det} (зелёные) | GT (жёлтые пунктир)"),
        (eigen_ov, "EigenCAM — общее внимание шеи сети"),
        (gcam_ov, "Grad-CAM — что подтверждает детекции"),
        (bgr, "Исходное изображение"),
        (sg_img, "SmoothGrad — пиксельная значимость"),
        (ig_img, "Integrated Gradients — пиксельная значимость"),
    ]
    for ax, (img, title) in zip(axes.flat, items):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=13)
        ax.axis("off")
    if faith:
        fig.suptitle(
            f"{stem}   |   faithfulness: маскирование зоны Grad-CAM роняет conf "
            f"{faith['conf_orig']:.2f} → {faith['conf_cam_masked']:.2f}, "
            f"случайной зоны — лишь до {faith['conf_random_masked']:.2f}",
            fontsize=15)
    else:
        fig.suptitle(stem, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / f"{stem}__panel.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Интерпретация YOLO-детектора")
    ap.add_argument("--model", required=True, help="путь к best.pt")
    ap.add_argument("--source", required=True, help="изображение или папка")
    ap.add_argument("--labels", default=None, help="папка с YOLO-разметкой (опционально, для GT)")
    ap.add_argument("--out", default="results", help="папка результатов")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--fast", action="store_true", help="пропустить SmoothGrad и Integrated Gradients")
    ap.add_argument("--max-images", type=int, default=0, help="ограничить число изображений (0 = все)")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    src = Path(args.source)
    files = sorted([src] if src.is_file() else
                   [p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS])
    if args.max_images:
        files = files[: args.max_images]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    ex = YoloExplainer(args.model, imgsz=args.imgsz, conf=args.conf)
    print(f"Модель: {args.model} | классы: {ex.names} | изображений: {len(files)}")

    faith_rows = []
    panel_paths = []
    for i, f in enumerate(files, 1):
        t0 = time.time()
        try:
            bgr = imread_safe(f)
            if bgr is None:
                print(f"[{i}/{len(files)}] {f.name}: не удалось прочитать, пропуск")
                continue
            h0, w0 = bgr.shape[:2]
            x, lb_params = preprocess(bgr, args.imgsz)
            img_dir = out_root / f.stem
            img_dir.mkdir(exist_ok=True)

            # --- Grad-CAM (+ детекции) и EigenCAM
            gcam, per_det_cams, det = ex.gradcam(x)
            eig = ex.eigencam(x)
            gcam_o = cam_to_original(gcam, lb_params)
            eig_o = cam_to_original(eig, lb_params)

            boxes_o, confs = [], []
            if len(det):
                boxes_o = scale_boxes((args.imgsz, args.imgsz), det[:, :4].clone(), (h0, w0)).cpu().numpy()
                confs = [float(c) for c in det[:, 4]]
            det_img = draw_boxes(bgr, boxes_o, confs)
            gt = load_gt_boxes(Path(args.labels) / f"{f.stem}.txt" if args.labels else None, w0, h0)
            for g in gt:
                x1, y1, x2, y2 = [int(v) for v in g]
                cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 230, 230), 2, lineType=cv2.LINE_AA)

            eig_ov = overlay_jet(bgr, eig_o)
            gcam_ov = overlay_jet(bgr, gcam_o)
            imwrite_safe(img_dir / "01_detections.jpg", det_img)
            imwrite_safe(img_dir / "02_eigencam.jpg", eig_ov)
            imwrite_safe(img_dir / "03_gradcam_global.jpg", gcam_ov)

            # --- Grad-CAM по каждой детекции
            for j, cam_j in enumerate(per_det_cams[:6]):
                cam_jo = cam_to_original(cam_j, lb_params)
                ov = overlay_jet(bgr, cam_jo)
                ov = draw_boxes(ov, [boxes_o[j]], [confs[j]], color=(255, 255, 255), thickness=2)
                imwrite_safe(img_dir / f"04_gradcam_det{j + 1}.jpg", ov)

            # --- пиксельные методы
            if not args.fast:
                sg = cam_to_original(ex.smoothgrad(x), lb_params)
                ig = cam_to_original(ex.integrated_gradients(x), lb_params)
                sg_img = render_pixel_map(sg, "gray")
                ig_img = render_pixel_map(ig, "hot")
                imwrite_safe(img_dir / "05_smoothgrad.jpg", sg_img)
                imwrite_safe(img_dir / "06_integrated_gradients.jpg", ig_img)
            else:
                sg_img = ig_img = np.zeros_like(bgr)

            # --- faithfulness
            faith = None
            if len(det):
                faith, masked_img = ex.faithfulness(bgr, gcam_o)
                faith_rows.append({"image": f.name, "n_detections": len(det), **faith})
                imwrite_safe(img_dir / "07_faithfulness_masked.jpg", masked_img)

            make_panel(out_root, f.stem, bgr, det_img, eig_ov, gcam_ov, sg_img, ig_img,
                       len(det), faith)
            panel_paths.append(out_root / f"{f.stem}__panel.png")
            print(f"[{i}/{len(files)}] {f.name}: det={len(det)} "
                  f"conf={[round(c, 2) for c in confs]} | {time.time() - t0:.1f} c")
        except Exception as e:
            print(f"[{i}/{len(files)}] {f.name}: ошибка ({type(e).__name__}: {e}), пропуск")

    # --- сводка faithfulness
    if faith_rows:
        with open(out_root / "faithfulness.csv", "w", newline="", encoding="utf-8-sig") as fp:
            wr = csv.DictWriter(fp, fieldnames=list(faith_rows[0].keys()))
            wr.writeheader()
            wr.writerows(faith_rows)
        d_cam = np.mean([r["drop_cam"] for r in faith_rows])
        d_rnd = np.mean([r["drop_random"] for r in faith_rows])
        print(f"\nFaithfulness (среднее по {len(faith_rows)} изобр.): "
              f"маскирование зоны CAM роняет conf на {d_cam:.3f}, "
              f"случайной зоны — на {d_rnd:.3f}")
    print(f"Готово. Результаты: {out_root.resolve()}")


if __name__ == "__main__":
    main()
