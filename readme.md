# YOLO Marker Detection: Training, Augmentation, Comparison & Interpretability

Полный пайплайн компьютерного зрения для детекции маркеров с использованием моделей семейства YOLO (YOLOv8 / YOLO11). 
Репозиторий содержит скрипты для базового обучения, сравнения архитектур, исследования влияния аугментаций, проверки робастности (устойчивости) к деградациям, глубокой интерпретации решений нейросети (XAI), а также готовый ноутбук для инференса на реальных «плохих» кадрах.

## Структура проекта

| Файл | Назначение |
|------|------------|
|`predict_marker.ipynb` | Быстрый старт: распаковка датасета, обучение YOLOv8n, валидация, инференс |
|`train_compare.py` | Сравнение архитектур (YOLOv8n/s, YOLO11n, YOLO11n-tuned) |
|`train_aug.py` | Исследование влияния аугментаций Albumentations + тест на робастность |
|`yolo_interpret.py` | XAI-движок: EigenCAM, Grad-CAM, SmoothGrad, Integrated Gradients, Faithfulness |
|`interpret_compare.py` | Сравнение интерпретируемости baseline vs aug моделей |
|`test_bad_photos.ipynb` | Инференс на «плохих» кадрах: отбор лучшей детекции и упаковка в ZIP |

---

## Установка

```bash
pip install ultralytics opencv-python numpy pandas matplotlib albumentations torch torchvision
````

Требования: Python 3.9+, CUDA (рекомендуется), минимум 8 GB VRAM для `batch=16, imgsz=640`.

---

## Пайплайн использования

### Шаг 1. Подготовка данных и базовое обучение (`predict_marker.ipynb`)

Jupyter-ноутбук для быстрого старта в Google Colab. Автоматически выполняет:

1. Монтирование Google Drive и распаковку датасета
2. Обновление `data.yaml` с абсолютными путями
3. Обучение базовой модели `yolov8n.pt` (50 эпох, `imgsz=640`, `batch=16`)
4. Валидацию на тестовой выборке (`mAP@0.5`, `Precision`, `Recall`)
5. Визуализацию предсказаний на 10 случайных изображениях
6. Сохранение `best.pt` в `/content/drive/MyDrive/yolo_models/`

> Подходит для первичной проверки работоспособности датасета и получения baseline-модели.

---

### Шаг 2. Сравнение архитектур (`train_compare.py`)

Скрипт обучает **4 конфигурации** на одном датасете и оценивает их на единой тестовой выборке:

| Конфиг | Архитектура | Оптимизатор / LR | Особенности |
|--------|-------------|------------------|-------------|
| `v8n` | YOLOv8 nano | SGD, lr0=0.01 | Базовые гиперпараметры |
| `v8s` | YOLOv8 small | SGD, lr0=0.01 | Крупнее модель |
| `v11n` | YOLO11 nano | SGD, lr0=0.01 | Новее архитектура |
| `v11n_tuned` | YOLO11 nano | **AdamW, lr0=0.002, cos_lr=True** | mixup=0.10, copy_paste=0.10, hsv_h=0.02, degrees=5.0 |

**Запуск:**
```bash
python train_compare.py --data data.yaml --epochs 60 --imgsz 640 --batch 16
# Обучить только конкретные модели:
python train_compare.py --data data.yaml --only v8n,v11n_tuned
```

**На выходе в `comparison/`:**
- `metrics_comparison.csv` — таблица метрик (mAP50, mAP50-95, Precision, Recall, Params, GFLOPs, Speed)
- `compare_metrics_bars.png` — столбчатое сравнение метрик
- `compare_loss_curves.png` — кривые ошибок (box/cls, train/val)
- `compare_map_curves.png` — рост mAP@50 и mAP@50-95 на валидации
- `compare_confusion_matrices.png` — матрицы ошибок на тесте
- `compare_speed_size.png` — компромисс "Точность vs Размер/Скорость"
- `report.md` — текстовый отчёт с выводами

---

### Шаг 3. Аугментации и Робастность (`train_aug.py`)

Скрипт сравнивает **baseline** (без аугментаций) и **aug** (с сильными аугментациями).

**Особенность реализации:** используется *monkey-patching* класса `ultralytics.data.augment.Albumentations` через флаг `_STRONG["on"]`, чтобы гибко включать/выключать Albumentations без изменения исходного кода Ultralytics.

**Сильные аугментации (Albumentations):**
- `Blur`, `CLAHE` (p=0.30)
- `ToGray` (p=0.05)
- `RandomBrightnessContrast`, `RandomGamma` (p=0.30/0.20)
- `ImageCompression` (quality 60–100, p=0.20)
- `CoarseDropout` (Cutout, 1–8 дыр, p=0.30)

**Синтетическая деградация теста:** скрипт автоматически создаёт деградированную копию тестовой выборки (`GaussianBlur` + `convertScaleAbs` + гауссов шум) для проверки устойчивости моделей.

**Запуск:**
```bash
python train_aug.py --data data.yaml --epochs 50 --imgsz 640 --batch 16 \
    --bad-photos path/to/bad_photos
# Пропустить обучение и использовать готовые веса:
python train_aug.py --data data.yaml --eval-only --bad-photos path/to/bad_photos
```

**На выходе в `results/detection/`:**
- `metrics.csv` — метрики на чистом и деградированном тесте
- `compare_metrics.png` — столбцы метрик (clean vs degraded)
- `robustness_drop.png` — падение mAP при деградации
- `bad_photos_robustness.csv` — детекции/уверенность на реальных "плохих" фото
- `compare_loss_curves.png`, `compare_confusion.png`, `report.md`

---

### Шаг 4. Интерпретация моделей (XAI) (`yolo_interpret.py`)

Кастомный XAI-движок для объяснения решений YOLO-детектора. Поддерживаются:

| Метод | Описание |
|-------|----------|
| **EigenCAM** | Карта внимания по активациям шеи сети (P3/P4/P5) без градиентов |
| **Grad-CAM (global)** | Карта по градиентам от суммарного скора всех детекций |
| **Grad-CAM (per-det)** | Что именно "зажгло" конкретный бокс |
| **SmoothGrad** | Пиксельная значимость (усреднённый градиент по зашумлённым копиям) |
| **Integrated Gradients** | Интеграл градиента от размытого базлайна |
| **Faithfulness** | Маскирование горячей зоны CAM → проверка падения уверенности |

**Локализационные метрики:**
- `pointing_game` — попал ли самый горячий пиксель CAM внутрь бокса
- `energy_in_box` — доля "массы" карты внутри боксов

**Запуск:**
```bash
python yolo_interpret.py --model runs/baseline/weights/best.pt \
    --source path/to/images --out results/interpretation/baseline
# Быстрый режим (без SmoothGrad/IG):
python yolo_interpret.py --model best.pt --source images/ --out results/ --fast
```

**На выходе для каждого изображения (`results/<stem>/`):**
- `01_detections.jpg` — предсказания (+ GT, если есть разметка)
- `02_eigencam.jpg`, `03_gradcam_global.jpg`
- `04_gradcam_det{j}.jpg` — CAM по каждой детекции
- `05_smoothgrad.jpg`, `06_integrated_gradients.jpg`
- `07_faithfulness_masked.jpg` — изображение с замаскированной горячей зоной
- `<stem>__panel.png` — сводная панель всех методов
- `faithfulness.csv` — сводка по всем изображениям

---

### Шаг 5. Сравнение интерпретируемости (`interpret_compare.py`)

Скрипт сравнивает, насколько правильно "смотрят" baseline и augmented модели на **хороших** (чистый тест) и **плохих** (наличие посторонних шумов) изображениях.

**Метрики для каждой модели и набора:**
- `faith_drop_cam` / `faith_drop_random` — падение уверенности при маскировании CAM vs случайной зоны
- `pointing_game_rate` — доля кадров, где пик CAM попал в бокс
- `energy_in_box` — доля энергии CAM внутри боксов
- `detect_rate` / `mean_conf` — доля детекций и средняя уверенность

**Запуск:**
```bash
python interpret_compare.py \
    --baseline runs/baseline/weights/best.pt \
    --aug runs/aug/weights/best.pt \
    --good path/to/clean_test \
    --bad path/to/bad_photos \
    --out results/interpretation \
    --n-good 10 --n-bad 15 --n-panels 6
```

**На выходе:**
- `interp_metrics.csv` — сводные метрики (model × set)
- `compare_interp_metrics.png` — столбчатое сравнение метрик интерпретации
- `<set>/<stem>__compare.png` — Grad-CAM baseline vs aug бок о бок

---

### Шаг 6. Практический инференс на реальных кадрах (`test_bad_photos.ipynb`)

Jupyter-ноутбук для быстрого визуального теста и демонстрации работы модели "в дикой природе" на реальных «плохих» фотографиях (например, сырых кадрах с камер в формате BMP).

**Что делает:**
1. Загружает обученную модель (`best.pt`) с Google Drive.
2. Последовательно обрабатывает папку с исходными BMP-файлами (`conf=0.5`).
3. Если модель нашла несколько объектов, скрипт находит **одну детекцию с максимальной уверенностью** (`np.argmax(confs)`).
4. Рисует на изображении *только* этот лучший bounding box и его скор.
5. Сохраняет результаты в формате JPG.
6. Автоматически архивирует результаты (папки `single` и `exp`) и скачивает `all_results.zip` через интерфейс Colab.

**Запуск:** 
Откройте ноутбук в Google Colab, укажите пути к модели и папке с `bad_photos`, затем последовательно выполните все ячейки.

---

## Ожидаемые результаты

1. **Сравнение архитектур:** `v11n_tuned` обычно показывает лучший баланс точности и скорости; тюнинг гиперпараметров (AdamW + Cosine LR) даёт прирост ~2–5% mAP по сравнению с базовым SGD.
2. **Аугментации:** модель `aug` показывает меньший **drop** mAP на деградированном тесте и более высокую долю детекций на реальных "плохих" кадрах — доказательство робастности.
3. **Интерпретация:** `aug`-модель имеет более высокие `pointing_game_rate` и `energy_in_box`, а метрика `faith_drop_cam ≫ faith_drop_random` подтверждает, что модель действительно "смотрит" на маркер, а не на фон.
4. **Реальный инференс:** Ноутбук `test_bad_photos.ipynb` наглядно демонстрирует, что даже на зашумленных BMP-кадрах модель способна уверенно (`conf > 0.8`) локализовать единственный главный маркер без ложных срабатываний.

---

## Технические заметки

- **Формат разметки:** YOLO (`class_id cx cy w h`, нормализованные координаты).
- **Монки-патчинг Albumentations:** в `train_aug.py` перехватывается `Albumentations.__init__`, что позволяет переключать пайплайны аугментаций без модификации файлов библиотеки.
- **XAI Hooks:** в `yolo_interpret.py` регистрируются `forward` и `backward` hooks на слои `Detect` (P3, P4, P5 шеи), что позволяет корректно считать градиенты для детекторных голов YOLO.
- **Дифференцируемый скор детекции:** `_det_score()` выбирает max вероятности класса среди анкеров с IoU > порога — это позволяет строить CAM для каждой детекции отдельно.
- **Безопасный IO:** `imread_safe` / `imwrite_safe` работают через `np.fromfile` + `cv2.imdecode`, чтобы корректно обрабатывать пути с кириллицей на Windows.
- **Обработка реальных кадров:** в `test_bad_photos.ipynb` используется жесткий отбор `argmax(conf)`, чтобы исключить дубликаты и ложные срабатывания на артефактах сжатия/шума при визуальном анализе.

---

## Лицензия и ссылки

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Albumentations](https://github.com/albumentations-team/albumentations)
- Методы XAI: EigenCAM, Grad-CAM, SmoothGrad, Integrated Gradients, Faithfulness
```