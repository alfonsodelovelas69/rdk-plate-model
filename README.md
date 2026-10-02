# RDK Plate Model
## YOLOv8 Nano License Plate Detection for RDK S100

[🇺🇦 Українська](#українська) | [🇬🇧 English](#english)

---

## English

### Overview
This project provides a **YOLOv8 nano license plate detection model** in ONNX format, optimized for CPU execution on the **RDK S100** edge AI device. The model detects vehicle license plates in real-time with minimal computational requirements, making it ideal for embedded vision applications.

### Quick Start

#### 1️⃣ Deploy the Model to Your RDK S100 Board
1. Download the `best.onnx` file
2. Transfer it to your board's root directory using SFTP (e.g., WinSCP):
   ```
   /root/best.onnx
   ```

#### 2️⃣ Update Your Python Script
Open the file `/root/video_plate_watch.py` and replace the old `find_plate_candidate()` function (which used OpenCV contour detection) with the YOLOv8 ONNX implementation below.

### Implementation

```python
import onnxruntime as ort
import numpy as np
import cv2

# Load model once at startup
onnx_session = ort.InferenceSession("/root/best.onnx", providers=['CPUExecutionProvider'])
onnx_input_name = onnx_session.get_inputs()[0].name

def find_plate_candidate(car_crop, conf_threshold=0.30):
    """
    Detects a license plate within a cropped car image using YOLOv8 ONNX.
    
    Args:
        car_crop: Input image (cropped car region)
        conf_threshold: Confidence threshold for detections (default: 0.30)
    
    Returns:
        Cropped license plate region or None if not detected
    """
    if car_crop is None or car_crop.size == 0:
        return None

    h_orig, w_orig = car_crop.shape[:2]

    # Step 1: Preprocess image for YOLOv8 (640x640, RGB, NCHW format, normalized)
    img_resized = cv2.resize(car_crop, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_tensor = img_rgb.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
    input_tensor = np.expand_dims(input_tensor, axis=0)   # CHW -> NCHW

    # Step 2: Run inference on CPU
    outputs = onnx_session.run(None, {onnx_input_name: input_tensor})[0]
    predictions = np.squeeze(outputs)  # Shape: (5, 8400)

    # Step 3: Parse detections
    boxes = predictions[:4, :]   # Coordinates [center_x, center_y, width, height]
    scores = predictions[4, :]   # Confidence scores

    # Filter by confidence threshold
    mask = scores > conf_threshold
    if not np.any(mask):
        return None

    valid_boxes = boxes[:, mask]
    valid_scores = scores[mask]

    # Select detection with highest confidence
    best_idx = np.argmax(valid_scores)
    cx, cy, w, h = valid_boxes[:, best_idx]

    # Scale coordinates from 640x640 back to original size
    scale_x = w_orig / 640.0
    scale_y = h_orig / 640.0

    x1 = int((cx - w / 2.0) * scale_x)
    y1 = int((cy - h / 2.0) * scale_y)
    x2 = int((cx + w / 2.0) * scale_x)
    y2 = int((cy + h / 2.0) * scale_y)

    # Clip coordinates to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_orig, x2), min(h_orig, y2)

    # Filter out too small detections
    if (x2 - x1) < 15 or (y2 - y1) < 10:
        return None

    # Return cropped license plate for OCR processing
    return car_crop[y1:y2, x1:x2]
```

### Getting the Model (Google Colab)

If you need to download and convert the model yourself:

1. Create a file `download_model.py`
2. Add your Roboflow API key
3. Run the script:

```python
from roboflow import Roboflow
from ultralytics import YOLO
import os

# 1. Authenticate with your API key
rf = Roboflow(api_key="YOUR_API_KEY")

# 2. Connect to the project
# Workspace: "ml-sdznj", Project: "yolov8-number-plate-detection"
project = rf.workspace("ml-sdznj").project("yolov8-number-plate-detection")

# Get the first model version
model_rf = project.version(1).models()[0]

# Download weights
print("Downloading weights from Roboflow...")
model_rf.download() 

# 3. Convert from .pt to .onnx format
if os.path.exists("weights.pt"):
    print("Converting to ONNX...")
    model = YOLO("weights.pt")
    
    # Export as ONNX
    # imgsz=640: Standard YOLO size
    # dynamic=False: For stability on weak CPUs
    model.export(format="onnx", imgsz=640, dynamic=False)
    
    print("Done! File weights.onnx created.")
else:
    print("Error: weights.pt not found.")
```

---

## Українська

### Опис
Цей проект надає **модель детекції автомобільних номерів YOLOv8 nano** у форматі ONNX, оптимізовану для роботи на CPU пристрою **RDK S100**. Модель детектує номерні знаки в режимі реального часу з мінімальними обчислювальними затратами, що робить її ідеальною для вбудованих додатків комп'ютерного зору.

### Швидкий старт

#### 1️⃣ Розгорніть модель на плату RDK S100
1. Завантажте файл `best.onnx`
2. Передайте його у кореневу директорію вашої плати за допомогою SFTP (наприклад, WinSCP):
   ```
   /root/best.onnx
   ```

#### 2️⃣ Оновіть свій Python-скрипт
Відкрийте файл `/root/video_plate_watch.py` і замініть старою функцію `find_plate_candidate()` (яка використовувала контурний пошук OpenCV) на реалізацію YOLOv8 ONNX наведену нижче.

### Реалізація

```python
import onnxruntime as ort
import numpy as np
import cv2

# Завантажуємо модель один раз при старті
onnx_session = ort.InferenceSession("/root/best.onnx", providers=['CPUExecutionProvider'])
onnx_input_name = onnx_session.get_inputs()[0].name

def find_plate_candidate(car_crop, conf_threshold=0.30):
    """
    Детектує номерний знак у вирізаному зображенні автомобіля за допомогою YOLOv8 ONNX.
    
    Параметри:
        car_crop: Вхідне зображення (вирізана область автомобіля)
        conf_threshold: Поріг впевненості для детекцій (за замовчуванням: 0.30)
    
    Повертає:
        Вирізаний номерний знак або None, якщо не детектовано
    """
    if car_crop is None or car_crop.size == 0:
        return None

    h_orig, w_orig = car_crop.shape[:2]

    # Крок 1: Підготовка зображення для YOLOv8 (640x640, RGB, NCHW, нормалізація)
    img_resized = cv2.resize(car_crop, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_tensor = img_rgb.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
    input_tensor = np.expand_dims(input_tensor, axis=0)   # CHW -> NCHW

    # Крок 2: Запуск нейромережі на CPU
    outputs = onnx_session.run(None, {onnx_input_name: input_tensor})[0]
    predictions = np.squeeze(outputs)  # Розмір: (5, 8400)

    # Крок 3: Аналіз детекцій
    boxes = predictions[:4, :]   # Координати [center_x, center_y, width, height]
    scores = predictions[4, :]   # Показники впевненості

    # Фільтрація за порогом впевненості
    mask = scores > conf_threshold
    if not np.any(mask):
        return None

    valid_boxes = boxes[:, mask]
    valid_scores = scores[mask]

    # Вибір детекції з найвищим показником впевненості
    best_idx = np.argmax(valid_scores)
    cx, cy, w, h = valid_boxes[:, best_idx]

    # Масштабування координат з 640x640 на оригінальний розмір
    scale_x = w_orig / 640.0
    scale_y = h_orig / 640.0

    x1 = int((cx - w / 2.0) * scale_x)
    y1 = int((cy - h / 2.0) * scale_y)
    x2 = int((cx + w / 2.0) * scale_x)
    y2 = int((cy + h / 2.0) * scale_y)

    # Обрізка координат в межах зображення
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_orig, x2), min(h_orig, y2)

    # Фільтрація надто малих детекцій
    if (x2 - x1) < 15 or (y2 - y1) < 10:
        return None

    # Повертаємо вирізаний номерний знак для подальшої обробки OCR
    return car_crop[y1:y2, x1:x2]
```

### Отримання моделі (Google Colab)

Якщо вам потрібно завантажити та конвертувати модель самостійно:

1. Створіть файл `download_model.py`
2. Додайте свій API ключ Roboflow
3. Запустіть скрипт:

```python
from roboflow import Roboflow
from ultralytics import YOLO
import os

# 1. Авторизація через ваш API ключ
rf = Roboflow(api_key="ВАШ_API_KEY")

# 2. Підключення до проекту
# Робочий простір: "ml-sdznj", Проект: "yolov8-number-plate-detection"
project = rf.workspace("ml-sdznj").project("yolov8-number-plate-detection")

# Отримуємо першу версію моделі
model_rf = project.version(1).models()[0]

# Завантажуємо ваги
print("Завантаження ваг з Roboflow...")
model_rf.download() 

# 3. Конвертація з .pt в формат .onnx
if os.path.exists("weights.pt"):
    print("Конвертація в ONNX...")
    model = YOLO("weights.pt")
    
    # Експорт у формат ONNX
    # imgsz=640: Стандартний розмір для YOLO
    # dynamic=False: Для стабільності на слабких процесорах
    model.export(format="onnx", imgsz=640, dynamic=False)
    
    print("Готово! Файл weights.onnx створено.")
else:
    print("Помилка: файл weights.pt не знайдено.")
```

---

### 📋 Requirements
- Python 3.8+
- OpenCV (`cv2`)
- NumPy
- ONNX Runtime
- Roboflow (для завантаження моделі)
- Ultralytics YOLO (для конвертації)

### 🚀 Features
✅ Optimized for RDK S100 CPU execution  
✅ ONNX format for cross-platform compatibility  
✅ Real-time license plate detection  
✅ Automatic coordinate scaling  
✅ Confidence-based filtering  
✅ Integration-ready Python implementation  

---

**Created for edge AI license plate detection on RDK S100**
