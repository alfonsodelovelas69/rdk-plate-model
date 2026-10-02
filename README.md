# rdk-plate-model
Модель YOLOv8 nano для детекції автомобільних номерів (ONNX), оптимізована для роботи на CPU в системі RDK S100.  
YOLOv8n license plate detection model in ONNX format for RDK S100 edge AI deployment.




1. Перенесення моделі на плату
Завантажте файл best.onnx у корінь системи вашої плати за шляхом /root/best.onnx (за допомогою SFTP-клієнта, наприклад WinSCP, FileZilla або командного рядка scp).



3. Заміна функції пошуку номера в Python-скрипті
Відкрийте файл /root/video_plate_watch.py і замініть стару функцію find_plate_candidate (яка шукала контури через OpenCV) на новий алгоритм інференсу ONNX:





import onnxruntime as ort
import numpy as np
import cv2

# Завантаження моделі в пам'ять (виконується один раз при старті скрипта)
onnx_session = ort.InferenceSession("/root/best.onnx", providers=['CPUExecutionProvider'])
onnx_input_name = onnx_session.get_inputs()[0].name

def find_plate_candidate(car_crop, conf_threshold=0.30):
    """
    Знаходить номерний знак всередині вирізаного автомобіля за допомогою YOLOv8 ONNX
    """
    if car_crop is None or car_crop.size == 0:
        return None

    h_orig, w_orig = car_crop.shape[:2]

    # 1. Підготовка зображення під стандарт YOLOv8 (640x640, RGB, NCHW, нормалізація)
    img_resized = cv2.resize(car_crop, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_tensor = img_rgb.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
    input_tensor = np.expand_dims(input_tensor, axis=0)   # CHW -> NCHW

    # 2. Виконання нейромережі на CPU
    outputs = onnx_session.run(None, {onnx_input_name: input_tensor})[0]
    predictions = np.squeeze(outputs)  # Масив розмірністю (5, 8400)

    # 3. Аналіз детекцій
    boxes = predictions[:4, :]   # Координати [center_x, center_y, width, height]
    scores = predictions[4, :]  # Впевненість моделі (confidence)

    # Фільтрація за порогом впевненості
    mask = scores > conf_threshold
    if not np.any(mask):
        return None

    valid_boxes = boxes[:, mask]
    valid_scores = scores[mask]

    # Вибір детекції з найвищим показником впевненості
    best_idx = np.argmax(valid_scores)
    cx, cy, w, h = valid_boxes[:, best_idx]

    # Перерахунок координат з розміру 640x640 на оригінальний розмір car_crop
    scale_x = w_orig / 640.0
    scale_y = h_orig / 640.0

    x1 = int((cx - w / 2.0) * scale_x)
    y1 = int((cy - h / 2.0) * scale_y)
    x2 = int((cx + w / 2.0) * scale_x)
    y2 = int((cy + h / 2.0) * scale_y)

    # Обрізка координат по межах кадру
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_orig, x2), min(h_orig, y2)

    if (x2 - x1) < 15 or (y2 - y1) < 10:
        return None

    # Повертаємо точний кроп номера для подальшого розпізнавання FastPlateOCR
    return car_crop[y1:y2, x1:x2]








Google Colab


Скрипт для завантаження та конвертації
Створіть файл download_model.py, вставте в нього цей код, підставивши свій API-ключ, і запустіть:


from roboflow import Roboflow
from ultralytics import YOLO
import os

# 1. Авторизація через ваш API ключ
rf = Roboflow(api_key="ВАШ_API_KEY")

# 2. Підключення до проекту з вашого посилання
# Робочий простір: "ml-sdznj", проект: "yolov8-number-plate-detection"
project = rf.workspace("ml-sdznj").project("yolov8-number-plate-detection")

# Отримуємо першу версію моделі
model_rf = project.version(1).models()[0]

# Завантажуємо ваги (файл збережеться як 'weights.pt' у поточній папці)
print("Завантаження ваг з Roboflow...")
model_rf.download() 

# 3. Конвертація завантаженого .pt у формат .onnx
if os.path.exists("weights.pt"):
    print("Конвертація в ONNX...")
    model = YOLO("weights.pt")
    
    # Експорт у формат ONNX
    # imgsz=640 - стандартний розмір для YOLO, dynamic=False - для стабільності на слабких процесорах
    model.export(format="onnx", imgsz=640, dynamic=False)
    
    print("Готово! Ваш файл weights.onnx створено.")
else:
    print("Помилка: файл weights.pt не знайдено.")









    
