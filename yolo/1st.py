import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Model i ścieżka do obrazu
MODEL_NAME = "yolo11n.pt"
IMAGE_PATH = "/Users/pawelw/miniconda3/envs/nano/lib/python3.9/site-packages/ultralytics/assets/image.jpg"

# Inicjalizacja modelu YOLO
model = YOLO(MODEL_NAME)

# Wczytanie obrazu
image = cv2.imread(IMAGE_PATH)

# Wykonanie detekcji obiektów
results = model(image)


# Funkcja do wyliczenia średniego koloru w obrębie ramki
def get_average_color(image, box):
    x1, y1, x2, y2 = map(int, box)  # Zamiana współrzędnych na liczby całkowite
    region = image[y1:y2, x1:x2]  # Wycinamy region z obrazka
    average_color = cv2.mean(region)[:3]  # Średni kolor (RGB)
    return average_color


# Rysowanie ramek w kolorze odpowiadającym wykrytemu kolorowi
annotated_image = image.copy()

for result in results[0].boxes:  # Iteracja przez wykryte obiekty
    box = result.xyxy[0].numpy()  # Współrzędne ramki
    conf = result.conf[0].item()  # Prawdopodobieństwo detekcji
    label = result.cls[0].item()  # Klasa obiektu
    print(label)
    class_name = "Car" if label == 2 else "Object"  # Zakładamy, że "Car" to klasa 0

    avg_color = get_average_color(image, box)  # Średni kolor w ramce
    color = tuple(map(int, avg_color))  # Konwersja na int dla cv2
    x1, y1, x2, y2 = map(int, box)

    # Rysowanie ramki w kolorze obiektu
    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

    # Rysowanie tekstu w oryginalnym kolorze (np. białym)
    text = f"{class_name}: {conf:.2f}"
    cv2.putText(annotated_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Wyświetlenie wyników
plt.figure(figsize=(20, 20))
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))  # Wyświetlenie z zaznaczonymi obiektami
plt.axis('off')
plt.title("Wyniki detekcji YOLO z ramkami w kolorze obiektu")
plt.show()
