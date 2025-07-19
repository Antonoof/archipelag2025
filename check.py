import cv2
import numpy as np
from my_solution.solution import predict
import matplotlib.pyplot as plt

def load_ground_truth(txt_path):
    """Загружает ground truth из текстового файла в формате [class x_center y_center width height]"""
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:  # Проверяем, что строка содержит все необходимые значения
            boxes.append({
                'label': int(parts[0]),
                'xc': float(parts[1]),
                'yc': float(parts[2]),
                'w': float(parts[3]),
                'h': float(parts[4]),
                'score': 1.0  # Для GT уверенность всегда 1.0
            })
    return boxes

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    """Рисует bounding boxes на изображении"""
    h, w = image.shape[:2]
    for box in boxes:
        xc = int(box['xc'] * w)
        yc = int(box['yc'] * h)
        width = int(box['w'] * w)
        height = int(box['h'] * h)
        
        x1 = xc - width // 2
        y1 = yc - height // 2
        x2 = xc + width // 2
        y2 = yc + height // 2
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        label = f"{box['label']}" if 'label' in box else f"{box['score']:.2f}"
        cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def visualize_and_save_results(image_path, gt_txt_path, output_path="comparison_result.png"):
    """
    Основная функция визуализации и сохранения результатов
    Args:
        image_path: путь к исходному изображению
        gt_txt_path: путь к файлу с ground truth
        output_path: путь для сохранения результата
    """
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Загрузка ground truth
    gt_boxes = load_ground_truth(gt_txt_path)
    
    # Получение предсказаний
    predictions = predict([image])[0]
    
    # Создаем копии изображения для визуализации
    pred_image = image.copy()
    gt_image = image.copy()
    
    # Рисуем предсказанные bounding boxes (зеленый)
    draw_boxes(pred_image, predictions, color=(0, 255, 0))
    
    # Рисуем ground truth (синий)
    draw_boxes(gt_image, gt_boxes, color=(255, 0, 0))
    
    # Создаем фигуру для сохранения
    fig = plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(pred_image)
    plt.title("Predictions")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(gt_image)
    plt.title("Ground Truth")
    plt.axis('off')
    
    plt.tight_layout()
    
    # Сохраняем результат
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Результат сохранен в файл: {output_path}")

visualize_and_save_results(
    image_path="01_1_000001.JPG",
    gt_txt_path="01_1_000001.txt",
    output_path="my_comparison.png"
)