import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Union


def generate_full_image_box(image: np.ndarray) -> dict:
    """Генерирует один большой бокс, покрывающий 90% изображения"""
    return {
        'xc': 0.5,          # Центр по x (середина изображения)
        'yc': 0.5,          # Центр по y (середина изображения)
        'w': 0.9,           # Ширина 90% от ширины изображения
        'h': 0.9,           # Высота 90% от высоты изображения
        'label': 0,          # Класс (предполагаем 0 - человек)
        'score': 0.9,       # Высокая уверенность
    }


def model_predict_one_image(image: np.ndarray) -> list:
    """Возвращает один большой бокс для изображения"""
    time.sleep(0.1)  # Имитация времени обработки
    return [generate_full_image_box(image)]


def predict(images: Union[List[np.ndarray], np.ndarray]) -> List[List[dict]]:
    """Функция производит инференс модели на изображении(ях)"""
    results = []
    if isinstance(images, np.ndarray):
        images = [images]
    
    for image in images:
        image_results = model_predict_one_image(image)
        results.append(image_results)
    
    return results


def process_images(images_dir: str, output_csv: str):
    """Обрабатывает все изображения в директории и сохраняет результаты"""
    results = []
    
    for img_path in Path(images_dir).glob('*.*'):
        if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h_img, w_img = img.shape[:2]
                
                start_time = time.time()
                preds = predict([img])[0]
                time_spent = time.time() - start_time
                
                for pred in preds:
                    results.append({
                        'image_id': img_path.stem,
                        'xc': round(pred['xc'], 4),
                        'yc': round(pred['yc'], 4),
                        'w': round(pred['w'], 4),
                        'h': round(pred['h'], 4),
                        'label': pred['label'],
                        'score': round(pred['score'], 4),
                        'time_spent': round(time_spent, 4),
                        'w_img': w_img,
                        'h_img': h_img
                    })
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    
    pd.DataFrame(results).to_csv(output_csv, index=False)


if __name__ == "__main__":
    import argparse
    import cv2
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    args = parser.parse_args()
    
    process_images(args.images_dir, args.output_csv)
