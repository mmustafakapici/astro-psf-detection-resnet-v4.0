import numpy as np
import cv2
import random
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import csv

from utils import load_config


def generate_uniform_slope_star_trails(image_size=(512, 512), line_length=100, thickness=2, slope=45, brightness_min=30, brightness_max=255, power_scale=2, num_lines=50, blur_ksize=(5, 5)):
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    annotations = []
    
    for _ in range(num_lines):
        start_x = int(random.randint(0, image_size[0] - line_length))
        start_y = int(random.randint(0, image_size[1] - line_length))
        random_value = random.random()
        brightness = int(brightness_min + (brightness_max - brightness_min) * (random_value ** power_scale))
        
        end_x = int(start_x + line_length)
        end_y = int(start_y + line_length * np.tan(np.radians(slope)))
        
        end_x = min(max(0, end_x), image_size[0])
        end_y = min(max(0, end_y), image_size[1])
        
        cv2.line(image, (start_x, start_y), (end_x, end_y), (brightness, brightness, brightness), thickness)
        
        annotations.append((start_x, start_y, end_x, end_y))
    
    image = cv2.GaussianBlur(image, blur_ksize, 0)
    
    return image, annotations

def save_star_trail_image(index, output_dir, annotation_dir):
    slope = random.randint(10, 80)
    num_lines = random.randint(50, 400)
    power_scale = random.uniform(32, 64)
        
    image, annotations = generate_uniform_slope_star_trails(
        image_size=(2048, 2048), 
        line_length=64, 
        thickness=2, 
        slope=slope, 
        brightness_min=30, 
        brightness_max=255, 
        power_scale=power_scale, 
        num_lines=num_lines, 
        blur_ksize=(9, 9) #ksize.width > 0 && ksize.width % 2 == 1 && ksize.height > 0 && ksize.height % 2 == 1 
    )
    
    filename = f'image_{index+1:04d}.png'
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image)
    
    annotation_file = f'image_{index+1:04d}.csv'
    annotation_path = os.path.join(annotation_dir, annotation_file)
    with open(annotation_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["x1", "y1", "x2", "y2"])
        for annot in annotations:
            writer.writerow(annot)

def generate_multiple_star_trail_images(num_train=1000, num_val=250, num_test=250, output_dir='generated_star_trails'):
    train_dir = os.path.join(output_dir, 'training')
    val_dir = os.path.join(output_dir, 'validation')
    test_dir = os.path.join(output_dir, 'test')
    
    train_annot_dir = os.path.join(output_dir, 'annotations', 'training')
    val_annot_dir = os.path.join(output_dir, 'annotations', 'validation')
    test_annot_dir = os.path.join(output_dir, 'annotations', 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    os.makedirs(train_annot_dir, exist_ok=True)
    os.makedirs(val_annot_dir, exist_ok=True)
    os.makedirs(test_annot_dir, exist_ok=True)
    
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(save_star_trail_image, range(num_train), [train_dir]*num_train, [train_annot_dir]*num_train), total=num_train))
        list(tqdm(executor.map(save_star_trail_image, range(num_val), [val_dir]*num_val, [val_annot_dir]*num_val), total=num_val))
        list(tqdm(executor.map(save_star_trail_image, range(num_test), [test_dir]*num_test, [test_annot_dir]*num_test), total=num_test))

def main():


    config = load_config()
    

    num_train = config['lines']['num_train']
    num_val = config['lines']['num_val']
    num_test = config['lines']['num_test']
    output_dir = config['lines']['output_dir']
    
    generate_multiple_star_trail_images(num_train=num_train, num_val=num_val, num_test=num_test, output_dir=output_dir)

if __name__ == "__main__":
    main()
