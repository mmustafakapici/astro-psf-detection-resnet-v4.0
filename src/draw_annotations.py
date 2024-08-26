import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from utils import load_config, load_fits_images
from concurrent.futures import ProcessPoolExecutor

def get_annotations(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    bboxes = []
    labels = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(label)
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bboxes.append([xmin, ymin, xmax, ymax])
    return bboxes, labels

def draw_boxes(image, bboxes, labels, class_colors, save_path):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image, cmap='gray')

    for bbox, label in zip(bboxes, labels):
        color = class_colors[label]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=0.5,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)

    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_set(set_type, config, class_colors):
    raw_data_dir = config['data'][f'{set_type}_set_dir']
    annotations_dir = config['data'][f'{set_type}_annotations_dir']
    annotated_images_dir = config['data'][f'annotated_{set_type}_images_dir']
    bands = config['data']['bands']

    if not os.path.exists(annotated_images_dir):
        os.makedirs(annotated_images_dir)

    set_folders = [os.path.join(raw_data_dir, f) for f in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, f))]
    for set_folder in set_folders:
        set_name = os.path.basename(set_folder)
        fits_paths = [os.path.join(set_folder, f'img_{band}.fits') for band in bands]
        images = load_fits_images(fits_paths)

        annotation_path = os.path.join(annotations_dir, f"{set_name}.xml")
        if os.path.exists(annotation_path):
            bboxes, labels = get_annotations(annotation_path)

            for band, image in zip(bands, images):
                save_path = os.path.join(annotated_images_dir, f"{set_name}_annotated_{band}.png")
                draw_boxes(image, bboxes, labels, class_colors, save_path)
                print(f"Saved annotated image for band {band}: {save_path}")
        else:
            print(f"Annotation not found for {set_name}")

if __name__ == "__main__":
    config = load_config()
    class_colors = {'star': 'red', 'galaxy': 'blue' , 'line': 'green' , 'source': 'yellow'}

    set_types = ['training', 'validation', 'test']

    
    # todo padding or  çerçeve çizimi + biçiminde 
    with ProcessPoolExecutor() as executor:
        executor.submit(process_set, set_types[0], config, class_colors)
        executor.submit(process_set, set_types[1], config, class_colors)
        executor.submit(process_set, set_types[2], config, class_colors)
