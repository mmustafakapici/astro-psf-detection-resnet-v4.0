from astropy.io import fits
from numpy import resize
import yaml
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import xml.etree.ElementTree as ET
import sep




def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def process_fits(data: np.ndarray, target_size=(512, 512), power: float = 4.0, sigma: float = 1.0 , backsize : int = 64 , backfiltersize : int = 3  ) -> np.ndarray:
    data = np.nan_to_num(data)  # NaN değerlerini 0 ile değiştir

    # %99.5 ölçekleme
    lower_percentile = np.percentile(data, 0.5)
    upper_percentile = np.percentile(data, 99.5)

    data = np.clip(data, lower_percentile, upper_percentile)  # Değerleri %0.5 ve %99.5 arasında kırp
    data = (data - lower_percentile) / (upper_percentile - lower_percentile)  # Bu aralığı normalize et
    
    
    # SEP ile arkaplan temizleme
    bkg = sep.Background(data,  bw=backsize, bh=backsize, fw=backfiltersize, fh=backfiltersize)
    data = data - bkg

    # Power scale uygulaması
    data = np.power(data, power)

    # Min-Max normalizasyonu
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Görüntüyü yeniden boyutlandır
    data = resize(data, target_size)

    return data

def load_fits_images(fits_paths: list) -> np.ndarray:
    images = []
    for path in fits_paths:
        with fits.open(path) as hdul:
            data = hdul[0].data
            images.append(process_fits(data))
    return np.stack(images, axis=0)

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=2):
    """Calculate mean Average Precision (mAP)."""
    average_precisions = []
    for c in range(num_classes):
        detections = [box for box in pred_boxes if box[1] == c]
        ground_truths = [box for box in true_boxes if box[1] == c]

        num_gt = len(ground_truths)
        if num_gt == 0:
            continue

        # Sort detections by confidence
        detections.sort(key=lambda x: x[2], reverse=True)

        TP = np.zeros(len(detections))
        FP = np.zeros(len(detections))

        # Used to determine if a bounding box was covered
        matched_boxes = []

        for d in range(len(detections)):
            detection = detections[d]
            gt_boxes = [box for box in ground_truths if box[0] == detection[0]]

            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_boxes:
                    continue
                iou = calculate_iou(detection[3:], gt[3:])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou > iou_threshold:
                TP[d] = 1
                matched_boxes.append(best_gt_idx)
            else:
                FP[d] = 1

        TP_cumsum = np.cumsum(TP)
        FP_cumsum = np.cumsum(FP)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)
        recalls = TP_cumsum / num_gt

        precisions = np.concatenate(([1], precisions))
        recalls = np.concatenate(([0], recalls))

        average_precisions.append(np.trapz(precisions, recalls))

    if len(average_precisions) == 0:
        return 0.0

    mAP = sum(average_precisions) / len(average_precisions)
    return mAP


def visualize_results(images, outputs, save_dir, threshold=0.5, dpi=300):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, (image, output) in enumerate(zip(images, outputs)):
        fig, ax = plt.subplots(1, figsize=(9, 9), dpi=dpi)  # DPI değeri ayarlandı
        ax.imshow(image.permute(1, 2, 0).cpu().numpy(), cmap='gray')

        boxes = output['boxes'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        scores = output['scores'].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            if score > threshold:
                color = 'r' if label == 1 else 'b'
                ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color=color, linewidth=1))
                ax.text(box[0], box[1] - 2, f'{label}: {score:.3f}', color=color, fontsize=6, bbox=dict(facecolor='white', alpha=0.1))

        ax.axis('off')  # Kenarlıkları kaldır
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Kenarlıkları tamamen kaldırmak için margin ayarı
        plt.savefig(os.path.join(save_dir, f'image_{i}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

def save_results_to_file(results, save_path, format='xaml'):
    if format == 'yaml':
        with open(save_path, 'w') as f:
            yaml.dump(results, f)
    elif format == 'xaml':
        root = ET.Element("Results")
        for result in results:
            entry = ET.SubElement(root, "Entry")
            for key, value in result.items():
                subelement = ET.SubElement(entry, key)
                subelement.text = str(value)
        tree = ET.ElementTree(root)
        tree.write(save_path)

def calculate_and_visualize_metrics(all_pred_boxes, all_true_boxes, save_dir):
    mAP = mean_average_precision(all_pred_boxes, all_true_boxes, iou_threshold=0.5, num_classes=2)
    print(f"mAP: {mAP:.4f}")

    # Prepare labels and scores for precision-recall curve
    labels = []
    scores = []
    selected_class = 1  # PR eğrisi için hangi sınıfın değerlendirileceğini seçin
    
    for pred_box in all_pred_boxes:
        label = pred_box[1]
        score = pred_box[2]
        if label == selected_class:
            labels.append(1)
        else:
            labels.append(0)
        scores.append(score)
    
     
    precisions, recalls, _ = precision_recall_curve(labels, scores)
    ap_score = average_precision_score(labels, scores)

    # Save mAP value
    with open(os.path.join(save_dir, 'mAP.txt'), 'w') as f:
        f.write(f"mAP: {mAP:.4f}\n")

    # Save PR curve
    save_precision_recall_curve(precisions, recalls, ap_score, os.path.join(save_dir, 'pr_curve.png'))
    
def save_precision_recall_curve(precisions, recalls, ap_score, save_path):
    """Save the Precision-Recall curve to a file."""
    plt.figure()
    plt.step(recalls, precisions, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve: AP={ap_score:.4f}')
    plt.savefig(save_path)
    plt.close()

def save_source_info(image_idx, pred_boxes, save_dir):
    source_info = []
    for box in pred_boxes:
        source_info.append({
            'image_index': image_idx,
            'label': box[1],
            'score': box[2],
            'box': box[3:]
        })
    save_results_to_file(source_info, os.path.join(save_dir, f'source_info_image_{image_idx}.xaml'), format='xaml')



