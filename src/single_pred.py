import os
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.io import fits
from model import faster_rcnn_resnet50_model
from utils import load_config

def load_image(image_path):
    if image_path.endswith('.fits'):
        hdul = fits.open(image_path)
        image_data = hdul[0].data
        hdul.close()
        image = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # RGB formatına dönüştürme
    else:
        image = cv2.imread(image_path)
    return image

def predict_single_image(model, image, device):
    model.eval()
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).permute(0, 3, 1, 2) / 255.0  # [B, C, H, W] formatında normalleştirilmiş tensor
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
    
    return outputs[0]  # Tek bir görüntü olduğu için [0] indeksini kullanıyoruz

def draw_predictions(image, outputs, class_mapping, save_path):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    boxes = outputs['boxes'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score >= 0.5:  # Eşik değer (threshold) 0.5
            xmin, ymin, xmax, ymax = box
            width, height = xmax - xmin, ymax - ymin
            color = 'red' if class_mapping[label] == 'source' else 'blue'

            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f"{class_mapping[label]}: {score:.2f}", color=color, fontsize=12, weight='bold')

    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_images_in_folder(folder_path, save_folder, model, device, class_mapping):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith('.fits') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            save_path = os.path.join(save_folder, filename.replace('.fits', '.png'))
            
            image = load_image(image_path)
            outputs = predict_single_image(model, image, device)
            draw_predictions(image, outputs, class_mapping, save_path)
            print(f"Processed {filename}, saved to {save_path}")

def remove_module_prefix(state_dict):
    """DDP ile eğitilmiş modeldeki 'module.' önekini kaldırır."""
    return {key.replace("module.", ""): value for key, value in state_dict.items()}

def main():
    config = load_config()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Modeli yükleme
    model = faster_rcnn_resnet50_model(config['model']['num_classes'])
    state_dict = torch.load(f"{config['training']['checkpoint_dir']}/model_final.pth", map_location=device)
    state_dict = remove_module_prefix(state_dict)  # Önekleri kaldır
    model.load_state_dict(state_dict)
    model.to(device)

    # Sınıf eşlemesi
    class_mapping = {0: 'source', 1: 'line'}

    # Tahminlerin kaydedileceği klasör
    save_folder = os.path.join(config['results']['outputs_dir'], 'predictions')

    # Klasördeki tüm görüntüleri işleme
    folder_path = config['pred_dir']  # FITS veya PNG dosyalarını içeren klasör yolu
    process_images_in_folder(folder_path, save_folder, model, device, class_mapping)

if __name__ == "__main__":
    main()
