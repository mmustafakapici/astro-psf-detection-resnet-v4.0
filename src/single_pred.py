import os
import torch
from PIL import Image
import numpy as np
from astropy.io import fits
from utils import load_config, visualize_results
from model import faster_rcnn_resnet50_model

def preprocess_image(image, config):
    """Görüntüyü model girişi için ön işler."""
    # Görüntüyü normalize et ve uint8 tipine dönüştür
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize to [0, 1]
    image = (image * 255).astype(np.uint8)  # Convert to uint8
    
    if config['model']['input_size']:
        target_size = config['model']['input_size']
        image = np.array(Image.fromarray(image).resize((target_size, target_size), Image.BILINEAR))
    
    return image

def load_image(image_path, config):
    if image_path.endswith('.png'):
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
    elif image_path.endswith('.fits'):
        with fits.open(image_path) as hdul:
            image = hdul[0].data.astype(np.float32)
            image = np.stack([image, image, image], axis=-1)  # RGB formatında 3 kanal olarak yükle
    else:
        raise ValueError(f"Unsupported image format: {image_path}")
    
    return preprocess_image(image, config)

def predict_and_visualize(model, image_path, device, save_dir, config):
    image = load_image(image_path, config)
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)

    pred_boxes = outputs[0]['boxes'].cpu().numpy()
    pred_labels = outputs[0]['labels'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()

    # Sonuçları kaydetme ve görselleştirme
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(save_dir, f"{file_name}_predictions.png")
    
    image = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    visualize_results([image], [outputs[0]], save_path)
    print(f"Saved predictions to {save_path}")

def remove_module_prefix(state_dict):
    """DDP ile eğitilmiş modeldeki 'module.' önekini kaldırır."""
    return {key.replace("module.", ""): value for key, value in state_dict.items()}

def main():
    config = load_config()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = faster_rcnn_resnet50_model(config['model']['num_classes'])
    
    # Model state_dict'i yükle ve DDP'den gelen 'module.' önekini kaldır
    state_dict = torch.load(f"{config['training']['checkpoint_dir']}/model_final.pth", map_location=device)
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)

    # Tahmin yapılacak dosya dizini
    input_dir = config['inference']['input_dir']
    save_dir = config['inference']['output_dir']
    os.makedirs(save_dir, exist_ok=True)

    # Dizin içindeki her bir fotoğraf için tahmin yap
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if file_path.endswith(('.png', '.fits')):
            print(f"Processing file: {file_name}")
            predict_and_visualize(model, file_path, device, save_dir, config)
        else:
            print(f"Skipped unsupported file: {file_name}")

if __name__ == "__main__":
    main()
