import os
import torch
import torchvision.transforms as T
from PIL import Image
from utils import load_config, visualize_results
from model import faster_rcnn_resnet50_model
from matplotlib import pyplot as plt




def load_image(image_path, device):
    """Görüntüyü yükler, tek kanala çevirir ve modele uygun hale getirir."""
    image = Image.open(image_path).convert("L")  # Tek kanala çevir
    
    transform = T.Compose([
        T.ToTensor(),  # Pytorch tensor formatına çevir
    ])
    image = transform(image).to(device)
    return image.unsqueeze(0)  # Batch dimension ekle

def predict_single_image(model, image, device):
    """Verilen görüntü üzerinde tahmin yapar."""
    model.eval()
    with torch.no_grad():
        predictions = model(image)
    return predictions

def save_predictions(image_path, image, output, save_dir):
    """Tahmin sonuçlarını kaydeder."""
    os.makedirs(save_dir, exist_ok=True)
    image_name = os.path.basename(image_path)
    image_save_path = os.path.join(save_dir, f"predicted_{image_name}")

    # Tahminleri kaydet ve görselleştir
    visualize_results(image, output, image_save_path , threshold=0.1)

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
    
    input_dir = config['inference']['input_dir']  # PNG görüntülerin bulunduğu ana dizin
    output_dir = config['inference']['output_dir']  # Tahmin sonuçlarının kaydedileceği ana dizin
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.png'):
                image_path = os.path.join(root, file)
                image = load_image(image_path, device)
                output = predict_single_image(model, image, device)

                # Kaydetmek için aynı alt klasör yapısını koruyarak yeni bir yol oluştur
                relative_path = os.path.relpath(root, input_dir)
                save_dir = os.path.join(output_dir, relative_path)
                save_predictions(image_path, image, output, save_dir)

if __name__ == "__main__":
    main()
