import os
from torch.utils.data import Dataset
import torch
import xml.etree.ElementTree as ET
from utils import load_config, load_fits_images
import csv
import cv2

class AstroDataset(Dataset):
    def __init__(self, config: dict, set_type: str = 'training'):
        self.config = config
        self.set_type = set_type
        self.data_dir = config['data'][f'{set_type}_set_dir']
        self.bands = config['data']['bands']
        self.annotations_dir = config['data'][f'{set_type}_annotations_dir']
        self.set_folders = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))]

        self.imgs = list(sorted(os.listdir(os.path.join(self.data_dir))))
        if set_type in ['training', 'validation', 'test']:
            self.annotations = list(sorted(os.listdir(self.annotations_dir)))
            print(f"Found {len(self.annotations)} annotations for {set_type} set")
        else:
            self.annotations = None
            print(f"No annotations found for {set_type} set")

        if len(self.set_folders) == 0:
            raise ValueError(f"No folders found in {self.data_dir}. Please check the directory structure and set_type parameter.")

    def __getitem__(self, idx):
        if idx >= len(self.set_folders):
            raise IndexError(f"Index {idx} is out of bounds for dataset with length {len(self.set_folders)}")

        set_folder = self.set_folders[idx]
        set_name = os.path.basename(set_folder)
        fits_paths = [os.path.join(set_folder, f'img_{band}.fits') for band in self.bands]
        images = load_fits_images(fits_paths)
        images_tensor = torch.from_numpy(images).float()

        target = {}

        if self.set_type in ['training', 'validation', 'test'] and self.annotations:
            annotation_path = os.path.join(self.annotations_dir, f"{set_name}.xml")
            boxes, labels = self.load_annotation(annotation_path)
            
            # Class mapping: String etiketleri sayısal değerlere dönüştürme
            class_mapping = {'source': 1}
            numeric_labels = [class_mapping[label] for label in labels]
            
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(numeric_labels, dtype=torch.int64)

        if hasattr(self, 'transforms') and self.transforms:
            images_tensor, target = self.transforms(images_tensor, target)

        return images_tensor, target

    def load_annotation(self, annotation_path):
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

    def __len__(self):
        return len(self.set_folders)

class LineDataset(Dataset):
    def __init__(self, config: dict, set_type: str = 'training'):
        self.config = config
        self.set_type = set_type
        self.data_dir = config['lines']['output_dir']
        self.full_data_dir = os.path.join(self.data_dir, set_type)  # Tam yol oluşturma

        if not os.path.exists(self.full_data_dir):
            raise ValueError(f"The directory {self.full_data_dir} does not exist. Please check the directory structure and set_type parameter.")

        # Tüm resimleri ve annotation dosyalarını tek bir dizinde arıyoruz
        self.imgs = list(sorted([f for f in os.listdir(self.full_data_dir) if f.endswith('.png')]))
        self.annotations_dir = os.path.join(self.data_dir, 'annotations', set_type)
        
        if set_type in ['training', 'validation', 'test']:
            self.annotations = list(sorted([f for f in os.listdir(self.annotations_dir) if f.endswith('.csv')]))
            print(f"Found {len(self.annotations)} annotations for {set_type} set")
            print(f"Found {len(self.imgs)} images for {set_type} set")
        else:
            self.annotations = None
            print(f"No annotations found for {set_type} set")

        if len(self.imgs) == 0:
            raise ValueError(f"No images found in {self.full_data_dir}. Please check the directory structure and set_type parameter.")

    def __getitem__(self, idx):
        if idx >= len(self.imgs):
            raise IndexError(f"Index {idx} is out of bounds for dataset with length {len(self.imgs)}")

        img_name = self.imgs[idx]
        img_path = os.path.join(self.full_data_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)

        target = {}

        if self.set_type in ['training', 'validation', 'test'] and self.annotations:
            annotation_name = img_name.replace('.png', '.csv')
            annotation_path = os.path.join(self.annotations_dir, annotation_name)
            boxes, labels = self.load_annotation(annotation_path)
            
            # Class mapping: String etiketleri sayısal değerlere dönüştürme
            class_mapping = {'line': 2}
            numeric_labels = [class_mapping[label] for label in labels]
            
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(numeric_labels, dtype=torch.int64)

        if hasattr(self, 'transforms') and self.transforms:
            image_tensor, target = self.transforms(image_tensor, target)

        return image_tensor, target

    def load_annotation(self, annotation_path):
        bboxes = []
        labels = []

        with open(annotation_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                xmin, ymin, xmax, ymax = map(int, row)
                bboxes.append([xmin, ymin, xmax, ymax])
                labels.append('line')  # String olarak ekleme
        return bboxes, labels

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    return tuple(zip(*batch))
def write_dataset_summary(loader, dataset_name, file_path):
    with open(file_path, 'a') as file:
        file.write(f"{dataset_name} - Toplam Örnek Sayısı: {len(loader.dataset)}\n")

def write_sample_data_to_file(loader, dataset_name, file_path):
    with open(file_path, 'a') as file:
        file.write(f"\n{dataset_name} - İlk 5 veri:\n")
        for i, (images, targets) in enumerate(loader):
            if i >= 5:  # İlk 5 örneği göstermek için
                break
            file.write(f"Örnek {i + 1}:\n")
            file.write(f"Set Name: {dataset_name}\n")
            file.write(f"Image Tensor Shape: {images[0].shape}\n")
            file.write(f"Target Boxes: {targets[0]['boxes']}\n")
            file.write(f"Target Labels: {targets[0]['labels']}\n")
            file.write(f"Boxes dtype: {targets[0]['boxes'].dtype}\n")
            file.write(f"Labels dtype: {targets[0]['labels'].dtype}\n")
            file.write("-" * 30 + "\n")

def main():
    config = load_config()

    train_dataset = AstroDataset(config=config, set_type='training')
    val_dataset = AstroDataset(config=config, set_type='validation')
    test_dataset = AstroDataset(config=config, set_type='test')
    
    line_train_dataset = LineDataset(config=config, set_type='training')
    line_val_dataset = LineDataset(config=config, set_type='validation')
    line_test_dataset = LineDataset(config=config, set_type='test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    line_train_loader = torch.utils.data.DataLoader(line_train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    line_val_loader = torch.utils.data.DataLoader(line_val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)
    line_test_loader = torch.utils.data.DataLoader(line_test_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Bilgilerin kaydedileceği dosya yolu
    file_path = "sample_data_output.txt"

    # Mevcut dosyayı temizle (eğer varsa)
    open(file_path, 'w').close()

    # Her veri seti için özet bilgileri dosyaya yaz
    write_dataset_summary(train_loader, "AstroDataset Training Set", file_path)
    write_dataset_summary(val_loader, "AstroDataset Validation Set", file_path)
    write_dataset_summary(test_loader, "AstroDataset Test Set", file_path)

    write_dataset_summary(line_train_loader, "LineDataset Training Set", file_path)
    write_dataset_summary(line_val_loader, "LineDataset Validation Set", file_path)
    write_dataset_summary(line_test_loader, "LineDataset Test Set", file_path)

    # Her veri seti için örnek bilgileri dosyaya yaz
    write_sample_data_to_file(train_loader, "AstroDataset Training Set", file_path)
    write_sample_data_to_file(val_loader, "AstroDataset Validation Set", file_path)
    write_sample_data_to_file(test_loader, "AstroDataset Test Set", file_path)

    write_sample_data_to_file(line_train_loader, "LineDataset Training Set", file_path)
    write_sample_data_to_file(line_val_loader, "LineDataset Validation Set", file_path)
    write_sample_data_to_file(line_test_loader, "LineDataset Test Set", file_path)

if __name__ == "__main__":
    main()
