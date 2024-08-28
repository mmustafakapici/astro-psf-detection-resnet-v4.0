import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from utils import load_config, mean_average_precision
from data_preprocessing import AstroDataset, LineDataset, collate_fn
from model import faster_rcnn_resnet50_model
from concurrent.futures import ThreadPoolExecutor
import yaml
import matplotlib.pyplot as plt
import glob
import numpy as np
import wandb
import os
import shutil

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch}"):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # wandb log the loss (only log on rank 0)
        if torch.distributed.get_rank() == 0:
            wandb.log({"loss": losses.item()}, step=epoch)

def evaluate(model, data_loader, device):
    model.eval()
    all_pred_boxes = []
    all_true_boxes = []
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(images)

        for target, output in zip(targets, outputs):
            true_boxes = target['boxes'].cpu().numpy()
            true_labels = target['labels'].cpu().numpy()
            true_boxes = [(0, label, 1, *box) for box, label in zip(true_boxes, true_labels)]
            all_true_boxes.extend(true_boxes)

            pred_boxes = output['boxes'].cpu().numpy()
            pred_labels = output['labels'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            pred_boxes = [(0, label, score, *box) for box, label, score in zip(pred_boxes, pred_labels, scores)]
            all_pred_boxes.extend(pred_boxes)

    return all_true_boxes, all_pred_boxes

def map_scores(all_true_boxes, all_pred_boxes):
    map30 = mean_average_precision(all_pred_boxes, all_true_boxes, iou_threshold=0.3, num_classes=2)
    map50 = mean_average_precision(all_pred_boxes, all_true_boxes, iou_threshold=0.5, num_classes=2)
    map75 = mean_average_precision(all_pred_boxes, all_true_boxes, iou_threshold=0.75, num_classes=2)
    return map30, map50, map75

class NumpyDumper(yaml.SafeDumper):
    def represent_data(self, data):
        if isinstance(data, np.ndarray):
            return self.represent_list(data.tolist())
        elif isinstance(data, np.generic):
            return self.represent_float(float(data))
        return super().represent_data(data)

def save_map_scores(map30, map50, map75, epoch, checkpoint_dir):
    map_scores = {
        'epoch': epoch,
        'mAP@0.3': map30,
        'mAP@0.5': map50,
        'mAP@0.75': map75
    }

    with open(f"{checkpoint_dir}/map_scores_epoch_{epoch}.yaml", 'w') as f:
        yaml.dump(map_scores, f, Dumper=NumpyDumper)

    with open(f"{checkpoint_dir}/map_scores.txt", 'a') as f:
        f.write(f"Epoch {epoch}:\n")
        f.write(f"mAP@0.3: {map30}\n")
        f.write(f"mAP@0.5: {map50}\n")
        f.write(f"mAP@0.75: {map75}\n")
        f.write("\n")

    # wandb log the mAP scores (only log on rank 0)
    if torch.distributed.get_rank() == 0:
        wandb.log({
            "mAP@0.3": map30,
            "mAP@0.5": map50,
            "mAP@0.75": map75,
        }, step=epoch)

def plot_map_scores(scores_dir, output_path):
    epochs = []
    map_30 = []
    map_50 = []
    map_75 = []

    for yaml_file in sorted(glob.glob(f"{scores_dir}/map_scores_epoch_*.yaml")):
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            epochs.append(data['epoch'])
            map_30.append(data['mAP@0.3'])
            map_50.append(data['mAP@0.5'])
            map_75.append(data['mAP@0.75'])

    plt.plot(epochs, map_30, label='mAP@0.3')
    plt.plot(epochs, map_50, label='mAP@0.5')
    plt.plot(epochs, map_75, label='mAP@0.75')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Mean Average Precision (mAP) Over Epochs')
    plt.legend()
    plt.grid(True)

    plt.xticks(ticks=epochs, labels=epochs)

    output_path = f"{output_path}/map_scores.png"

    plt.savefig(output_path)
    plt.close()

    # wandb log the final mAP scores plot (only log on rank 0)
    if torch.distributed.get_rank() == 0:
        wandb.log({"mAP Scores Plot": wandb.Image(output_path)})


def main():
    # Initialize process group for DDP
    torch.distributed.init_process_group(backend='nccl')
    config = load_config()

    wanb_api_key_path = config['wandb']['api_key_path']
    def set_wandb_api_key():
        with open(wanb_api_key_path, 'r') as f:
            api_key = f.readline().strip()
        os.environ['WANDB_API_KEY'] = api_key

    # wandb API key ayarı (only set on rank 0)
    if torch.distributed.get_rank() == 0:
        set_wandb_api_key()

    device = torch.device('cuda', torch.distributed.get_rank())

    # wandb başlat (only initialize on rank 0)
    if torch.distributed.get_rank() == 0:
        wandb.init(project="astro-object-detection", config=config)

    # Datasetlerin yüklenmesi
    train_dataset = AstroDataset(config=config, set_type='training')
    val_dataset = AstroDataset(config=config, set_type='validation')
    line_train_dataset = LineDataset(config=config, set_type='training')
    line_val_dataset = LineDataset(config=config, set_type='validation')

    # Distributed Sampler'ların eklenmesi
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    line_train_sampler = DistributedSampler(line_train_dataset)
    line_val_sampler = DistributedSampler(line_val_dataset)

    # DataLoader'ların oluşturulması
    train_loader = DataLoader(train_dataset, batch_size=config['model']['batch_size'], sampler=train_sampler, num_workers=config['model']['num_workers'], collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['model']['batch_size'], sampler=val_sampler, num_workers=config['model']['num_workers'], collate_fn=collate_fn)
    line_train_loader = DataLoader(line_train_dataset, batch_size=config['model']['batch_size'], sampler=line_train_sampler, num_workers=config['model']['num_workers'], collate_fn=collate_fn)
    line_val_loader = DataLoader(line_val_dataset, batch_size=config['model']['batch_size'], sampler=line_val_sampler, num_workers=config['model']['num_workers'], collate_fn=collate_fn)

    model = faster_rcnn_resnet50_model(config['model']['num_classes'])
    model.to(device)

    # Modeli DDP ile sarmalama
    model = DDP(model, device_ids=[torch.distributed.get_rank()])

    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=config['model']['learning_rate'], momentum=0.9, weight_decay=0.0005)

    num_epochs = config['model']['num_epochs']

    for epoch in range(num_epochs):
        # Sampler'ları yeniden düzenleme (her epoch başında gerekli)
        train_sampler.set_epoch(epoch)
        line_train_sampler.set_epoch(epoch)

        # Yıldızlar için eğitim
        train_one_epoch(model, optimizer, train_loader, device, epoch)


        
        # Çizgiler için eğitim
        train_one_epoch(model, optimizer, line_train_loader, device, epoch)
        


        # Evaluate and log mAP scores every 5 epochs
        if epoch % config['evaluation']['eval_interval'] == 0:
            # Yıldızlar için değerlendirme
            all_true_boxes, all_pred_boxes = evaluate(model, val_loader, device)
            map_results = map_scores(all_true_boxes, all_pred_boxes)
            if torch.distributed.get_rank() == 0:
                save_map_scores(*map_results, epoch, config['results']['log_dir'])

            
            # Çizgiler için değerlendirme
            all_true_boxes, all_pred_boxes = evaluate(model, line_val_loader, device)
            map_results = map_scores(all_true_boxes, all_pred_boxes)
            if torch.distributed.get_rank() == 0:
                save_map_scores(*map_results, epoch, config['results']['log_dir'])
            

        # Model checkpoint kaydetme (only save on rank 0)
        if epoch % config['training']['save_interval'] == 0 and torch.distributed.get_rank() == 0:
            torch.save(model.state_dict(), f"{config['training']['checkpoint_dir']}/model_epoch_{epoch}.pth")


   
    if torch.distributed.get_rank() == 0:
        torch.save(model.state_dict(), f"{config['training']['checkpoint_dir']}/model_final.pth")

        plot_map_scores(config['results']['log_dir'], config['results']['outputs_dir'])

        wandb.finish()

    # DDP'yi sonlandırma
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()

