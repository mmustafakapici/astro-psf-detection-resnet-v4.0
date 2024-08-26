import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from utils import load_config, mean_average_precision
from data_preprocessing import AstroDataset, collate_fn
from model import faster_rcnn_resnet50_model
from concurrent.futures import ThreadPoolExecutor
import yaml
import matplotlib.pyplot as plt
import glob
import numpy as np
import wandb
import os
import shutil
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_one_epoch(model, optimizer, data_loader, device, epoch, rank):
    model.train()
    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch} (Rank {rank})", disable=rank!=0):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if rank == 0:  # Sadece rank 0'da WandB loglaması yap
            wandb.log({"loss": losses.item()}, step=epoch)

def evaluate(model, data_loader, device, rank):
    model.eval()
    all_pred_boxes = []
    all_true_boxes = []
    for images, targets in tqdm(data_loader, desc=f"Evaluating (Rank {rank})", disable=rank!=0):
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
    map30 = mean_average_precision(all_true_boxes, all_pred_boxes, iou_threshold=0.3, num_classes=2)
    map50 = mean_average_precision(all_true_boxes, all_pred_boxes, iou_threshold=0.5, num_classes=2)
    map75 = mean_average_precision(all_true_boxes, all_pred_boxes, iou_threshold=0.75, num_classes=2)
    return map30, map50, map75

class NumpyDumper(yaml.SafeDumper):
    def represent_data(self, data):
        if isinstance(data, np.ndarray):
            return self.represent_list(data.tolist())
        elif isinstance(data, np.generic):
            return self.represent_float(float(data))
        return super().represent_data(data)

def save_map_scores(map30, map50, map75, epoch, checkpoint_dir, rank):
    if rank == 0:  # Sadece rank 0'da kayıt yap
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

        wandb.log({
            "mAP@0.3": map30,
            "mAP@0.5": map50,
            "mAP@0.75": map75,
        }, step=epoch)

def plot_map_scores(scores_dir, output_path, rank):
    if rank == 0:  # Sadece rank 0'da plot oluştur
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

        wandb.log({"mAP Scores Plot": wandb.Image(output_path)})

def train(rank, world_size, config):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # Sadece rank 0'da WandB başlat
    if rank == 0:
        wandb.init(project="astro-object-detection", config=config)

    train_dataset = AstroDataset(config=config, set_type='training')
    val_dataset = AstroDataset(config=config, set_type='validation')

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=config['model']['batch_size'], shuffle=False, num_workers=config['model']['num_workers'], collate_fn=collate_fn, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=config['model']['batch_size'], shuffle=False, num_workers=config['model']['num_workers'], collate_fn=collate_fn, sampler=val_sampler)

    model = faster_rcnn_resnet50_model(config['model']['num_classes'])
    model.to(device)
    model = DDP(model, device_ids=[rank])

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=config['model']['learning_rate'], momentum=0.9, weight_decay=0.0005)

    num_epochs = config['model']['num_epochs']

    with ThreadPoolExecutor() as executor:
        future_map_scores = None

        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            train_one_epoch(model, optimizer, train_loader, device, epoch, rank)

            if future_map_scores is not None:
                map_results = future_map_scores.result()
                save_map_scores(*map_results, epoch-1, config['results']['log_dir'], rank)

            all_true_boxes, all_pred_boxes = evaluate(model, val_loader, device, rank)
            future_map_scores = executor.submit(map_scores, all_true_boxes, all_pred_boxes)

            if epoch % config['training']['save_interval'] == 0 and rank == 0:
                model_checkpoint_path = f"{config['training']['checkpoint_dir']}/model_epoch_{epoch}.pth"
                torch.save(model.state_dict(), model_checkpoint_path)
                wandb.log({"model_checkpoint_path": model_checkpoint_path})

        if future_map_scores is not None:
            map_results = future_map_scores.result()
            save_map_scores(*map_results, num_epochs-1, config['results']['log_dir'], rank)

    if rank == 0:
        final_model_path = f"{config['training']['checkpoint_dir']}/model_final.pth"
        torch.save(model.state_dict(), final_model_path)
        wandb.log({"final_model_path": final_model_path})
        plot_map_scores(config['results']['log_dir'], config['results']['outputs_dir'], rank)
        wandb.finish()

    cleanup()

def main():
    config = load_config()
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
