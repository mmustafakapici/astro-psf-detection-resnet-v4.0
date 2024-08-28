import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import load_config, calculate_and_visualize_metrics, visualize_results, save_results_to_file, save_source_info, save_precision_recall_curve
from data_preprocessing import AstroDataset, LineDataset, collate_fn
from model import faster_rcnn_resnet50_model
from sklearn.metrics import precision_recall_curve, average_precision_score

def test(model, astro_loader, line_loader, device, save_dir):
    model.eval()
    
    class_mapping = {1: 'source', 2: 'line'}
    
    class_scores = {v: [] for v in class_mapping.keys()}
    class_labels = {v: [] for v in class_mapping.keys()}
    all_pred_boxes = []
    all_true_boxes = []

    for loader_type, data_loader in [('astro', astro_loader),
    
     ('line', line_loader)
     ]:
        for image_idx, (images, targets) in enumerate(tqdm(data_loader, desc=f"Testing {loader_type} data")):
            images = list(image.to(device) for image in images)

            with torch.no_grad():
                outputs = model(images)

            for target, output, image in zip(targets, outputs, images):
                true_boxes = target['boxes'].cpu().numpy()
                true_labels = target['labels'].cpu().numpy()
                pred_boxes = output['boxes'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                scores = output['scores'].cpu().numpy()

                # Class label and score alignment
                if len(true_labels) > 0:
                    for true_label, pred_label, score in zip(true_labels, pred_labels, scores):
                        true_class = int(true_label)
                        pred_class = int(pred_label)

                        # Debugging outputs
                        print(f"True class: {true_class}, Pred class: {pred_class}, Score: {score}")

                        class_labels[true_class].append(1)
                        class_scores[true_class].append(score)

                        for other_class in class_scores.keys():
                            if other_class != true_class:
                                class_labels[other_class].append(0)
                                class_scores[other_class].append(0)

                all_true_boxes.extend(true_boxes)
                all_pred_boxes.extend(pred_boxes)

                # Save source info and visualize results
                images_folder = os.path.join(save_dir, 'images')
                os.makedirs(images_folder, exist_ok=True)
                image_save_dir = os.path.join(images_folder, f'{loader_type}_image_{image_idx+1:04d}')
                os.makedirs(image_save_dir, exist_ok=True)
                image_pred_boxes = [(pred_class, score, *box) for box, pred_class, score in zip(pred_boxes, pred_labels, scores)]

                save_source_info(image_idx, image_pred_boxes, image_save_dir)
                visualize_results([image.cpu()], [output], image_save_dir)
                results_xaml_path = os.path.join(image_save_dir, 'results.xaml')
                save_results_to_file([output], results_xaml_path, format='xaml')

    metrics_save_dir = os.path.join(save_dir, 'metrics')
    os.makedirs(metrics_save_dir, exist_ok=True)
    print(f"Saving metrics to {metrics_save_dir}")
    calculate_and_visualize_metrics(all_pred_boxes, all_true_boxes, metrics_save_dir)
    
    # Precision-Recall curve calculation
    for class_id in class_mapping.keys():
        if len(class_labels[class_id]) != len(class_scores[class_id]):
            print(f"Warning: Length mismatch for class {class_id}. Skipping Precision-Recall curve calculation.")
            continue
        if len(class_labels[class_id]) == 0 or len(class_scores[class_id]) == 0:
            print(f"No valid data for class {class_id}. Skipping Precision-Recall curve calculation.")
            continue
        precisions, recalls, _ = precision_recall_curve(class_labels[class_id], class_scores[class_id])
        ap_score = average_precision_score(class_labels[class_id], class_scores[class_id])
        pr_curve_path = os.path.join(metrics_save_dir, f'precision_recall_curve_class_{class_mapping[class_id]}.png')
        print(f"Saving Precision-Recall curve for class {class_mapping[class_id]} to {pr_curve_path}")
        save_precision_recall_curve(precisions, recalls, ap_score, pr_curve_path)

def remove_module_prefix(state_dict):
    """DDP ile eğitilmiş modeldeki 'module.' önekini kaldırır."""
    return {key.replace("module.", ""): value for key, value in state_dict.items()}

def main():
    config = load_config()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_dataset = AstroDataset(config=config, set_type='test')
    test_loader = DataLoader(test_dataset, batch_size=config['evaluation']['eval_batch_size'], shuffle=False, num_workers=config['model']['num_workers'], collate_fn=collate_fn)
    
    line_test_dataset = LineDataset(config=config, set_type='test')
    line_test_loader = DataLoader(line_test_dataset, batch_size=config['evaluation']['eval_batch_size'], shuffle=False, num_workers=config['model']['num_workers'], collate_fn=collate_fn)

    model = faster_rcnn_resnet50_model(config['model']['num_classes'])
    
    # Model state_dict'i yükle ve DDP'den gelen 'module.' önekini kaldır
    state_dict = torch.load(f"{config['training']['checkpoint_dir']}/model_final.pth", map_location=device)
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    
    model.to(device)

    save_dir = config['results']['outputs_dir']
    test(model, test_loader, line_test_loader, device, save_dir)

if __name__ == "__main__":
    main()
