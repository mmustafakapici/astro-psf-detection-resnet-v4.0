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
    all_pred_boxes = []
    all_true_boxes = []
    class_scores = {0: [], 
                    1: [] 
                    }
    class_labels = {0: [],
                    1: [] 
                    }

    for loader_type, data_loader in [('astro', astro_loader), ('line', line_loader)]:
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

                # İkili sınıflandırma için etiketler ve skorları ayarlayın
                for label, score in zip(pred_labels, scores):
                    if label in class_scores:
                        class_labels[label].append(1)
                        class_scores[label].append(score)
                        other_class = 1 - label
                        class_labels[other_class].append(0)
                        class_scores[other_class].append(score)

                all_true_boxes.extend(true_boxes)
                all_pred_boxes.extend(pred_boxes)

                # Save source info and results for each image
                image_save_dir = os.path.join(save_dir, f'{loader_type}_image_{image_idx+1:04d}')
                os.makedirs(image_save_dir, exist_ok=True)
                image_pred_boxes = [(0, label, score, *box) for box, label, score in zip(pred_boxes, pred_labels, scores)]
                
                save_source_info(image_idx, image_pred_boxes, image_save_dir)

                # Save and visualize results
                visualize_results([image.cpu()], [output], image_save_dir)
                
                # Save results in XAML format
                results_xaml_path = os.path.join(image_save_dir, 'results.xaml')
                save_results_to_file([output], results_xaml_path, format='xaml')

    # Calculate and visualize metrics for the entire dataset
    metrics_save_dir = os.path.join(save_dir, 'metrics')
    os.makedirs(metrics_save_dir, exist_ok=True)
    print(f"Saving metrics to {metrics_save_dir}")
    calculate_and_visualize_metrics(all_pred_boxes, all_true_boxes, metrics_save_dir)
    
    # Save Precision-Recall curves for each class
    for class_id in class_scores.keys():
        precisions, recalls, _ = precision_recall_curve(class_labels[class_id], class_scores[class_id])
        ap_score = average_precision_score(class_labels[class_id], class_scores[class_id])
        pr_curve_path = os.path.join(metrics_save_dir, f'precision_recall_curve_class_{class_id}.png')
        print(f"Saving Precision-Recall curve for class {class_id} to {pr_curve_path}")
        save_precision_recall_curve(precisions, recalls, ap_score, pr_curve_path)

def main():
    config = load_config()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_dataset = AstroDataset(config=config, set_type='test')
    test_loader = DataLoader(test_dataset, batch_size=config['evaluation']['eval_batch_size'], shuffle=False, num_workers=config['model']['num_workers'], collate_fn=collate_fn)
    
    line_test_dataset = LineDataset(config=config, set_type='test')
    line_test_loader = DataLoader(line_test_dataset, batch_size=config['evaluation']['eval_batch_size'], shuffle=False, num_workers=config['model']['num_workers'], collate_fn=collate_fn)

    model = faster_rcnn_resnet50_model(config['model']['num_classes'])
    model.load_state_dict(torch.load(f"{config['training']['checkpoint_dir']}/model_final.pth"))
    model.to(device)

    save_dir = config['results']['outputs_dir']
    test(model, test_loader, line_test_loader, device, save_dir)

if __name__ == "__main__":
    main()
