import os
from utils import load_config

def create_dirs(dir_list):
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def main():
    config = load_config()

    data_dirs = [
        config['data']['raw_data_dir'],
        config['data']['processed_data_dir'],
        config['data']['annotations_dir'],
        config['data']['training_annotations_dir'],
        config['data']['test_annotations_dir'],
        config['data']['validation_annotations_dir'],
        config['data']['annotated_images_dir'],
        config['data']['annotated_training_images_dir'],
        config['data']['annotated_test_images_dir'],
        config['data']['annotated_validation_images_dir'],
        config['data']['training_set_dir'],
        config['data']['validation_set_dir'],
        config['data']['test_set_dir'],
        config['data']['real_set_dir']
    ]

    results_dirs = [
        config['results']['results_dir'],
        config['results']['model_dir'],
        config['results']['log_dir'],
        config['results']['tensorboard_dir'],
        config['results']['wandb_dir'],
        config['results']['outputs_dir'],
        
    ]

    training_dirs = [
        config['training']['checkpoint_dir'],
        
    ]

    lines_dirs=[
        config['lines']['output_dir'],
    ]

    create_dirs(data_dirs)
    create_dirs(results_dirs)
    create_dirs(training_dirs)
    create_dirs(lines_dirs)

if __name__ == "__main__":
    main()
