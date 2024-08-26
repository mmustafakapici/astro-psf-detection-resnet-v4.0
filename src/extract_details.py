import os
import yaml
import xml.etree.ElementTree as ET
from collections import defaultdict
from utils import load_config

def count_sources_in_band(annotation_file, bands):
    tree = ET.parse(annotation_file)
    root = tree.getroot()

    source_count = {}

    for obj in root.findall('object'):
        band = obj.find('band').text
        label = obj.find('name').text
        if band in bands:
            if band not in source_count:
                source_count[band] = {}
            if label not in source_count[band]:
                source_count[band][label] = 0
            source_count[band][label] += 1

    return source_count

def process_annotation_files(annotation_dir, bands):
    set_details = {}

    for annotation_file in os.listdir(annotation_dir):
        if annotation_file.endswith('.xml'):
            set_name = os.path.splitext(annotation_file)[0]  # Set adını dosya adından al
            file_path = os.path.join(annotation_dir, annotation_file)
            source_count = count_sources_in_band(file_path, bands)
            set_details[set_name] = source_count

    return set_details

def save_results_to_yaml(results, output_file):
    with open(output_file, 'w') as yamlfile:
        yaml.dump(results, yamlfile, default_flow_style=False)

def main():
    config = load_config()

    sets = {
        'training': config['data']['training_annotations_dir'],
        'validation': config['data']['validation_annotations_dir'],
        'test': config['data']['test_annotations_dir']
    }

    bands = config['data']['bands']
    output_dir = config['results']['outputs_dir']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_results = {}

    for set_name, annotation_dir in sets.items():
        print(f"Processing {set_name} set...")
        set_details = process_annotation_files(annotation_dir, bands)
        all_results[set_name] = set_details

    output_file = os.path.join(output_dir, 'source_counts.yaml')
    save_results_to_yaml(all_results, output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
