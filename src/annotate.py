import os
from concurrent.futures import  ThreadPoolExecutor
from tqdm import tqdm
from xml.etree.ElementTree import Element, SubElement, ElementTree
from source_finders import detect_stars, detect_galaxies , detect_lines , detect_sources_sep
from utils import load_config, load_fits_images

def create_annotation(filename, width, height, depth, bboxes, labels, bands, save_path):
    annotation = Element('annotation')
    folder = SubElement(annotation, 'folder')
    folder.text = config['data']['training_set_dir']
    file = SubElement(annotation, 'filename')
    file.text = filename

    size = SubElement(annotation, 'size')
    width_elem = SubElement(size, 'width')
    width_elem.text = str(width)
    height_elem = SubElement(size, 'height')
    height_elem.text = str(height)
    depth_elem = SubElement(size, 'depth')
    depth_elem.text = str(depth)

    for bbox, label, band in zip(bboxes, labels, bands):
        obj = SubElement(annotation, 'object')
        name = SubElement(obj, 'name')
        name.text = label

        band_elem = SubElement(obj, 'band')
        band_elem.text = band

        bndbox = SubElement(obj, 'bndbox')
        xmin = SubElement(bndbox, 'xmin')
        xmin.text = str(bbox[0])
        ymin = SubElement(bndbox, 'ymin')
        ymin.text = str(bbox[1])
        xmax = SubElement(bndbox, 'xmax')
        xmax.text = str(bbox[2])
        ymax = SubElement(bndbox, 'ymax')
        ymax.text = str(bbox[3])

    tree = ElementTree(annotation)
    tree.write(save_path, encoding='utf-8', xml_declaration=True)

def generate_annotations(config, set_type='training'):
    data_dir = config['data'][f'{set_type}_set_dir']
    annotations_dir = config['data'][f'{set_type}_annotations_dir']
    bands = config['data']['bands']

    set_folders = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    for set_folder in tqdm(set_folders, desc=f"Generating {set_type} annotations"):
        set_name = os.path.basename(set_folder)
        fits_paths = [os.path.join(set_folder, f'img_{band}.fits') for band in bands]
        images = load_fits_images(fits_paths)

        all_bboxes = []
        all_labels = []
        all_bands = []

        for image, band in zip(images, bands):
            star_bboxes = detect_stars(image)
            galaxy_bboxes = detect_galaxies(image)
            line_bboxes = detect_lines(image)
            source_bboxes = detect_sources_sep(image)
            """
            all_bboxes.extend(star_bboxes)
            all_labels.extend(['star'] * len(star_bboxes))
            all_bands.extend([band] * len(star_bboxes))

            all_bboxes.extend(galaxy_bboxes)
            all_labels.extend(['galaxy'] * len(galaxy_bboxes))
            all_bands.extend([band] * len(galaxy_bboxes))
            
            all_bboxes.extend(line_bboxes)
            all_labels.extend(['line'] * len(line_bboxes))
            all_bands.extend([band] * len(line_bboxes))
            """
            all_bboxes.extend(source_bboxes)
            all_labels.extend(['source'] * len(source_bboxes))
            all_bands.extend([band] * len(source_bboxes))
           
        filename = f"{set_name}.xml"
        path = os.path.join(annotations_dir, filename)
        create_annotation(set_name, images.shape[2], images.shape[1], images.shape[0], all_bboxes, all_labels, all_bands, path)

if __name__ == "__main__":
    config = load_config()

    # Multi processing
    with ThreadPoolExecutor() as executor:
        executor.submit(generate_annotations, config, 'training')
        executor.submit(generate_annotations, config, 'validation')
        executor.submit(generate_annotations, config, 'test')
