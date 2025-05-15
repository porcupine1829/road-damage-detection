# scripts/dataset_preparation.py
import os
import shutil
import random
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Class mapping based on label_map.pbtxt
CLASS_MAP = {
    'D00': 0,  # Longitudinal cracks
    'D10': 1,  # Transverse cracks
    'D20': 2,  # Alligator cracks
    'D40': 3   # Potholes
}

def convert_xml_to_yolo(xml_path, img_path):
    """
    Convert Pascal VOC XML annotation to YOLO format with multiple damage classes
    
    Args:
        xml_path: Path to the XML annotation file
        img_path: Path to the corresponding image file
        
    Returns:
        str: YOLO format annotations as a string with each line as: class_id x_center y_center width height
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions
        img = cv2.imread(img_path)
        if img is None:
            img_height, img_width = int(root.find('./size/height').text), int(root.find('./size/width').text)
        else:
            img_height, img_width = img.shape[:2]
        
        result = []
        
        for obj in root.findall('./object'):
            # Get class
            name_element = obj.find('name')
            if name_element is None:
                continue
                
            class_name = name_element.text
            
            # Map class name to class_id using CLASS_MAP
            if class_name in CLASS_MAP:
                class_id = CLASS_MAP[class_name]
            else:
                print(f"Warning: Unknown class {class_name} in {xml_path}, skipping.")
                continue
            
            # Get bounding box coordinates
            bbox = obj.find('bndbox')
            if bbox is None:
                continue
                
            xmin_elem = bbox.find('xmin')
            ymin_elem = bbox.find('ymin')
            xmax_elem = bbox.find('xmax')
            ymax_elem = bbox.find('ymax')
            
            if None in (xmin_elem, ymin_elem, xmax_elem, ymax_elem):
                continue
                
            try:
                xmin = float(xmin_elem.text)
                ymin = float(ymin_elem.text)
                xmax = float(xmax_elem.text)
                ymax = float(ymax_elem.text)
            except (ValueError, TypeError):
                continue
            
            # Validate coordinates
            if xmin >= xmax or ymin >= ymax or xmin < 0 or ymin < 0 or xmax > img_width or ymax > img_height:
                print(f"Warning: Invalid bbox coordinates in {xml_path}, skipping.")
                continue
            
            # Convert to YOLO format (x_center, y_center, width, height) normalized
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # Add to result
            result.append(f"{class_id} {x_center} {y_center} {width} {height}")
        
        return "\n".join(result)
    except Exception as e:
        print(f"Error processing {xml_path}: {str(e)}")
        return ""

def organize_dataset(base_dir, output_dir, test_size=0.2, seed=42):
    """
    Combine data from all countries and split into train and test sets
    
    Args:
        base_dir: Base directory containing the RDD2022 dataset
        output_dir: Output directory for the processed dataset
        test_size: Fraction of images to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        bool: True if successful
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Create directories
    os.makedirs(os.path.join(output_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test", "labels"), exist_ok=True)
    
    # Get all countries
    countries = [d for d in os.listdir(os.path.join(base_dir, "data")) 
                if os.path.isdir(os.path.join(base_dir, "data", d))]
    
    print(f"Found countries: {countries}")
    
    all_image_paths = []
    
    # Collect all images and annotations
    for country in countries:
        image_dir = os.path.join(base_dir, "data", country, "train", "images")
        if os.path.exists(image_dir):
            country_images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) 
                             if img.endswith((".jpg", ".png", ".jpeg"))]
            all_image_paths.extend(country_images)
            print(f"Added {len(country_images)} images from {country}")
    
    print(f"Total images found: {len(all_image_paths)}")
    
    # Split into train and test
    train_images, test_images = train_test_split(all_image_paths, test_size=test_size, random_state=seed)
    
    print(f"Training set: {len(train_images)} images")
    print(f"Test set: {len(test_images)} images")
    
    # Process the split data
    for split_name, image_list in [("train", train_images), ("test", test_images)]:
        for img_path in tqdm(image_list, desc=f"Processing {split_name} set"):
            # Get corresponding XML path
            img_name = os.path.basename(img_path)
            img_base = os.path.splitext(img_name)[0]
            country = img_path.split(os.sep)[-4]
            xml_path = os.path.join(base_dir, "data", country, "train", "annotations", "xmls", f"{img_base}.xml")
            
            if not os.path.exists(xml_path):
                print(f"Warning: Annotation not found for {img_path}, skipping.")
                continue
            
            # Copy image
            dest_img_path = os.path.join(output_dir, split_name, "images", img_name)
            shutil.copy(img_path, dest_img_path)
            
            # Convert XML to YOLO format
            yolo_txt = convert_xml_to_yolo(xml_path, img_path)
            
            # Skip if no valid annotations were found
            if not yolo_txt:
                print(f"Warning: No valid annotations in {xml_path}, skipping.")
                os.remove(dest_img_path)  # Remove the copied image
                continue
            
            # Save YOLO label
            txt_path = os.path.join(output_dir, split_name, "labels", f"{img_base}.txt")
            with open(txt_path, 'w') as f:
                f.write(yolo_txt)
    
    # Create dataset.yaml for YOLOv8
    dataset_yaml = os.path.join(output_dir, "dataset.yaml")
    with open(dataset_yaml, 'w') as f:
        f.write(f"# Road Damage Detection Dataset (RDD2022)\n\n")
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: test/images\n\n")
        f.write(f"# Classes\n")
        f.write(f"names:\n")
        f.write(f"  0: D00  # Longitudinal cracks\n")
        f.write(f"  1: D10  # Transverse cracks\n")
        f.write(f"  2: D20  # Alligator cracks\n")
        f.write(f"  3: D40  # Potholes\n")
    
    print(f"Dataset preparation complete. Output saved to {output_dir}")
    print(f"Dataset configuration saved to {dataset_yaml}")
    
    return True

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Road Damage Detection dataset for YOLOv8")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to RDD2022 dataset base directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed dataset")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data for testing (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    organize_dataset(base_dir=args.input_dir, output_dir=args.output_dir, test_size=args.test_size, seed=args.seed)
