# scripts/training.py
import glob
import json
import os
import shutil
import argparse

import torch
from ultralytics import YOLO

def train_model(dataset_yaml, model_path="yolov8x", img_size=640, batch_size=32, epochs=50, output_dir="./outputs"):
    """
    Train a YOLOv8 model on the road damage dataset

    Args:
        dataset_yaml: Path to the dataset yaml file
        model_path: Base model name (e.g., "yolov8x") or path to a .pt file
        epochs: Number of training epochs
        img_size: Image size for training
        batch_size: Batch size for training
        output_dir: Directory to save outputs
        
    Returns:
        str: Path to the trained model file, or None if training fails
    """
    # Set up GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model path for outputs
    weights_dir = os.path.join(output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Initialize model
    model = YOLO(model_path)

    # Save training configuration for future reference
    config = {
        "dataset_yaml": dataset_yaml,
        "model_path": model_path,
        "epochs": epochs,
        "img_size": img_size,
        "batch_size": batch_size,
        "output_dir": output_dir,
        "device": str(device),
    }

    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Prepare training arguments
    train_args = {
        'data': dataset_yaml,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'project': output_dir,
        'name': "road_damage_detector",
        'verbose': True,
        'workers': 2,  # Adjust based on available CPU cores
        'cache': False,  # Set to True for faster training if enough RAM
        'device': str(device)
    }

    # Train the model
    model.train(**train_args)

    # Get the path to the best model
    best_model_path = os.path.join(output_dir, "road_damage_detector", "weights", "best.pt")

    # Save a copy to a more accessible location
    if os.path.exists(best_model_path):
        final_model_path = os.path.join(weights_dir, "road_damage_model.pt")
        shutil.copy(best_model_path, final_model_path)
        print(f"Best model saved to {final_model_path}")

        # Save training checkpoint information
        checkpoint_info = {
            "model_path": final_model_path,
            "training_completed": True,
            "epochs_completed": epochs
        }

        checkpoint_file = os.path.join(output_dir, "training_checkpoint.json")
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_info, f, indent=4)

        return final_model_path
    else:
        print(f"Warning: Best model not found at {best_model_path}")
        # Try to find any model weights that might have been saved
        model_files = glob.glob(os.path.join(output_dir, "road_damage_detector/weights/*.pt"))
        if model_files:
            latest_model = sorted(model_files, key=os.path.getmtime)[-1]
            print(f"Found alternative model file: {latest_model}")
            final_model_path = os.path.join(weights_dir, "road_damage_model.pt")
            shutil.copy(latest_model, final_model_path)

            # Save partial training checkpoint
            checkpoint_info = {
                "model_path": final_model_path,
                "model_source": latest_model,
                "training_completed": False
            }

            checkpoint_file = os.path.join(output_dir, "training_checkpoint.json")
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_info, f, indent=4)

            return final_model_path

        # Save failed training information
        checkpoint_info = {
            "model_path": None,
            "training_completed": False,
            "error": "No model weights found"
        }

        checkpoint_file = os.path.join(output_dir, "training_checkpoint.json")
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_info, f, indent=4)

        return None

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for road damage detection")
    parser.add_argument("--dataset_yaml", type=str, required=True, help="Path to dataset YAML file")
    parser.add_argument("--model_path", type=str, default="yolov8x", help="YOLOv8 model or pretrained weights path")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save outputs")
    parser.add_argument("--img_size", type=int, default=640, help="Training image size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Train the model
    model_path = train_model(
        dataset_yaml=args.dataset_yaml,
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        epochs=args.epochs
    )
    
    if model_path:
        print(f"Model saved to: {model_path}")
