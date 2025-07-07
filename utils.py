# data_utils.py
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from collections import defaultdict
import os
import json
from config import num_labels, alpha
from PIL import Image
import numpy as np  # 追加
import glob


LABEL_ASSIGN_PATH = "label_assignments.json"
DATA_DIR = "./Plant_leave_diseases_dataset_with_augmentation"
PROCESSED_DATA_DIR = "./processed_dataset"


def generate_and_save_dirichlet_partitioned_data(num_clients: int, alpha: float = alpha):
    # Memory-efficient: Check existence more thoroughly
    if os.path.exists(PROCESSED_DATA_DIR):
        try:
            train_dir = os.path.join(PROCESSED_DATA_DIR, "train")
            test_dir = os.path.join(PROCESSED_DATA_DIR, "test")
            if os.path.exists(train_dir) and os.path.exists(test_dir):
                client_dirs = [d for d in os.listdir(train_dir) if d.startswith("client_")]
                if len(client_dirs) >= num_clients:
                    print(f"{PROCESSED_DATA_DIR} 内にクライアントデータが既に存在するため処理をスキップします。")
                    return
        except Exception as e:
            print(f"Error checking existing data: {e}, regenerating...")

    dataset = ImageFolder(root=DATA_DIR)
    total_samples = len(dataset.samples)
    print(f"元のデータセット総数: {total_samples}")
    class_to_idx = dataset.class_to_idx
    num_classes = len(class_to_idx)

    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        label_to_indices[label].append(idx)

    client_indices = defaultdict(list)
    client_labels = defaultdict(set)
    client_indices_per_label = {client_id: defaultdict(list) for client_id in range(num_clients)}

    for label in range(num_classes):
        indices = label_to_indices[label]
        np.random.shuffle(indices)

        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(indices)).astype(int)

        # Fix: Ensure proper distribution of remaining indices
        remaining = len(indices) - proportions.sum()
        if remaining > 0:
            # Distribute remaining indices to clients with largest proportions
            for _ in range(remaining):
                proportions[np.argmax(proportions)] += 1
        elif remaining < 0:
            # Remove excess indices from clients with smallest non-zero proportions
            for _ in range(-remaining):
                non_zero_idx = np.where(proportions > 0)[0]
                if len(non_zero_idx) > 0:
                    proportions[non_zero_idx[np.argmin(proportions[non_zero_idx])]] -= 1

        start = 0
        for client_id, count in enumerate(proportions):
            if count == 0:
                continue
            subset = indices[start:start + count]
            client_indices[client_id].extend(subset)
            client_labels[client_id].add(label)
            client_indices_per_label[client_id][label].extend(subset)
            start += count
    assigned_total = sum(len(indices) for indices in client_indices.values())
    print(f"クライアントへの割り当て総数: {assigned_total}")
    
    # Use configurable image size for memory efficiency
    from config import image_size
    transform = transforms.Resize((image_size, image_size))
    for client_id in range(num_clients):
        for mode in ["train", "test"]:
            save_base = os.path.join(PROCESSED_DATA_DIR, mode, f"client_{client_id}")
            os.makedirs(save_base, exist_ok=True)

        train_indices, test_indices = [], []

        for label, indices in client_indices_per_label[client_id].items():
            np.random.shuffle(indices)
            split = int(0.8 * len(indices))
            train_indices.extend(indices[:split])
            test_indices.extend(indices[split:])

        def save_images(subset, base_dir):
            for idx in subset:
                try:
                    path, label = dataset.samples[idx]
                    # Error handling: Check if image file exists and is valid
                    if not os.path.exists(path):
                        print(f"Warning: Image not found: {path}")
                        continue
                    
                    img = Image.open(path).convert("RGB")
                    img = transform(img)
                    class_dir = os.path.join(base_dir, f"class_{label}")
                    os.makedirs(class_dir, exist_ok=True)
                    filename = os.path.basename(path)
                    img.save(os.path.join(class_dir, filename))
                except Exception as e:
                    print(f"Error processing image {path}: {e}")
                    continue

        save_images(train_indices, os.path.join(PROCESSED_DATA_DIR, "train", f"client_{client_id}"))
        save_images(test_indices, os.path.join(PROCESSED_DATA_DIR, "test", f"client_{client_id}"))

    # 割り当てラベルを保存
    label_assignments = {cid: sorted(list(labels)) for cid, labels in client_labels.items()}
    label_to_clients = defaultdict(list)
    for cid, labels in label_assignments.items():
        for label in labels:
            label_to_clients[label].append(cid)

    with open(LABEL_ASSIGN_PATH, "w") as f:
        json.dump({
            "label_assignments": {str(k): v for k, v in label_assignments.items()},
            "label_to_clients": {str(k): v for k, v in label_to_clients.items()},
            "num_total_labels": num_classes,
        }, f, indent=2)
    print("\n=== フォルダに保存された画像枚数（クライアント別） ===")
    for cid in range(num_clients):
        train_dir = os.path.join(PROCESSED_DATA_DIR, "train", f"client_{cid}")
        test_dir = os.path.join(PROCESSED_DATA_DIR, "test", f"client_{cid}")

        train_count = len(glob.glob(os.path.join(train_dir, "class_*", "*")))
        test_count = len(glob.glob(os.path.join(test_dir, "class_*", "*")))

        total_count = train_count + test_count
        print(f"Client {cid}: Train {train_count}枚, Test {test_count}枚, 合計 {total_count}枚")
    print("=== Dirichlet-based Client Label Assignments ===")
    for cid in range(num_clients):
        print(f"Client {cid}: Labels {sorted(label_assignments[cid])}")


def get_partitioned_data(client_id: int, num_clients: int):
    # Memory-efficient data loading with proper transforms
    from config import image_size
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    
    train_dir = os.path.join(PROCESSED_DATA_DIR, "train", f"client_{client_id}")
    test_dir = os.path.join(PROCESSED_DATA_DIR, "test", f"client_{client_id}")
    
    # Error handling for missing client data
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(f"Client {client_id} data not found. Please run data generation first.")
    
    try:
        train_dataset = ImageFolder(root=train_dir, transform=train_transform)
        test_dataset = ImageFolder(root=test_dir, transform=test_transform)
        
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            raise ValueError(f"Client {client_id} has empty dataset")
            
        return train_dataset, test_dataset
    except Exception as e:
        raise RuntimeError(f"Error loading data for client {client_id}: {e}")

