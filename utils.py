# data_utils.py
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from collections import defaultdict
import random
import os
import json
from typing import Tuple, Dict
from config import num_labels, is_iid
from PIL import Image
import numpy as np  # 追加

LABEL_ASSIGN_PATH = "label_assignments.json"
LABEL_INDICES_PATH = "label_indices.json"
DATA_DIR = "./Plant_leave_diseases_dataset_with_augmentation"
PROCESSED_DATA_DIR = "./processed_dataset"


def generate_and_save_dirichlet_partitioned_data(num_clients: int, alpha: float = 5.0):
    if os.path.exists(PROCESSED_DATA_DIR):
        client_dirs = [d for d in os.listdir(os.path.join(PROCESSED_DATA_DIR, "train")) if d.startswith("client_")]
        if len(client_dirs) >= 1:
            print(f"{PROCESSED_DATA_DIR} 内にクライアントデータが既に存在するため処理をスキップします。")
            return

    dataset = ImageFolder(root=DATA_DIR)
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

        while proportions.sum() < len(indices):
            proportions[np.argmax(proportions)] += 1

        start = 0
        for client_id, count in enumerate(proportions):
            if count == 0:
                continue
            subset = indices[start:start + count]
            client_indices[client_id].extend(subset)
            client_labels[client_id].add(label)
            client_indices_per_label[client_id][label].extend(subset)
            start += count

    transform = transforms.Resize((64, 64))
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
                path, label = dataset.samples[idx]
                img = Image.open(path).convert("RGB")
                img = transform(img)
                class_dir = os.path.join(base_dir, f"class_{label}")
                os.makedirs(class_dir, exist_ok=True)
                filename = os.path.basename(path)
                img.save(os.path.join(class_dir, filename))

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

    print("=== Dirichlet-based Client Label Assignments ===")
    for cid in range(num_clients):
        print(f"Client {cid}: Labels {sorted(label_assignments[cid])}")


def get_partitioned_data(client_id: int, num_clients: int):
    # 加工済みデータのフォルダ読み込み
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 学習時のみのaugmentation
        transforms.ToTensor(),
    ])
    train_dir = os.path.join(PROCESSED_DATA_DIR, "train", f"client_{client_id}")
    test_dir = os.path.join(PROCESSED_DATA_DIR, "test", f"client_{client_id}")

    train_dataset = ImageFolder(root=train_dir, transform=transform)
    test_dataset = ImageFolder(root=test_dir, transform=transform)

    return train_dataset, test_dataset

