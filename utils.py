import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from collections import defaultdict
import os
import json
from config import num_labels, alpha

# ★ run_clients.py と同じフラグを config に追加しておくこと
from config import backbone_name    # "resnet18" or "small"

from PIL import Image
import numpy as np
import random

LABEL_ASSIGN_PATH = "label_assignments.json"

DATA_DIR = "./Plant_leave_diseases_dataset_with_augmentation"

# バックボーンに応じて processed フォルダを分けると事故防止
if backbone_name == "resnet18":
    PROCESSED_DATA_DIR = "./processed_dataset_0.5alpha_resnet"
else:
    PROCESSED_DATA_DIR = "./processed_dataset_0.5alpha"


def generate_and_save_dirichlet_partitioned_data(
    num_clients: int,
    alpha: float = alpha,
    target_crops=None
):
    # =========================================================
    # 乱数シード固定
    # =========================================================
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # =========================================================
    # 既存チェック
    # =========================================================
    if os.path.exists(PROCESSED_DATA_DIR):
        client_dirs = [d for d in os.listdir(os.path.join(PROCESSED_DATA_DIR, "train")) if d.startswith("client_")]
        if len(client_dirs) >= 1:
            print(f"{PROCESSED_DATA_DIR} 内にクライアントデータが既に存在するためスキップ")
            return

    # =========================================================
    # Load dataset
    # =========================================================
    dataset = ImageFolder(root=DATA_DIR)
    all_classes = dataset.classes

    # =========================================================
    # crop filter
    # =========================================================
    if target_crops is not None:
        target_crops = set(target_crops)
        selected_classes = [cls for cls in all_classes if cls.split("___")[0] in target_crops]

        selected_indices = [
            i for i, (path, label) in enumerate(dataset.samples)
            if dataset.classes[label].split("___")[0] in target_crops
        ]

        old_to_new = {}
        new_classes = sorted(selected_classes)
        for new_idx, cls_name in enumerate(new_classes):
            old_to_new[dataset.class_to_idx[cls_name]] = new_idx

        new_samples = []
        for i in selected_indices:
            path, old_label = dataset.samples[i]
            if old_label in old_to_new:
                new_label = old_to_new[old_label]
                new_samples.append((path, new_label))

        dataset.samples = new_samples
        dataset.classes = new_classes
        dataset.class_to_idx = {cls: i for i, cls in enumerate(new_classes)}
    else:
        selected_classes = all_classes

    num_classes = len(selected_classes)
    total_samples = len(dataset.samples)

    # =========================================================
    # label grouping
    # =========================================================
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        label_to_indices[label].append(idx)

    client_indices = defaultdict(list)
    client_labels = defaultdict(set)
    client_indices_per_label = {client_id: defaultdict(list) for client_id in range(num_clients)}

    # =========================================================
    # Dirichlet
    # =========================================================
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
            subset = indices[start:start+count]
            client_indices[client_id].extend(subset)
            client_labels[client_id].add(label)
            client_indices_per_label[client_id][label].extend(subset)
            start += count

    # =========================================================
    # output resize rule
    # =========================================================
    if backbone_name == "resnet18":
        # ResNet expected input
        resize_transform = transforms.Resize((224, 224))
    else:
        # legacy small CNN
        resize_transform = transforms.Resize((128, 128))

    # =========================================================
    # save imgs
    # =========================================================
    for client_id in range(num_clients):
        for mode in ["train", "test"]:
            save_base = os.path.join(PROCESSED_DATA_DIR, mode, f"client_{client_id}")
            os.makedirs(save_base, exist_ok=True)

        train_indices, test_indices = [], []
        for label, indices in client_indices_per_label[client_id].items():
            np.random.shuffle(indices)
            split = int(0.8*len(indices))
            train_indices.extend(indices[:split])
            test_indices.extend(indices[split:])

        def save_images(subset, base_dir):
            for idx in subset:
                path, label = dataset.samples[idx]
                img = Image.open(path).convert("RGB")
                img = resize_transform(img)
                class_name = dataset.classes[label]
                class_dir = os.path.join(base_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                img.save(os.path.join(class_dir, os.path.basename(path)))

        save_images(train_indices, os.path.join(PROCESSED_DATA_DIR, "train", f"client_{client_id}"))
        save_images(test_indices, os.path.join(PROCESSED_DATA_DIR, "test", f"client_{client_id}"))

    # =========================================================
    # meta info
    # =========================================================
    label_assignments = {cid: sorted(list(labels)) for cid, labels in client_labels.items()}
    label_to_clients = defaultdict(list)
    for cid, labels in label_assignments.items():
        for label in labels:
            label_to_clients[label].append(cid)

    with open(LABEL_ASSIGN_PATH, "w") as f:
        json.dump({
            "target_crops": sorted(list(target_crops)) if target_crops else "ALL",
            "num_total_labels": num_classes,
            "label_assignments": {str(k): v for k, v in label_assignments.items()},
            "label_to_clients": {str(k): v for k, v in label_to_clients.items()},
        }, f, indent=2)


def get_partitioned_data(client_id: int, num_clients: int):
    # ---------------------------------------------------------
    # transforms
    # ---------------------------------------------------------
    if backbone_name == "resnet18":
        # Common data augmentation for pretrained backbone
        train_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])
    else:
        # legacy
        train_transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor()
        ])

    train_dir = os.path.join(PROCESSED_DATA_DIR, "train", f"client_{client_id}")
    test_dir = os.path.join(PROCESSED_DATA_DIR, "test", f"client_{client_id}")

    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = ImageFolder(root=test_dir, transform=test_transform)

    return train_dataset, test_dataset
