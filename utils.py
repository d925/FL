import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from collections import defaultdict
import random
import os
import json
from typing import Tuple, Dict
from config import num_labels,is_iid

LABEL_ASSIGN_PATH = "label_assignments.json"
LABEL_INDICES_PATH = "label_indices.json"
DATA_DIR = "./Plant_leave_diseases_dataset_with_augmentation"


def group_labels_by_crop(class_to_idx):
    # ラベル名から作物名（Crop名）を抽出する
    crop_to_labels = defaultdict(list)
    for i, class_name in enumerate(class_to_idx):
        crop = class_name.split("___")[0]
        crop_to_labels[crop].append(i)
    return crop_to_labels

def generate_label_assignments(num_clients: int) -> Tuple[Dict[int, list], Dict[int, list]]:
    if os.path.exists(LABEL_ASSIGN_PATH):
        with open(LABEL_ASSIGN_PATH, "r") as f:
            data = json.load(f)
        label_assignments = {int(k): v for k, v in data["label_assignments"].items()}
        label_to_clients = {int(k): v for k, v in data["label_to_clients"].items()}
        return label_assignments, label_to_clients
    
    dataset = ImageFolder(root=DATA_DIR)
    class_to_idx = dataset.class_to_idx
    crop_to_labels = group_labels_by_crop(class_to_idx)
    crop_list = list(crop_to_labels.keys())

    label_assignments = defaultdict(list)
    label_to_clients = defaultdict(list)


        # Non-IID構成：各クライアントに1〜3作物を割り当て、その作物のラベルを取得
    crop_assignments = {}
    for client_id in range(num_clients):
        if is_iid:
            selected_crops = crop_list[:]
        else:
            selected_crops = random.sample(crop_list, random.randint(1, 3))
        crop_assignments[client_id] = selected_crops

        for crop in selected_crops:
            for label in crop_to_labels[crop]:
                label_assignments[client_id].append(label)
                label_to_clients[label].append(client_id)
    used_labels = set()
    for labels in label_assignments.values():
        used_labels.update(labels)
    
    num_total_labels = len(used_labels)

        
    with open(LABEL_ASSIGN_PATH, "w") as f:
        json.dump({
            "label_assignments": {str(k): v for k, v in label_assignments.items()},
            "label_to_clients": {str(k): v for k, v in label_to_clients.items()},
            "num_total_labels": num_total_labels,
        }, f, indent=2)

    print("=== Client Label Assignments ===")
    for cid in range(num_clients):
        print(f"Client {cid}: Labels {sorted(label_assignments[cid])}")

    return label_assignments, label_to_clients

# データをラベルごとに80:20で分割して保存
def prepare_label_indices():
    if os.path.exists(LABEL_INDICES_PATH):
        print("Label indices already prepared.")
        return

    dataset = ImageFolder(root=DATA_DIR)

    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[label].append(idx)

    train = {}
    test = {}

    for label, indices in label_to_indices.items():
        random.shuffle(indices)
        split = int(0.8 * len(indices))
        train[label] = indices[:split]
        test[label] = indices[split:]

    with open(LABEL_INDICES_PATH, "w") as f:
        json.dump({
            "train": {str(k): v for k, v in train.items()},
            "test": {str(k): v for k, v in test.items()},
        }, f, indent=2)

    print("Saved label index mapping to JSON.")


# クライアントごとに割り当てられたラベルを元にデータを取り出す
class RemappedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, label_map):
        self.subset = subset
        self.label_map = label_map

    def __getitem__(self, index):
        image, label = self.subset[index]
        remapped_label = self.label_map[label]
        return image, remapped_label

    def __len__(self):
        return len(self.subset)

def get_partitioned_data(client_id: int, num_clients: int):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    full_dataset = ImageFolder(root=DATA_DIR, transform=transform)

    with open(LABEL_INDICES_PATH, "r") as f:
        data = json.load(f)
    train_label_indices = {int(k): v for k, v in data["train"].items()}
    test_label_indices = {int(k): v for k, v in data["test"].items()}

    label_assignments, label_to_clients = generate_label_assignments(num_clients)
    assigned_labels = label_assignments[client_id]  # グローバルラベル

    # グローバルラベル -> ローカル連番ラベル（例: 12, 14, 30 -> 0, 1, 2）
    label_map = {global_label: local_id for local_id, global_label in enumerate(sorted(assigned_labels))}

    def extract_and_remap(label_indices_dict):
        indices = []
        for label in assigned_labels:
            all_indices = label_indices_dict[label]
            client_list = label_to_clients[label]
            client_index = client_list.index(client_id)
            per_client = len(all_indices) // len(client_list)
            start = client_index * per_client
            end = start + per_client
            indices.extend(all_indices[start:end])

        subset = torch.utils.data.Subset(full_dataset, indices)
        remapped_subset = RemappedDataset(subset, label_map)
        return remapped_subset

    client_train = extract_and_remap(train_label_indices)
    client_test = extract_and_remap(test_label_indices)

    return client_train, client_test