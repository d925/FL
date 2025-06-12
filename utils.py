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


LABEL_ASSIGN_PATH = "label_assignments.json"
LABEL_INDICES_PATH = "label_indices.json"
DATA_DIR = "./Plant_leave_diseases_dataset_with_augmentation"
PROCESSED_DATA_DIR = "./processed_dataset"

def group_labels_by_crop(class_to_idx):
    """
    辞書のキー（クラス名）から「作物名」を抽出し、作物ごとのラベル（正しいインデックス）を集約する。
    修正前は enumerate() により添字を使用していたため、正しいインデックスが得られなかった。
    """
    crop_to_labels = defaultdict(list)
    for class_name, idx in class_to_idx.items():
        crop = class_name.split("___")[0]
        crop_to_labels[crop].append(idx)
    return crop_to_labels

def generate_label_assignments(num_clients: int) -> Tuple[Dict[int, list], Dict[int, list]]:
    """
    各クライアントにラベルを割り当てる。
    IID の場合は全作物（＝すべてのラベル）を割り当て、非IID の場合はランダムに 1〜min(3, len(crop_list)) 個の作物を選択する。
    """
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

    # Non-IID構成：各クライアントに1〜min(3, len(crop_list))作物を割り当て、その作物のラベルを取得
    crop_assignments = {}
    for client_id in range(num_clients):
        if is_iid:
            selected_crops = crop_list[:]  # すべての作物を割り当てる
        else:
            num_to_sample = random.randint(4, min(6, len(crop_list)))
            selected_crops = random.sample(crop_list, num_to_sample)
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

def prepare_label_indices():
    """
    各ラベルのインデックスリストを 80:20 で分割して JSON に保存。
    既にファイルが存在する場合は再計算を行わない。
    """
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

def prepare_processed_data(client_id: int, num_clients: int):
    train_dir = os.path.join(PROCESSED_DATA_DIR, "train", f"client_{client_id}")
    test_dir = os.path.join(PROCESSED_DATA_DIR, "test", f"client_{client_id}")
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print(f"Processed data for client {client_id} already exists, skipping generation.")
        return

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        # ランダム反転は学習時だけなのでここでは入れない
        transforms.ToTensor(),
    ])

    full_dataset = ImageFolder(root=DATA_DIR)
    with open("label_indices.json", "r") as f:
        data = json.load(f)
    train_label_indices = {int(k): v for k, v in data["train"].items()}
    test_label_indices = {int(k): v for k, v in data["test"].items()}
    label_assignments, _ = generate_label_assignments(num_clients)
    assigned_labels = label_assignments[client_id]

    def save_subset(indices, base_dir):
        for idx in indices:
            path, label = full_dataset.samples[idx]
            if label not in assigned_labels:
                continue
            img = Image.open(path).convert("RGB")
            img = transforms.Resize((64, 64))(img)  # transform でToTensorはせずPILのまま
            class_dir = os.path.join(base_dir, f"class_{label}")
            os.makedirs(class_dir, exist_ok=True)
            filename = os.path.basename(path)
            save_path = os.path.join(class_dir, filename)
            img.save(save_path)

    print(f"Preparing processed train data for client {client_id} ...")
    save_subset([idx for label in assigned_labels for idx in train_label_indices.get(label, [])], train_dir)
    print(f"Preparing processed test data for client {client_id} ...")
    save_subset([idx for label in assigned_labels for idx in test_label_indices.get(label, [])], test_dir)


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