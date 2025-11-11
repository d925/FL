import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from collections import defaultdict
import os
import json
from config import num_labels, alpha
from PIL import Image
import numpy as np
import random

LABEL_ASSIGN_PATH = "label_assignments.json"
DATA_DIR = "./Plant_leave_diseases_dataset_with_augmentation"
PROCESSED_DATA_DIR = "./processed_dataset_0.1alpha_sub"


def generate_and_save_dirichlet_partitioned_data(
    num_clients: int,
    alpha: float = alpha,
    target_crops=None
):
    """
    PlantVillage データセットを Dirichlet 分布に基づいてクライアントに非IID分割して保存。

    Args:
        num_clients (int): クライアント数
        alpha (float): Dirichlet 分布のパラメータ（小さいほど非IIDが強い）
        target_crops (list[str] or None): ["Apple", "Corn", "Grape", "Tomato"] など。
                                          None の場合は全クラスを使用。
    """
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
    # 既存データチェック
    # =========================================================
    if os.path.exists(PROCESSED_DATA_DIR):
        client_dirs = [d for d in os.listdir(os.path.join(PROCESSED_DATA_DIR, "train")) if d.startswith("client_")]
        if len(client_dirs) >= 1:
            print(f"{PROCESSED_DATA_DIR} 内にクライアントデータが既に存在するため処理をスキップします。")
            return

    # =========================================================
    # 1️⃣ データロード
    # =========================================================
    dataset = ImageFolder(root=DATA_DIR)
    all_classes = dataset.classes
    print(f"全クラス数: {len(all_classes)}")

    # =========================================================
    # 2️⃣ 作物フィルタ（必要なら）
    # =========================================================
# 2️⃣ 作物フィルタ（必要なら）
    if target_crops is not None:
        target_crops = set(target_crops)
        selected_classes = [cls for cls in all_classes if cls.split("___")[0] in target_crops]
        print(f"🎯 対象作物クラス数: {len(selected_classes)} / {len(all_classes)}")
        print(f"→ {sorted(target_crops)}")

        # 該当クラスのみに限定
        selected_indices = [
            i for i, (path, label) in enumerate(dataset.samples)
            if dataset.classes[label].split("___")[0] in target_crops
        ]

        # ✅ 新しいサンプル構築＋再ラベル付け
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
    print(f"最終的に使用するクラス数: {num_classes}, 画像数: {total_samples}")

    # =========================================================
    # 3️⃣ クラスごとにサンプルを整理
    # =========================================================
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        label_to_indices[label].append(idx)

    client_indices = defaultdict(list)
    client_labels = defaultdict(set)
    client_indices_per_label = {client_id: defaultdict(list) for client_id in range(num_clients)}

    # =========================================================
    # 4️⃣ Dirichlet による非IID分割
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
            subset = indices[start:start + count]
            client_indices[client_id].extend(subset)
            client_labels[client_id].add(label)
            client_indices_per_label[client_id][label].extend(subset)
            start += count

    assigned_total = sum(len(indices) for indices in client_indices.values())
    print(f"クライアントへの割り当て総数: {assigned_total}")

    # =========================================================
    # 5️⃣ 各クライアントごとに train/test 分割して保存
    # =========================================================
    transform = transforms.Resize((128, 128))

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
                class_name = dataset.classes[label]
                class_dir = os.path.join(base_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                img.save(os.path.join(class_dir, os.path.basename(path)))

        save_images(train_indices, os.path.join(PROCESSED_DATA_DIR, "train", f"client_{client_id}"))
        save_images(test_indices, os.path.join(PROCESSED_DATA_DIR, "test", f"client_{client_id}"))

    # =========================================================
    # 6️⃣ メタ情報保存
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

    print("✅ Dirichlet分割完了")
    if target_crops:
        print(f"→ 対象作物: {sorted(target_crops)}")
    else:
        print("→ 全作物を使用しました。")


def get_partitioned_data(client_id: int, num_clients: int):
    # 加工済みデータのフォルダ読み込み
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