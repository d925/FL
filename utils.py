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
PROCESSED_DATA_DIR = "./processed_dataset_4crops_0.1alpha"


def generate_and_save_dirichlet_partitioned_data(
    num_clients: int,
    alpha: float = alpha,
    target_crops=["Apple", "Corn", "Grape", "Tomato"]
):
    """
    PlantVillage データセットを Dirichlet 分布に基づいてクライアントに非IID分割して保存。
    各クライアントの総サンプル数が大きく偏らないように補正付き。

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
    # データロード
    # =========================================================
    dataset = ImageFolder(root=DATA_DIR)
    all_classes = dataset.classes
    print(f"全クラス数: {len(all_classes)}")

    # =========================================================
    # 作物フィルタ（必要なら）
    # =========================================================
    if target_crops is not None:
        target_crops = set(target_crops)
        selected_classes = [cls for cls in all_classes if cls.split("___")[0] in target_crops]
        print(f"🎯 対象作物クラス数: {len(selected_classes)} / {len(all_classes)}")
        print(f"→ {sorted(target_crops)}")

        selected_indices = [
            i for i, (_, label) in enumerate(dataset.samples)
            if dataset.classes[label].split("___")[0] in target_crops
        ]
        dataset.samples = [dataset.samples[i] for i in selected_indices]
        dataset.classes = selected_classes
        dataset.class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}
    else:
        selected_classes = all_classes  # 全部使う

    num_classes = len(selected_classes)
    total_samples = len(dataset.samples)
    print(f"最終的に使用するクラス数: {num_classes}, 画像数: {total_samples}")

    # =========================================================
    # クラスごとにインデックス整理
    # =========================================================
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        label_to_indices[label].append(idx)

    client_indices = defaultdict(list)
    client_labels = defaultdict(set)
    client_indices_per_label = {cid: defaultdict(list) for cid in range(num_clients)}

    # =========================================================
    # Dirichlet 分割 + 均等化補正
    # =========================================================
    avg_per_client = total_samples // num_clients
    client_sample_counts = np.zeros(num_clients, dtype=int)

    for label in range(num_classes):
        indices = label_to_indices[label]
        np.random.shuffle(indices)

        # 基本の Dirichlet 比率
        proportions = np.random.dirichlet([alpha] * num_clients)

        # サンプル数補正（多く持っているクライアントには割り当て減らす）
        remaining_quota = np.maximum(avg_per_client * 1.2 - client_sample_counts, 1e-6)
        proportions = proportions * remaining_quota
        proportions = proportions / proportions.sum()

        counts = (proportions * len(indices)).astype(int)

        # 総和を調整（誤差分を最大クライアントに配分）
        while counts.sum() < len(indices):
            counts[np.argmax(proportions)] += 1

        # 割り当て実行
        start = 0
        for cid, count in enumerate(counts):
            if count <= 0:
                continue
            subset = indices[start:start + count]
            client_indices[cid].extend(subset)
            client_labels[cid].add(label)
            client_indices_per_label[cid][label].extend(subset)
            client_sample_counts[cid] += count
            start += count

    print(f"クライアントごとのサンプル数（均等化後）:")
    for cid, count in enumerate(client_sample_counts):
        print(f"  Client {cid}: {count} 枚")

    print(f"合計 = {client_sample_counts.sum()} (理論値 {total_samples})")

    # =========================================================
    # クライアントごとに train/test 分割して保存
    # =========================================================
    transform = transforms.Resize((128, 128))

    for cid in range(num_clients):
        for mode in ["train", "test"]:
            save_base = os.path.join(PROCESSED_DATA_DIR, mode, f"client_{cid}")
            os.makedirs(save_base, exist_ok=True)

        train_indices, test_indices = [], []
        for label, indices in client_indices_per_label[cid].items():
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

        save_images(train_indices, os.path.join(PROCESSED_DATA_DIR, "train", f"client_{cid}"))
        save_images(test_indices, os.path.join(PROCESSED_DATA_DIR, "test", f"client_{cid}"))

    # =========================================================
    # メタ情報保存
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
            "client_sample_counts": client_sample_counts.tolist()
        }, f, indent=2)

    print("✅ Dirichlet分割完了（均等化補正あり）")
    print(f"→ 対象作物: {sorted(target_crops) if target_crops else 'ALL'}")
    print(f"→ alpha={alpha}")


def get_partitioned_data(client_id: int, num_clients: int):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dir = os.path.join(PROCESSED_DATA_DIR, "train", f"client_{client_id}")
    test_dir = os.path.join(PROCESSED_DATA_DIR, "test", f"client_{client_id}")

    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = ImageFolder(root=test_dir, transform=test_transform)

    return train_dataset, test_dataset
