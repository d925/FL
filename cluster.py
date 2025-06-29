import torch
import numpy as np
from torch.nn.utils import parameters_to_vector
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import SpectralClustering
from model import CNN
from config import num_clients, num_clusters
from utils import get_partitioned_data

def extract_layerwise_weights(cid, device, input_size=(3, 128, 128), batches=3):
    """クライアントのローカルモデルをforward＆仮学習後、flattenしたパラメータを取得"""
    model = CNN().to(device)
    dummy_input = torch.randn(1, *input_size).to(device)
    with torch.no_grad():
        model(dummy_input)  # Lazyレイヤ初期化

    trainset, _ = get_partitioned_data(cid, num_clients)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for i, (data, target) in enumerate(trainloader):
        if i >= batches:
            break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    model.eval()
    flat_weights = parameters_to_vector(model.parameters()).detach().cpu().numpy()
    flat_weights /= np.linalg.norm(flat_weights) + 1e-8  # L2正規化
    return flat_weights

def cluster_clients(num_clients, num_clusters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_vectors = []

    print("🔍 クライアントごとに仮学習→重み抽出中...")
    for cid in range(num_clients):
        w = extract_layerwise_weights(cid, device)
        weight_vectors.append(w)

    weight_vectors = np.stack(weight_vectors)
    print(f"📉 PCAで次元圧縮中（{weight_vectors.shape[1]} → 50次元）...")
    pca = PCA(n_components=50, random_state=42)
    reduced_vectors = pca.fit_transform(weight_vectors)

    print("📐 コサイン距離行列を計算中...")
    distance_matrix = cosine_distances(reduced_vectors)

    print("🔗 スペクトルクラスタリング中...")
    spectral = SpectralClustering(
        n_clusters=num_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42
    )
    cluster_ids = spectral.fit_predict(distance_matrix)

    print("✅ クラスタリング完了：クラスタ割当結果")
    for cid, clid in enumerate(cluster_ids):
        print(f"  Client {cid:2d} → Cluster {clid}")

    return {cid: int(cluster_ids[cid]) for cid in range(num_clients)}
