import torch
import numpy as np
from torch.nn.utils import parameters_to_vector
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import SpectralClustering
from model import CNN  # お前のモデルクラス
from config import num_clients, num_clusters
from utils import get_partitioned_data

def extract_layerwise_weights(cid, device, input_size=(3, 64, 64)):
    """クライアントのローカルモデルを一度forwardし、パラメータをflattenして取得"""
    model = CNN().to(device)
    dummy_input = torch.randn(1, *input_size).to(device)

    # モデル初期化（LazyLinear対応）
    with torch.no_grad():
        model(dummy_input)

    # ローカルデータ取得（この時点でtrainせずに構造だけ取る）
    trainset, _ = get_partitioned_data(cid, num_clients)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32)

    # 仮学習（少しだけやる：LayerCFLは“モデルの初期適応”を見る）
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for data, target in trainloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        break  # ほんの1バッチだけ学習させて層を個別化
    model.eval()

    # パラメータを1ベクトルにflatten
    return parameters_to_vector(model.parameters()).detach().cpu().numpy()

def cluster_clients(num_clients, num_clusters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_vectors = []

    print("🔍 各クライアントのローカルモデルから重み抽出中...")
    for cid in range(num_clients):
        w = extract_layerwise_weights(cid, device)
        weight_vectors.append(w)

    print("📐 コサイン距離行列を計算中...")
    distance_matrix = cosine_distances(weight_vectors)

    print("🔗 スペクトルクラスタリングでクライアントをグルーピング中...")
    spectral = SpectralClustering(
        n_clusters=num_clusters,
        affinity='precomputed',
        random_state=42
    )
    cluster_ids = spectral.fit_predict(distance_matrix)

    return {cid: int(cluster_ids[cid]) for cid in range(num_clients)}
