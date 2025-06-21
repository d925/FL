from torchvision.models import resnet18
import torch.nn as nn
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader

from utils import get_partitioned_data

def extract_features(client_id, model, device):
    dataset, _ = get_partitioned_data(client_id, num_clients=15)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    features = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            feat = model(x)
            features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0).mean(axis=0)


def cluster_clients(num_clients, num_clusters, feature_extractor=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if feature_extractor is None:
        model = resnet18(pretrained=True)
        model.fc = nn.Identity()  # 最終層を除去して特徴抽出器に
    else:
        model = feature_extractor
    model.to(device)

    client_features = []
    for cid in range(num_clients):
        feat = extract_features(cid, model, device)
        client_features.append(feat)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(client_features)

    return {cid: int(cluster_ids[cid]) for cid in range(num_clients)}
