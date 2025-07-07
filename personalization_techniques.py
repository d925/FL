# Model Personalization Techniques for Non-IID Federated Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import OrderedDict
import copy

class FedRepPersonalization:
    """
    FedRep: Separates representation and personalization layers
    Memory impact: Low (only stores personalization layers)
    """
    def __init__(self, model: nn.Module, personalization_layers: List[str]):
        self.global_model = model
        self.personalization_layers = personalization_layers
        self.client_personalization = {}
        
    def get_representation_parameters(self) -> Dict[str, torch.Tensor]:
        """Get parameters for representation layers (shared globally)"""
        repr_params = {}
        for name, param in self.global_model.named_parameters():
            if not any(pers_layer in name for pers_layer in self.personalization_layers):
                repr_params[name] = param
        return repr_params
    
    def get_personalization_parameters(self, client_id: int) -> Dict[str, torch.Tensor]:
        """Get parameters for personalization layers (client-specific)"""
        if client_id not in self.client_personalization:
            # Initialize client personalization layers
            self.client_personalization[client_id] = {}
            for name, param in self.global_model.named_parameters():
                if any(pers_layer in name for pers_layer in self.personalization_layers):
                    self.client_personalization[client_id][name] = param.clone()
        
        return self.client_personalization[client_id]
    
    def update_representation(self, new_params: Dict[str, torch.Tensor]):
        """Update global representation parameters"""
        for name, param in self.global_model.named_parameters():
            if name in new_params:
                param.data.copy_(new_params[name])
    
    def update_personalization(self, client_id: int, new_params: Dict[str, torch.Tensor]):
        """Update client personalization parameters"""
        if client_id not in self.client_personalization:
            self.client_personalization[client_id] = {}
        
        for name, param in new_params.items():
            if any(pers_layer in name for pers_layer in self.personalization_layers):
                self.client_personalization[client_id][name] = param.clone()
    
    def get_client_model(self, client_id: int) -> nn.Module:
        """Get personalized model for specific client"""
        client_model = copy.deepcopy(self.global_model)
        
        # Update with client-specific personalization
        if client_id in self.client_personalization:
            for name, param in client_model.named_parameters():
                if name in self.client_personalization[client_id]:
                    param.data.copy_(self.client_personalization[client_id][name])
        
        return client_model

class FedPACPersonalization:
    """
    FedPAC: Personalized Aggregation and Clustering
    Memory impact: Low (stores client similarities and cluster assignments)
    """
    def __init__(self, similarity_threshold: float = 0.5):
        self.similarity_threshold = similarity_threshold
        self.client_similarities = {}
        self.client_clusters = {}
        self.cluster_models = {}
        
    def update_client_similarity(self, client_id: int, other_client_id: int, similarity: float):
        """Update similarity between two clients"""
        if client_id not in self.client_similarities:
            self.client_similarities[client_id] = {}
        self.client_similarities[client_id][other_client_id] = similarity
    
    def get_similar_clients(self, client_id: int) -> List[int]:
        """Get clients similar to the given client"""
        if client_id not in self.client_similarities:
            return []
        
        similar_clients = []
        for other_id, similarity in self.client_similarities[client_id].items():
            if similarity >= self.similarity_threshold:
                similar_clients.append(other_id)
        
        return similar_clients
    
    def personalized_aggregate(self, client_id: int, client_updates: Dict[int, Dict[str, torch.Tensor]], 
                             client_weights: Dict[int, float]) -> Dict[str, torch.Tensor]:
        """
        Aggregate updates from similar clients only
        """
        similar_clients = self.get_similar_clients(client_id)
        
        # Include the client itself
        if client_id not in similar_clients:
            similar_clients.append(client_id)
        
        # Filter updates and weights
        filtered_updates = {cid: client_updates[cid] for cid in similar_clients if cid in client_updates}
        filtered_weights = {cid: client_weights[cid] for cid in similar_clients if cid in client_weights}
        
        if not filtered_updates:
            return client_updates.get(client_id, {})
        
        # Weighted aggregation
        return self._weighted_average(filtered_updates, filtered_weights)
    
    def _weighted_average(self, updates: Dict[int, Dict[str, torch.Tensor]], 
                         weights: Dict[int, float]) -> Dict[str, torch.Tensor]:
        """Compute weighted average of parameter updates"""
        if not updates:
            return {}
        
        total_weight = sum(weights.values())
        if total_weight == 0:
            return list(updates.values())[0]
        
        result = {}
        first_update = list(updates.values())[0]
        
        for key in first_update:
            weighted_sum = sum(weights[cid] * updates[cid][key] for cid in updates.keys())
            result[key] = weighted_sum / total_weight
        
        return result

class MetaLearningPersonalization:
    """
    Meta-learning based personalization (MAML-style)
    Memory impact: Moderate (stores meta-gradients and client adaptations)
    """
    def __init__(self, model: nn.Module, meta_lr: float = 0.01, adaptation_steps: int = 5):
        self.meta_model = model
        self.meta_lr = meta_lr
        self.adaptation_steps = adaptation_steps
        self.client_adaptations = {}
        
    def meta_update(self, client_tasks: List[Tuple[int, torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Perform meta-learning update across multiple client tasks
        
        Args:
            client_tasks: List of (client_id, support_data, query_data) tuples
        """
        meta_gradients = {}
        
        for client_id, support_data, query_data in client_tasks:
            # Clone model for client adaptation
            adapted_model = copy.deepcopy(self.meta_model)
            
            # Inner loop: adapt to client's support set
            support_x, support_y = support_data
            optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.meta_lr)
            
            for _ in range(self.adaptation_steps):
                optimizer.zero_grad()
                pred = adapted_model(support_x)
                loss = F.cross_entropy(pred, support_y)
                loss.backward()
                optimizer.step()
            
            # Outer loop: compute meta-gradients on query set
            query_x, query_y = query_data
            query_pred = adapted_model(query_x)
            query_loss = F.cross_entropy(query_pred, query_y)
            
            # Compute gradients w.r.t. meta-parameters
            meta_grads = torch.autograd.grad(query_loss, self.meta_model.parameters(), 
                                           create_graph=True, retain_graph=True)
            
            # Accumulate meta-gradients
            for i, (name, param) in enumerate(self.meta_model.named_parameters()):
                if name not in meta_gradients:
                    meta_gradients[name] = torch.zeros_like(param)
                meta_gradients[name] += meta_grads[i]
        
        # Average meta-gradients
        num_tasks = len(client_tasks)
        if num_tasks > 0:
            for name in meta_gradients:
                meta_gradients[name] /= num_tasks
        
        return meta_gradients
    
    def adapt_to_client(self, client_id: int, client_data: Tuple[torch.Tensor, torch.Tensor]) -> nn.Module:
        """
        Adapt meta-model to specific client
        """
        adapted_model = copy.deepcopy(self.meta_model)
        
        # Fast adaptation using client data
        client_x, client_y = client_data
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.meta_lr)
        
        for _ in range(self.adaptation_steps):
            optimizer.zero_grad()
            pred = adapted_model(client_x)
            loss = F.cross_entropy(pred, client_y)
            loss.backward()
            optimizer.step()
        
        # Store client adaptation
        self.client_adaptations[client_id] = {
            name: param.clone() for name, param in adapted_model.named_parameters()
        }
        
        return adapted_model

class FeatureAlignmentPersonalization:
    """
    Feature alignment for domain adaptation in non-IID FL
    Memory impact: Low (only stores alignment statistics)
    """
    def __init__(self, feature_dim: int, alignment_weight: float = 0.1):
        self.feature_dim = feature_dim
        self.alignment_weight = alignment_weight
        self.global_feature_stats = None
        self.client_feature_stats = {}
        
    def update_global_stats(self, features: torch.Tensor):
        """Update global feature statistics"""
        batch_mean = torch.mean(features, dim=0)
        batch_var = torch.var(features, dim=0)
        
        if self.global_feature_stats is None:
            self.global_feature_stats = {
                'mean': batch_mean,
                'var': batch_var,
                'count': features.size(0)
            }
        else:
            # Update running statistics
            old_count = self.global_feature_stats['count']
            new_count = old_count + features.size(0)
            
            # Update mean
            self.global_feature_stats['mean'] = (
                old_count * self.global_feature_stats['mean'] + 
                features.size(0) * batch_mean
            ) / new_count
            
            # Update variance
            self.global_feature_stats['var'] = (
                old_count * self.global_feature_stats['var'] + 
                features.size(0) * batch_var
            ) / new_count
            
            self.global_feature_stats['count'] = new_count
    
    def update_client_stats(self, client_id: int, features: torch.Tensor):
        """Update client-specific feature statistics"""
        client_mean = torch.mean(features, dim=0)
        client_var = torch.var(features, dim=0)
        
        self.client_feature_stats[client_id] = {
            'mean': client_mean,
            'var': client_var
        }
    
    def alignment_loss(self, client_id: int, features: torch.Tensor) -> torch.Tensor:
        """
        Compute feature alignment loss between client and global distributions
        """
        if self.global_feature_stats is None or client_id not in self.client_feature_stats:
            return torch.tensor(0.0, device=features.device)
        
        global_mean = self.global_feature_stats['mean'].to(features.device)
        global_var = self.global_feature_stats['var'].to(features.device)
        
        client_mean = self.client_feature_stats[client_id]['mean'].to(features.device)
        client_var = self.client_feature_stats[client_id]['var'].to(features.device)
        
        # MMD-style alignment loss
        mean_diff = torch.norm(client_mean - global_mean)
        var_diff = torch.norm(client_var - global_var)
        
        return self.alignment_weight * (mean_diff + var_diff)

class AdaptivePersonalization:
    """
    Adaptive personalization that combines multiple techniques
    Memory impact: Moderate (combines multiple personalization methods)
    """
    def __init__(self, model: nn.Module, personalization_config: Dict):
        self.model = model
        self.config = personalization_config
        
        # Initialize personalization techniques
        self.techniques = {}
        
        if 'fedrep' in personalization_config:
            self.techniques['fedrep'] = FedRepPersonalization(
                model, personalization_config['fedrep']['layers']
            )
        
        if 'fedpac' in personalization_config:
            self.techniques['fedpac'] = FedPACPersonalization(
                personalization_config['fedpac']['similarity_threshold']
            )
        
        if 'meta_learning' in personalization_config:
            self.techniques['meta_learning'] = MetaLearningPersonalization(
                model, 
                personalization_config['meta_learning']['meta_lr'],
                personalization_config['meta_learning']['adaptation_steps']
            )
        
        if 'feature_alignment' in personalization_config:
            self.techniques['feature_alignment'] = FeatureAlignmentPersonalization(
                personalization_config['feature_alignment']['feature_dim'],
                personalization_config['feature_alignment']['alignment_weight']
            )
        
        # Technique weights
        self.technique_weights = personalization_config.get('weights', {})
        
    def personalize_model(self, client_id: int, client_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> nn.Module:
        """
        Create personalized model for client using selected techniques
        """
        personalized_model = copy.deepcopy(self.model)
        
        # Apply FedRep if enabled
        if 'fedrep' in self.techniques:
            personalized_model = self.techniques['fedrep'].get_client_model(client_id)
        
        # Apply meta-learning adaptation if enabled and data available
        if 'meta_learning' in self.techniques and client_data is not None:
            personalized_model = self.techniques['meta_learning'].adapt_to_client(client_id, client_data)
        
        return personalized_model
    
    def personalized_aggregation(self, client_id: int, client_updates: Dict[int, Dict[str, torch.Tensor]], 
                               client_weights: Dict[int, float]) -> Dict[str, torch.Tensor]:
        """
        Perform personalized aggregation for client
        """
        # Use FedPAC if enabled
        if 'fedpac' in self.techniques:
            return self.techniques['fedpac'].personalized_aggregate(client_id, client_updates, client_weights)
        
        # Default to weighted average
        return self._weighted_average(client_updates, client_weights)
    
    def compute_personalization_loss(self, client_id: int, features: torch.Tensor) -> torch.Tensor:
        """
        Compute additional personalization losses
        """
        total_loss = torch.tensor(0.0, device=features.device)
        
        # Add feature alignment loss if enabled
        if 'feature_alignment' in self.techniques:
            alignment_loss = self.techniques['feature_alignment'].alignment_loss(client_id, features)
            weight = self.technique_weights.get('feature_alignment', 1.0)
            total_loss += weight * alignment_loss
        
        return total_loss
    
    def _weighted_average(self, updates: Dict[int, Dict[str, torch.Tensor]], 
                         weights: Dict[int, float]) -> Dict[str, torch.Tensor]:
        """Compute weighted average of parameter updates"""
        if not updates:
            return {}
        
        total_weight = sum(weights.values())
        if total_weight == 0:
            return list(updates.values())[0]
        
        result = {}
        first_update = list(updates.values())[0]
        
        for key in first_update:
            weighted_sum = sum(weights[cid] * updates[cid][key] for cid in updates.keys())
            result[key] = weighted_sum / total_weight
        
        return result
    
    def get_memory_usage_estimate(self) -> Dict[str, float]:
        """
        Estimate memory usage for personalization techniques
        """
        base_model_size = sum(p.numel() * 4 for p in self.model.parameters()) / (1024**2)  # MB
        
        memory_usage = {
            'base_model': base_model_size,
            'fedrep_overhead': base_model_size * 0.2 if 'fedrep' in self.techniques else 0,
            'fedpac_overhead': 1.0 if 'fedpac' in self.techniques else 0,  # Similarity matrices
            'meta_learning_overhead': base_model_size * 0.5 if 'meta_learning' in self.techniques else 0,
            'feature_alignment_overhead': 0.1 if 'feature_alignment' in self.techniques else 0,
        }
        
        memory_usage['total'] = sum(memory_usage.values())
        
        return memory_usage