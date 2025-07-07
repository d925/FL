# Advanced Aggregation Strategies for Non-IID Federated Learning
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, OrderedDict
import math

class FedNovaAggregator:
    """
    FedNova: Normalizes and scales local updates to handle non-IID data
    Memory impact: Minimal (only stores momentum terms)
    """
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.v = None  # Momentum buffer
        
    def aggregate(self, client_updates: List[Dict], client_weights: List[float]) -> Dict:
        """
        Aggregate client updates using FedNova normalization
        
        Args:
            client_updates: List of client parameter updates
            client_weights: List of client weights (typically number of samples)
        """
        if not client_updates:
            return {}
            
        # Normalize client updates by their effective learning rates
        normalized_updates = []
        effective_weights = []
        
        for i, (update, weight) in enumerate(zip(client_updates, client_weights)):
            # Calculate effective learning rate (assumes local SGD steps)
            tau_i = len(update)  # Number of local steps approximation
            coeff = tau_i / sum(client_weights)  # Normalization coefficient
            
            normalized_update = {}
            for key, param in update.items():
                normalized_update[key] = param * coeff
            
            normalized_updates.append(normalized_update)
            effective_weights.append(weight * coeff)
        
        # Weighted aggregation
        aggregated = self._weighted_average(normalized_updates, effective_weights)
        
        # Apply momentum
        if self.v is None:
            self.v = {k: torch.zeros_like(v) for k, v in aggregated.items()}
        
        for key in aggregated:
            self.v[key] = self.momentum * self.v[key] + aggregated[key]
            aggregated[key] = self.v[key]
        
        return aggregated
    
    def _weighted_average(self, updates: List[Dict], weights: List[float]) -> Dict:
        """Compute weighted average of parameter updates"""
        if not updates:
            return {}
            
        total_weight = sum(weights)
        if total_weight == 0:
            return updates[0]
            
        result = {}
        for key in updates[0]:
            result[key] = sum(w * update[key] for w, update in zip(weights, updates)) / total_weight
        
        return result

class SCAFFOLDAggregator:
    """
    SCAFFOLD: Uses control variates to handle client drift in non-IID settings
    Memory impact: Moderate (stores control variates per client)
    """
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.server_control = None
        self.client_controls = {}
        
    def aggregate(self, client_updates: List[Dict], client_ids: List[int], 
                 client_weights: List[float]) -> Dict:
        """
        Aggregate using SCAFFOLD with control variates
        
        Args:
            client_updates: List of client parameter updates
            client_ids: List of client IDs
            client_weights: List of client weights
        """
        if not client_updates:
            return {}
            
        # Initialize control variates if needed
        if self.server_control is None:
            self.server_control = {k: torch.zeros_like(v) for k, v in client_updates[0].items()}
            
        # SCAFFOLD aggregation with control variates
        aggregated = self._weighted_average(client_updates, client_weights)
        
        # Update server control variate
        total_weight = sum(client_weights)
        for key in aggregated:
            client_control_sum = sum(self.client_controls.get(cid, {}).get(key, 0) 
                                   for cid in client_ids)
            self.server_control[key] += (client_control_sum / total_weight) / len(client_ids)
            
        return aggregated
    
    def update_client_control(self, client_id: int, old_params: Dict, new_params: Dict, 
                            learning_rate: float, local_steps: int):
        """Update client control variate"""
        if client_id not in self.client_controls:
            self.client_controls[client_id] = {k: torch.zeros_like(v) for k, v in old_params.items()}
            
        for key in old_params:
            # Option I: Simple control update
            param_diff = new_params[key] - old_params[key]
            self.client_controls[client_id][key] = param_diff / (local_steps * learning_rate)
    
    def _weighted_average(self, updates: List[Dict], weights: List[float]) -> Dict:
        """Compute weighted average of parameter updates"""
        if not updates:
            return {}
            
        total_weight = sum(weights)
        if total_weight == 0:
            return updates[0]
            
        result = {}
        for key in updates[0]:
            result[key] = sum(w * update[key] for w, update in zip(weights, updates)) / total_weight
        
        return result

class AdaptiveAggregator:
    """
    Adaptive aggregation that adjusts based on client similarity and performance
    Memory impact: Low (only stores client similarity metrics)
    """
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.client_similarities = {}
        self.client_performances = {}
        
    def aggregate(self, client_updates: List[Dict], client_ids: List[int], 
                 client_weights: List[float], client_accuracies: List[float]) -> Dict:
        """
        Aggregate based on client similarity and performance
        """
        if not client_updates:
            return {}
            
        # Update client performance tracking
        for cid, acc in zip(client_ids, client_accuracies):
            self.client_performances[cid] = acc
            
        # Calculate client similarities
        similarities = self._calculate_similarities(client_updates, client_ids)
        
        # Adjust weights based on similarity and performance
        adjusted_weights = []
        for i, (cid, weight, acc) in enumerate(zip(client_ids, client_weights, client_accuracies)):
            # Performance-based adjustment
            perf_factor = 1.0 + (acc - 0.5) * 0.5  # Boost good performers
            
            # Similarity-based adjustment
            sim_factor = 1.0
            if cid in similarities:
                avg_sim = np.mean(list(similarities[cid].values()))
                sim_factor = 1.0 + (avg_sim - 0.5) * 0.3  # Boost similar clients
            
            adjusted_weights.append(weight * perf_factor * sim_factor)
        
        return self._weighted_average(client_updates, adjusted_weights)
    
    def _calculate_similarities(self, updates: List[Dict], client_ids: List[int]) -> Dict:
        """Calculate cosine similarity between client updates"""
        similarities = defaultdict(dict)
        
        for i, cid_i in enumerate(client_ids):
            for j, cid_j in enumerate(client_ids):
                if i != j:
                    sim = self._cosine_similarity(updates[i], updates[j])
                    similarities[cid_i][cid_j] = sim
                    
        return similarities
    
    def _cosine_similarity(self, update1: Dict, update2: Dict) -> float:
        """Calculate cosine similarity between two parameter updates"""
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0
        
        for key in update1:
            if key in update2:
                u1_flat = update1[key].flatten()
                u2_flat = update2[key].flatten()
                
                dot_product += torch.dot(u1_flat, u2_flat).item()
                norm1 += torch.norm(u1_flat).item() ** 2
                norm2 += torch.norm(u2_flat).item() ** 2
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))
    
    def _weighted_average(self, updates: List[Dict], weights: List[float]) -> Dict:
        """Compute weighted average of parameter updates"""
        if not updates:
            return {}
            
        total_weight = sum(weights)
        if total_weight == 0:
            return updates[0]
            
        result = {}
        for key in updates[0]:
            result[key] = sum(w * update[key] for w, update in zip(weights, updates)) / total_weight
        
        return result

class LayerWiseAggregator:
    """
    Layer-wise adaptive aggregation for different parts of the network
    Memory impact: Low (only stores layer-wise weights)
    """
    def __init__(self, layer_weights: Optional[Dict[str, float]] = None):
        self.layer_weights = layer_weights or {}
        self.layer_importance = {}
        
    def aggregate(self, client_updates: List[Dict], client_weights: List[float], 
                 layer_importances: Optional[Dict[str, float]] = None) -> Dict:
        """
        Aggregate with layer-wise importance weighting
        """
        if not client_updates:
            return {}
            
        # Update layer importance if provided
        if layer_importances:
            self.layer_importance.update(layer_importances)
            
        # Standard weighted average with layer-wise adjustments
        total_weight = sum(client_weights)
        if total_weight == 0:
            return client_updates[0]
            
        result = {}
        for key in client_updates[0]:
            # Get layer importance (default 1.0)
            layer_name = self._get_layer_name(key)
            importance = self.layer_importance.get(layer_name, 1.0)
            
            # Weighted aggregation with layer importance
            weighted_sum = sum(w * importance * update[key] 
                             for w, update in zip(client_weights, client_updates))
            result[key] = weighted_sum / (total_weight * importance)
        
        return result
    
    def _get_layer_name(self, param_name: str) -> str:
        """Extract layer name from parameter name"""
        # Simple heuristic: take first part before '.'
        return param_name.split('.')[0] if '.' in param_name else param_name
    
    def update_layer_importance(self, layer_gradients: Dict[str, torch.Tensor]):
        """Update layer importance based on gradient magnitudes"""
        for layer_name, grad in layer_gradients.items():
            grad_norm = torch.norm(grad).item()
            self.layer_importance[layer_name] = grad_norm / (grad_norm + 1e-8)  # Normalized importance