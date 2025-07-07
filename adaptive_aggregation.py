import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from flwr.common import Parameters, Scalar
import json
import os

class AdaptiveAggregation:
    """
    Adaptive aggregation strategy for non-IID federated learning.
    Adjusts client weights based on performance, staleness, and data quality.
    """
    
    def __init__(self, memory_efficient=True):
        self.memory_efficient = memory_efficient
        self.client_history = {}  # Track client performance history
        self.round_metrics = {}   # Track round-level metrics
        self.alpha_performance = 0.7  # Weight for performance-based aggregation
        self.alpha_staleness = 0.2   # Weight for staleness penalty
        self.alpha_diversity = 0.1   # Weight for diversity bonus
        
    def aggregate_parameters(self, client_results: List[Tuple[Parameters, int, Dict[str, Scalar]]], 
                           round_num: int) -> Tuple[Parameters, Dict[str, Scalar]]:
        """
        Adaptive parameter aggregation with performance-based weighting.
        
        Args:
            client_results: List of (parameters, num_examples, metrics) from clients
            round_num: Current federated learning round
            
        Returns:
            Aggregated parameters and aggregation metrics
        """
        if not client_results:
            raise ValueError("No client results to aggregate")
        
        # Extract data from client results
        parameters_list = [params for params, _, _ in client_results]
        num_examples_list = [num_examples for _, num_examples, _ in client_results]
        metrics_list = [metrics for _, _, metrics in client_results]
        
        # Calculate adaptive weights
        adaptive_weights = self._calculate_adaptive_weights(
            num_examples_list, metrics_list, round_num
        )
        
        # Aggregate parameters using adaptive weights
        aggregated_params = self._weighted_average_parameters(
            parameters_list, adaptive_weights
        )
        
        # Calculate aggregation metrics
        aggregation_metrics = self._calculate_aggregation_metrics(
            adaptive_weights, metrics_list, num_examples_list
        )
        
        # Update history for next round
        self._update_client_history(metrics_list, round_num)
        
        if self.memory_efficient:
            # Clear unnecessary data to save memory
            self._cleanup_memory()
        
        return aggregated_params, aggregation_metrics
    
    def _calculate_adaptive_weights(self, num_examples_list: List[int], 
                                  metrics_list: List[Dict[str, Scalar]], 
                                  round_num: int) -> np.ndarray:
        """Calculate adaptive weights based on multiple factors."""
        num_clients = len(num_examples_list)
        
        # 1. Base weights (proportional to data size)
        base_weights = np.array(num_examples_list, dtype=np.float32)
        base_weights = base_weights / base_weights.sum()
        
        # 2. Performance-based weights (inverse loss)
        performance_weights = np.ones(num_clients, dtype=np.float32)
        for i, metrics in enumerate(metrics_list):
            loss = metrics.get('loss', 1.0)
            # Higher weight for lower loss (better performance)
            performance_weights[i] = 1.0 / (1.0 + float(loss))
        performance_weights = performance_weights / performance_weights.sum()
        
        # 3. Staleness penalty (clients that haven't participated recently get higher weight)
        staleness_weights = self._calculate_staleness_weights(num_clients, round_num)
        
        # 4. Diversity bonus (clients with different data distributions)
        diversity_weights = self._calculate_diversity_weights(metrics_list)
        
        # Combine all weights
        final_weights = (
            self.alpha_performance * performance_weights +
            self.alpha_staleness * staleness_weights +
            self.alpha_diversity * diversity_weights +
            (1 - self.alpha_performance - self.alpha_staleness - self.alpha_diversity) * base_weights
        )
        
        # Ensure weights sum to 1
        final_weights = final_weights / final_weights.sum()
        
        return final_weights
    
    def _calculate_staleness_weights(self, num_clients: int, round_num: int) -> np.ndarray:
        """Calculate staleness-based weights."""
        staleness_weights = np.ones(num_clients, dtype=np.float32)
        
        for i in range(num_clients):
            client_id = f"client_{i}"
            if client_id in self.client_history:
                last_round = self.client_history[client_id].get('last_round', 0)
                staleness = round_num - last_round
                # Higher weight for more stale clients
                staleness_weights[i] = 1.0 + 0.1 * staleness
        
        return staleness_weights / staleness_weights.sum()
    
    def _calculate_diversity_weights(self, metrics_list: List[Dict[str, Scalar]]) -> np.ndarray:
        """Calculate diversity-based weights."""
        num_clients = len(metrics_list)
        diversity_weights = np.ones(num_clients, dtype=np.float32)
        
        # Simple diversity measure based on accuracy variance
        accuracies = [float(metrics.get('accuracy', 0.5)) for metrics in metrics_list]
        mean_acc = np.mean(accuracies)
        
        for i, acc in enumerate(accuracies):
            # Higher weight for clients with different performance (more diverse)
            diversity_weights[i] = 1.0 + abs(acc - mean_acc)
        
        return diversity_weights / diversity_weights.sum()
    
    def _weighted_average_parameters(self, parameters_list: List[Parameters], 
                                   weights: np.ndarray) -> Parameters:
        """Perform weighted average of parameters."""
        # Convert parameters to numpy arrays
        weights_sum = weights.sum()
        
        # Get the first parameter set to initialize the structure
        first_params = parameters_list[0]
        
        # Initialize aggregated parameters
        aggregated_arrays = []
        
        for layer_idx in range(len(first_params)):
            # Initialize with zeros
            layer_shape = first_params[layer_idx].shape
            aggregated_layer = np.zeros(layer_shape, dtype=np.float32)
            
            # Weighted sum
            for client_idx, params in enumerate(parameters_list):
                weight = weights[client_idx]
                aggregated_layer += weight * params[layer_idx]
            
            aggregated_arrays.append(aggregated_layer)
        
        return aggregated_arrays
    
    def _calculate_aggregation_metrics(self, weights: np.ndarray, 
                                     metrics_list: List[Dict[str, Scalar]], 
                                     num_examples_list: List[int]) -> Dict[str, Scalar]:
        """Calculate metrics for the aggregation process."""
        # Weighted average metrics
        total_examples = sum(num_examples_list)
        
        weighted_accuracy = sum(
            weights[i] * float(metrics.get('accuracy', 0)) 
            for i, metrics in enumerate(metrics_list)
        )
        
        weighted_loss = sum(
            weights[i] * float(metrics.get('loss', 1.0)) 
            for i, metrics in enumerate(metrics_list)
        )
        
        # Weight distribution statistics
        weight_entropy = -np.sum(weights * np.log(weights + 1e-8))
        weight_std = np.std(weights)
        
        return {
            'aggregated_accuracy': weighted_accuracy,
            'aggregated_loss': weighted_loss,
            'total_samples': total_examples,
            'num_clients': len(metrics_list),
            'weight_entropy': float(weight_entropy),
            'weight_std': float(weight_std),
            'max_weight': float(np.max(weights)),
            'min_weight': float(np.min(weights))
        }
    
    def _update_client_history(self, metrics_list: List[Dict[str, Scalar]], round_num: int):
        """Update client history for next round."""
        for i, metrics in enumerate(metrics_list):
            client_id = f"client_{i}"
            if client_id not in self.client_history:
                self.client_history[client_id] = {}
            
            self.client_history[client_id].update({
                'last_round': round_num,
                'last_accuracy': float(metrics.get('accuracy', 0)),
                'last_loss': float(metrics.get('loss', 1.0)),
                'participation_count': self.client_history[client_id].get('participation_count', 0) + 1
            })
    
    def _cleanup_memory(self):
        """Clean up memory for memory-efficient operation."""
        # Keep only last 10 rounds of history
        max_history = 10
        if len(self.round_metrics) > max_history:
            oldest_rounds = sorted(self.round_metrics.keys())[:-max_history]
            for round_num in oldest_rounds:
                del self.round_metrics[round_num]
    
    def save_history(self, filepath: str):
        """Save aggregation history to file."""
        history_data = {
            'client_history': self.client_history,
            'round_metrics': self.round_metrics
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def load_history(self, filepath: str):
        """Load aggregation history from file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                history_data = json.load(f)
                self.client_history = history_data.get('client_history', {})
                self.round_metrics = history_data.get('round_metrics', {})