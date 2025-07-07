# Smart Client Selection Strategies for Non-IID Federated Learning
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random
import math

class DiversityBasedSelector:
    """
    Selects clients based on data diversity to ensure representative training
    Memory impact: Low (only stores client statistics)
    """
    def __init__(self, selection_fraction: float = 0.3):
        self.selection_fraction = selection_fraction
        self.client_stats = {}
        self.client_label_distributions = {}
        
    def select_clients(self, available_clients: List[int], 
                      client_performances: Dict[int, float],
                      client_label_counts: Dict[int, Dict[int, int]]) -> List[int]:
        """
        Select diverse clients based on label distribution and performance
        
        Args:
            available_clients: List of available client IDs
            client_performances: Dict of client_id -> accuracy
            client_label_counts: Dict of client_id -> {label: count}
        """
        if not available_clients:
            return []
            
        num_to_select = max(1, int(len(available_clients) * self.selection_fraction))
        
        # Update client statistics
        for client_id in available_clients:
            if client_id in client_label_counts:
                self.client_label_distributions[client_id] = client_label_counts[client_id]
        
        # Calculate diversity scores
        diversity_scores = self._calculate_diversity_scores(available_clients)
        
        # Combine diversity and performance
        combined_scores = {}
        for client_id in available_clients:
            performance = client_performances.get(client_id, 0.5)
            diversity = diversity_scores.get(client_id, 0.5)
            
            # Weighted combination (favor diversity slightly for non-IID)
            combined_scores[client_id] = 0.3 * performance + 0.7 * diversity
        
        # Select top clients
        selected = sorted(combined_scores.keys(), 
                         key=lambda x: combined_scores[x], reverse=True)[:num_to_select]
        
        return selected
    
    def _calculate_diversity_scores(self, clients: List[int]) -> Dict[int, float]:
        """Calculate diversity scores based on label distribution differences"""
        diversity_scores = {}
        
        if len(clients) <= 1:
            return {client: 1.0 for client in clients}
        
        # Calculate pairwise diversity
        for client_id in clients:
            if client_id not in self.client_label_distributions:
                diversity_scores[client_id] = 0.5
                continue
                
            client_dist = self.client_label_distributions[client_id]
            diversities = []
            
            for other_id in clients:
                if other_id != client_id and other_id in self.client_label_distributions:
                    other_dist = self.client_label_distributions[other_id]
                    diversity = self._calculate_distribution_distance(client_dist, other_dist)
                    diversities.append(diversity)
            
            # Average diversity with other clients
            diversity_scores[client_id] = np.mean(diversities) if diversities else 0.5
        
        return diversity_scores
    
    def _calculate_distribution_distance(self, dist1: Dict[int, int], 
                                       dist2: Dict[int, int]) -> float:
        """Calculate Jensen-Shannon divergence between two label distributions"""
        all_labels = set(dist1.keys()) | set(dist2.keys())
        
        if not all_labels:
            return 0.0
        
        # Normalize distributions
        total1 = sum(dist1.values())
        total2 = sum(dist2.values())
        
        if total1 == 0 or total2 == 0:
            return 0.0
        
        p1 = np.array([dist1.get(label, 0) / total1 for label in all_labels])
        p2 = np.array([dist2.get(label, 0) / total2 for label in all_labels])
        
        # Jensen-Shannon divergence
        m = 0.5 * (p1 + p2)
        
        # Avoid log(0)
        p1 = np.maximum(p1, 1e-10)
        p2 = np.maximum(p2, 1e-10)
        m = np.maximum(m, 1e-10)
        
        kl1 = np.sum(p1 * np.log(p1 / m))
        kl2 = np.sum(p2 * np.log(p2 / m))
        
        js_div = 0.5 * (kl1 + kl2)
        return js_div

class PowerOfChoiceSelector:
    """
    Power-of-choice client selection for better convergence in non-IID settings
    Memory impact: Minimal (only stores recent client metrics)
    """
    def __init__(self, selection_fraction: float = 0.3, choice_factor: int = 2):
        self.selection_fraction = selection_fraction
        self.choice_factor = choice_factor  # Sample choice_factor * target_num clients
        self.client_losses = {}
        self.client_staleness = {}
        
    def select_clients(self, available_clients: List[int], 
                      client_losses: Dict[int, float],
                      round_number: int) -> List[int]:
        """
        Select clients using power-of-choice with loss and staleness
        """
        if not available_clients:
            return []
            
        num_to_select = max(1, int(len(available_clients) * self.selection_fraction))
        
        # Update client information
        self.client_losses.update(client_losses)
        
        # Update staleness (rounds since last selection)
        for client_id in available_clients:
            if client_id not in self.client_staleness:
                self.client_staleness[client_id] = 0
            self.client_staleness[client_id] += 1
        
        # Power-of-choice selection
        candidate_pool_size = min(len(available_clients), 
                                 self.choice_factor * num_to_select)
        
        # Sample candidates
        candidates = random.sample(available_clients, candidate_pool_size)
        
        # Calculate selection scores (higher loss + higher staleness = higher priority)
        scores = {}
        for client_id in candidates:
            loss = self.client_losses.get(client_id, 1.0)
            staleness = self.client_staleness.get(client_id, 1)
            
            # Combine loss and staleness (both should be higher for selection)
            scores[client_id] = loss + 0.1 * staleness
        
        # Select top clients
        selected = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:num_to_select]
        
        # Reset staleness for selected clients
        for client_id in selected:
            self.client_staleness[client_id] = 0
        
        return selected

class ClusterAwareSelector:
    """
    Selects clients ensuring representation from different clusters
    Memory impact: Low (only stores cluster information)
    """
    def __init__(self, selection_fraction: float = 0.3):
        self.selection_fraction = selection_fraction
        self.client_clusters = {}
        
    def update_clusters(self, client_cluster_map: Dict[int, int]):
        """Update client cluster assignments"""
        self.client_clusters = client_cluster_map
        
    def select_clients(self, available_clients: List[int], 
                      client_performances: Dict[int, float]) -> List[int]:
        """
        Select clients ensuring cluster diversity
        """
        if not available_clients:
            return []
            
        num_to_select = max(1, int(len(available_clients) * self.selection_fraction))
        
        # Group clients by cluster
        cluster_clients = defaultdict(list)
        for client_id in available_clients:
            cluster_id = self.client_clusters.get(client_id, 0)
            cluster_clients[cluster_id].append(client_id)
        
        if not cluster_clients:
            return random.sample(available_clients, num_to_select)
        
        # Select clients from each cluster proportionally
        selected = []
        clusters = list(cluster_clients.keys())
        
        # Calculate per-cluster selections
        clients_per_cluster = num_to_select // len(clusters)
        remaining = num_to_select % len(clusters)
        
        for i, cluster_id in enumerate(clusters):
            cluster_clients_list = cluster_clients[cluster_id]
            
            # Number to select from this cluster
            to_select = clients_per_cluster
            if i < remaining:
                to_select += 1
            
            to_select = min(to_select, len(cluster_clients_list))
            
            # Select best performing clients from this cluster
            cluster_performances = {cid: client_performances.get(cid, 0.5) 
                                  for cid in cluster_clients_list}
            
            cluster_selected = sorted(cluster_performances.keys(), 
                                    key=lambda x: cluster_performances[x], 
                                    reverse=True)[:to_select]
            
            selected.extend(cluster_selected)
        
        return selected

class GradientBasedSelector:
    """
    Selects clients based on gradient similarity and magnitude
    Memory impact: Moderate (stores gradient statistics)
    """
    def __init__(self, selection_fraction: float = 0.3, gradient_threshold: float = 0.1):
        self.selection_fraction = selection_fraction
        self.gradient_threshold = gradient_threshold
        self.client_gradient_norms = {}
        self.global_gradient_norm = 0.0
        
    def select_clients(self, available_clients: List[int], 
                      client_gradients: Dict[int, Dict[str, torch.Tensor]]) -> List[int]:
        """
        Select clients based on gradient information
        """
        if not available_clients:
            return []
            
        num_to_select = max(1, int(len(available_clients) * self.selection_fraction))
        
        # Calculate gradient norms
        gradient_norms = {}
        for client_id in available_clients:
            if client_id in client_gradients:
                grad_norm = self._calculate_gradient_norm(client_gradients[client_id])
                gradient_norms[client_id] = grad_norm
                self.client_gradient_norms[client_id] = grad_norm
        
        if not gradient_norms:
            return random.sample(available_clients, num_to_select)
        
        # Select clients with significant gradients
        significant_clients = []
        for client_id, grad_norm in gradient_norms.items():
            if grad_norm > self.gradient_threshold:
                significant_clients.append(client_id)
        
        if len(significant_clients) >= num_to_select:
            # Select from significant clients
            selected = sorted(significant_clients, 
                            key=lambda x: gradient_norms[x], 
                            reverse=True)[:num_to_select]
        else:
            # Include all significant clients and fill with others
            remaining = num_to_select - len(significant_clients)
            other_clients = [c for c in available_clients if c not in significant_clients]
            
            additional = sorted(other_clients, 
                              key=lambda x: gradient_norms.get(x, 0), 
                              reverse=True)[:remaining]
            
            selected = significant_clients + additional
        
        return selected
    
    def _calculate_gradient_norm(self, gradients: Dict[str, torch.Tensor]) -> float:
        """Calculate L2 norm of gradients"""
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += torch.norm(grad).item() ** 2
        return math.sqrt(total_norm)

class AdaptiveSelector:
    """
    Adaptive client selection that combines multiple strategies
    Memory impact: Moderate (combines multiple selectors)
    """
    def __init__(self, selection_fraction: float = 0.3):
        self.selection_fraction = selection_fraction
        self.diversity_selector = DiversityBasedSelector(selection_fraction)
        self.power_selector = PowerOfChoiceSelector(selection_fraction)
        self.cluster_selector = ClusterAwareSelector(selection_fraction)
        
        # Strategy weights (can be learned/adapted)
        self.strategy_weights = {
            'diversity': 0.4,
            'power': 0.3,
            'cluster': 0.3
        }
        
    def select_clients(self, available_clients: List[int], 
                      client_performances: Dict[int, float],
                      client_losses: Dict[int, float],
                      client_label_counts: Dict[int, Dict[int, int]],
                      client_cluster_map: Dict[int, int],
                      round_number: int) -> List[int]:
        """
        Adaptive selection combining multiple strategies
        """
        if not available_clients:
            return []
            
        num_to_select = max(1, int(len(available_clients) * self.selection_fraction))
        
        # Update cluster information
        self.cluster_selector.update_clusters(client_cluster_map)
        
        # Get selections from each strategy
        diversity_selected = self.diversity_selector.select_clients(
            available_clients, client_performances, client_label_counts)
        
        power_selected = self.power_selector.select_clients(
            available_clients, client_losses, round_number)
        
        cluster_selected = self.cluster_selector.select_clients(
            available_clients, client_performances)
        
        # Combine selections with weighted voting
        client_scores = defaultdict(float)
        
        for client_id in diversity_selected:
            client_scores[client_id] += self.strategy_weights['diversity']
        
        for client_id in power_selected:
            client_scores[client_id] += self.strategy_weights['power']
        
        for client_id in cluster_selected:
            client_scores[client_id] += self.strategy_weights['cluster']
        
        # Select top clients
        selected = sorted(client_scores.keys(), 
                         key=lambda x: client_scores[x], reverse=True)[:num_to_select]
        
        # Fill with random selection if needed
        if len(selected) < num_to_select:
            remaining = [c for c in available_clients if c not in selected]
            additional = random.sample(remaining, 
                                     min(num_to_select - len(selected), len(remaining)))
            selected.extend(additional)
        
        return selected
    
    def update_strategy_weights(self, strategy_performances: Dict[str, float]):
        """Update strategy weights based on performance"""
        total_performance = sum(strategy_performances.values())
        if total_performance > 0:
            for strategy, performance in strategy_performances.items():
                if strategy in self.strategy_weights:
                    self.strategy_weights[strategy] = performance / total_performance