import numpy as np
import torch
import random
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import json

class SmartClientSelection:
    """
    Smart client selection strategy for non-IID federated learning.
    Improves convergence by selecting diverse and high-quality clients.
    """
    
    def __init__(self, num_clients: int, selection_fraction: float = 0.6):
        self.num_clients = num_clients
        self.selection_fraction = selection_fraction
        self.client_history = {}
        self.round_counter = 0
        
        # Selection strategies
        self.strategies = {
            'diversity': self._diversity_based_selection,
            'performance': self._performance_based_selection,
            'staleness': self._staleness_based_selection,
            'hybrid': self._hybrid_selection
        }
        
        # Strategy weights (will be adapted based on performance)
        self.strategy_weights = {
            'diversity': 0.4,
            'performance': 0.3,
            'staleness': 0.2,
            'hybrid': 0.1
        }
    
    def select_clients(self, available_clients: List[int], 
                      client_metrics: Dict[int, Dict] = None,
                      strategy: str = 'hybrid') -> List[int]:
        """
        Select clients for the current round.
        
        Args:
            available_clients: List of available client IDs
            client_metrics: Recent performance metrics for clients
            strategy: Selection strategy to use
            
        Returns:
            List of selected client IDs
        """
        self.round_counter += 1
        
        # Number of clients to select
        num_to_select = max(1, int(len(available_clients) * self.selection_fraction))
        
        if strategy not in self.strategies:
            strategy = 'hybrid'
        
        # Update client history if metrics provided
        if client_metrics:
            self._update_client_history(client_metrics)
        
        # Apply selection strategy
        selected_clients = self.strategies[strategy](available_clients, num_to_select)
        
        # Update selection history
        for client_id in selected_clients:
            if client_id not in self.client_history:
                self.client_history[client_id] = {}
            self.client_history[client_id]['last_selected_round'] = self.round_counter
            self.client_history[client_id]['selection_count'] = \
                self.client_history[client_id].get('selection_count', 0) + 1
        
        return selected_clients
    
    def _diversity_based_selection(self, available_clients: List[int], 
                                 num_to_select: int) -> List[int]:
        """Select clients to maximize diversity."""
        if not self.client_history:
            # Random selection for first few rounds
            return random.sample(available_clients, 
                               min(num_to_select, len(available_clients)))
        
        # Calculate diversity scores
        diversity_scores = {}
        
        for client_id in available_clients:
            score = 0.0
            
            # Prefer clients with different performance patterns
            if client_id in self.client_history:
                client_data = self.client_history[client_id]
                
                # Accuracy diversity (prefer outliers)
                avg_acc = np.mean([h.get('accuracy', 0.5) for h in self.client_history.values() 
                                 if 'accuracy' in h])
                client_acc = client_data.get('accuracy', 0.5)
                score += abs(client_acc - avg_acc) * 2.0
                
                # Loss diversity
                avg_loss = np.mean([h.get('loss', 1.0) for h in self.client_history.values() 
                                  if 'loss' in h])
                client_loss = client_data.get('loss', 1.0)
                score += abs(client_loss - avg_loss) * 1.0
                
                # Data size diversity
                avg_samples = np.mean([h.get('num_samples', 100) for h in self.client_history.values() 
                                     if 'num_samples' in h])
                client_samples = client_data.get('num_samples', 100)
                score += abs(client_samples - avg_samples) / avg_samples
            
            diversity_scores[client_id] = score
        
        # Select top diverse clients
        sorted_clients = sorted(available_clients, 
                              key=lambda x: diversity_scores.get(x, 0), 
                              reverse=True)
        
        return sorted_clients[:num_to_select]
    
    def _performance_based_selection(self, available_clients: List[int], 
                                   num_to_select: int) -> List[int]:
        """Select high-performing clients."""
        if not self.client_history:
            return random.sample(available_clients, 
                               min(num_to_select, len(available_clients)))
        
        # Calculate performance scores
        performance_scores = {}
        
        for client_id in available_clients:
            if client_id in self.client_history:
                client_data = self.client_history[client_id]
                
                # Combine accuracy and inverse loss
                accuracy = client_data.get('accuracy', 0.0)
                loss = client_data.get('loss', 10.0)
                
                # Performance score (higher is better)
                score = accuracy - 0.1 * loss
                
                # Bonus for consistent performance
                if 'accuracy_history' in client_data:
                    acc_std = np.std(client_data['accuracy_history'])
                    score += 0.1 / (1.0 + acc_std)  # Bonus for stability
                
                performance_scores[client_id] = score
            else:
                performance_scores[client_id] = 0.0
        
        # Select top performing clients
        sorted_clients = sorted(available_clients, 
                              key=lambda x: performance_scores.get(x, 0), 
                              reverse=True)
        
        return sorted_clients[:num_to_select]
    
    def _staleness_based_selection(self, available_clients: List[int], 
                                 num_to_select: int) -> List[int]:
        """Select clients that haven't participated recently."""
        staleness_scores = {}
        
        for client_id in available_clients:
            if client_id in self.client_history:
                last_round = self.client_history[client_id].get('last_selected_round', 0)
                staleness = self.round_counter - last_round
                
                # Higher score for more stale clients
                staleness_scores[client_id] = staleness
            else:
                # New clients get high priority
                staleness_scores[client_id] = float('inf')
        
        # Select most stale clients
        sorted_clients = sorted(available_clients, 
                              key=lambda x: staleness_scores.get(x, 0), 
                              reverse=True)
        
        return sorted_clients[:num_to_select]
    
    def _hybrid_selection(self, available_clients: List[int], 
                        num_to_select: int) -> List[int]:
        """Hybrid selection combining multiple strategies."""
        
        # Get selections from different strategies
        diversity_clients = self._diversity_based_selection(available_clients, num_to_select)
        performance_clients = self._performance_based_selection(available_clients, num_to_select)
        staleness_clients = self._staleness_based_selection(available_clients, num_to_select)
        
        # Combine with weighted scores
        combined_scores = defaultdict(float)
        
        # Diversity contribution
        for i, client_id in enumerate(diversity_clients):
            combined_scores[client_id] += self.strategy_weights['diversity'] * (num_to_select - i)
        
        # Performance contribution
        for i, client_id in enumerate(performance_clients):
            combined_scores[client_id] += self.strategy_weights['performance'] * (num_to_select - i)
        
        # Staleness contribution
        for i, client_id in enumerate(staleness_clients):
            combined_scores[client_id] += self.strategy_weights['staleness'] * (num_to_select - i)
        
        # Select top combined scores
        sorted_clients = sorted(available_clients, 
                              key=lambda x: combined_scores.get(x, 0), 
                              reverse=True)
        
        return sorted_clients[:num_to_select]
    
    def _update_client_history(self, client_metrics: Dict[int, Dict]):
        """Update client history with latest metrics."""
        for client_id, metrics in client_metrics.items():
            if client_id not in self.client_history:
                self.client_history[client_id] = {
                    'accuracy_history': [],
                    'loss_history': []
                }
            
            client_data = self.client_history[client_id]
            
            # Update current metrics
            client_data['accuracy'] = metrics.get('accuracy', 0.0)
            client_data['loss'] = metrics.get('loss', 10.0)
            client_data['num_samples'] = metrics.get('num_samples', 0)
            
            # Update history (keep last 10 values for memory efficiency)
            client_data['accuracy_history'].append(metrics.get('accuracy', 0.0))
            client_data['loss_history'].append(metrics.get('loss', 10.0))
            
            if len(client_data['accuracy_history']) > 10:
                client_data['accuracy_history'] = client_data['accuracy_history'][-10:]
                client_data['loss_history'] = client_data['loss_history'][-10:]
    
    def adapt_strategy_weights(self, global_performance: float):
        """Adapt strategy weights based on global performance."""
        # If performance is poor, emphasize diversity and staleness
        if global_performance < 0.3:
            self.strategy_weights.update({
                'diversity': 0.5,
                'performance': 0.2,
                'staleness': 0.3,
                'hybrid': 0.0
            })
        # If performance is moderate, balance all strategies
        elif global_performance < 0.7:
            self.strategy_weights.update({
                'diversity': 0.3,
                'performance': 0.4,
                'staleness': 0.2,
                'hybrid': 0.1
            })
        # If performance is good, emphasize performance
        else:
            self.strategy_weights.update({
                'diversity': 0.2,
                'performance': 0.6,
                'staleness': 0.1,
                'hybrid': 0.1
            })
    
    def get_selection_stats(self) -> Dict:
        """Get statistics about client selection."""
        if not self.client_history:
            return {}
        
        selection_counts = [data.get('selection_count', 0) 
                           for data in self.client_history.values()]
        
        return {
            'total_clients': len(self.client_history),
            'avg_selections': np.mean(selection_counts),
            'std_selections': np.std(selection_counts),
            'min_selections': np.min(selection_counts),
            'max_selections': np.max(selection_counts)
        }