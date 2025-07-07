# Comprehensive Non-IID Federated Learning System
# Integrates all improvements for plant disease classification

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import os
import logging
from collections import defaultdict
import copy

# Import our custom modules
from advanced_aggregation import FedNovaAggregator, SCAFFOLDAggregator, AdaptiveAggregator
from smart_client_selection import AdaptiveSelector
from plant_disease_augmentation import MemoryEfficientDataAugmentation
from personalization_techniques import AdaptivePersonalization
from improved_model import ModelFactory, ImprovedCNN, EfficientPlantDiseaseNet
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NonIIDFLSystem:
    """
    Comprehensive Non-IID Federated Learning System
    Integrates advanced aggregation, client selection, personalization, and augmentation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self._initialize_model()
        self._initialize_aggregator()
        self._initialize_client_selector()
        self._initialize_personalization()
        self._initialize_augmentation()
        
        # Training state
        self.current_round = 0
        self.client_metrics = defaultdict(dict)
        self.global_metrics = []
        
        # Memory management
        self._setup_memory_management()
        
    def _initialize_model(self):
        """Initialize model architecture"""
        model_config = self.config.get('model', {})
        model_type = model_config.get('type', 'improved')
        
        # Create model
        self.global_model = ModelFactory.create_model(
            model_type=model_type,
            num_classes=num_labels,
            **model_config.get('params', {})
        )
        
        # Optimize for memory if needed
        memory_target = self.config.get('memory_target_mb', 2000)
        self.global_model = ModelFactory.optimize_for_memory(self.global_model, memory_target)
        
        self.global_model.to(self.device)
        
        # Log model information
        model_info = ModelFactory.get_model_info(self.global_model)
        logger.info(f"Model initialized: {model_info}")
        
    def _initialize_aggregator(self):
        """Initialize aggregation strategy"""
        agg_config = self.config.get('aggregation', {})
        agg_type = agg_config.get('type', 'adaptive')
        
        if agg_type == 'fednova':
            self.aggregator = FedNovaAggregator(
                momentum=agg_config.get('momentum', 0.9)
            )
        elif agg_type == 'scaffold':
            self.aggregator = SCAFFOLDAggregator(
                num_clients=num_clients
            )
        elif agg_type == 'adaptive':
            self.aggregator = AdaptiveAggregator(
                similarity_threshold=agg_config.get('similarity_threshold', 0.7)
            )
        else:
            # Default to adaptive aggregation
            self.aggregator = AdaptiveAggregator()
            
    def _initialize_client_selector(self):
        """Initialize client selection strategy"""
        selection_config = self.config.get('client_selection', {})
        selection_fraction = selection_config.get('fraction', 0.3)
        
        self.client_selector = AdaptiveSelector(selection_fraction=selection_fraction)
        
    def _initialize_personalization(self):
        """Initialize personalization techniques"""
        pers_config = self.config.get('personalization', {})
        
        if pers_config.get('enabled', True):
            self.personalization = AdaptivePersonalization(
                self.global_model, 
                pers_config
            )
        else:
            self.personalization = None
            
    def _initialize_augmentation(self):
        """Initialize data augmentation"""
        aug_config = self.config.get('augmentation', {})
        
        self.augmentation = MemoryEfficientDataAugmentation(
            image_size=image_size,
            severity=aug_config.get('severity', 0.3)
        )
        
    def _setup_memory_management(self):
        """Setup memory management"""
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
            torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
            
    def run_federated_training(self, num_rounds: int = num_rounds):
        """
        Run complete federated training with all improvements
        """
        logger.info(f"Starting federated training for {num_rounds} rounds")
        
        # Initialize client data and statistics
        self._initialize_client_data()
        
        for round_num in range(num_rounds):
            self.current_round = round_num
            logger.info(f"\n=== Round {round_num + 1}/{num_rounds} ===")
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 1. Smart client selection
            selected_clients = self._select_clients()
            logger.info(f"Selected {len(selected_clients)} clients: {selected_clients}")
            
            # 2. Distribute global model
            global_params = self._get_global_parameters()
            
            # 3. Client training with personalization
            client_updates = self._train_clients(selected_clients, global_params)
            
            # 4. Advanced aggregation
            aggregated_params = self._aggregate_updates(client_updates, selected_clients)
            
            # 5. Update global model
            self._update_global_model(aggregated_params)
            
            # 6. Evaluation
            round_metrics = self._evaluate_round(selected_clients)
            self.global_metrics.append(round_metrics)
            
            # 7. Save progress
            self._save_round_results(round_num, round_metrics)
            
            # 8. Adaptive adjustments
            self._adapt_strategies(round_metrics)
            
        logger.info("Federated training completed!")
        return self.global_metrics
        
    def _initialize_client_data(self):
        """Initialize client data and statistics"""
        logger.info("Initializing client data and statistics...")
        
        # Get client label distributions for smart selection
        self.client_label_distributions = {}
        self.client_data_sizes = {}
        
        for client_id in range(num_clients):
            try:
                # Load client data to get statistics
                from utils import get_partitioned_data
                train_dataset, test_dataset = get_partitioned_data(client_id, num_clients)
                
                # Get label distribution
                label_counts = defaultdict(int)
                for _, label in train_dataset.samples:
                    label_counts[label] += 1
                
                self.client_label_distributions[client_id] = dict(label_counts)
                self.client_data_sizes[client_id] = len(train_dataset)
                
            except Exception as e:
                logger.warning(f"Could not load data for client {client_id}: {e}")
                self.client_label_distributions[client_id] = {}
                self.client_data_sizes[client_id] = 0
                
    def _select_clients(self) -> List[int]:
        """Smart client selection"""
        available_clients = list(range(num_clients))
        
        # Get client performance from previous rounds
        client_performances = {}
        client_losses = {}
        
        for client_id in available_clients:
            if client_id in self.client_metrics:
                recent_metrics = self.client_metrics[client_id]
                client_performances[client_id] = recent_metrics.get('accuracy', 0.5)
                client_losses[client_id] = recent_metrics.get('loss', 1.0)
            else:
                client_performances[client_id] = 0.5
                client_losses[client_id] = 1.0
        
        # Get cluster information if available
        client_cluster_map = getattr(self, 'client_cluster_map', {})
        
        # Select clients using adaptive strategy
        selected_clients = self.client_selector.select_clients(
            available_clients=available_clients,
            client_performances=client_performances,
            client_losses=client_losses,
            client_label_counts=self.client_label_distributions,
            client_cluster_map=client_cluster_map,
            round_number=self.current_round
        )
        
        return selected_clients
        
    def _get_global_parameters(self) -> Dict[str, torch.Tensor]:
        """Get global model parameters"""
        return {name: param.clone() for name, param in self.global_model.named_parameters()}
        
    def _train_clients(self, selected_clients: List[int], 
                      global_params: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, torch.Tensor]]:
        """Train selected clients with personalization"""
        client_updates = {}
        
        for client_id in selected_clients:
            try:
                # Get client-specific model (with personalization)
                client_model = self._get_client_model(client_id, global_params)
                
                # Train client
                client_update, client_metrics = self._train_single_client(
                    client_id, client_model, global_params
                )
                
                client_updates[client_id] = client_update
                self.client_metrics[client_id] = client_metrics
                
            except Exception as e:
                logger.error(f"Error training client {client_id}: {e}")
                
        return client_updates
        
    def _get_client_model(self, client_id: int, 
                         global_params: Dict[str, torch.Tensor]) -> nn.Module:
        """Get personalized model for client"""
        if self.personalization:
            # Use personalization
            client_model = self.personalization.personalize_model(client_id)
        else:
            # Use global model
            client_model = copy.deepcopy(self.global_model)
            
        # Load global parameters
        for name, param in client_model.named_parameters():
            if name in global_params:
                param.data.copy_(global_params[name])
                
        return client_model
        
    def _train_single_client(self, client_id: int, client_model: nn.Module,
                           global_params: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Train a single client"""
        # Load client data
        from utils import get_partitioned_data
        train_dataset, test_dataset = get_partitioned_data(client_id, num_clients)
        
        # Apply client-specific augmentation
        train_transform, val_transform = self.augmentation.get_client_transforms(
            client_id, num_clients
        )
        
        # Override transforms
        train_dataset.transform = train_transform
        test_dataset.transform = val_transform
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Reduce memory usage
            pin_memory=False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        # Setup training
        client_model.to(self.device)
        client_model.train()
        
        optimizer = optim.SGD(client_model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Apply batch augmentation if enabled
                if self.config.get('augmentation', {}).get('batch_augmentation', False):
                    data, target = self.augmentation.apply_batch_augmentation(
                        (data, target), technique='mixup'
                    )
                
                optimizer.zero_grad()
                
                # Forward pass
                if isinstance(client_model, dict):  # Multi-head model
                    outputs = client_model(data)
                    loss = criterion(outputs['disease'], target)
                else:
                    outputs = client_model(data)
                    loss = criterion(outputs, target)
                
                # Add personalization loss if enabled
                if self.personalization:
                    features = client_model.extract_features(data) if hasattr(client_model, 'extract_features') else outputs
                    pers_loss = self.personalization.compute_personalization_loss(client_id, features)
                    loss += pers_loss
                
                # Proximal term for FedProx
                if proximal_mu > 0:
                    prox_term = 0.0
                    for name, param in client_model.named_parameters():
                        if name in global_params:
                            prox_term += torch.sum((param - global_params[name].to(self.device)) ** 2)
                    loss += (proximal_mu / 2) * prox_term
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Memory management
                if batch_idx % cache_cleanup_interval == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # Evaluation
        client_model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if isinstance(client_model, dict):  # Multi-head model
                    outputs = client_model(data)
                    outputs = outputs['disease']
                else:
                    outputs = client_model(data)
                
                test_loss += criterion(outputs, target).item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        avg_test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        
        # Get parameter updates
        client_update = {}
        for name, param in client_model.named_parameters():
            if name in global_params:
                client_update[name] = param.data - global_params[name].to(self.device)
        
        client_metrics = {
            'accuracy': accuracy,
            'loss': avg_test_loss,
            'num_samples': len(train_dataset)
        }
        
        # Move model back to CPU to save GPU memory
        client_model.cpu()
        
        return client_update, client_metrics
        
    def _aggregate_updates(self, client_updates: Dict[int, Dict[str, torch.Tensor]], 
                          selected_clients: List[int]) -> Dict[str, torch.Tensor]:
        """Aggregate client updates using advanced aggregation"""
        if not client_updates:
            return self._get_global_parameters()
        
        # Prepare data for aggregation
        updates_list = list(client_updates.values())
        weights_list = [self.client_data_sizes.get(cid, 1) for cid in selected_clients]
        
        # Get client accuracies for adaptive aggregation
        accuracies_list = [self.client_metrics.get(cid, {}).get('accuracy', 0.5) 
                          for cid in selected_clients]
        
        # Aggregate based on aggregator type
        if isinstance(self.aggregator, AdaptiveAggregator):
            aggregated = self.aggregator.aggregate(
                updates_list, selected_clients, weights_list, accuracies_list
            )
        elif isinstance(self.aggregator, FedNovaAggregator):
            aggregated = self.aggregator.aggregate(updates_list, weights_list)
        else:
            # Default weighted average
            aggregated = self._weighted_average(updates_list, weights_list)
        
        return aggregated
        
    def _weighted_average(self, updates: List[Dict[str, torch.Tensor]], 
                         weights: List[float]) -> Dict[str, torch.Tensor]:
        """Compute weighted average of updates"""
        if not updates:
            return {}
        
        total_weight = sum(weights)
        if total_weight == 0:
            return updates[0]
        
        result = {}
        for key in updates[0]:
            result[key] = sum(w * update[key] for w, update in zip(weights, updates)) / total_weight
        
        return result
        
    def _update_global_model(self, aggregated_params: Dict[str, torch.Tensor]):
        """Update global model with aggregated parameters"""
        for name, param in self.global_model.named_parameters():
            if name in aggregated_params:
                param.data += aggregated_params[name].to(self.device)
                
    def _evaluate_round(self, selected_clients: List[int]) -> Dict[str, float]:
        """Evaluate the current round"""
        # Aggregate client metrics
        total_samples = 0
        total_correct = 0
        total_loss = 0.0
        
        for client_id in selected_clients:
            if client_id in self.client_metrics:
                metrics = self.client_metrics[client_id]
                samples = metrics.get('num_samples', 0)
                accuracy = metrics.get('accuracy', 0.0)
                loss = metrics.get('loss', 0.0)
                
                total_samples += samples
                total_correct += accuracy * samples
                total_loss += loss * samples
        
        round_metrics = {
            'round': self.current_round,
            'accuracy': total_correct / total_samples if total_samples > 0 else 0.0,
            'loss': total_loss / total_samples if total_samples > 0 else 0.0,
            'num_clients': len(selected_clients),
            'total_samples': total_samples
        }
        
        logger.info(f"Round {self.current_round + 1} - "
                   f"Accuracy: {round_metrics['accuracy']:.4f}, "
                   f"Loss: {round_metrics['loss']:.4f}")
        
        return round_metrics
        
    def _save_round_results(self, round_num: int, round_metrics: Dict[str, float]):
        """Save round results"""
        results_dir = "results/non_iid_fl_system"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save round metrics
        metrics_file = os.path.join(results_dir, "round_metrics.jsonl")
        with open(metrics_file, "a") as f:
            json.dump(round_metrics, f)
            f.write("\n")
        
        # Save model checkpoint every 10 rounds
        if (round_num + 1) % 10 == 0:
            checkpoint_file = os.path.join(results_dir, f"model_round_{round_num + 1}.pth")
            torch.save({
                'round': round_num,
                'model_state_dict': self.global_model.state_dict(),
                'metrics': round_metrics
            }, checkpoint_file)
            
    def _adapt_strategies(self, round_metrics: Dict[str, float]):
        """Adapt strategies based on performance"""
        # Adapt client selection weights based on performance
        current_accuracy = round_metrics.get('accuracy', 0.0)
        
        if hasattr(self.client_selector, 'update_strategy_weights'):
            # Simple performance-based adaptation
            if len(self.global_metrics) > 1:
                prev_accuracy = self.global_metrics[-2].get('accuracy', 0.0)
                improvement = current_accuracy - prev_accuracy
                
                # Update strategy weights based on improvement
                strategy_performances = {
                    'diversity': max(0.1, 0.5 + improvement),
                    'power': max(0.1, 0.5 + improvement * 0.5),
                    'cluster': max(0.1, 0.5 + improvement * 0.3)
                }
                
                self.client_selector.update_strategy_weights(strategy_performances)
                
    def get_final_results(self) -> Dict[str, Any]:
        """Get final training results"""
        if not self.global_metrics:
            return {}
        
        final_metrics = self.global_metrics[-1]
        
        # Calculate improvement over rounds
        if len(self.global_metrics) > 1:
            initial_accuracy = self.global_metrics[0].get('accuracy', 0.0)
            final_accuracy = final_metrics.get('accuracy', 0.0)
            improvement = final_accuracy - initial_accuracy
        else:
            improvement = 0.0
        
        # Get memory usage estimates
        memory_usage = {}
        if self.personalization:
            memory_usage.update(self.personalization.get_memory_usage_estimate())
        
        augmentation_usage = self.augmentation.get_memory_usage_estimate(batch_size)
        memory_usage.update(augmentation_usage)
        
        return {
            'final_accuracy': final_metrics.get('accuracy', 0.0),
            'final_loss': final_metrics.get('loss', 0.0),
            'total_rounds': len(self.global_metrics),
            'accuracy_improvement': improvement,
            'memory_usage_estimates': memory_usage,
            'model_info': ModelFactory.get_model_info(self.global_model),
            'all_round_metrics': self.global_metrics
        }

# Configuration templates for different scenarios
DEFAULT_CONFIG = {
    'model': {
        'type': 'improved',
        'params': {
            'dropout_rate': 0.3
        }
    },
    'aggregation': {
        'type': 'adaptive',
        'similarity_threshold': 0.7
    },
    'client_selection': {
        'fraction': 0.3
    },
    'personalization': {
        'enabled': True,
        'fedrep': {
            'layers': ['fc', 'fc2']
        },
        'feature_alignment': {
            'feature_dim': 256,
            'alignment_weight': 0.1
        },
        'weights': {
            'feature_alignment': 0.5
        }
    },
    'augmentation': {
        'severity': 0.3,
        'batch_augmentation': True
    },
    'memory_target_mb': 2000
}

MEMORY_EFFICIENT_CONFIG = {
    'model': {
        'type': 'efficient',
        'params': {
            'width_multiplier': 0.75
        }
    },
    'aggregation': {
        'type': 'adaptive',
        'similarity_threshold': 0.7
    },
    'client_selection': {
        'fraction': 0.2  # Fewer clients for memory efficiency
    },
    'personalization': {
        'enabled': True,
        'fedrep': {
            'layers': ['head']
        },
        'weights': {
            'fedrep': 1.0
        }
    },
    'augmentation': {
        'severity': 0.2,
        'batch_augmentation': False
    },
    'memory_target_mb': 1500
}

HIGH_PERFORMANCE_CONFIG = {
    'model': {
        'type': 'improved',
        'params': {
            'dropout_rate': 0.2
        }
    },
    'aggregation': {
        'type': 'fednova',
        'momentum': 0.9
    },
    'client_selection': {
        'fraction': 0.4
    },
    'personalization': {
        'enabled': True,
        'fedrep': {
            'layers': ['fc', 'fc2']
        },
        'fedpac': {
            'similarity_threshold': 0.6
        },
        'feature_alignment': {
            'feature_dim': 256,
            'alignment_weight': 0.15
        },
        'weights': {
            'fedrep': 0.4,
            'fedpac': 0.3,
            'feature_alignment': 0.3
        }
    },
    'augmentation': {
        'severity': 0.4,
        'batch_augmentation': True
    },
    'memory_target_mb': 3000
}

def run_experiment(config_name: str = 'default') -> Dict[str, Any]:
    """
    Run federated learning experiment with specified configuration
    
    Args:
        config_name: Configuration to use ('default', 'memory_efficient', 'high_performance')
    
    Returns:
        Dictionary containing final results
    """
    # Select configuration
    if config_name == 'memory_efficient':
        config = MEMORY_EFFICIENT_CONFIG
    elif config_name == 'high_performance':
        config = HIGH_PERFORMANCE_CONFIG
    else:
        config = DEFAULT_CONFIG
    
    # Create and run system
    fl_system = NonIIDFLSystem(config)
    
    # Run training
    fl_system.run_federated_training()
    
    # Get results
    results = fl_system.get_final_results()
    
    # Save results
    results_dir = "results/non_iid_fl_system"
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f"final_results_{config_name}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Experiment completed. Results saved to {results_file}")
    
    return results

if __name__ == "__main__":
    # Run experiments with different configurations
    configurations = ['default', 'memory_efficient', 'high_performance']
    
    for config_name in configurations:
        print(f"\n{'='*50}")
        print(f"Running experiment: {config_name}")
        print(f"{'='*50}")
        
        try:
            results = run_experiment(config_name)
            print(f"Final accuracy: {results['final_accuracy']:.4f}")
            print(f"Accuracy improvement: {results['accuracy_improvement']:.4f}")
            print(f"Memory usage: {results['memory_usage_estimates']}")
        except Exception as e:
            logger.error(f"Error running {config_name}: {e}")
            continue