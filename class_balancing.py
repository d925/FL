import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
import torch.nn.functional as F

class ClassBalancingLoss:
    """
    Class balancing techniques for handling imbalanced datasets in federated learning.
    """
    
    def __init__(self, num_classes: int = 38, strategy: str = 'focal'):
        self.num_classes = num_classes
        self.strategy = strategy
        self.class_weights = None
        self.beta = 0.9999  # For effective number based weighting
        
    def calculate_class_weights(self, class_counts: Dict[int, int]) -> torch.Tensor:
        """Calculate class weights based on class distribution."""
        
        # Convert to tensor
        counts = torch.zeros(self.num_classes)
        for class_id, count in class_counts.items():
            if class_id < self.num_classes:
                counts[class_id] = count
        
        # Prevent division by zero
        counts = torch.clamp(counts, min=1)
        
        if self.strategy == 'inverse_freq':
            # Inverse frequency weighting
            total_samples = counts.sum()
            weights = total_samples / (self.num_classes * counts)
            
        elif self.strategy == 'effective_num':
            # Effective number based weighting
            effective_nums = (1.0 - torch.pow(self.beta, counts)) / (1.0 - self.beta)
            weights = 1.0 / effective_nums
            
        elif self.strategy == 'sqrt_inv':
            # Square root inverse frequency
            total_samples = counts.sum()
            weights = torch.sqrt(total_samples / (self.num_classes * counts))
            
        else:  # uniform
            weights = torch.ones(self.num_classes)
        
        # Normalize weights
        weights = weights / weights.sum() * self.num_classes
        
        return weights
    
    def get_loss_function(self, class_counts: Dict[int, int] = None):
        """Get the appropriate loss function with class balancing."""
        
        if class_counts:
            self.class_weights = self.calculate_class_weights(class_counts)
        
        if self.strategy == 'focal':
            return FocalLoss(alpha=self.class_weights, gamma=2.0, num_classes=self.num_classes)
        elif self.strategy == 'label_smooth':
            return LabelSmoothingLoss(num_classes=self.num_classes, smoothing=0.1)
        elif self.strategy == 'weighted_ce':
            return nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            return nn.CrossEntropyLoss()

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha=None, gamma=2.0, num_classes=38, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction
        
        if isinstance(alpha, (float, int)):
            self.alpha = torch.ones(num_classes) * alpha
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif alpha is None:
            self.alpha = None
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) where N is batch size and C is number of classes
            targets: (N,) where N is batch size
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Calculate focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            at = self.alpha.gather(0, targets)
            focal_weight = at * focal_weight
        
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss to prevent overconfidence.
    """
    
    def __init__(self, num_classes=38, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        """
        Args:
            pred: (N, C) predictions
            target: (N,) ground truth labels
        """
        pred = F.log_softmax(pred, dim=1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=1))

class BalancedSampler:
    """
    Balanced sampling strategies for federated learning.
    """
    
    def __init__(self, strategy='oversample'):
        self.strategy = strategy
    
    def get_balanced_indices(self, targets: List[int], target_samples_per_class: int = None) -> List[int]:
        """
        Get balanced sampling indices.
        
        Args:
            targets: List of target labels
            target_samples_per_class: Target number of samples per class
            
        Returns:
            List of indices for balanced sampling
        """
        class_counts = Counter(targets)
        unique_classes = list(class_counts.keys())
        
        if target_samples_per_class is None:
            if self.strategy == 'oversample':
                target_samples_per_class = max(class_counts.values())
            elif self.strategy == 'undersample':
                target_samples_per_class = min(class_counts.values())
            else:  # balanced
                target_samples_per_class = int(np.mean(list(class_counts.values())))
        
        # Group indices by class
        class_indices = {cls: [] for cls in unique_classes}
        for idx, target in enumerate(targets):
            class_indices[target].append(idx)
        
        balanced_indices = []
        
        for cls in unique_classes:
            cls_indices = class_indices[cls]
            current_count = len(cls_indices)
            
            if current_count >= target_samples_per_class:
                # Undersample or keep as is
                selected_indices = np.random.choice(
                    cls_indices, 
                    target_samples_per_class, 
                    replace=False
                ).tolist()
            else:
                # Oversample
                selected_indices = cls_indices.copy()
                remaining = target_samples_per_class - current_count
                
                # Sample with replacement for the remaining
                additional_indices = np.random.choice(
                    cls_indices, 
                    remaining, 
                    replace=True
                ).tolist()
                
                selected_indices.extend(additional_indices)
            
            balanced_indices.extend(selected_indices)
        
        # Shuffle the final indices
        np.random.shuffle(balanced_indices)
        return balanced_indices

class ProgressiveResizing:
    """
    Progressive resizing for curriculum learning in federated setting.
    """
    
    def __init__(self, start_size=32, end_size=64, total_rounds=200):
        self.start_size = start_size
        self.end_size = end_size
        self.total_rounds = total_rounds
    
    def get_current_size(self, current_round: int) -> int:
        """Get current image size based on round number."""
        if current_round >= self.total_rounds:
            return self.end_size
        
        # Linear progression
        progress = current_round / self.total_rounds
        current_size = int(self.start_size + (self.end_size - self.start_size) * progress)
        
        # Ensure size is multiple of 4 for better GPU efficiency
        current_size = ((current_size + 3) // 4) * 4
        
        return min(max(current_size, self.start_size), self.end_size)

class CurriculumLearning:
    """
    Curriculum learning strategies for federated learning.
    """
    
    def __init__(self, strategy='difficulty_based'):
        self.strategy = strategy
        self.sample_difficulties = {}
    
    def calculate_sample_difficulty(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[int, float]:
        """
        Calculate difficulty score for each sample.
        Higher score = more difficult
        """
        difficulties = {}
        
        # Convert to probabilities
        probs = F.softmax(predictions, dim=1)
        
        for i, (prob, target) in enumerate(zip(probs, targets)):
            # Difficulty based on prediction confidence
            target_prob = prob[target].item()
            
            # More difficult if model is less confident about correct class
            difficulty = 1.0 - target_prob
            
            difficulties[i] = difficulty
        
        return difficulties
    
    def get_curriculum_samples(self, sample_difficulties: Dict[int, float], 
                             current_round: int, total_rounds: int, 
                             easy_fraction: float = 0.3) -> List[int]:
        """
        Get samples for curriculum learning based on current round.
        
        Args:
            sample_difficulties: Dictionary mapping sample_idx -> difficulty_score
            current_round: Current federated learning round
            total_rounds: Total number of rounds
            easy_fraction: Fraction of easy samples to include initially
            
        Returns:
            List of sample indices to use for training
        """
        # Calculate curriculum progress (0 to 1)
        progress = min(current_round / (total_rounds * 0.8), 1.0)  # Reach full curriculum at 80% of training
        
        # Sort samples by difficulty (easy to hard)
        sorted_samples = sorted(sample_difficulties.items(), key=lambda x: x[1])
        
        # Calculate how many samples to include
        total_samples = len(sorted_samples)
        
        if self.strategy == 'difficulty_based':
            # Start with easy samples, gradually include harder ones
            num_samples = int(total_samples * (easy_fraction + (1 - easy_fraction) * progress))
            selected_samples = [idx for idx, _ in sorted_samples[:num_samples]]
            
        elif self.strategy == 'anti_curriculum':
            # Start with hard samples, gradually include easier ones
            num_samples = int(total_samples * (easy_fraction + (1 - easy_fraction) * progress))
            selected_samples = [idx for idx, _ in sorted_samples[-num_samples:]]
            
        else:  # random
            num_samples = int(total_samples * (easy_fraction + (1 - easy_fraction) * progress))
            all_indices = [idx for idx, _ in sorted_samples]
            selected_samples = np.random.choice(all_indices, num_samples, replace=False).tolist()
        
        return selected_samples