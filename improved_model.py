# Improved Model Architecture for Non-IID Plant Disease Classification
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math

class AttentionBlock(nn.Module):
    """
    Lightweight attention mechanism for feature enhancement
    Memory impact: Low (adds minimal parameters)
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super(AttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Lightweight MLP for attention
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * attention

class ResidualBlock(nn.Module):
    """
    Lightweight residual block with batch normalization
    Memory impact: Low (efficient residual connections)
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class ImprovedCNN(nn.Module):
    """
    Improved CNN architecture with attention and residual connections
    Designed for plant disease classification in non-IID FL settings
    """
    def __init__(self, num_classes: int = 38, dropout_rate: float = 0.3):
        super(ImprovedCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual blocks with attention
        self.layer1 = self._make_layer(32, 64, 2, stride=1)
        self.attention1 = AttentionBlock(64)
        
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.attention2 = AttentionBlock(128)
        
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.attention3 = AttentionBlock(256)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier with dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, in_channels: int, out_channels: int, 
                   num_blocks: int, stride: int = 1) -> nn.Sequential:
        """Create a layer with multiple residual blocks"""
        layers = []
        
        # First block may have different stride
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks with attention
        x = self.layer1(x)
        x = self.attention1(x)
        
        x = self.layer2(x)
        x = self.attention2(x)
        
        x = self.layer3(x)
        x = self.attention3(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def extract_features(self, x):
        """Extract features for clustering/similarity computation"""
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.attention1(x)
        
        x = self.layer2(x)
        x = self.attention2(x)
        
        x = self.layer3(x)
        x = self.attention3(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        return x

class EfficientPlantDiseaseNet(nn.Module):
    """
    Ultra-efficient network for memory-constrained environments
    Optimized for 8GB GPU / 32GB RAM
    """
    def __init__(self, num_classes: int = 38, width_multiplier: float = 1.0):
        super(EfficientPlantDiseaseNet, self).__init__()
        
        # Adjust channel numbers based on width multiplier
        def make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        # Define channel sizes
        channels = [int(c * width_multiplier) for c in [16, 32, 64, 128, 256]]
        channels = [make_divisible(c, 8) for c in channels]
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU6(inplace=True)
        )
        
        # Efficient blocks
        self.blocks = nn.ModuleList([
            self._make_efficient_block(channels[0], channels[1], stride=1),
            self._make_efficient_block(channels[1], channels[2], stride=2),
            self._make_efficient_block(channels[2], channels[3], stride=2),
            self._make_efficient_block(channels[3], channels[4], stride=2),
        ])
        
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(channels[4], num_classes)
        )
        
        self._initialize_weights()
        
    def _make_efficient_block(self, in_channels: int, out_channels: int, 
                            stride: int = 1) -> nn.Module:
        """Create efficient depthwise separable block"""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, 
                     padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.head(x)
        return x
    
    def extract_features(self, x):
        """Extract features for clustering/similarity computation"""
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        # Global pooling
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        
        return x

class MultiHeadModel(nn.Module):
    """
    Model with multiple heads for different aspects of plant disease classification
    """
    def __init__(self, num_classes: int = 38, num_plant_types: int = 8):
        super(MultiHeadModel, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = ImprovedCNN(num_classes=256)  # Use as feature extractor
        
        # Remove the final classification layer
        self.feature_extractor.fc = nn.Identity()
        
        # Multiple heads
        self.disease_head = nn.Linear(256, num_classes)  # Disease classification
        self.plant_head = nn.Linear(256, num_plant_types)  # Plant type classification
        self.severity_head = nn.Linear(256, 3)  # Severity: mild, moderate, severe
        
    def forward(self, x):
        features = self.feature_extractor(x)
        
        disease_pred = self.disease_head(features)
        plant_pred = self.plant_head(features)
        severity_pred = self.severity_head(features)
        
        return {
            'disease': disease_pred,
            'plant': plant_pred,
            'severity': severity_pred
        }
    
    def extract_features(self, x):
        """Extract shared features"""
        return self.feature_extractor.extract_features(x)

class ModelFactory:
    """
    Factory for creating different model architectures based on requirements
    """
    @staticmethod
    def create_model(model_type: str, num_classes: int = 38, **kwargs) -> nn.Module:
        """
        Create model based on type and requirements
        
        Args:
            model_type: Type of model ('basic', 'improved', 'efficient', 'multihead')
            num_classes: Number of output classes
            **kwargs: Additional model-specific parameters
        """
        if model_type == 'basic':
            from model import CNN
            return CNN(num_classes=num_classes)
        
        elif model_type == 'improved':
            dropout_rate = kwargs.get('dropout_rate', 0.3)
            return ImprovedCNN(num_classes=num_classes, dropout_rate=dropout_rate)
        
        elif model_type == 'efficient':
            width_multiplier = kwargs.get('width_multiplier', 1.0)
            return EfficientPlantDiseaseNet(num_classes=num_classes, 
                                          width_multiplier=width_multiplier)
        
        elif model_type == 'multihead':
            num_plant_types = kwargs.get('num_plant_types', 8)
            return MultiHeadModel(num_classes=num_classes, num_plant_types=num_plant_types)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, float]:
        """
        Get model information including parameter count and memory usage
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory usage (in MB)
        param_memory = total_params * 4 / (1024**2)  # 4 bytes per parameter
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_memory_mb': param_memory,
            'estimated_training_memory_mb': param_memory * 4,  # Rough estimate for gradients, etc.
        }
    
    @staticmethod
    def optimize_for_memory(model: nn.Module, target_memory_mb: float = 2000) -> nn.Module:
        """
        Optimize model for memory constraints
        """
        current_info = ModelFactory.get_model_info(model)
        
        if current_info['estimated_training_memory_mb'] <= target_memory_mb:
            return model
        
        # Apply model compression techniques
        # 1. Reduce model width
        if hasattr(model, 'width_multiplier'):
            reduction_factor = target_memory_mb / current_info['estimated_training_memory_mb']
            new_width = model.width_multiplier * math.sqrt(reduction_factor)
            return ModelFactory.create_model(
                'efficient', 
                num_classes=38, 
                width_multiplier=new_width
            )
        
        # 2. Use efficient architecture
        return ModelFactory.create_model('efficient', num_classes=38, width_multiplier=0.75)

class TransferLearningModel(nn.Module):
    """
    Transfer learning model for plant disease classification
    """
    def __init__(self, num_classes: int = 38, pretrained_backbone: str = 'mobilenet'):
        super(TransferLearningModel, self).__init__()
        
        if pretrained_backbone == 'mobilenet':
            import torchvision.models as models
            self.backbone = models.mobilenet_v2(pretrained=True)
            
            # Freeze early layers
            for param in self.backbone.features[:10].parameters():
                param.requires_grad = False
            
            # Replace classifier
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.backbone.last_channel, num_classes)
            )
        
        elif pretrained_backbone == 'efficientnet':
            # Use efficient architecture as pretrained backbone
            self.backbone = EfficientPlantDiseaseNet(num_classes=num_classes)
            
    def forward(self, x):
        return self.backbone(x)
    
    def extract_features(self, x):
        if hasattr(self.backbone, 'extract_features'):
            return self.backbone.extract_features(x)
        else:
            # For MobileNet, extract features before final classifier
            x = self.backbone.features(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.view(x.size(0), -1)
            return x