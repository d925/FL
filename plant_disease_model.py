import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, EfficientNet_B0_Weights
import math

class PlantDiseaseClassifier(nn.Module):
    """
    专为植物病害分类设计的高性能模型
    基于预训练模型，针对PlantVillage数据集优化
    """
    
    def __init__(self, num_classes=38, model_type='efficientnet_b0', pretrained=True):
        super(PlantDiseaseClassifier, self).__init__()
        self.num_classes = num_classes
        self.model_type = model_type
        
        if model_type == 'efficientnet_b0':
            # EfficientNet-B0: 高效且适合移动端
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            
            # 获取特征维度
            feature_dim = self.backbone.classifier[1].in_features
            
            # 替换分类器
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            
        elif model_type == 'resnet18':
            # ResNet18: 平衡性能和计算量
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            
            # 获取特征维度
            feature_dim = self.backbone.fc.in_features
            
            # 替换分类器
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
            
        elif model_type == 'custom_cnn':
            # 自定义CNN，针对植物病害优化
            self.backbone = self._build_custom_backbone()
            
        # 添加注意力机制
        self.attention = SpatialAttentionModule()
        
        # 冻结预训练层（初期训练）
        self._freeze_backbone(freeze=True)
        
    def _build_custom_backbone(self):
        """构建自定义CNN骨干网络"""
        return nn.Sequential(
            # Block 1: 捕获基本纹理
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 2: 病害特征提取
            self._make_residual_block(64, 128, 2),
            self._make_residual_block(128, 256, 2),
            self._make_residual_block(256, 512, 2),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # 分类器
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )
    
    def _make_residual_block(self, in_channels, out_channels, stride=1):
        """创建残差块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _freeze_backbone(self, freeze=True):
        """冻结或解冻预训练层"""
        if self.model_type in ['efficientnet_b0', 'resnet18']:
            # 冻结除分类器外的所有层
            for name, param in self.backbone.named_parameters():
                if 'classifier' not in name and 'fc' not in name:
                    param.requires_grad = not freeze
    
    def unfreeze_backbone(self):
        """解冻所有层进行微调"""
        self._freeze_backbone(freeze=False)
    
    def forward(self, x):
        if self.model_type == 'custom_cnn':
            return self.backbone(x)
        
        # 对于预训练模型，添加注意力机制
        if self.model_type == 'efficientnet_b0':
            # EfficientNet特征提取
            x = self.backbone.features(x)
            x = self.attention(x)  # 添加注意力
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.backbone.classifier(x)
            
        elif self.model_type == 'resnet18':
            # ResNet特征提取
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            
            x = self.attention(x)  # 添加注意力
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.backbone.fc(x)
        
        return x

class SpatialAttentionModule(nn.Module):
    """空间注意力模块，突出病害区域"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 计算通道维度的最大值和平均值
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        # 拼接并生成注意力图
        attention_map = torch.cat([max_out, avg_out], dim=1)
        attention_map = self.conv(attention_map)
        attention_map = self.sigmoid(attention_map)
        
        return x * attention_map

class PlantDiseaseEnsemble(nn.Module):
    """植物病害分类集成模型"""
    
    def __init__(self, num_classes=38, num_models=3):
        super(PlantDiseaseEnsemble, self).__init__()
        self.models = nn.ModuleList([
            PlantDiseaseClassifier(num_classes, 'efficientnet_b0'),
            PlantDiseaseClassifier(num_classes, 'resnet18'),
            PlantDiseaseClassifier(num_classes, 'custom_cnn')
        ])
        
        # 集成权重（可学习）
        self.ensemble_weights = nn.Parameter(torch.ones(num_models) / num_models)
        
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # 加权平均
        ensemble_output = torch.zeros_like(outputs[0])
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        for i, output in enumerate(outputs):
            ensemble_output += weights[i] * output
        
        return ensemble_output

class ProgressiveTrainingScheduler:
    """渐进式训练调度器"""
    
    def __init__(self, total_rounds=200):
        self.total_rounds = total_rounds
        self.phases = {
            'freeze': (0, 30),      # 冻结预训练层
            'unfreeze': (30, 100),  # 解冻进行微调
            'fine_tune': (100, 200) # 精细调优
        }
    
    def get_training_config(self, current_round):
        """获取当前轮次的训练配置"""
        config = {
            'learning_rate': 0.001,
            'unfreeze_backbone': False,
            'use_mixup': False,
            'use_cutmix': False,
            'augmentation_strength': 0.3
        }
        
        # 阶段1: 冻结预训练层，只训练分类器
        if self.phases['freeze'][0] <= current_round < self.phases['freeze'][1]:
            config.update({
                'learning_rate': 0.001,
                'unfreeze_backbone': False,
                'augmentation_strength': 0.2
            })
        
        # 阶段2: 解冻进行端到端微调
        elif self.phases['unfreeze'][0] <= current_round < self.phases['unfreeze'][1]:
            config.update({
                'learning_rate': 0.0001,  # 更小的学习率
                'unfreeze_backbone': True,
                'use_mixup': True,
                'augmentation_strength': 0.4
            })
        
        # 阶段3: 精细调优
        else:
            config.update({
                'learning_rate': 0.00005,
                'unfreeze_backbone': True,
                'use_mixup': True,
                'use_cutmix': True,
                'augmentation_strength': 0.5
            })
        
        return config

class PlantDiseaseMetrics:
    """植物病害分类专用评估指标"""
    
    def __init__(self, num_classes=38):
        self.num_classes = num_classes
        self.class_names = self._get_class_names()
    
    def _get_class_names(self):
        """获取PlantVillage类别名称"""
        return [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
            'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 
            'Corn___Northern_Leaf_Blight', 'Corn___healthy',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
            'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
    
    def calculate_plant_wise_accuracy(self, predictions, targets):
        """计算按植物种类分组的准确率"""
        plant_groups = {
            'Apple': [0, 1, 2, 3],
            'Corn': [7, 8, 9, 10],
            'Grape': [11, 12, 13, 14],
            'Tomato': [28, 29, 30, 31, 32, 33, 34, 35, 36, 37],
            'Others': [4, 5, 6, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        }
        
        plant_accuracies = {}
        for plant, class_indices in plant_groups.items():
            mask = torch.isin(targets, torch.tensor(class_indices))
            if mask.sum() > 0:
                plant_pred = predictions[mask]
                plant_true = targets[mask]
                accuracy = (plant_pred == plant_true).float().mean().item()
                plant_accuracies[plant] = accuracy
        
        return plant_accuracies
    
    def calculate_disease_vs_healthy_accuracy(self, predictions, targets):
        """计算健康vs病害分类准确率"""
        healthy_classes = [3, 4, 6, 10, 14, 17, 19, 22, 23, 24, 27, 37]  # healthy类别
        
        # 转换为二分类问题
        healthy_mask_true = torch.isin(targets, torch.tensor(healthy_classes))
        healthy_mask_pred = torch.isin(predictions, torch.tensor(healthy_classes))
        
        accuracy = (healthy_mask_true == healthy_mask_pred).float().mean().item()
        return accuracy