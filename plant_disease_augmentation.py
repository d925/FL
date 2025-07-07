import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import random
from PIL import Image, ImageFilter, ImageEnhance
import cv2

class PlantDiseaseAugmentation:
    """
    Plant disease specific data augmentation for improving non-IID federated learning.
    Focused on realistic plant disease variations while being memory efficient.
    """
    
    def __init__(self, severity=0.5, enable_advanced=True):
        self.severity = severity  # 0.0 to 1.0, controls augmentation intensity
        self.enable_advanced = enable_advanced
        
    def get_training_transforms(self, image_size=64):
        """Get training-time augmentation transforms."""
        base_transforms = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),  # Plants can be oriented differently
        ]
        
        # Add plant-specific augmentations
        augmentation_transforms = [
            PlantColorVariation(severity=self.severity),
            PlantLightingVariation(severity=self.severity),
            PlantTextureVariation(severity=self.severity),
            LeafDeformation(severity=self.severity),
        ]
        
        if self.enable_advanced:
            augmentation_transforms.extend([
                DiseaseSpotSimulation(severity=self.severity),
                EnvironmentalEffects(severity=self.severity),
                PlantAgingEffect(severity=self.severity),
            ])
        
        # Randomly apply augmentations
        random_augment = transforms.RandomApply(augmentation_transforms, p=0.7)
        
        final_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))  # Simulate missing leaf parts
        ]
        
        return transforms.Compose(base_transforms + [random_augment] + final_transforms)
    
    def get_mixup_transform(self, alpha=0.4):
        """Get MixUp augmentation for better generalization."""
        return MixUpTransform(alpha=alpha)
    
    def get_cutmix_transform(self, alpha=1.0):
        """Get CutMix augmentation for better generalization."""
        return CutMixTransform(alpha=alpha)

class PlantColorVariation(nn.Module):
    """Simulate natural color variations in plants."""
    
    def __init__(self, severity=0.5):
        super().__init__()
        self.severity = severity
    
    def forward(self, img):
        if random.random() > self.severity:
            return img
        
        # Convert to PIL for color manipulation
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        
        # Simulate seasonal color changes
        if random.random() < 0.3:
            # Autumn-like color shift
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(random.uniform(0.7, 1.3))
            
            # Slight yellow/brown tint
            img = np.array(img)
            img[:, :, 1] = np.clip(img[:, :, 1] * random.uniform(1.0, 1.2), 0, 255)  # Green
            img[:, :, 0] = np.clip(img[:, :, 0] * random.uniform(1.1, 1.3), 0, 255)  # Red
            img = Image.fromarray(img.astype(np.uint8))
        
        # Simulate different health levels
        if random.random() < 0.4:
            # Simulate chlorophyll loss (yellowing)
            img = np.array(img)
            yellow_factor = random.uniform(0.8, 1.2)
            img[:, :, 1] = np.clip(img[:, :, 1] * yellow_factor, 0, 255)
            img = Image.fromarray(img.astype(np.uint8))
        
        return img

class PlantLightingVariation(nn.Module):
    """Simulate different lighting conditions."""
    
    def __init__(self, severity=0.5):
        super().__init__()
        self.severity = severity
    
    def forward(self, img):
        if random.random() > self.severity:
            return img
            
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        
        # Simulate different times of day
        brightness_factor = random.uniform(0.7, 1.4)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
        
        # Simulate shadow effects
        if random.random() < 0.3:
            img = np.array(img)
            # Create gradient shadow
            h, w = img.shape[:2]
            shadow = np.ones((h, w)) * random.uniform(0.7, 0.9)
            # Random shadow direction
            if random.random() < 0.5:
                shadow = np.minimum(shadow, np.linspace(1.0, 0.6, w).reshape(1, -1))
            else:
                shadow = np.minimum(shadow, np.linspace(1.0, 0.6, h).reshape(-1, 1))
            
            img = img * shadow[:, :, np.newaxis]
            img = np.clip(img, 0, 255)
            img = Image.fromarray(img.astype(np.uint8))
        
        return img

class PlantTextureVariation(nn.Module):
    """Simulate texture variations in plant surfaces."""
    
    def __init__(self, severity=0.5):
        super().__init__()
        self.severity = severity
    
    def forward(self, img):
        if random.random() > self.severity:
            return img
            
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        
        # Add subtle texture noise
        if random.random() < 0.4:
            img = np.array(img)
            noise = np.random.normal(0, random.uniform(2, 8), img.shape)
            img = np.clip(img + noise, 0, 255)
            img = Image.fromarray(img.astype(np.uint8))
        
        # Simulate surface roughness
        if random.random() < 0.3:
            # Slight blur for smoother surfaces
            blur_radius = random.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        return img

class LeafDeformation(nn.Module):
    """Simulate natural leaf deformations and orientations."""
    
    def __init__(self, severity=0.5):
        super().__init__()
        self.severity = severity
    
    def forward(self, img):
        if random.random() > self.severity:
            return img
            
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        
        # Random rotation within natural range
        if random.random() < 0.6:
            angle = random.uniform(-15, 15)
            img = TF.rotate(img, angle)
        
        # Slight perspective changes
        if random.random() < 0.3:
            # Simple shear transformation
            img = TF.affine(img, 
                          angle=0,
                          translate=(0, 0),
                          scale=1.0,
                          shear=random.uniform(-5, 5))
        
        return img

class DiseaseSpotSimulation(nn.Module):
    """Simulate disease spots and abnormalities."""
    
    def __init__(self, severity=0.5):
        super().__init__()
        self.severity = severity
    
    def forward(self, img):
        if random.random() > self.severity * 0.5:  # Less frequent but impactful
            return img
            
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        
        img = np.array(img)
        h, w = img.shape[:2]
        
        # Add random disease-like spots
        num_spots = random.randint(1, 4)
        for _ in range(num_spots):
            # Random spot location and size
            center_x = random.randint(int(w * 0.2), int(w * 0.8))
            center_y = random.randint(int(h * 0.2), int(h * 0.8))
            radius = random.randint(3, 12)
            
            # Create circular mask
            y, x = np.ogrid[:h, :w]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            # Disease spot color (brown, yellow, or dark)
            spot_colors = [
                [139, 69, 19],   # Brown
                [255, 255, 0],   # Yellow
                [50, 50, 50],    # Dark
                [255, 165, 0],   # Orange
            ]
            color = random.choice(spot_colors)
            
            # Apply spot with some transparency
            alpha = random.uniform(0.3, 0.7)
            for c in range(3):
                img[mask, c] = img[mask, c] * (1 - alpha) + color[c] * alpha
        
        img = np.clip(img, 0, 255)
        return Image.fromarray(img.astype(np.uint8))

class EnvironmentalEffects(nn.Module):
    """Simulate environmental effects like dust, water drops, etc."""
    
    def __init__(self, severity=0.5):
        super().__init__()
        self.severity = severity
    
    def forward(self, img):
        if random.random() > self.severity * 0.3:
            return img
            
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        
        img = np.array(img)
        
        # Simulate dust particles
        if random.random() < 0.4:
            h, w = img.shape[:2]
            num_particles = random.randint(5, 20)
            for _ in range(num_particles):
                x = random.randint(0, w-1)
                y = random.randint(0, h-1)
                size = random.randint(1, 3)
                # Dust color (light gray/brown)
                dust_color = [random.randint(180, 220)] * 3
                
                y_start, y_end = max(0, y-size), min(h, y+size+1)
                x_start, x_end = max(0, x-size), min(w, x+size+1)
                img[y_start:y_end, x_start:x_end] = dust_color
        
        img = np.clip(img, 0, 255)
        return Image.fromarray(img.astype(np.uint8))

class PlantAgingEffect(nn.Module):
    """Simulate plant aging effects."""
    
    def __init__(self, severity=0.5):
        super().__init__()
        self.severity = severity
    
    def forward(self, img):
        if random.random() > self.severity * 0.4:
            return img
            
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        
        # Simulate aging by reducing saturation and slight browning
        enhancer = ImageEnhance.Color(img)
        saturation_factor = random.uniform(0.7, 1.0)
        img = enhancer.enhance(saturation_factor)
        
        # Add slight brown tint for aging
        img = np.array(img)
        aging_factor = random.uniform(0.05, 0.15)
        img[:, :, 0] = np.clip(img[:, :, 0] * (1 + aging_factor), 0, 255)  # Red
        img[:, :, 1] = np.clip(img[:, :, 1] * (1 - aging_factor * 0.5), 0, 255)  # Green
        img = Image.fromarray(img.astype(np.uint8))
        
        return img

class MixUpTransform:
    """MixUp augmentation for batch-level mixing."""
    
    def __init__(self, alpha=0.4):
        self.alpha = alpha
    
    def __call__(self, batch_data, batch_labels):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch_data.size(0)
        index = torch.randperm(batch_size)
        
        mixed_data = lam * batch_data + (1 - lam) * batch_data[index, :]
        mixed_labels = lam * batch_labels + (1 - lam) * batch_labels[index]
        
        return mixed_data, mixed_labels

class CutMixTransform:
    """CutMix augmentation for batch-level mixing."""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch_data, batch_labels):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch_data.size(0)
        index = torch.randperm(batch_size)
        
        # Generate random bounding box
        W, H = batch_data.size(-1), batch_data.size(-2)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        mixed_data = batch_data.clone()
        mixed_data[:, :, bby1:bby2, bbx1:bbx2] = batch_data[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual cut size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        mixed_labels = lam * batch_labels + (1 - lam) * batch_labels[index]
        
        return mixed_data, mixed_labels