import os

# Core FL Configuration
num_clients = int(os.getenv('FL_NUM_CLIENTS', 50))
num_rounds = int(os.getenv('FL_NUM_ROUNDS', 200))
num_labels = int(os.getenv('FL_NUM_LABELS', 38))
alpha = float(os.getenv('FL_ALPHA', 0.1))  # Stronger non-IID for better heterogeneity
is_cluster = os.getenv('FL_USE_CLUSTER', 'True').lower() == 'true'

# Memory-efficient settings with improved parameters
batch_size = int(os.getenv('FL_BATCH_SIZE', 16))  # Reduced from 32 for memory
learning_rate = float(os.getenv('FL_LEARNING_RATE', 0.03))  # Increased for faster convergence
local_epochs = int(os.getenv('FL_LOCAL_EPOCHS', 5))  # Increased for better local learning
proximal_mu = float(os.getenv('FL_PROXIMAL_MU', 0.2))  # Increased for stronger regularization

# Learning rate scheduling
lr_scheduler_step = int(os.getenv('FL_LR_SCHEDULER_STEP', 50))  # Decay every 50 rounds
lr_scheduler_gamma = float(os.getenv('FL_LR_SCHEDULER_GAMMA', 0.8))  # Decay factor
warmup_rounds = int(os.getenv('FL_WARMUP_ROUNDS', 10))  # Warmup period

# Clustering Configuration
max_clusters = int(os.getenv('FL_MAX_CLUSTERS', 8))
pca_components = int(os.getenv('FL_PCA_COMPONENTS', 32))  # Reduced for memory

# Memory Management
gpu_memory_fraction = float(os.getenv('FL_GPU_MEMORY_FRACTION', 0.3))  # Conservative
cache_cleanup_interval = int(os.getenv('FL_CACHE_CLEANUP_INTERVAL', 10))

# Data Configuration
train_test_split = float(os.getenv('FL_TRAIN_TEST_SPLIT', 0.8))
image_size = int(os.getenv('FL_IMAGE_SIZE', 64))  # Reduced from 128 for memory

# Non-IID Enhancement Configuration
class_imbalance_ratio = float(os.getenv('FL_CLASS_IMBALANCE_RATIO', 0.7))  # Introduce class imbalance
min_samples_per_client = int(os.getenv('FL_MIN_SAMPLES_PER_CLIENT', 10))  # Minimum samples per client

# Plant Disease Augmentation Configuration
augmentation_severity = float(os.getenv('FL_AUGMENTATION_SEVERITY', 0.6))  # Augmentation intensity
enable_advanced_augmentation = os.getenv('FL_ENABLE_ADVANCED_AUG', 'True').lower() == 'true'

# Validation
if num_clients <= 0:
    raise ValueError("num_clients must be positive")
if num_rounds <= 0:
    raise ValueError("num_rounds must be positive")
if not 0 < train_test_split < 1:
    raise ValueError("train_test_split must be between 0 and 1")
if gpu_memory_fraction <= 0 or gpu_memory_fraction > 1:
    raise ValueError("gpu_memory_fraction must be between 0 and 1")