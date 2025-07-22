# Hardware Configuration
no_cuda = False  # If True, disables GPU (CUDA) training.
no_mps = False  # If True, disables macOS GPU (MPS) training.

# Mode: set to 'train' or 'test'
mode = 'train'             # Change to 'test' for testing mode.
test_epochs = 20          # Total number of testing episodes.
train_epochs = 50000      # Total number of training episodes.

# Batch and Episode Settings
class_num = 5             # Number of classes per episode (e.g., 5-way classification).
train_batch_size = 1      # Training batch size per class. (Actual training batch size = train_batch_size * class_num.)
test_batch_size = 15      # Testing batch size per class. (Actual test batch size = test_batch_size * class_num.)
test_episodes = 200       # Number of evaluation episodes (each episode is an independent test run).

# Model Architecture Parameters
feature_dim = 64          # Feature dimension output by the encoder.
hidden_unit = 84          # Number of hidden units in the relation network’s fully-connected layer.

# Training Hyperparameters
learning_rate = 0.001     # Learning rate for the optimizer.
scheduler_step_size = 300 # Scheduler step size: number of episodes between learning rate decays.
validate_each = 100       # Frequency (in episodes) at which to run validation.
patience = 10             # Early stopping patience (number of validation rounds with no improvement before stopping training).

# Data Configuration
dataset_type = 'rgb'  # Change to 'bw' to use the black & white (Omniglot) code path.
# Data Configuration
if dataset_type == 'rgb':
    ######################
    # model_prefix = 'CUB100'
    # data_dir = 'data/CUB100'  # RGB dataset folder structure (expects subfolders "train" and "val"/"test")
    ######################
    # model_prefix = 'CIFAR100'
    # data_dir = 'data/CIFAR100'  
    ######################
    model_prefix = 'Imagenet'
    data_dir = 'data/Imagenet_ravi_larochelle_split'  
    ######################
else:
    model_prefix = 'Omniglot'
    data_dir = 'data/Omniglot' 

data_resize_shape = 84      # Resize input images to this size (both width and height) when creating the PyTorch dataset.
seed = 50                   # Random seed for reproducibility.