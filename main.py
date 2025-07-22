import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

# Import settings.
import settings
# Import task generator functions.
import task_generator as tg
# Import utility functions.
from utils import mean_confidence_interval, init_device

# Set random seeds for reproducibility.
random.seed(settings.seed)
np.random.seed(settings.seed)
torch.manual_seed(settings.seed)
if not settings.no_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed_all(settings.seed)

# Determine device.
device = init_device(settings.no_cuda, settings.no_mps)
print(f"Using device: {device}")

# Import the network modules.
from relation_network import CNNEncoder, RelationNetwork, weights_init

# Training Function
def train():
    # Depending on dataset type, select folder splitting and task class.
    if settings.dataset_type == 'bw':
        # For BW (Omniglot), use the Omniglot folder split and task.
        metatrain_folders, metatest_folders = tg.get_omniglot_metafolders(settings.data_dir)
        TaskClass = tg.FewShotTask 
        in_channels = 1
    else:
        # For RGB, use the standard folder split and task.
        metatrain_folders, metatest_folders = tg.get_metafolders(settings.data_dir, mode="train")
        TaskClass = tg.FewShotTask
        in_channels = 3

    # Create models.
    feature_encoder = CNNEncoder(in_channels=in_channels).to(device)
    relation_network = RelationNetwork(settings.feature_dim, settings.hidden_unit, dataset_type=settings.dataset_type).to(device)
    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    # Set up optimizers and schedulers.
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=settings.learning_rate)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=settings.learning_rate)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=settings.scheduler_step_size, gamma=0.5)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=settings.scheduler_step_size, gamma=0.5)

    last_accuracy = 0.0
    no_improvement_count = 0  # Counter for early stopping

    for episode in range(settings.train_epochs):
        # For BW, use a random rotation; for RGB, no rotation.
        rotation = random.choice([0, 90, 180, 270]) if settings.dataset_type == 'bw' else 0

        # Create a training task.
        task = TaskClass(metatrain_folders, settings.class_num,
                         train_num=settings.train_batch_size,
                         test_num=settings.test_batch_size)
        sample_dataloader = tg.get_data_loader(task,
                                               num_per_class=settings.train_batch_size,
                                               split="train", shuffle=False,
                                               use_old_sampler=False,
                                               rotation=rotation)
        batch_dataloader = tg.get_data_loader(task,
                                              num_per_class=settings.test_batch_size,
                                              split="test", shuffle=True,
                                              use_old_sampler=False,
                                              rotation=rotation)
        samples, sample_labels = next(iter(sample_dataloader))
        batches, batch_labels = next(iter(batch_dataloader))
        samples = samples.to(device)
        batches = batches.to(device)

        # Forward pass.
        sample_features = feature_encoder(Variable(samples))
        batch_features = feature_encoder(Variable(batches))
        # Get spatial dimensions dynamically.
        _, _, H, W = sample_features.shape

        sample_features_ext = sample_features.unsqueeze(0).repeat(settings.test_batch_size * settings.class_num, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(settings.train_batch_size * settings.class_num, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2)
        relation_pairs = relation_pairs.view(-1, settings.feature_dim * 2, H, W)
        relations = relation_network(relation_pairs)
        relations = relations.view(-1, settings.class_num * settings.train_batch_size)

        mse = nn.MSELoss()
        one_hot_labels = torch.zeros(settings.test_batch_size * settings.class_num, settings.class_num).to(device)
        one_hot_labels = one_hot_labels.scatter_(1, batch_labels.view(-1, 1).to(device), 1)
        loss = mse(relations, one_hot_labels)

        # Backward pass.
        feature_encoder.zero_grad()
        relation_network.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(relation_network.parameters(), 0.5)
        feature_encoder_optim.step()
        relation_network_optim.step()

        # Step schedulers after optimizer steps.
        feature_encoder_scheduler.step()
        relation_network_scheduler.step()

        if (episode + 1) % 1 == 0:
            print("Episode:", episode + 1, "Loss:", loss.item())

        # Validation.
        if (episode + 1) % settings.validate_each == 0:
            print("Validation Testing...")
            accuracies = []
            for _ in range(settings.test_episodes):
                total_rewards = 0
                counter = 0
                test_task = TaskClass(metatest_folders, settings.class_num,
                                      train_num=1, test_num=settings.test_batch_size)
                sample_dataloader_val = tg.get_data_loader(test_task,
                                                           num_per_class=1,
                                                           split="train",
                                                           shuffle=False,
                                                           use_old_sampler=False,
                                                           rotation=rotation)
                test_dataloader = tg.get_data_loader(test_task,
                                                     num_per_class=3,
                                                     split="test",
                                                     shuffle=True,
                                                     use_old_sampler=False,
                                                     rotation=rotation)
                sample_images, _ = next(iter(sample_dataloader_val))
                sample_images = sample_images.to(device)
                for test_images, test_labels in test_dataloader:
                    batch_size = test_labels.shape[0]
                    test_images = test_images.to(device)
                    s_feats = feature_encoder(Variable(sample_images))
                    t_feats = feature_encoder(Variable(test_images))
                    s_feats_ext = s_feats.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
                    t_feats_ext = t_feats.unsqueeze(0).repeat(settings.class_num, 1, 1, 1, 1)
                    t_feats_ext = torch.transpose(t_feats_ext, 0, 1)
                    # Use dynamic shape.
                    _, _, H_val, W_val = s_feats.shape
                    relation_pairs = torch.cat((s_feats_ext, t_feats_ext), 2)
                    relation_pairs = relation_pairs.view(-1, settings.feature_dim * 2, H_val, W_val)
                    relations = relation_network(relation_pairs)
                    relations = relations.view(-1, settings.class_num)
                    _, predict_labels = torch.max(relations.data, 1)
                    rewards = [1 if predict_labels[j] == test_labels[j].to(device) else 0 for j in range(batch_size)]
                    total_rewards += np.sum(rewards)
                    counter += batch_size
                accuracy = total_rewards / float(counter)
                accuracies.append(accuracy)
            val_acc, ci = mean_confidence_interval(accuracies)
            print("Validation Accuracy:", val_acc, "CI:", ci)

            # Early stopping.
            if val_acc > last_accuracy:
                last_accuracy = val_acc
                no_improvement_count = 0
                os.makedirs("./models", exist_ok=True)
                model_feat_path = f"./models/{settings.model_prefix}_feature_encoder_{settings.class_num}way_{settings.train_batch_size}shot.pkl"
                model_rel_path = f"./models/{settings.model_prefix}_relation_network_{settings.class_num}way_{settings.train_batch_size}shot.pkl"
                torch.save(feature_encoder.state_dict(), model_feat_path)
                torch.save(relation_network.state_dict(), model_rel_path)
                print("Saved models at episode:", episode + 1)
            else:
                no_improvement_count += 1
                print("No improvement count:", no_improvement_count)
                if no_improvement_count >= settings.patience:
                    print("Early stopping: no improvement for", settings.patience, "validation rounds.")
                    break

# Testing Function
def test():
    if settings.dataset_type == 'bw':
        metatrain_folders, metatest_folders = tg.get_omniglot_metafolders(settings.data_dir)
        TaskClass = tg.FewShotTask
        in_channels = 1
    else:
        metatrain_folders, metatest_folders = tg.get_metafolders(settings.data_dir, mode="test")
        TaskClass = tg.FewShotTask
        in_channels = 3

    feature_encoder = CNNEncoder(in_channels=in_channels).to(device)
    relation_network = RelationNetwork(settings.feature_dim, settings.hidden_unit, dataset_type=settings.dataset_type).to(device)
    
    model_feat_path = f"./models/{settings.model_prefix}_feature_encoder_{settings.class_num}way_{settings.train_batch_size}shot.pkl"
    model_rel_path = f"./models/{settings.model_prefix}_relation_network_{settings.class_num}way_{settings.train_batch_size}shot.pkl"
    if os.path.exists(model_feat_path):
        feature_encoder.load_state_dict(torch.load(model_feat_path, map_location=device))
        print("Loaded feature encoder.")
    if os.path.exists(model_rel_path):
        relation_network.load_state_dict(torch.load(model_rel_path, map_location=device))
        print("Loaded relation network.")

    total_accuracy = 0.0
    # Get a dummy sample to extract feature map size.
    dummy = torch.zeros(1, in_channels, settings.data_resize_shape, settings.data_resize_shape).to(device)
    dummy_feats = feature_encoder(dummy)
    _, _, H, W = dummy_feats.shape

    for episode in range(settings.test_epochs):
        print("Testing Episode:", episode + 1)
        accuracies = []
        for _ in range(settings.test_episodes):
            total_rewards = 0
            counter = 0
            test_task = TaskClass(metatest_folders, settings.class_num,
                                  train_num=1, test_num=settings.test_batch_size)
            # In test mode, for the support set use the old sampler.
            rotation = random.choice([0, 90, 180, 270]) if settings.dataset_type == 'bw' else 0
            sample_dataloader = tg.get_data_loader(test_task,
                                                   num_per_class=1,
                                                   split="train",
                                                   shuffle=False,
                                                   use_old_sampler=True,
                                                   rotation=rotation)
            test_dataloader = tg.get_data_loader(test_task,
                                                 num_per_class=1,
                                                 split="test",
                                                 shuffle=True,
                                                 use_old_sampler=False,
                                                 rotation=rotation)
            sample_images, _ = next(iter(sample_dataloader))
            sample_images = sample_images.to(device)
            for test_images, test_labels in test_dataloader:
                batch_size = test_labels.shape[0]
                test_images = test_images.to(device)
                s_feats = feature_encoder(Variable(sample_images))
                t_feats = feature_encoder(Variable(test_images))
                s_feats_ext = s_feats.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
                t_feats_ext = t_feats.unsqueeze(0).repeat(settings.class_num, 1, 1, 1, 1)
                t_feats_ext = torch.transpose(t_feats_ext, 0, 1)
                relation_pairs = torch.cat((s_feats_ext, t_feats_ext), 2)
                relation_pairs = relation_pairs.view(-1, settings.feature_dim * 2, H, W)
                relations = relation_network(relation_pairs)
                relations = relations.view(-1, settings.class_num)
                _, predict_labels = torch.max(relations.data, 1)
                rewards = [1 if predict_labels[j] == test_labels[j].to(device) else 0 for j in range(batch_size)]
                total_rewards += np.sum(rewards)
                counter += batch_size
            accuracy = total_rewards / float(counter)
            accuracies.append(accuracy)
        ep_acc, ci = mean_confidence_interval(accuracies)
        print("Test Accuracy:", ep_acc, "CI:", ci)
        total_accuracy += ep_acc
    avg_acc = total_accuracy / settings.test_epochs
    print("Average Test Accuracy:", avg_acc)

#   Main 
if __name__ == '__main__':
    if settings.mode == 'train':
        train()
    else:
        test()
