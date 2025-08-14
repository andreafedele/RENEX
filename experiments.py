import os
import time
import uuid
import torch
import random
import numpy as np
import pandas as pd

# Plot utils
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from PIL import Image
from torchvision import transforms
from skimage.segmentation import quickshift, felzenszwalb, slic

import settings
from utils import init_device, get_dataset_mean_std
from task_generator import get_omniglot_metafolders

# Import the network modules.
from relation_network import CNNEncoder, RelationNetwork

# Lime Explainer
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

#############
### UTILS ###
#############

# Loads images and apply transformation accordingly
def load_image(image_path, transform, device):
    try:
        if settings.dataset_type == 'bw':
            image = Image.open(image_path).convert('L')
        else:
            image = Image.open(image_path).convert('RGB')
             
        image_tensor = transform(image).unsqueeze(0).to(device)
        return image_tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Gets the model according transformation
def get_transform(model_prefix):
    # Get dataset mean and std for normalization purpouses 
    mean, std = get_dataset_mean_std(model_prefix)

    if settings.dataset_type == 'rgb':
        transform = transforms.Compose([
            transforms.Resize((settings.data_resize_shape, settings.data_resize_shape)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
        ])
    else:
        # black and white
        transform = transforms.Compose([
            transforms.Resize((settings.data_resize_shape, settings.data_resize_shape)),
            transforms.ToTensor(),
        ])

    return transform

# Computes Network relation scores in a given n-way k-shot task
def compute_relation_score_in_task(query_set, support_set, feature_encoder, relation_network):
    query_feature = feature_encoder(query_set)
    support_set_feature = feature_encoder(support_set)
   
    sample_features_ext = query_feature.unsqueeze(0).repeat(support_set_feature.shape[0],1,1,1,1)
    test_features_ext = support_set_feature.unsqueeze(0).repeat(1*settings.class_num,1,1,1,1)
    test_features_ext = torch.transpose(test_features_ext,0,1)

    relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,settings.feature_dim*2,19,19)
    relations = relation_network(relation_pairs).view(-1,settings.class_num)

    return relations

# Replace superpixels with the replace value for a specific segment
def remove_superpixel(image_np, segments, segment_value, replace_value):
    perturbed_image = image_np.copy()
    perturbed_image[segments == segment_value] = replace_value 
    return perturbed_image

####################################
### INSERTION & DELETION (RENEX) ###
####################################

def compute_insertion_score(query_idx, supp_idx, query_set, support_set,
                            orig_support_img, segmentation, heatmap,
                            feature_encoder, relation_network, device, replace_value):
    """
    Parameters:
      query_idx (int): index of the query image.
      supp_idx (int): index of the support image in the support_set.
      query_set (torch.Tensor): tensor of query images [5, 3, H, W].
      support_set (torch.Tensor): tensor of support images [5, 3, H, W].
      orig_support_img (numpy.ndarray): the original support image in [0,1] with shape [H, W, 3].
      segmentation (numpy.ndarray): segmentation mask for the support image, shape [H, W].
      heatmap (numpy.ndarray): the heatmap (per-pixel delta values) for this query-support pair.
      feature_encoder, relation_network: the respective models.
      device: torch device.
      replace_value: color to use as starting canvas before adding segments
      
    Returns:
      insertion_auc (float): area under the curve of the insertion process.
      scores_list (list): list of relation scores as segments are inserted.
    """
    # Build a dictionary of segment deltas.
    seg_vals = np.unique(segmentation)
    seg_deltas = {}
    for seg in seg_vals:
        mask = (segmentation == seg)
        # Since in our heatmap each pixel in the segment holds (delta / num_pixels),
        # we can recover the segment delta as:
        seg_val = np.unique(heatmap[mask])[0]  # constant value
        seg_deltas[seg] = seg_val * np.sum(mask)
    
    # Order segments from highest delta (most important) to lowest.
    ordered_segments = sorted(seg_deltas.keys(), key=lambda k: seg_deltas[k], reverse=True)
    
    # Create an "empty" (i.e. full gray) support image.
    composite = np.full_like(orig_support_img, replace_value)  # shape [H, W, 3]
    
    # Number of segments
    scores_list = []
    
    # Build the modified support set: start with composite replacing supp_idx.
    def build_support_set(modified_img):
        # modified_img should be a numpy array in [0,1] of shape [H, W, 3].
        modified_tensor = torch.tensor(modified_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        new_support = []
        for i in range(support_set.shape[0]):
            if i == supp_idx:
                new_support.append(modified_tensor)
            else:
                new_support.append(support_set[i].unsqueeze(0))
        return torch.cat(new_support, dim=0)  # shape [5,3,H,W]
    
    # Compute initial insertion score using the empty (gray) image.
    support_modified = build_support_set(composite)
    relations = compute_relation_score_in_task(query_set, support_modified, feature_encoder, relation_network)
    score = relations[query_idx, supp_idx].item()
    scores_list.append(score)
    
    # Iteratively add segments.
    for i, seg in enumerate(ordered_segments):
        mask = (segmentation == seg)
        # Paste the original support pixel values from orig_support_img into composite.
        composite[mask] = orig_support_img[mask]
        support_modified = build_support_set(composite)
        relations = compute_relation_score_in_task(query_set, support_modified, feature_encoder, relation_network)

        # Verbose plots
        cmap = 'grey' if settings.dataset_type == 'bw' else None
        # plt.imshow(support_modified[supp_idx].permute(1, 2, 0), cmap=cmap)
        # plt.imshow(support_modified[supp_idx].cpu().permute(1, 2, 0), cmap=cmap)
        # plt.show()

        score = relations[query_idx, supp_idx].item()
        scores_list.append(score)
    
    # x-axis: fraction of segments inserted (0 to 1)
    fractions = np.linspace(0, 1, len(scores_list))
    # Compute AUC (using the trapezoidal rule)
    insertion_auc = np.trapz(scores_list, fractions)
    return insertion_auc, scores_list

def compute_deletion_score(query_idx, supp_idx, query_set, support_set,
                           orig_support_img, segmentation, heatmap,
                           feature_encoder, relation_network, device, replace_value):
    """
    Parameters:
      query_idx (int): index of the query image.
      supp_idx (int): index of the support image.
      query_set (torch.Tensor): tensor of query images [5, 3, H, W].
      support_set (torch.Tensor): tensor of support images [5, 3, H, W].
      orig_support_img (numpy.ndarray): the original support image in [0,1] with shape [H, W, 3].
      segmentation (numpy.ndarray): segmentation mask for the support image, shape [H, W].
      heatmap (numpy.ndarray): the heatmap for this query-support pair.
      feature_encoder, relation_network: the respective models.
      device: torch device.
      replace_value: color to use while deleting segments.
      
    Returns:
      deletion_auc (float): area under the curve of the deletion process.
      scores_list (list): list of relation scores as segments are removed.
    """
    # Build dictionary of segment deltas.
    seg_vals = np.unique(segmentation)
    seg_deltas = {}
    for seg in seg_vals:
        mask = (segmentation == seg)
        seg_val = np.unique(heatmap[mask])[0]
        seg_deltas[seg] = seg_val * np.sum(mask)
    
    # Order segments from highest delta to lowest.
    ordered_segments = sorted(seg_deltas.keys(), key=lambda k: seg_deltas[k], reverse=True)
    
    # Start with a full support image (original image).
    composite = orig_support_img.copy()
    scores_list = []
    
    def build_support_set(modified_img):
        modified_tensor = torch.tensor(modified_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        new_support = []
        for i in range(support_set.shape[0]):
            if i == supp_idx:
                new_support.append(modified_tensor)
            else:
                new_support.append(support_set[i].unsqueeze(0))
        return torch.cat(new_support, dim=0)
    
    # Compute initial deletion score using the full support image.
    support_modified = build_support_set(composite)
    relations = compute_relation_score_in_task(query_set, support_modified, feature_encoder, relation_network)
    score = relations[query_idx, supp_idx].item()
    scores_list.append(score)
    
    # Iteratively remove segments (set them to gray or any replacing value)
    for i, seg in enumerate(ordered_segments):
        mask = (segmentation == seg)
        composite[mask] = replace_value 
        support_modified = build_support_set(composite)
        relations = compute_relation_score_in_task(query_set, support_modified, feature_encoder, relation_network)
        score = relations[query_idx, supp_idx].item()
        scores_list.append(score)

        # Verbose plots
        cmap = 'grey' if settings.dataset_type == 'bw' else None
        # plt.imshow(support_modified[supp_idx].permute(1, 2, 0), cmap=cmap)
        # plt.imshow(support_modified[supp_idx].cpu().permute(1, 2, 0), cmap=cmap)
        # plt.show()
    
    fractions = np.linspace(0, 1, len(scores_list))
    deletion_auc = np.trapz(scores_list, fractions)
    return deletion_auc, scores_list

####################################
### INSERTION & DELETION (LIME) ###
####################################

def compute_insertion_lime(query_idx, supp_idx, query_set, support_set,
                             orig_support_img, lime_segments, relevant_segments,
                             feature_encoder, relation_network, device, replace_value):
    
    # Start with an empty (full gray) image.
    composite = np.full_like(orig_support_img, replace_value)  # shape [H, W, 3]
    scores_list = []
    
    # Helper function to rebuild the support set with the modified support image.
    def build_support_set(modified_img):
        modified_tensor = torch.tensor(modified_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        new_support = []
        for i in range(support_set.shape[0]):
            if i == supp_idx:
                new_support.append(modified_tensor)
            else:
                new_support.append(support_set[i].unsqueeze(0))
        return torch.cat(new_support, dim=0)  # shape [5, 3, H, W]
    
    # Compute initial insertion score using the empty (gray) image.
    support_modified = build_support_set(composite)
    relations = compute_relation_score_in_task(query_set, support_modified, feature_encoder, relation_network)
    score = relations[query_idx, supp_idx].item()
    scores_list.append(score)
    
    # Iteratively "insert" segments according to the provided ordering.
    for seg in relevant_segments:
        # Update composite: replace gray pixels with the original ones for this segment.
        mask = (lime_segments == seg)
        composite[mask] = orig_support_img[mask]
        
        support_modified = build_support_set(composite)
        relations = compute_relation_score_in_task(query_set, support_modified, feature_encoder, relation_network)
        score = relations[query_idx, supp_idx].item()
        scores_list.append(score)

        # Verbose plots
        cmap = 'grey' if settings.dataset_type == 'bw' else None
        # plt.imshow(support_modified[supp_idx].permute(1, 2, 0), cmap=cmap)
        # plt.imshow(support_modified[supp_idx].cpu().permute(1, 2, 0), cmap=cmap)
        # plt.show()
    
    fractions = np.linspace(0, 1, len(scores_list))
    insertion_auc = np.trapz(scores_list, fractions)
    return insertion_auc, scores_list

def compute_deletion_lime(query_idx, supp_idx, query_set, support_set,
                          orig_support_img, lime_segments, relevant_segments,
                          feature_encoder, relation_network, device, replace_value):
    
    # Start with the full original image.
    composite = orig_support_img.copy()
    scores_list = []
    
    def build_support_set(modified_img):
        modified_tensor = torch.tensor(modified_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        new_support = []
        for i in range(support_set.shape[0]):
            if i == supp_idx:
                new_support.append(modified_tensor)
            else:
                new_support.append(support_set[i].unsqueeze(0))
        return torch.cat(new_support, dim=0)
    
    # Compute initial deletion score using the full image.
    support_modified = build_support_set(composite)
    relations = compute_relation_score_in_task(query_set, support_modified, feature_encoder, relation_network)
    score = relations[query_idx, supp_idx].item()
    scores_list.append(score)
    
    # Iteratively "delete" segments in the given order.
    for seg in relevant_segments:
        mask = (lime_segments == seg)
        composite[mask] = replace_value  # 0.5 Set these pixels to gray.
        
        support_modified = build_support_set(composite)
        relations = compute_relation_score_in_task(query_set, support_modified, feature_encoder, relation_network)
        score = relations[query_idx, supp_idx].item()
        scores_list.append(score)

        # Verbose plots
        cmap = 'grey' if settings.dataset_type == 'bw' else None
        # plt.imshow(support_modified[supp_idx].permute(1, 2, 0), cmap=cmap)
        # plt.imshow(support_modified[supp_idx].cpu().permute(1, 2, 0), cmap=cmap)
        # plt.show()
    
    fractions = np.linspace(0, 1, len(scores_list))
    deletion_auc = np.trapz(scores_list, fractions)
    return deletion_auc, scores_list

#############
### RENEX ###
#############

def RENEX(query_set, support_set, feature_encoder, relation_network, segmentation_func, segmentation_params, replace_value, device, verbose=False):
    # Compute original relation scores (from unperturbed samples)
    original_relations = compute_relation_score_in_task(query_set, support_set, feature_encoder, relation_network)

    # Generate heatmap explanations for all query-support pairs.
    # We will loop over all query indices and, for each, loop over all support samples.
    # We store heatmaps in a dictionary keyed by (query_idx, supp_idx).
    heatmaps = {}

    # Each entry will be a dictionary with keys "auc" and "scores".
    insertion_results = {}
    deletion_results = {}

    # Computational time
    execution_times = []

    # Predicted relation score
    predicted_scores = []

    if settings.dataset_type == 'bw':
        replace_value = replace_value[0]

    for query_idx in range(5):
        # Compute RENEX on correct classification tasks w/ rel score higher than given threshold
        query_scores = original_relations[query_idx, :]  
        predicted_support_idx = torch.argmax(query_scores).item()

        if predicted_support_idx == query_idx and query_scores[predicted_support_idx].item() >= RELATION_SCORE_THRESHOLD:
            print(f"Query sample {query_idx+1} correctly classified (highest score at support {predicted_support_idx+1}).")
            predicted_scores.append(query_scores[predicted_support_idx].item())
        
            for supp_idx in range(5):
                # Convert the current support image to a NumPy array 
                support_image_np = support_set[supp_idx].permute(1, 2, 0).cpu().numpy()  # shape: [84,84,3] in [0,1]

                # Compute segments on the support image.
                if settings.dataset_type == 'bw':
                    # Applying Mask to Black drawn character on top of white background, so to only filter on those images
                    support_image_np_no_axis = support_image_np.squeeze(axis=2)
                    segmentation_params['mask'] = support_image_np_no_axis < 0.5 # Mask filters out white background, to only focus on black drawn characters.
                    segments = segmentation_func(support_image_np_no_axis, **segmentation_params)
                else:
                    segments = segmentation_func(support_image_np, **segmentation_params)
                
                unique_segments = np.unique(segments)
                if verbose:
                    print(f"Query {query_idx+1}, Support {supp_idx+1}: {len(unique_segments)} superpixels")

                # --- Start process time measurement here ---
                start_time = time.process_time()
                
                # Get the original relation score for the pair (query_idx, supp_idx)
                original_score = original_relations[query_idx, supp_idx].item()
                print(f"Original Score for Query {query_idx+1} and Support {supp_idx+1}: {original_score:.3f}")
                
                # Create an empty heatmap (same spatial dimensions as the support image)
                heatmap = np.zeros(support_image_np.shape[:2], dtype=np.float32)
                
                # Loop over each superpixel in the support image.
                for segment_value in unique_segments:
                    # Perturb the support image for the current segment
                    perturbed_image = remove_superpixel(support_image_np, segments, segment_value, replace_value=replace_value)
                    # Convert the perturbed image back to a tensor of shape [1, 3, 84, 84]
                    perturbed_image_tensor = torch.tensor(perturbed_image).permute(2, 0, 1).unsqueeze(0).float().to(device)

                    if verbose == True:
                        cmap = 'grey' if settings.dataset_type == 'bw' else None
                        plt.imshow(perturbed_image, cmap=cmap)
                        plt.show()
                    
                    # Rebuild the support set where only the current support sample (at supp_idx) is replaced by its perturbed version.
                    support_set_modified = []
                    for i in range(5):
                        if i == supp_idx:
                            support_set_modified.append(perturbed_image_tensor)
                        else:
                            support_set_modified.append(support_set[i].unsqueeze(0))
                    support_set_modified = torch.cat(support_set_modified, dim=0)  # [5, 3, 84, 84]
                    
                    # Compute the relation score for the modified pair (query_idx, supp_idx)
                    relations_modified = compute_relation_score_in_task(query_set, support_set_modified, feature_encoder, relation_network)
                    new_score = relations_modified[query_idx, supp_idx].item()
                    
                    # Compute the difference (delta) between original and new score.
                    delta = original_score - new_score
                    if verbose:
                        print(f" Query {query_idx+1}, Support {supp_idx+1} - Segment {segment_value}: New Score: {new_score:.3f}, Delta: {delta:.3f}")
                    
                    # Build/update the heatmap: distribute the delta over the pixels of this segment.
                    segment_mask = (segments == segment_value)
                    num_pixels = np.sum(segment_mask)
                    heatmap[segment_mask] = delta / num_pixels
                    
                # Store the heatmap for this (query, support) pair.
                heatmaps[(query_idx, supp_idx)] = heatmap

                # --- End process time measurement here ---
                end_time = time.process_time()
                if query_idx == supp_idx:
                    execution_times.append(end_time - start_time)

                # Li metto qui per comodità, in realtà andrebbero calcolati al di fuori dell'explainer così da lasciarlo pulito
                ## li salvo però solo su quegli esempi che so appartenere alla stessa classe
                if query_idx == supp_idx:
                    insertion_auc, insertion_scores = compute_insertion_score(query_idx, supp_idx,
                                            query_set, support_set,
                                            support_image_np, segments, heatmap,
                                            feature_encoder, relation_network, device, replace_value)
                    

                    deletion_auc, deletion_scores = compute_deletion_score(query_idx, supp_idx,
                                            query_set, support_set,
                                            support_image_np, segments, heatmap,
                                            feature_encoder, relation_network, device, replace_value)
                    
                    insertion_results[(query_idx, supp_idx)] = {"auc": insertion_auc, "scores": insertion_scores}
                    deletion_results[(query_idx, supp_idx)] = {"auc": deletion_auc, "scores": deletion_scores}
            
    return heatmaps, insertion_results, deletion_results, execution_times, predicted_scores

#############
### LIME ###
#############

def LIME(query_set, support_set, feature_encoder, relation_network, segmentation_func, segmentation_params, replace_value, device, verbose=False):
    # Compute original relation scores (from unperturbed samples)
    original_relations = compute_relation_score_in_task(query_set, support_set, feature_encoder, relation_network)

    # Generate heatmap explanations for all query-support pairs.
    # We will loop over all query indices and, for each, loop over all support samples.
    # We store heatmaps in a dictionary keyed by (query_idx, supp_idx).
    heatmaps = {}

    # Each entry will be a dictionary with keys "auc" and "scores".
    insertion_results = {}
    deletion_results = {}

    # Computational time
    execution_times = []

    # Predicted relation score
    predicted_scores = []

    if settings.dataset_type == 'bw':
        replace_value = replace_value[0]

    for query_idx in range(5):
        # Compute RENEX on correct classification tasks w/ rel score higher than given threshold
        query_scores = original_relations[query_idx, :]  
        predicted_support_idx = torch.argmax(query_scores).item()

        if predicted_support_idx == query_idx and query_scores[predicted_support_idx].item() >= RELATION_SCORE_THRESHOLD:
            print(f"Query sample {query_idx+1} correctly classified (highest score at support {predicted_support_idx+1}).")
            predicted_scores.append(query_scores[predicted_support_idx].item())
        
            for supp_idx in range(5):
                # Convert the current support image to a NumPy array 
                support_image_np = support_set[supp_idx].permute(1, 2, 0).cpu().numpy()  # shape: [84,84,3] in [0,1]

                # Applying Mask to Black drawn character on top of white background, so to only filter on those images
                if settings.dataset_type == 'bw':
                    # Mask filters out white background, to only focus on black drawn characters.
                    segmentation_params['mask'] = support_image_np < 0.5
                    segmentation_func.set_params(**segmentation_params)

                explainer = lime_image.LimeImageExplainer()

                # --- Start process time measurement here ---
                start_time = time.process_time()
                
                def predict_fn_relscore(perturbed_images):
                    lime_scores = []
                    for perturbed_image in perturbed_images:
                        # Convert the perturbed image back to a tensor of shape [1, 3, 84, 84]
                        perturbed_image_tensor = torch.tensor(perturbed_image).permute(2, 0, 1).unsqueeze(0).float().to(device)
                        
                        # Rebuild the support set where only the current support sample (at supp_idx) is replaced by its perturbed version.
                        support_set_modified = []
                        for i in range(5):
                            if i == supp_idx:
                                support_set_modified.append(perturbed_image_tensor)
                            else:
                                support_set_modified.append(support_set[i].unsqueeze(0))
                        support_set_modified = torch.cat(support_set_modified, dim=0)  # [5, 3, 84, 84]
                        
                        # Compute the relation score for the modified pair (query_idx, supp_idx)
                        relations_modified = compute_relation_score_in_task(query_set, support_set_modified, feature_encoder, relation_network)
                        new_score = relations_modified[:, supp_idx].detach().cpu().numpy()
                        lime_scores.append(new_score)
                    
                    predict_proba = np.mean(lime_scores, axis=0).reshape(-1,1)
                    return predict_proba

                # image_to_explain = support_image_np if settings.dataset_type == 'RGB' else support_image_np_no_axis
                explanation = explainer.explain_instance(support_image_np, 
                                                        predict_fn_relscore, 
                                                        top_labels=5, 
                                                        # hide_color=0, 
                                                        hide_color=replace_value, 
                                                        num_samples=1000, 
                                                        batch_size=5,
                                                        segmentation_fn=segmentation_func)
                
                # --- End process time measurement here ---
                end_time = time.process_time()
                if query_idx == supp_idx:
                    execution_times.append(end_time - start_time)
                
                lime_segments = explanation.segments
                lime_local_exp = explanation.local_exp[explanation.top_labels[0]]
                lime_local_exp_sorted = sorted(lime_local_exp, key=lambda x: x[1], reverse=True)
                relevant_segments = [seg for seg, weight in lime_local_exp_sorted]

                # create heatmap shaped [H, W]
                heatmap = np.zeros(support_image_np.shape[:2], dtype=np.float32)

                for seg_id, weight in lime_local_exp:
                    mask = (lime_segments == seg_id)
                    if mask.ndim == 3:
                        mask = mask[:, :, 0]
                    num_pixels = np.sum(mask)
                    if num_pixels > 0:
                        # per-pixel value -> sum over segment = weight (same pattern as RENEX)
                        heatmap[mask] = weight / num_pixels

                # store it so function returns same structure as RENEX
                heatmaps[(query_idx, supp_idx)] = heatmap

                insertion_auc_lime, insertion_scores_lime = compute_insertion_lime(query_idx, supp_idx,
                                              query_set, support_set,
                                              support_image_np, lime_segments, relevant_segments,
                                              feature_encoder, relation_network, device, replace_value)

                deletion_auc_lime, deletion_scores_lime = compute_deletion_lime(query_idx, supp_idx,
                                                            query_set, support_set,
                                                            support_image_np, lime_segments, relevant_segments,
                                                            feature_encoder, relation_network, device, replace_value)
                
                insertion_results[(query_idx, supp_idx)] = {"auc": insertion_auc_lime, "scores": insertion_scores_lime}
                deletion_results[(query_idx, supp_idx)] = {"auc": deletion_auc_lime, "scores": deletion_scores_lime}

            
    return heatmaps, insertion_results, deletion_results, execution_times, predicted_scores

#####################
### Task Creation ###
#####################

def get_omniglot_random_test_samples(classes, num_classes=5):
    selected_classes = random.sample(classes, num_classes)
    
    query_paths = []
    support_paths = []
    
    for cls_dir in selected_classes:
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                 if f.lower().endswith(('.png'))]
        if len(files) < 2:
            class_name = selected_classes[0].split("/")[-1]
            print(f"Not enough images in class {class_name}. Using the same image for query and support.")
            query_paths.append(files[0])
            support_paths.append(files[0])
        else:
            q, s = random.sample(files, 2)
            query_paths.append(q)
            support_paths.append(s)
    
    return query_paths, support_paths

def get_random_test_samples(test_dir, num_classes=5):
    classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    selected_classes = random.sample(classes, num_classes)
    
    query_paths = []
    support_paths = []
    
    for cls in selected_classes:
        cls_dir = os.path.join(test_dir, cls)
        # Filter for common image file extensions.
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                 if f.lower().endswith(('.jpg'))]
        if len(files) < 2:
            print(f"Not enough images in class {cls}. Using the same image for query and support.")
            query_paths.append(files[0])
            support_paths.append(files[0])
        else:
            q, s = random.sample(files, 2)
            query_paths.append(q)
            support_paths.append(s)
    
    return query_paths, support_paths

########################
### PLOT EXPLANTIONS ###
########################
# Ensure mathtext uses bold correctly
plt.rcParams['mathtext.default'] = 'regular'
norm = Normalize(vmin=-1, vmax=1)  # This norm will be used in the visualization
FONT_SIZE = 20

def normalize_per_heatmap(hm):
    """Locally normalize hm so positives → [0,1] and negatives → [-1,0]."""
    hm = hm.copy()
    pos = hm>0; neg = hm<0
    max_pos = hm[pos].max() if pos.any() else 0.0
    min_neg = hm[neg].min() if neg.any() else 0.0

    out = np.zeros_like(hm)
    if max_pos>0: out[pos] = hm[pos] / max_pos
    if min_neg<0: out[neg] = hm[neg] / (-min_neg)
    return out

def export_explanation(query_idx, query_set, support_set, heatmaps, relation_scores):
    """
    Visualize the explanation for a given query sample (query_idx) against all 5 support images.

    Parameters:
      query_idx (int): index of the query sample.
      query_set (torch.Tensor): tensor of query images with shape [5, 3, H, W].
      support_set (torch.Tensor): tensor of support images with shape [5, 3, H, W].
      heatmaps (dict): dictionary with keys (query_idx, supp_idx) and values as NumPy arrays (heatmaps).
      relation_scores (array-like): list or array of 5 scores corresponding to the relation score for the
                                    pair (query_idx, support_sample) for each support sample.
    """
    beta = 0.25 # beta = 0 pure local, beta = 1 pure global
    
    # camp for RGB and greyscale plot
    cmap = 'grey' if settings.dataset_type == 'bw' else None
    
    # Convert query image to numpy (assumed to be [0,1])
    query_img = query_set[query_idx].permute(1, 2, 0).cpu().numpy()
    
    predicted_support_idx = np.argmax(relation_scores).item()
    if predicted_support_idx == query_idx:
        print(f"Query sample {query_idx+1} correctly classified (highest score at support {predicted_support_idx+1}).")
    else:
        print(f"Query sample {query_idx+1} misclassified (highest score at support {predicted_support_idx+1}, expected {query_idx+1}).")
    
    # Convert support images to numpy
    num_support = support_set.shape[0]
    support_imgs = [support_set[i].permute(1, 2, 0).cpu().numpy() for i in range(num_support)]

    # 1) local normalize each
    local_hms = [normalize_per_heatmap(heatmaps[(query_idx,i)]) for i in range(num_support)]

    # 2) compute raw max‐abs and global max
    raw_max = np.array([np.abs(heatmaps[(query_idx,i)]).max() for i in range(num_support)])
    global_max = max(raw_max.max(), 1e-6)

    # 3) build hybrid maps
    scales = (raw_max/global_max)**beta
    norm_hm_list = [local_hms[i] * scales[i] for i in range(num_support)]

    # Create the figure: 2 rows x 6 columns.
    fig, axes = plt.subplots(2, 6, figsize=(20, 6))
    
    # Row 1, Column 1: Query image.
    axes[0, 0].imshow(query_img, cmap=cmap)
    axes[0, 0].axis("off")
    axes[0, 0].set_title(f"Query x; $y=c_{{{query_idx+1}}}$", fontsize=FONT_SIZE, fontweight='bold')
    
    # Row 1, Columns 2-6: Support images with their relation scores.
    for supp_idx in range(num_support):
        ax = axes[0, supp_idx+1]
        ax.imshow(support_imgs[supp_idx], cmap=cmap)
        ax.axis("off")
        score = relation_scores[supp_idx]
        ax.set_title(f"$s_{{{supp_idx+1}}}$; score = {score:.3f}", fontsize=FONT_SIZE, fontweight='bold')

    # Row 2, Column 1: empty for colorbar.
    axes[1, 0].axis("off")

    # Row 2, Columns 2-6: the normalized heatmaps.
    for supp_idx in range(num_support):
        ax = axes[1, supp_idx+1]
        ax.imshow(support_imgs[supp_idx], cmap=cmap, alpha=0.7)
        im = ax.imshow(norm_hm_list[supp_idx], cmap='coolwarm', norm=norm, alpha=0.7)
        ax.axis("off")
        ax.set_title(f"$e_{{{supp_idx+1}}}$; $y=c_{{{supp_idx+1}}}$", fontsize=FONT_SIZE, fontweight='bold')
        # Optionally, add a colorbar for each heatmap.
        # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Single colorbar inset into axes[1,0]
    cax = inset_axes(axes[1, 0],
                     width="5%",  # width = 5% of parent axes
                     height="80%",  # height = 80% of parent axes
                     loc='center')
    
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=FONT_SIZE * 0.8)
    
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')
    
    # plt.suptitle(f"Explanations for Query Sample {query_idx+1}", fontsize=14)
    plt.tight_layout()
    
    # Generate unique filename and save plot
    unique_name = f"{uuid.uuid4().hex}.png"
    save_dir = os.path.join(RESULT_FOLDER, settings.model_prefix, EXPLAINER, EXPLANATION_PLOT_FOLDER)
    save_path = os.path.join(save_dir, unique_name)
    fig.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    # plt.show()


##########
## MAIN ##
##########

# Useful Experimental Params

REPLACING_COLORS = {
    'white_color': np.array([0, 0, 0], dtype=np.float32),
    'black_color': np.array([1, 1, 1], dtype=np.float32),
    'grey_color': np.array([128/255, 128/255, 128/255], dtype=np.float32),
    'violet_color': np.array([238/255, 130/255, 238/255], dtype=np.float32),
    'green_color': np.array([0, 1, 50/255], dtype=np.float32)
}

EXPORT_EXPLANATION = True
EXPLANATION_PLOT_FOLDER = 'explanations/'
NUM_RESULTS_NEEDED = 200
RELATION_SCORE_THRESHOLD = 0.95
RESULT_FOLDER = 'results/'
EXPLAINER = 'RENEX' # or LIME

def main():
    # Set random seeds for reproducibility.
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if not settings.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)

    # Determine device.
    device = init_device(settings.no_cuda, settings.no_mps)
    print(f"Using device: {device}")

    # Create reults folders if don't exist
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)

    if not os.path.exists(RESULT_FOLDER + settings.model_prefix):
        os.makedirs(RESULT_FOLDER + settings.model_prefix)
    
    if not os.path.exists(RESULT_FOLDER + settings.model_prefix + '/' + EXPLAINER):
        os.makedirs(RESULT_FOLDER + settings.model_prefix + '/' + EXPLAINER)
    
    if not os.path.exists(RESULT_FOLDER + settings.model_prefix + '/' + EXPLAINER + '/' +EXPLANATION_PLOT_FOLDER):
        os.makedirs(RESULT_FOLDER + settings.model_prefix + '/' + EXPLAINER + '/' +EXPLANATION_PLOT_FOLDER)

    # Determine in_channels for network initialization
    in_channels = 1 if settings.dataset_type == 'bw' else 3
    feature_encoder = CNNEncoder(in_channels=in_channels).to(device)
    relation_network = RelationNetwork(settings.feature_dim, settings.hidden_unit, dataset_type=settings.dataset_type).to(device)
    
    # Loads models and trained weights
    model_feat_path = f"./models/{settings.model_prefix}_feature_encoder_{settings.class_num}way_{settings.train_batch_size}shot.pkl"
    model_rel_path = f"./models/{settings.model_prefix}_relation_network_{settings.class_num}way_{settings.train_batch_size}shot.pkl"
    if os.path.exists(model_feat_path):
        feature_encoder.load_state_dict(torch.load(model_feat_path, map_location=device))
        print("Loaded feature encoder.")
    if os.path.exists(model_rel_path):
        relation_network.load_state_dict(torch.load(model_rel_path, map_location=device))
        print("Loaded relation network.")

    print("Quantitative Analysis for dataset: ", settings.model_prefix)
    
    # Get image loading transformation
    transform = get_transform(settings.model_prefix)

    # Define explainer function
    EXPLAINER_FUNCTION = RENEX if EXPLAINER == 'RENEX' else LIME

    if settings.dataset_type == 'bw':
        # Black and white Omniglet has different metatest folders split than RGB datasets
        metatrain_folders, metatest_folders = get_omniglot_metafolders(settings.data_dir)

        # Mask Slic
        segmentation_func = slic
        segmentation_params = {'n_segments':10, 'compactness':100, 'sigma':1, 'mask': None, 'channel_axis': None}

        if EXPLAINER == 'LIME':
            # A Wrapper for Segmenter is needed in order to use custom segmentation params
            segmenter_name = segmentation_func.__name__
            segmentation_func = SegmentationAlgorithm(segmenter_name, **segmentation_params)

        # RENEX call
        color_key = 'black_color'# BLACK & WHITE obscure pixels with black_color (although counterintuitive)
        REPLACE_COLOR = REPLACING_COLORS[color_key] 
        
        insertion_auc_all, deletion_auc_all = [], []
        insertion_curves_all, deletion_curves_all = [], []
        execution_times_all = []

        df = []

        # Loop over tasks.
        while len(insertion_auc_all) < NUM_RESULTS_NEEDED:
            # Get random query and support image paths (one per class) from the test directory.
            query_paths, support_paths = get_omniglot_random_test_samples(metatest_folders, num_classes=settings.class_num)
            
            # Load images.
            query_images = [load_image(p,transform,device) for p in query_paths]
            support_images = [load_image(p,transform,device) for p in support_paths]
            
            # Build the query and support sets.
            query_set = torch.cat(query_images, dim=0)   # shape [5, 3, 84, 84]
            support_set = torch.cat(support_images, dim=0) # shape [5, 3, 84, 84]
            
            # Compute heatmaps and insertion/deletion scores.
            heatmaps, insertion_results, deletion_results, exec_times, predicted_scores = EXPLAINER_FUNCTION(
                                                                query_set,
                                                                support_set, 
                                                                feature_encoder, 
                                                                relation_network, 
                                                                segmentation_func, 
                                                                segmentation_params, 
                                                                REPLACE_COLOR,
                                                                device,
                                                                False)

            # Plot explanation
            if EXPORT_EXPLANATION and bool(heatmaps):
                query_idx = list(heatmaps.keys())[0][0]
                original_relations = compute_relation_score_in_task(query_set, support_set, feature_encoder, relation_network)
                export_explanation(query_idx, query_set, support_set, heatmaps, original_relations[query_idx, :].cpu().detach().numpy())
    
            
            # For each same-class pair (i.e. query_idx == supp_idx) in this task, store the scores.
            for i in range(settings.class_num):
                key = (i, i)
                if key in insertion_results:
                    insertion_auc_all.append(insertion_results[key]["auc"])
                    deletion_auc_all.append(deletion_results[key]["auc"])
                    insertion_curves_all.append(insertion_results[key]["scores"])
                    deletion_curves_all.append(deletion_results[key]["scores"])
                    
                    # Store execution times
                    execution_times_all.extend(exec_times)
                    # predicted_scores_all.extend(predicted_scores)

        # Compute average AUC values.
        avg_insertion_auc = np.mean(insertion_auc_all)
        std_insertion_auc = np.std(insertion_auc_all)
        avg_deletion_auc = np.mean(deletion_auc_all)
        std_deletion_auc = np.std(deletion_auc_all)
        avg_exec_times = np.mean(execution_times_all)
        std_exec_times = np.std(execution_times_all)

        # Storing segmentation function name for export reasons
        segmentation_function_name = segmentation_func.__name__ if hasattr(segmentation_func, '__name__') else segmentation_func.algo_type
        
        data = {
            'dataset': settings.model_prefix,
            'explainer': EXPLAINER,
            'segmentation_function': segmentation_function_name,
            'replace_color': color_key,
            'avg_insertion_auc': round(avg_insertion_auc,3),
            'std_insertion_auc': round(std_insertion_auc,3),
            'avg_deletion_auc': round(avg_deletion_auc,3),
            'std_deletion_auc': round(std_deletion_auc,3),
            'avg_exec_times': round(avg_exec_times,3),
            'std_exec_times': round(std_exec_times,3)
        }

        df.append(data)
        # Saving insertion and deletion curves to be plotted later
        ins_path = f"{RESULT_FOLDER}/{settings.model_prefix}/{EXPLAINER}/ins_{segmentation_function_name}.npz"
        del_path = f"{RESULT_FOLDER}/{settings.model_prefix}/{EXPLAINER}/del_{segmentation_function_name}.npz"  

        np.savez(ins_path, np.array(insertion_curves_all, dtype=object))
        np.savez(del_path, np.array(deletion_curves_all, dtype=object))

        csv_save_path = f"{RESULT_FOLDER}/{settings.model_prefix}_{EXPLAINER}.csv"
        pd.DataFrame(df).to_csv(csv_save_path) 
    else:
        # test directory
        test_dir = f"./data/{settings.model_prefix}/test/"

        # Define replace colors to use in experiments
        COLOR_KEYS = ['violet_color', 'green_color', 'grey_color', 'white_color', 'black_color']
        df = []
        SEGMENTERS = [
            {'f': quickshift,   'p': {"kernel_size": 5, "max_dist": 10, "ratio": 0.5}},
            {'f': felzenszwalb, 'p': {"scale": 100, "sigma": 0.5, "min_size": 50}},
            {'f': slic,         'p': {"n_segments": 30, "compactness": 100, "start_label": 1}},
        ]

        # loop over replacing colors
        for color_key in COLOR_KEYS:
            REPLACE_COLOR = REPLACING_COLORS[color_key] 
            # loop over segmenters
            for segmenter in SEGMENTERS:
                segmentation_func = segmenter['f']
                segmentation_params = segmenter['p']

                if EXPLAINER == 'LIME':
                    # A Wrapper for Segmenter is needed in order to use custom segmentation params
                    segmenter_name = segmentation_func.__name__
                    segmentation_func = SegmentationAlgorithm(segmenter_name, **segmentation_params)
            
                insertion_auc_all, deletion_auc_all = [], []
                insertion_curves_all, deletion_curves_all = [], []
                execution_times_all = []
                # predicted_scores_all = []

                # Loop over tasks.
                while len(insertion_auc_all) < NUM_RESULTS_NEEDED:
                    # Get random query and support image paths (one per class) from the test directory.
                    query_paths, support_paths = get_random_test_samples(test_dir, num_classes=settings.class_num)
                    
                    # Load images.
                    query_images = [load_image(p,transform,device) for p in query_paths]
                    support_images = [load_image(p,transform,device) for p in support_paths]
                    
                    # Build the query and support sets.
                    query_set = torch.cat(query_images, dim=0)   # shape [5, 3, 84, 84]
                    support_set = torch.cat(support_images, dim=0) # shape [5, 3, 84, 84]
                    
                    # Compute heatmaps and insertion/deletion scores.
                    heatmaps, insertion_results, deletion_results, exec_times, predicted_scores = EXPLAINER_FUNCTION(
                                                                        query_set,
                                                                        support_set, 
                                                                        feature_encoder, 
                                                                        relation_network, 
                                                                        segmentation_func, 
                                                                        segmentation_params, 
                                                                        REPLACE_COLOR,
                                                                        device,
                                                                        False)
                    
                    # Plot explanation
                    if EXPORT_EXPLANATION and bool(heatmaps):
                        query_idx = list(heatmaps.keys())[0][0]
                        original_relations = compute_relation_score_in_task(query_set, support_set, feature_encoder, relation_network)
                        export_explanation(query_idx, query_set, support_set, heatmaps, original_relations[query_idx, :].cpu().detach().numpy())
                    
                    # For each same-class pair (i.e. query_idx == supp_idx) in this task, store the scores.
                    for i in range(settings.class_num):
                        key = (i, i)
                        if key in insertion_results:
                            insertion_auc_all.append(insertion_results[key]["auc"])
                            deletion_auc_all.append(deletion_results[key]["auc"])
                            insertion_curves_all.append(insertion_results[key]["scores"])
                            deletion_curves_all.append(deletion_results[key]["scores"])
                            
                            # Store execution times
                            execution_times_all.extend(exec_times)
                            # predicted_scores_all.extend(predicted_scores)

                # Compute average AUC values.
                avg_insertion_auc = np.mean(insertion_auc_all)
                std_insertion_auc = np.std(insertion_auc_all)
                avg_deletion_auc = np.mean(deletion_auc_all)
                std_deletion_auc = np.std(deletion_auc_all)
                avg_exec_times = np.mean(execution_times_all)
                std_exec_times = np.std(execution_times_all)

                # Storing the name of the segmentation function for export reasons
                segmentation_function_name = segmentation_func.__name__ if hasattr(segmentation_func, '__name__') else segmentation_func.algo_type
                
                data = {
                    'dataset': settings.model_prefix,
                    'explainer': EXPLAINER,
                    'segmentation_function': segmentation_function_name,
                    'replace_color': color_key,
                    'avg_insertion_auc': round(avg_insertion_auc,3),
                    'std_insertion_auc': round(std_insertion_auc,3),
                    'avg_deletion_auc': round(avg_deletion_auc,3),
                    'std_deletion_auc': round(std_deletion_auc,3),
                    'avg_exec_times': round(avg_exec_times,3),
                    'std_exec_times': round(std_exec_times,3)
                }

                df.append(data)
                # Saving insertion and deletion curves to be plotted later
                ins_path = f"{RESULT_FOLDER}/{settings.model_prefix}/{EXPLAINER}/ins_{segmentation_function_name}_{color_key}.npz"
                del_path = f"{RESULT_FOLDER}/{settings.model_prefix}/{EXPLAINER}/del_{segmentation_function_name}_{color_key}.npz"

                np.savez(ins_path, np.array(insertion_curves_all, dtype=object))
                np.savez(del_path, np.array(deletion_curves_all, dtype=object))

        csv_save_path = f"{RESULT_FOLDER}/{settings.model_prefix}_{EXPLAINER}.csv"
        pd.DataFrame(df).to_csv(csv_save_path)
                
    
if __name__ == "__main__":
    main()
