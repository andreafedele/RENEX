import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from task_generator import get_omniglot_metafolders
from experiments import get_random_test_samples, get_omniglot_random_test_samples

FONT_SIZE = 20
RELATION_SCORE_THRESHOLD = 0.9

plt.rcParams['mathtext.default'] = 'regular'
norm = Normalize(vmin=-1, vmax=1)  # This norm will be used in the visualization

# Heatmap normalization 
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

# Plot explanation function
def plot_explanation(query_idx, query_set, support_set, heatmaps, relation_scores, settings):
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
    # plt.tight_layout()
    plt.show()

# Loads images and apply transformation accordingly
def load_image(image_path, transform, device, settings):
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
def get_transform(settings):
    if settings.dataset_type == 'rgb':
        transform = transforms.Compose([
            transforms.Resize((settings.data_resize_shape, settings.data_resize_shape)),
            transforms.ToTensor(),
        ])
    else:
        # black and white
        transform = transforms.Compose([
            transforms.Resize((settings.data_resize_shape, settings.data_resize_shape)),
            transforms.ToTensor(),
        ])

    return transform

# Computes Network relation scores in a given n-way k-shot task
def compute_relation_score_in_task(query_set, support_set, feature_encoder, relation_network, settings):
    query_feature = feature_encoder(query_set)
    support_set_feature = feature_encoder(support_set)
   
    sample_features_ext = query_feature.unsqueeze(0).repeat(support_set_feature.shape[0],1,1,1,1)
    test_features_ext = support_set_feature.unsqueeze(0).repeat(1*settings.class_num,1,1,1,1)
    test_features_ext = torch.transpose(test_features_ext,0,1)

    relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,settings.feature_dim*2,19,19)
    relations = relation_network(relation_pairs).view(-1,settings.class_num)

    return relations

# Returns task of a correctly classified query-support sample
def get_correctly_classified_task(settings, test_dir, transform, device, feature_encoder, relation_network):
    # Get random query and support image paths (one per class) from the test directory.
    if settings.dataset_type == 'rgb':
        query_paths, support_paths = get_random_test_samples(test_dir, num_classes=settings.class_num)
    else:
        # Black and white Omniglet has different metatest folders split than RGB datasets
        _, metatest_folders = get_omniglot_metafolders(settings.data_dir)
        query_paths, support_paths = get_omniglot_random_test_samples(metatest_folders, num_classes=settings.class_num)
    
    # Load images.
    query_images = [load_image(p,transform,device,settings) for p in query_paths]
    support_images = [load_image(p,transform,device,settings) for p in support_paths]
    
    # Build the query and support sets.
    query_set = torch.cat(query_images, dim=0)
    support_set = torch.cat(support_images, dim=0)
    
    # Compute original relation scores (from unperturbed samples)
    original_relations = compute_relation_score_in_task(query_set, support_set, feature_encoder, relation_network, settings)
    
    for query_idx in range(settings.class_num):
        # Compute RENEX on correct classification tasks w/ rel score higher than given threshold
        query_scores = original_relations[query_idx, :]  
        predicted_support_idx = torch.argmax(query_scores).item()
    
        if predicted_support_idx == query_idx and query_scores[predicted_support_idx].item() >= RELATION_SCORE_THRESHOLD:
            return query_set, support_set, original_relations

    return None, None, None