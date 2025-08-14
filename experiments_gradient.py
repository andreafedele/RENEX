import os
import time
import uuid
import random
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients

import settings
from utils import init_device
from task_generator import get_omniglot_metafolders
from relation_network import CNNEncoder, RelationNetwork

# Plot utils
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Ensure mathtext uses bold correctly
plt.rcParams['mathtext.default'] = 'regular'
norm = Normalize(vmin=-1, vmax=1)  # This norm will be used in the visualization
FONT_SIZE = 20

from experiments import (
    load_image, get_transform, compute_relation_score_in_task,
    REPLACING_COLORS, 
    get_random_test_samples, get_omniglot_random_test_samples,
    normalize_per_heatmap
)

def export_ig_explanation_strong_rb_new(query_idx, query_set, support_set, heatmaps, relation_scores,
                                    out_dir, explainer_folder,
                                    bg_alpha=0.06,
                                    cmap_signed='coolwarm',
                                    max_alpha=1.0,
                                    gamma=4.0,
                                    top_percentile=85.0,
                                    outline=False,
                                    dilate_size=5,
                                    dpi=250):
    try:
        from scipy.ndimage import maximum_filter, binary_dilation, binary_erosion
        has_scipy = True
    except Exception:
        has_scipy = False
        maximum_filter = None
        binary_dilation = None
        binary_erosion = None

    beta = 0.25
    bg_cmap = 'gray' if settings.dataset_type == 'bw' else None

    query_img = query_set[query_idx].permute(1,2,0).cpu().numpy()
    num_support = support_set.shape[0]
    support_imgs = [support_set[i].permute(1,2,0).cpu().numpy() for i in range(num_support)]

    # normalized local maps in [-1,1] preserving sign
    local_hms = [normalize_per_heatmap(heatmaps[(query_idx,i)]) for i in range(num_support)]
    raw_max = np.array([np.abs(heatmaps[(query_idx,i)]).max() for i in range(num_support)])
    global_max = max(raw_max.max(), 1e-6)
    scales = (raw_max/global_max)**beta
    norm_hm_list = [local_hms[i] * scales[i] for i in range(num_support)]

    fig, axes = plt.subplots(2, 6, figsize=(20, 6))

    # Row 1: query + supports
    axes[0,0].imshow(query_img, cmap=bg_cmap, interpolation='nearest')
    axes[0,0].axis("off")
    axes[0,0].set_title(f"Query x; $y=c_{{{query_idx+1}}}$", fontsize=FONT_SIZE, fontweight='bold')
    for supp_idx in range(num_support):
        ax = axes[0, supp_idx+1]
        ax.imshow(support_imgs[supp_idx], cmap=bg_cmap, interpolation='nearest')
        ax.axis("off")
        score = relation_scores[supp_idx]
        ax.set_title(f"$s_{{{supp_idx+1}}}$; score = {score:.3f}", fontsize=FONT_SIZE, fontweight='bold')

    axes[1,0].axis("off")

    # ScalarMappable for colorbar (signed -1..1)
    mappable = ScalarMappable(norm=norm, cmap=cmap_signed)
    mappable.set_array(np.linspace(-1.0, 1.0, 256))

    cmap = plt.get_cmap(cmap_signed)

    for supp_idx in range(num_support):
        ax = axes[1, supp_idx+1]
        ax.imshow(support_imgs[supp_idx], cmap=bg_cmap, alpha=bg_alpha, interpolation='nearest')

        signed_map = norm_hm_list[supp_idx]  # [-1,1]
        pos = np.clip(signed_map, 0.0, 1.0)
        neg = -np.clip(signed_map, None, 0.0)
        mag = np.abs(signed_map)  # 0..1

        # amplify magnitude non-linearly (stronger than before)
        mag_enh = np.clip(mag, 0.0, 1.0) ** gamma

        # threshold for top pixels (more inclusive)
        thresh = np.percentile(mag_enh, top_percentile) if mag_enh.size > 0 else 1.0

        # background alpha (faint) and strong alpha (for tops)
        background_alpha = mag_enh.copy()
        # make low-level tint slightly stronger so non-top pixels are more visible
        background_alpha[background_alpha >= thresh] *= 0.05   # top pixels reduced in bg tint
        background_alpha[background_alpha < thresh] *= 0.35    # low-level tint a bit stronger than before

        strong_alpha = np.zeros_like(mag_enh)
        if mag_enh.max() > thresh:
            strong_alpha[mag_enh >= thresh] = (mag_enh[mag_enh >= thresh] - thresh) / (mag_enh.max() - thresh + 1e-12)
            strong_alpha = np.clip(strong_alpha, 0.0, 1.0) * max_alpha

        # dilation to make top pixels chunkier (if scipy available)
        if has_scipy and dilate_size and dilate_size > 1:
            try:
                strong_alpha = maximum_filter(strong_alpha, size=dilate_size)
                background_alpha = maximum_filter(background_alpha, size=1)
            except Exception:
                pass

        # Map positives to [0.5,1.0] and negatives to [0.5,0.0] of cmap
        if pos.max() > 0:
            pos_norm = pos
            # slight contrast stretch before mapping to colorvals (helps reds pop)
            pos_norm = np.clip((pos_norm - pos_norm.min()) / (pos_norm.max() - pos_norm.min() + 1e-12), 0.0, 1.0)
            pos_norm = pos_norm ** 1.1  # small extra boost
            pos_color_vals = 0.5 + 0.5 * pos_norm  # 0.5..1.0
            pos_rgba = cmap(pos_color_vals)
            pos_alpha = (background_alpha * (pos > 0)) + (strong_alpha * (pos > 0))
            pos_rgba[..., -1] = np.clip(pos_alpha, 0.0, 1.0)
            ax.imshow(pos_rgba, interpolation='nearest', zorder=3)

        if neg.max() > 0:
            neg_norm = neg
            neg_norm = np.clip((neg_norm - neg_norm.min()) / (neg_norm.max() - neg_norm.min() + 1e-12), 0.0, 1.0)
            neg_norm = neg_norm ** 1.1
            neg_color_vals = 0.5 - 0.5 * neg_norm  # 0.5..0.0
            neg_rgba = cmap(neg_color_vals)
            neg_alpha = (background_alpha * (neg > 0)) + (strong_alpha * (neg > 0))
            neg_rgba[..., -1] = np.clip(neg_alpha, 0.0, 1.0)
            ax.imshow(neg_rgba, interpolation='nearest', zorder=3)

        # optional outline around strong regions to boost contrast
        if outline:
            strong_mask = strong_alpha > (0.15 * max_alpha)
            if has_scipy and binary_dilation is not None:
                try:
                    # make a thin boundary
                    dil = binary_dilation(strong_mask, iterations=1)
                    er = binary_erosion(strong_mask, iterations=1)
                    boundary = np.logical_and(dil, ~er)
                except Exception:
                    boundary = strong_mask
            else:
                # fallback: boundary = perimeter pixels (rough)
                # from scipy import ndimage as _ndi if False else None
                boundary = strong_mask

            if boundary.any():
                # draw a subtle dark border to separate overlay from background
                border_overlay = np.zeros((boundary.shape[0], boundary.shape[1], 4), dtype=float)
                border_overlay[..., :3] = 0.0  # black
                border_overlay[..., -1] = boundary.astype(float) * 0.6
                ax.imshow(border_overlay, interpolation='nearest', zorder=3.5)

        # faint halo below very-strong pixels to increase legibility
        halo_mask = strong_alpha > (0.85 * max_alpha)
        if halo_mask.any():
            halo = np.zeros((halo_mask.shape[0], halo_mask.shape[1], 4), dtype=float)
            halo[..., :3] = 1.0
            halo[..., -1] = (strong_alpha * 0.12) * halo_mask
            ax.imshow(halo, interpolation='nearest', zorder=2.5)

        ax.axis("off")
        ax.set_title(f"$e_{{{supp_idx+1}}}$; $y=c_{{{supp_idx+1}}}$", fontsize=FONT_SIZE, fontweight='bold')

    # colorbar
    cax = inset_axes(axes[1, 0], width="5%", height="80%", loc='center')
    cbar = fig.colorbar(mappable, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=FONT_SIZE * 0.8)
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')

    plt.tight_layout()

    save_dir = os.path.join(out_dir, explainer_folder)
    os.makedirs(save_dir, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}.png"
    save_path = os.path.join(save_dir, unique_name)
    fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f"IG explanation (strong RB tuned) saved to {save_path}")
    return save_path

# Gradient-based explainer using Integrated Gradients
def compute_ig_heatmap(query_idx, supp_idx, query_set, support_set,
                       feature_encoder, relation_network, device, baseline):
    # feature_encoder.eval()
    # relation_network.eval()
    
    def rel_score(inputs):
        # inputs: batch of support images
        # Build feature for query repeated and inputs
        batch = inputs
        # Compute features
        q_feat = feature_encoder(query_set)  # [Nq, C, Hf, Wf]
        s_feat = feature_encoder(batch)      # [Nb, C, Hf, Wf]
        # Expand to relation pairs
        sample_ext = q_feat.unsqueeze(0).repeat(s_feat.size(0),1,1,1,1)
        test_ext = s_feat.unsqueeze(0).repeat(query_set.size(0),1,1,1,1)
        test_ext = torch.transpose(test_ext,0,1)
        pairs = torch.cat((sample_ext, test_ext),2).view(-1, settings.feature_dim*2, 19, 19)
        # relation scores
        rels = relation_network(pairs).view(-1, settings.class_num)
        # return scores for supp_idx
        return rels[:, supp_idx]

    ig = IntegratedGradients(rel_score)
    support_img = support_set[supp_idx].unsqueeze(0).to(device)
    attributions, _ = ig.attribute(
        inputs=support_img,
        baselines=baseline.unsqueeze(0).to(device),
        return_convergence_delta=True
    )

    # Sum signed attributions across channels to preserve positive vs negative importance
    hm = attributions.sum(dim=1).squeeze(0).cpu().detach().numpy()

    return hm # not normalized

# Pixel-level insertion/deletion based on IG heatmap
def compute_insertion_score_ig(query_idx, supp_idx, query_set, support_set,
                               orig_support_img, heatmap,
                               feature_encoder, relation_network, device, replace_value):
    
    H, W, _ = orig_support_img.shape
    # Flatten pixel indices sorted by descending heatmap value
    flat_idx = np.argsort(-heatmap.flatten())
    # Start with blank canvas
    composite = np.full_like(orig_support_img, replace_value)
    scores = []

    def build_support(modified_img):
        mod_t = torch.tensor(modified_img).permute(2,0,1).unsqueeze(0).float().to(device)
        batch = []
        for i in range(support_set.shape[0]):
            batch.append(mod_t if i==supp_idx else support_set[i].unsqueeze(0))
        return torch.cat(batch,0)

    # initial score
    relations = compute_relation_score_in_task(query_set, build_support(composite), feature_encoder, relation_network)
    scores.append(relations[query_idx, supp_idx].item())

    # Insert pixels one by one
    for idx in flat_idx:
        i, j = divmod(idx, W)
        composite[i,j,:] = orig_support_img[i,j,:]
        relations = compute_relation_score_in_task(query_set, build_support(composite), feature_encoder, relation_network)
        scores.append(relations[query_idx, supp_idx].item())

    fractions = np.linspace(0,1,len(scores))
    auc = np.trapz(scores, fractions)
    return auc, scores

def compute_deletion_score_ig(query_idx, supp_idx, query_set, support_set,
                              orig_support_img, heatmap,
                              feature_encoder, relation_network, device, replace_value):
    H, W, _ = orig_support_img.shape
    flat_idx = np.argsort(-heatmap.flatten())
    composite = orig_support_img.copy()
    scores = []

    def build_support(modified_img):
        mod_t = torch.tensor(modified_img).permute(2,0,1).unsqueeze(0).float().to(device)
        batch = []
        for i in range(support_set.shape[0]):
            batch.append(mod_t if i==supp_idx else support_set[i].unsqueeze(0))
        return torch.cat(batch,0)

    relations = compute_relation_score_in_task(query_set, build_support(composite), feature_encoder, relation_network)
    scores.append(relations[query_idx, supp_idx].item())

    for idx in flat_idx:
        i, j = divmod(idx, W)
        composite[i,j,:] = replace_value
        relations = compute_relation_score_in_task(query_set, build_support(composite), feature_encoder, relation_network)
        scores.append(relations[query_idx, supp_idx].item())

    fractions = np.linspace(0,1,len(scores))
    auc = np.trapz(scores, fractions)
    return auc, scores

##########
## MAIN ##
##########

EXPORT_EXPLANATION = True
EXPLANATION_PLOT_FOLDER = 'explanations/'
NUM_RESULTS_NEEDED = 200
RELATION_SCORE_THRESHOLD = 0.9
RESULT_FOLDER = 'results/'
EXPLAINER = 'IntegratedGradients'

def main():
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if not settings.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)

    device = init_device(settings.no_cuda, settings.no_mps)
    print(f"Using device: {device}")

    os.makedirs(RESULT_FOLDER, exist_ok=True)
    out_dir = os.path.join(RESULT_FOLDER, settings.model_prefix, EXPLAINER)
    os.makedirs(os.path.join(out_dir, EXPLANATION_PLOT_FOLDER), exist_ok=True)

    in_ch = 1 if settings.dataset_type=='bw' else 3
    fe = CNNEncoder(in_channels=in_ch).to(device)
    rn = RelationNetwork(settings.feature_dim, settings.hidden_unit, dataset_type=settings.dataset_type).to(device)
    feat_path = f"./models/{settings.model_prefix}_feature_encoder_{settings.class_num}way_{settings.train_batch_size}shot.pkl"
    rel_path  = f"./models/{settings.model_prefix}_relation_network_{settings.class_num}way_{settings.train_batch_size}shot.pkl"
    if os.path.exists(feat_path): fe.load_state_dict(torch.load(feat_path,map_location=device))
    if os.path.exists(rel_path):  rn.load_state_dict(torch.load(rel_path,map_location=device))

    transform = get_transform(settings.model_prefix)
    replace_value = REPLACING_COLORS['grey_color'][0] if settings.dataset_type=='bw' else REPLACING_COLORS['grey_color']

    insertion_all, deletion_all, times_all = [], [], []
    deletion_curves_all, insertion_curves_all = [], []

    sampler_bw = get_omniglot_random_test_samples if settings.dataset_type=='bw' else None
    sampler_rgb = get_random_test_samples if settings.dataset_type!='bw' else None

    while len(insertion_all) < NUM_RESULTS_NEEDED:
        print(len(insertion_all))
        if settings.dataset_type=='bw':
            qps, sps = sampler_bw(get_omniglot_metafolders(settings.data_dir)[1], settings.class_num)
        else:
            qps, sps = sampler_rgb(f"./data/{settings.model_prefix}/test/", settings.class_num)

        qs = [load_image(p, transform, device) for p in qps]
        ss = [load_image(p, transform, device) for p in sps]
        query_set = torch.cat(qs); 
        support_set = torch.cat(ss)

        # Compute original relation scores once per task
        original_relations = compute_relation_score_in_task(query_set, support_set, fe, rn)
        for q in range(settings.class_num):
            # Check correct classification & threshold
            q_scores = original_relations[q]
            pred_idx = torch.argmax(q_scores).item()
            if pred_idx != q or q_scores[pred_idx].item() < RELATION_SCORE_THRESHOLD:
                continue

            H, W = query_set.shape[-2:]
            if settings.dataset_type == 'bw':
                # single-channel baseline for black-and-white
                baseline = torch.full((1, H, W), replace_value, device=device)
            else:
                # 3-channel baseline for RGB
                baseline = torch.tensor(replace_value, device=device).view(3,1,1).expand(3, H, W)

            # store IG heatmaps
            heatmaps = {}
            print("Computing heatmaps")

            # Only explain the correctly classified query
            for s in range(settings.class_num):
                # compute and time IG attribution
                start = time.process_time()
                hm = compute_ig_heatmap(q, s, query_set, support_set, fe, rn, device, baseline)
                end = time.process_time()

                # collect per-pair heatmap
                heatmaps[(q, s)] = hm

                if q == s:
                    ins_auc, ins_scores = compute_insertion_score_ig(
                        q, s, query_set, support_set,
                        support_set[s].permute(1,2,0).cpu().numpy(), hm,
                        fe, rn, device, replace_value)
                    
                    del_auc, del_scores = compute_deletion_score_ig(
                        q, s, query_set, support_set,
                        support_set[s].permute(1,2,0).cpu().numpy(), hm,
                        fe, rn, device, replace_value)
                    
                    insertion_all.append(ins_auc)
                    deletion_all.append(del_auc)
                    insertion_curves_all.append(ins_scores)
                    deletion_curves_all.append(del_scores)
                    times_all.append(end - start)

            # Export the multi-support explanation 
            if EXPORT_EXPLANATION and bool(heatmaps):
                # original relations for titles / scores
                original_relations = original_relations  # already computed above for task
                export_ig_explanation_strong_rb_new(q, query_set, support_set, heatmaps, original_relations[q, :].cpu().detach().numpy(), out_dir, EXPLANATION_PLOT_FOLDER)

    # Save curves for later plotting
    ins_path = os.path.join(out_dir, f"ins_IG.npz")
    del_path = os.path.join(out_dir, f"del_IG.npz")
    np.savez(ins_path, np.array(insertion_curves_all, dtype=object))
    np.savez(del_path, np.array(deletion_curves_all, dtype=object))

    # Aggregate and save summary CSV
    data = {
        'dataset': settings.model_prefix,
        'explainer': 'IntegratedGradients',
        'avg_insertion_auc': round(np.mean(insertion_all),3),
        'std_insertion_auc': round(np.std(insertion_all),3),
        'avg_deletion_auc': round(np.mean(deletion_all),3),
        'std_deletion_auc': round(np.std(deletion_all),3),
        'avg_exec_times': round(np.mean(times_all),3),
        'std_exec_times': round(np.std(times_all),3)
    }

    pd.DataFrame([data]).to_csv(f"{RESULT_FOLDER}/{settings.model_prefix}_IG.csv", index=False)

if __name__=='__main__':
    main()
