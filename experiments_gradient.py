import os
import time
import random
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients

import settings
from utils import init_device
from task_generator import get_omniglot_metafolders
from relation_network import CNNEncoder, RelationNetwork

from experiments import (
    load_image, get_transform, compute_relation_score_in_task,
    REPLACING_COLORS, 
    get_random_test_samples, get_omniglot_random_test_samples
)

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

EXPORT_EXPLANATION = False
EXPLANATION_PLOT_FOLDER = 'explanations/'
NUM_RESULTS_NEEDED = 100
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


            # Only explain the correctly classified query
            for s in range(settings.class_num):
                # compute and time IG attribution
                start = time.process_time()
                hm = compute_ig_heatmap(q, s, query_set, support_set, fe, rn, device, baseline)
                end = time.process_time()

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
