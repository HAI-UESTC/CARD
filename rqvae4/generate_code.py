import collections
import json
import logging
import argparse
import os
import sys

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rqvae4.datasets import EmbDataset
from rqvae4.models.rqvae import RQVAE
import pandas as pd

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

parser = argparse.ArgumentParser(description='Generate discrete codes with trained RQ-VAE')
parser.add_argument('--ckpt_path', type=str, default='', help='Path to RQVAE checkpoint .pth')
parser.add_argument('--ckpt_dir', type=str, default='', help='Directory containing checkpoint; will use best_collision_model.pth if found')
parser.add_argument('--data_path', type=str, required=True, help='Path to item_emb.parquet for target dataset')
parser.add_argument('--out_path', type=str, required=True, help='Output .npy path for generated codes')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=64)
cli = parser.parse_args()

ckpt_path = cli.ckpt_path
if not ckpt_path:
    if cli.ckpt_dir:
        cand = os.path.join(cli.ckpt_dir, 'best_collision_model.pth')
        if os.path.exists(cand):
            ckpt_path = cand
        else:
            raise FileNotFoundError(f'Checkpoint not found: {cand}. Provide --ckpt_path explicitly.')
    else:
        raise ValueError('Please provide --ckpt_path or --ckpt_dir')

device = torch.device(cli.device)

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
ckpt_args = ckpt["args"]
state_dict = ckpt["state_dict"]

data = EmbDataset(cli.data_path)

model = RQVAE(in_dim=data.dim,
                  num_emb_list=ckpt_args.num_emb_list,
                  e_dim=ckpt_args.e_dim,
                  layers=ckpt_args.layers,
                  dropout_prob=ckpt_args.dropout_prob,
                  bn=ckpt_args.bn,
                  loss_type=ckpt_args.loss_type,
                  quant_loss_weight=ckpt_args.quant_loss_weight,
                  kmeans_init=ckpt_args.kmeans_init,
                  kmeans_iters=ckpt_args.kmeans_iters,
                  sk_epsilons=ckpt_args.sk_epsilons,
                  sk_iters=ckpt_args.sk_iters,
                  )

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(model)

num_workers = getattr(ckpt_args, 'num_workers', 4)
data_loader = DataLoader(data, num_workers=num_workers,
                             batch_size=cli.batch_size, shuffle=False,
                             pin_memory=True)

all_indices = []
all_indices_str = []

for d in tqdm(data_loader):
    d = d.to(device)
    indices = model.get_indices(d, use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = index.tolist()  # pure integer code, e.g. [c1, c2, c3, c4]
        all_indices.append(code)
        all_indices_str.append(str(code))

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)

for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon=0.0

tt = 0
#There are often duplicate items in the dataset, and we no longer differentiate them
while True:
    if tt >= 30 or check_collision(all_indices_str):
        break

    collision_item_groups = get_collision_item(all_indices_str)
    print(collision_item_groups)
    print(len(collision_item_groups))
    for collision_items in collision_item_groups:
        d = data[collision_items].to(device)

        indices = model.get_indices(d, use_sk=True)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices):
            code = index.tolist()
            all_indices[item] = code
            all_indices_str[item] = str(code)
    tt += 1


print("All indices number: ", len(all_indices))
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
print("Collision Rate", (tot_item - tot_indice) / tot_item)

# Directly use integer codes, shape [N, L]
codes_array = all_indices

# Derive ItemID list for each row of codes_array.
"""We save a parallel array of ItemIDs to make code-to-item mapping explicit.

If the source parquet (cli.data_path) has an 'ItemID' column, we trust it and
use that as the item identifier for each row. Otherwise, we fall back to the
original convention that row index i corresponds to ItemID i+1.
"""
try:
    df_ids = pd.read_parquet(cli.data_path)
    if "ItemID" in df_ids.columns:
        item_ids = df_ids["ItemID"].to_numpy()
        if len(item_ids) != codes_array.shape[0]:
            print(f"[WARN] ItemID count {len(item_ids)} != codes count {codes_array.shape[0]}, falling back to 1..N.")
            item_ids = np.arange(1, codes_array.shape[0] + 1, dtype=int)
    else:
        print("[INFO] No 'ItemID' column in parquet; using 1..N as ItemIDs.")
        item_ids = np.arange(1, codes_array.shape[0] + 1, dtype=int)
except Exception as e:
    print(f"[WARN] Failed to read ItemID from parquet ({e}); using 1..N as ItemIDs.")
    item_ids = np.arange(1, codes_array.shape[0] + 1, dtype=int)

# save the codes and item_ids to numpy files
print(f"Saving codes to {cli.out_path}")
print(f"the first 5 codes: {codes_array[:5]}")
np.save(cli.out_path, codes_array)

base, ext = os.path.splitext(cli.out_path)
ids_out = base + "_item_ids.npy"
print(f"Saving item IDs to {ids_out}")
print(f"the first 5 item IDs: {item_ids[:5]}")
np.save(ids_out, item_ids)

