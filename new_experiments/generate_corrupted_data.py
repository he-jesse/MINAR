import sys
sys.path.append('../')
sys.path.append('./SALSA-CLRS/')
from salsaclrs import SALSACLRSDataset
from salsaclrs.data import SALSA_CLRS_DATASETS

import os
import torch

algorithms = ['bfs', 'dfs', 'dijkstra', 'mst_prim', 'bellman_ford', 'articulation_points', 'bridges']
local_dir = './data/'
clean_data_dir = './data_clean/'
corr_data_dir = './data_corrupted/'
rand_data_dir = './data_random/'

for algorithm in algorithms:
    print(f'Generating corrupted data for {algorithm}...')
    os.makedirs(f'{clean_data_dir}/{algorithm}', exist_ok=True)
    os.makedirs(f'{corr_data_dir}/{algorithm}', exist_ok=True)
    os.makedirs(f'{rand_data_dir}/{algorithm}', exist_ok=True)
    test_data = SALSACLRSDataset(ignore_all_hints=True, root=local_dir, split="test",
                                algorithm=algorithm, num_samples=1000, graph_generator="er",
                                graph_generator_kwargs=SALSA_CLRS_DATASETS["test"]["er_16"],
                                nickname="er_16")
    for i, data in enumerate(test_data):
        torch.save(data, f'{clean_data_dir}/{algorithm}/data_{i}.pt')
        data_corr = data.clone()
        for key in data.inputs:
            corrupted_input = torch.zeros_like(data[key])
            data_corr[key] = corrupted_input
        if hasattr(data, 'weights') :
            corrupted_weights = torch.zeros_like(data.weights)
            data_corr.weights = corrupted_weights
        torch.save(data_corr, f'{corr_data_dir}/{algorithm}/data_{i}.pt')

        data_rand = data.clone()
        for key in data.inputs:
            random_input = torch.rand_like(data[key])
            data_rand[key] = random_input
        if hasattr(data, 'weights') :
            random_weights = torch.rand_like(data.weights)
            data_rand.weights = random_weights
        torch.save(data_rand, f'{rand_data_dir}/{algorithm}/data_{i}.pt')