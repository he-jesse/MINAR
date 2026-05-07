import sys
sys.path.append('../')
sys.path.append('./SALSA-CLRS/')
from salsaclrs import SALSACLRSDataset
from salsaclrs.data import SALSA_CLRS_DATASETS, er_probabilities

train_generators = {
    "er" : {'p_range': er_probabilities(16), 'n' : [4, 7, 11, 13, 16], 'low' : 1, 'high' : 5},
    "ba" : {'n' : [4, 7, 11, 13, 16], 'm' : [1, 2, 3], 'low' : 1, 'high' : 5},
    "sbm" : {'n' : [4, 7, 11, 13, 16], 'k' : [2, 3, 4], 'p_in' : [0.5, 0.7], 'p_out' : [0.01, 0.1], 'low' : 1, 'high' : 5},
    "complete" : {'n' : [4, 7, 11, 13, 16], 'low' : 1, 'high' : 5},
    # "path" : {'n' : [4, 7, 11, 13, 16], 'low' : 1, 'high' : 5},
    # "tree" : {'n' : [4, 7, 11, 13, 16], 'r' : [2, 3, 4], 'low' : 1, 'high' : 5},
}

algorithms = ['bfs', 'dijkstra', 'mst_prim', 'bellman_ford']
local_dir = './data_new/'
cores = -1 # Don't use multiprocessing - this avoids issues with random seeding across processes
seed = 42

for algorithm in algorithms:
    for k in SALSA_CLRS_DATASETS["test"]:
        if k[-2:] == '16' or k[-2:] == '80' or k[-3:] == '160':
            SALSACLRSDataset(ignore_all_hints=True, root=local_dir, split="test",
                             algorithm=algorithm, num_samples=1000, graph_generator=k.split("_")[0],
                             graph_generator_kwargs=SALSA_CLRS_DATASETS["test"][k],
                             nickname=k, max_cores=cores,
                             **{'seed': seed})
    print(f"Generated {algorithm} test set")
    
    SALSACLRSDataset(ignore_all_hints=True, root=local_dir, split="val", algorithm=algorithm,
                     num_samples=1000, graph_generator="er",
                     graph_generator_kwargs=SALSA_CLRS_DATASETS["val"], max_cores=cores,
                     **{'seed': seed})
    print(f"Generated {algorithm} val set")
    for generator, graph_kwargs in train_generators.items():
        SALSACLRSDataset(ignore_all_hints=False, root=local_dir, split="train", algorithm=algorithm,
                        num_samples=250, graph_generator=generator,
                        graph_generator_kwargs=graph_kwargs, max_cores=cores,
                        **{'seed': seed})
    print(f"Generated {algorithm} train set")