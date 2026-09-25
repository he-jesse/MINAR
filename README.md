# MINAR: Mechanistic Interpretability for Neural Algorithmic Reasoning

This contains the code for the paper MINAR: Mechanistic Interpretability for Neural Algorithmic Reasoning. The code was developed and tested using Python 3.12.3. The package requirements are listed in `environment.yml`.

## Instructions

### Bellman-Ford

Data generation can be found in the folder `bellman_ford_experiments/data`.

Bellman-Ford experiments are divided into subfolders. Each subfolder contains 
1. A training notebook
2. A circuit analysis notebook, and
3. A plotting notebook.
In each notebook, simply run all cells. Note that model training may take up to 30 min on the hardware reported in the paper.

### SALSA-CLRS
The SALSA-CLRS experiments are contained in the folder `new_experiments`.
1. Data for BFS, DFS, Dijkstra's algorithm, and Prim's MST are from SALSA-CLRS (https://arxiv.org/abs/2309.12253)
2. A data generation script for Bellman-Ford, Articulation Points, and Bridges is provided in `generate_new_salsa_clrs_data.py`.
3. Clean/corrupted data generation is in `generate_corrupted_data.py`.
4. The model training script used is `train_salsa_clrs_distributed_l1_schedule.py`. (NOTE: The script assumes access to 7 GPUs to perform multi-GPU parallel training.)
5. Plotting of model progress is in `salsa_clrs_plots.ipynb`
6. Circuit analysis is performed in `salsa-clrs_circuits.ipynb`. Since identifying large circuits can be time-consuming, an alternative script is given in `compute_salsa-clrs_circuits.py` which can be run in the background or in parallel.
7. Evaluation of the BFS circuit is performed in `salsa-clrs_bfs_performance.ipynb`
8. Ablation studies can be found in `salsa-clrs_ablations.ipynb` and `salsa-clrs_circuits_corruptions.ipynb`

                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830
