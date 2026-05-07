import sys
sys.path.append('../')
sys.path.append('./SALSA-CLRS/')

import os
import argparse

import torch
from model.GINE import GINE
from model.RecGINE import RecGINE

from SALSACLRSComputationGraph import SALSACLRSComputationGraph
from SALSACLRSComputationGraph import SALSACLRSCircuit
import networkx as nx

from loguru import logger
from baselines.core.models.encoder import Encoder
from baselines.core.models.decoder import Decoder, grab_outputs, output_mask
from baselines.core.models.processor import Processor
from baselines.core.loss import CLRSLoss
from salsaclrs import specs
from baselines.core.metrics import calc_metrics
from salsaclrs.data import SALSACLRSDataLoader
from torch_geometric.loader import DataLoader
from EncodeProcessDecode import EncodeProcessDecode

args = argparse.ArgumentParser()
args.add_argument('--checkpoint_name', type=str, help='Name of the checkpoint (without .pt extension) to use for naming the circuit file')
args.add_argument('--model_checkpoint', type=str, help='Path to the model checkpoint to use for circuit construction')
args.add_argument('--computation_graph', type=str, help='Path to the scored computation graph to use for circuit construction')
args.add_argument('--circuit_algorithm', type=str, default='longest_path', help='Algorithm to use for circuit construction')
args.add_argument('--prune', type=bool, default=True, help='Whether to prune the circuit after construction')
args.add_argument('--K', type=int, default=100, help='Number of edges to add the circuit')
args.add_argument('--score', type=str, default='weight', help='Scoring method to use')
args = args.parse_args()

print(f'Using {args.circuit_algorithm} to construct {args.score} circuit with {args.K} edges from model checkpoint: {args.checkpoint_name}')

device = 'cpu'
algorithms = ['bfs', 'dfs', 'dijkstra', 'mst_prim', 'bellman_ford', 'articulation_points', 'bridges']
output_types = {
    'bfs' : 'pointer',
    'dfs' : 'pointer',
    'dijkstra' : 'pointer',
    'mst_prim' : 'pointer',
    'bellman_ford' : 'pointer',
    'articulation_points' : 'mask',
    'bridges' : 'pointer',
}
logger.disable('baselines.core.models.encoder')
logger.disable('baselines.core.models.decoder')

hidden_dim = 128
encoders = torch.nn.ModuleDict({
    task : Encoder(specs=specs.SPECS[task]) for task in algorithms
})

decoders = torch.nn.ModuleDict({
    task : Decoder(specs=specs.SPECS[task], 
                   hidden_dim = hidden_dim * 2,
                   no_hint=False) for task in algorithms
})

for encoder in encoders.values():
    encoder.to(device)
for decoder in decoders.values():
    decoder.to(device)
processor = GINE(3*128, 128, 2, 128, edge_dim=1, aggr='max')
processor.to(device)
model = EncodeProcessDecode(encoders, decoders, processor, device=device)

args.model_checkpoint
model_state = torch.load(args.model_checkpoint, map_location=device)
model.load_state_dict(model_state)
model.eval()
model.to(device)

G = SALSACLRSComputationGraph(model, special_modules=['convs.0.lin', 'convs.1.lin'])
G.add_module('convs.0.lin', processor.convs[0].lin,
            module_inputs='edge_attr',
            module_outputs=0,
            layer=0)
G.add_module('convs.1.lin', processor.convs[1].lin,
            module_inputs='edge_attr',
            module_outputs=2,
            layer=0)
G.correct_layers()

G_scores = torch.load(args.computation_graph, weights_only=False, map_location=device)
for edge, data in G_scores.items():
    G.add_edge(*edge, **data)

circuit_path = f'salsa_clrs_circuits_{args.circuit_algorithm}_new_prune={args.prune}_{args.checkpoint_name}_{args.score}_{args.K}.pt'
circuits = {}
if os.path.exists(f'circuits/{circuit_path}'):
    circuits = torch.load(f'circuits/{circuit_path}', weights_only=False, map_location=device)

if args.score == 'weight':
    weight_circuit = SALSACLRSCircuit(model, G, args.K, key='weight', circuit_algorithm=args.circuit_algorithm, prune=args.prune)

for algorithm in algorithms:
    if (algorithm, args.K, args.score) in circuits:
        print(f'Circuit for {algorithm} with K={args.K} and score method {args.score} already exists. Skipping...')
    elif args.score == 'weight':
        circuits[(algorithm, args.K, args.score)] = weight_circuit
        print(f'{algorithm} circuit with K={args.K} and score method {args.score}: {circuits[(algorithm, args.K, args.score)].number_of_edges()} edges')
    else:
        circuits[(algorithm, args.K, args.score)] = SALSACLRSCircuit(model, G, args.K, key=f'{args.score}_{algorithm}', circuit_algorithm=args.circuit_algorithm, prune=args.prune)
        print(f'{algorithm} circuit with K={args.K} and score method {args.score}: {circuits[(algorithm, args.K, args.score)].number_of_edges()} edges')

# Remove model and G references to avoid pickling the model
for circuit in circuits.values():
    circuit.G = None
    circuit.model = None
    circuit.EncodeProcessDecode = None
torch.save(circuits, f'circuits/{circuit_path}')