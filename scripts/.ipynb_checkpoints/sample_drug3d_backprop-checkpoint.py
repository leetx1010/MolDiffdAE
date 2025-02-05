import os
import shutil
import argparse
import sys
sys.path.append('.')

import torch
from torch.nn import Module
from torch.nn import functional as F

from models.transition import ContigousTransition, GeneralCategoricalTransition
from models.graph import NodeEdgeNet
from models.common import *
from models.diffusion import *
from models.model_ae import *
from models.property_predictor import *

from tqdm import tqdm
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from models.model import MolDiff
from utils.dataset import get_dataset
from utils.transforms import FeaturizeMol, Compose
from utils.transforms import *
from utils.misc import *
from utils.reconstruct import *
from utils.evaluation import *
from utils.sample import seperate_outputs, seperate_outputs_no_traj

from easydict import EasyDict

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')    

import time
import concurrent.futures

def process_outputs(outputs, batch_node_raw, halfedge_index_raw, batch_halfedge_raw, n_graphs=1, add_edge=None):
    batch_node, halfedge_index, batch_halfedge = batch_node_raw.cpu().numpy(), halfedge_index_raw.cpu().numpy(), batch_halfedge_raw.cpu().numpy()
    output_list = seperate_outputs_no_traj(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge)

    gen_list = []
    pool = EasyDict({
        'failed': [],
        'finished': [],
    })
    for i_mol, output_mol in enumerate(output_list):
        mol_info = featurizer.decode_output(
            pred_node=output_mol['pred'][0],
            pred_pos=output_mol['pred'][1],
            pred_halfedge=output_mol['pred'][2],
            halfedge_index=output_mol['halfedge_index'],
        )  # note: traj is not used
        try:
            rdmol = reconstruct_from_generated_with_edges(mol_info, add_edge=add_edge)
        except MolReconsError:
            pool.failed.append(mol_info)
            continue
        mol_info['rdmol'] = rdmol
        smiles = Chem.MolToSmiles(rdmol)
        mol_info['smiles'] = smiles
        if '.' in smiles:
            pool.failed.append(mol_info)
        else:   # Pass checks!
            gen_list.append(mol_info) 

    return gen_list
    
def predictor_backprop(data, predictor, target, steps=10, n_graphs=1, gui_scale = 0.5):
    batch = Batch.from_data_list([data.clone() for _ in range(n_graphs)], follow_batch = ['halfedge_type','node_type']).to(device)

    node_type = batch.node_type
    node_pos = batch.node_pos
    batch_node = batch.node_type_batch
    halfedge_type = batch.halfedge_type
    halfedge_index = batch.halfedge_index
    batch_halfedge = batch.halfedge_type_batch

    h_node = F.one_hot(node_type, predictor.num_node_types).float()
    pos_node = batch.node_pos
    
        
    edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
    batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)

    h_node_prev = h_node
    pos_prev = pos_node
    h_halfedge_prev = F.one_hot(halfedge_type, predictor.num_edge_types).float()
    #h_edge_prev = torch.cat([h_halfedge_prev, h_halfedge_prev], dim=0)
    
    time_step = None
    for i in range(steps):
        h_node_in = h_node_prev.detach().requires_grad_(True)
        pos_in = pos_prev.detach().requires_grad_(True)
        #h_edge_in = h_edge_prev.detach().requires_grad_(True)
        h_halfedge_in = h_halfedge_prev.detach().requires_grad_(True)
        h_edge_in = torch.cat([h_halfedge_in, h_halfedge_in], dim=0)
        
        pred_prop = predictor(h_node_in, 
                              pos_in, 
                              batch_node,
                              h_edge_in,
                              edge_index, 
                              batch_edge, 
                              time_step)
        
        prop = torch.tensor([[target]]).float().to(device)    

        mse_loss = nn.MSELoss()
        loss = mse_loss(pred_prop, prop)
        
        #loss = pred_prop
        pos_delta = torch.autograd.grad(loss, pos_in, retain_graph=True)[0] * gui_scale
        h_node_delta = torch.autograd.grad(loss, h_node_in, retain_graph=True)[0] * gui_scale
        h_halfedge_delta = torch.autograd.grad(loss, h_halfedge_in)[0] * gui_scale
        
        pos_prev = pos_prev + pos_delta
        h_node_prev = h_node_prev + h_node_delta
        h_halfedge_prev = h_halfedge_prev + h_halfedge_delta

    outputs ={
                'pred': [h_node_prev, pos_prev, h_halfedge_prev],
            }
    
    outputs = {key:[v.detach().cpu().numpy() for v in value] for key, value in outputs.items() if len(value)>0}
    
    gen_list = process_outputs(outputs, batch_node, halfedge_index, batch_halfedge, n_graphs=n_graphs, add_edge=None)
    return gen_list, pred_prop.detach().cpu().numpy()



#for task in ["qed+", "Asphericity+", "Asphericity-", "fr_halogen+", "fr_halogen-", "MolLogP+", "MolLogP-", "RadiusOfGyration+", "RadiusOfGyration-", "SAS-"]:
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/train/train_MolDiffAE_discrete.yml')
parser.add_argument('--task', type=str, default='qed+')
parser.add_argument('--gui_scale', type=float, default=10)
parser.add_argument('--name', type=str, default='drug3d')
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--property_set', type=str, default='logs')
parser.add_argument('--logdir', type=str, default='logs')
args = parser.parse_args()

args.delta = ('delta' in args.property_set) # temporary

config = load_config(args.config)

if args.delta:
    pp = args.task[:-1]
    with open(args.property_set, 'rb') as f:
        test_split, descriptor_names = pickle.load(f)
    idx_test, targets, _ = test_split[args.task]
else:
    pp = args.task[:-1]
    with open(args.property_set, 'rb') as f:
        test_split, descriptor_names = pickle.load(f)
    idx_test, target = test_split[args.task]

idx_pp = descriptor_names.index(pp)

logdir = args.logdir
task = args.task
gui_scale = args.gui_scale
steps = args.steps

print(args.name)

cls_prefix = str(idx_pp)
if 'drug3d' not in args.name:
    config.dataset.name = args.name
    if 'crossdocked' in args.name:
        config.dataset.root = '/home/mli/tili/mnt/MolDiffAE/data/crossdocked'
        config.dataset.split = 'split_by_key.pt'
        cls_prefix = f'{idx_pp}-crossdocked'

device='cuda'
print(task)
idx_pp = descriptor_names.index(task[:-1])
ckpt = torch.load(f'/home/mli/tili/mnt/MolDiffAE/model/property_predictor/PropertyPred-{cls_prefix}-clean-withedge/checkpoints/10000.pt', map_location='cuda')
config_cls = ckpt['config']

featurizer = FeaturizeMol(config_cls.chem.atomic_numbers, config_cls.chem.mol_bond_types,
                            use_mask_node=config_cls.transform.use_mask_node,
                            use_mask_edge=config_cls.transform.use_mask_edge
                            )
predictor = PropertyPredictorWithEdge(
    config=config_cls.model,
    num_node_types=featurizer.num_node_types,
    num_edge_types=featurizer.num_edge_types
).to(device)

transform = Compose([
    featurizer,
])

dataset, subsets = get_dataset(
        config = config.dataset,
        transform = transform,
    )
train_set, val_set, test_set = subsets['train'], subsets['val'], subsets['test']


predictor.load_state_dict(ckpt['model'], strict=False)
predictor.cuda()
predictor.eval()

gen_dict = {}

#if task[-1] == '-':
#    gui_scale = -gui_scale

target = test_split[task][1]

for n, i in tqdm(enumerate(idx_test)):
    if args.task not in ['random','orig']:
        if args.delta:
            target = targets[n]

    mol_list_manipulate, pred_prop = predictor_backprop(test_set[i], predictor, target, steps=steps, gui_scale=gui_scale)

    pp_list_manipulate = []

    if len(mol_list_manipulate) > 0:
        for mol in mol_list_manipulate:
            desc = get_descriptors(mol['rdmol'])
            pp_list_manipulate.append(desc)

    gen_dict[i] = (mol_list_manipulate, pp_list_manipulate)


save_name = f'property-{args.name}_{task}_backprop-{steps}-{gui_scale}'
if args.delta:
    save_name += '-delta'
if 'test' in args.property_set:
    save_name += '_test'       
with open(f'{logdir}/{save_name}.pkl', 'wb') as f:
    pickle.dump(gen_dict, f)




