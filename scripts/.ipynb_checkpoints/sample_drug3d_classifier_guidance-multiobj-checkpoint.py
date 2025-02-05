import os
import sys
import shutil
import argparse
sys.path.append('.')

from tqdm import tqdm
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

import torch
import numpy as np
import torch.utils.tensorboard
from easydict import EasyDict
from rdkit import Chem

from models.model import *
from utils.dataset import get_dataset
from utils.transforms import FeaturizeMol, Compose
from utils.transforms import *
from utils.misc import *
from utils.reconstruct import *
from utils.evaluation import *
from utils.sample import seperate_outputs, seperate_outputs_no_traj

from easydict import EasyDict
import pickle
from rdkit.Chem import Draw, AllChem


from tqdm import tqdm
import torch
from torch.nn import Module, Sequential, Linear, Conv1d, ModuleList
from torch.nn import functional as F
from models.transition import ContigousTransition, GeneralCategoricalTransition
from models.graph import NodeEdgeNet

from torch_scatter import scatter_sum, scatter_softmax
from torch_geometric.nn import radius_graph, knn_graph


from models.common import *
from models.diffusion import *
from models.model_ae import *
from models.property_predictor import *

task_target_dict = {"qed": [1.0],
                    "SAS": [0.0],
                    "SPS": [30.0],
                    "Asphericity": [1.0, 0.0],
                    "fr_halogen": [2,0, 0.0],
                    #"fr_halogen": [0.0],
                    "NumAromaticRings": [2,0, 0.0],
                    #"NumAromaticRings": [0.0],
                    "MolLogP": [5.0, 0.0],
                    "RadiusOfGyration": [6.0],
                    "PBF": [1.5],
                    "orig": [1.0],
                    "random": [1.0]}

task_list = [('qed+','SAS-'), 
             ('qed+', 'MolLogP-'), 
             ('MolLogP-','SAS-'), 
             ('qed+','fr_halogen-'),
             ('MolLogP-','fr_halogen-'),
             ('SAS-','fr_halogen-'),
             ('qed+','fr_halogen+'),
             ('MolLogP-','fr_halogen+'),
             ('SAS-','fr_halogen+'),
             ('qed+','SAS-','fr_halogen+'),
             ('qed+', 'MolLogP-','fr_halogen+'),
             ('MolLogP-','SAS-','fr_halogen+'),
             ('qed+','SAS-','MolLogP-'),
             ('qed+','SAS-','fr_halogen-'),
             ('qed+', 'MolLogP-','fr_halogen-'),
             ('MolLogP-','SAS-','fr_halogen-'),
             ('Asphericity+', 'RadiusOfGyration+'),
             ('Asphericity+', 'RadiusOfGyration-'),
             ('Asphericity-', 'RadiusOfGyration+'),
             ('Asphericity-', 'RadiusOfGyration-'),
            ]

guidance_strength_dict = {'MolLogP':0.01,
                         'qed':0.1,
                         'SAS':0.01,
                         'fr_halogen':0.1,
                         'Asphericity':0.1,
                         'RadiusOfGyration':0.1}

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  

def print_pool_status(pool, logger):
    logger.info('[Pool] Finished %d | Failed %d' % (
        len(pool.finished), len(pool.failed)
    ))


def data_exists(data, prevs):
    for other in prevs:
        if len(data.logp_history) == len(other.logp_history):
            if (data.ligand_context_element == other.ligand_context_element).all().item() and \
                (data.ligand_context_feature_full == other.ligand_context_feature_full).all().item() and \
                torch.allclose(data.ligand_context_pos, other.ligand_context_pos):
                return True
    return False

def reverse_map_noise(data, model, noise='deterministic', stride=1, n_graphs=1, max_size=None, start_step=None):
    if max_size is None:
        max_size = len(data.element)
    if start_step is None:
        time_sequence = list(range(0, model.num_timesteps-stride, stride))
    else:
        time_sequence = list(range(0, start_step, stride))
        
    batch = Batch.from_data_list([data.clone() for _ in range(n_graphs)], follow_batch = ['halfedge_type','node_type']).to(args.device)

    node_type = batch.node_type 
    node_pos = batch.node_pos
    batch_node = batch.node_type_batch
    halfedge_type = batch.halfedge_type
    halfedge_index = batch.halfedge_index
    batch_halfedge = batch.halfedge_type_batch
    num_mol = batch.num_graphs
    
    edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
    batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
    
    h_node_pert = F.one_hot(batch.node_type, model.num_node_types).float() / model.scaling[1]
    pos_pert = batch.node_pos / model.scaling[0]
    h_halfedge_pert = F.one_hot(batch.halfedge_type, model.num_edge_types).float()/ model.scaling[2]
    
    
    if noise == 'deterministic':
        for i, step in tqdm(enumerate(time_sequence), total=len(time_sequence)):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(args.device)
            h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)
            
            # # 1 inference
            preds = model(
                h_node_pert, pos_pert, batch_node,
                h_edge_pert, edge_index, batch_edge, 
                time_step, 
            )
            pred_node = preds['pred_node'].detach()  # (N, num_node_types)
            pred_pos = preds['pred_pos'].detach()  # (N, 3)
            pred_halfedge = preds['pred_halfedge'].detach()  # (E//2, num_bond_types)
        
            
            # # 2 get the t + 1 state
            pos_prev = model.pos_transition.reverse_sample_ddim(
                x_t=pos_pert, x_recon=pred_pos, t=time_step, s=time_step+stride,  batch=batch_node)
            
            h_node_prev = model.node_transition.reverse_sample_ddim(
                x_t=h_node_pert, x_recon=pred_node, t=time_step, s=time_step+stride, batch=batch_node)
            h_halfedge_prev = model.edge_transition.reverse_sample_ddim(
                x_t=h_halfedge_pert, x_recon=pred_halfedge, t=time_step, s=time_step+stride, batch=batch_halfedge)
            
            # # 3 update t-1
            pos_pert = pos_prev
            h_node_pert = h_node_prev
            h_halfedge_pert = h_halfedge_prev
            
    elif noise == 'random':
        n_nodes_all = node_type.shape[0]
        n_halfedges_all = halfedge_type.shape[0]
        h_node_init = model.node_transition.sample_init(n_nodes_all)
        pos_init = model.pos_transition.sample_init([n_nodes_all, 3])
        h_halfedge_init = model.edge_transition.sample_init(n_halfedges_all)

    
        h_node_pert = h_node_init
        pos_pert = pos_init
        h_halfedge_pert = h_halfedge_init    

    return None, h_node_pert, pos_pert, batch_node, h_halfedge_pert, halfedge_index, batch_halfedge
    
def reconstruct_ddim(h_node_pert, pos_pert, batch_node, h_halfedge_pert, halfedge_index, batch_halfedge, predictors, stride=1, n_graphs=1, start_step=None, with_edge=False, max_size=None, add_edge=None, guidance=None):
    if start_step is None:
        time_sequence = list(range(0, model.num_timesteps-stride, stride))
    else:
        time_sequence = list(range(0, start_step, stride))
    edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
    batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
    
    if max_size is None:
        max_size = h_node_pert.shape[0]    
    for i, step in tqdm(enumerate(time_sequence[::-1]), total=len(time_sequence)):
       
        time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(args.device)
        h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)
        
        # # 1 inference
        preds = model(
            h_node_pert, pos_pert, batch_node,
            h_edge_pert, edge_index, batch_edge, 
            time_step, 
        )
        pred_node = preds['pred_node'].detach()  # (N, num_node_types)
        pred_pos = preds['pred_pos'].detach()  # (N, 3)
        pred_halfedge = preds['pred_halfedge'].detach()  # (E//2, num_bond_types)
    
        # # 2 get the t + 1 state
        pos_prev = model.pos_transition.reverse_sample_ddim(
            x_t=pos_pert, x_recon=pred_pos, t=time_step, s=time_step-stride,  batch=batch_node)
        
        h_node_prev = model.node_transition.reverse_sample_ddim(
            x_t=h_node_pert, x_recon=pred_node, t=time_step, s=time_step-stride, batch=batch_node)
        h_halfedge_prev = model.edge_transition.reverse_sample_ddim(
            x_t=h_halfedge_pert, x_recon=pred_halfedge, t=time_step, s=time_step-stride, batch=batch_halfedge)
    
        pos_pert.detach().cpu()
        h_node_pert.detach().cpu()
        h_halfedge_pert.detach().cpu()

        if guidance is not None:
            for predictor, prop, gui_scale in predictors:
                prop = prop.repeat_interleave(n_graphs,0)

                gui_type, _ = guidance
                mse_loss = torch.nn.MSELoss()
                
                if (gui_scale > 0):
                    with torch.enable_grad():
                        if args.with_edge:
       
                            h_node_in = h_node_pert.detach().requires_grad_(True)
                            pos_in = pos_pert.detach().requires_grad_(True)
                            h_halfedge_in = h_halfedge_pert.detach().requires_grad_(True)
                            h_edge_in = torch.cat([h_halfedge_in, h_halfedge_in], dim=0)
    
                            pred_prop = predictor(h_node_in, 
                                                  pos_in, 
                                                  batch_node,
                                                  h_edge_in,
                                                  edge_index, 
                                                  batch_edge, 
                                                  time_step)
        
                            mse = mse_loss(pred_prop.float(), prop.float())
                            pos_delta = - torch.autograd.grad(mse, pos_in, retain_graph=True)[0] * gui_scale
                            h_node_delta = - torch.autograd.grad(mse, h_node_in, retain_graph=True)[0] * gui_scale
                            h_halfedge_delta = - torch.autograd.grad(mse, h_halfedge_in)[0] * gui_scale
                        else:
                            h_node_in = h_node_pert.detach().requires_grad_(True)
                            pos_in = pos_pert.detach().requires_grad_(True)
                            pred_prop = predictor(h_node_in, 
                                                  pos_in, 
                                                  batch_node,
                                                  edge_index, 
                                                  batch_edge, 
                                                  time_step)
        
                            mse = mse_loss(pred_prop.float(), prop.float())
                            pos_delta = - torch.autograd.grad(mse, pos_in, retain_graph=True)[0] * gui_scale
                            h_node_delta = - torch.autograd.grad(mse, h_node_in)[0] * gui_scale
                            h_halfedge_delta = 0
                        #print(pos_delta, h_node_delta)
                        
                    pos_prev = pos_prev + pos_delta
                    h_node_prev = h_node_prev + h_node_delta
                    h_halfedge_prev = h_halfedge_prev + h_halfedge_delta
        
        # # 3 update t-1
        pos_pert = pos_prev
        h_node_pert = h_node_prev
        h_halfedge_pert = h_halfedge_prev   
    
    outputs ={
                'pred': [pred_node, pred_pos, pred_halfedge],
            }
    
    outputs = {key:[v.detach().cpu().numpy() for v in value] for key, value in outputs.items() if len(value)>0}
    
    gen_list = process_outputs(outputs, batch_node, halfedge_index, batch_halfedge, n_graphs=n_graphs, add_edge=add_edge)
    return gen_list
    
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
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/sample/sample_MolDiff.yml')
    parser.add_argument('--name', type=str, default='drug3d')
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--property_set', type=str, default='logs')
    parser.add_argument('--cls_dataset', type=str, default='drug3d')
    parser.add_argument('--task_id', type=int, default=0)
    parser.add_argument('--target', type=float, default=1.0)
    parser.add_argument('--noise', type=str, default='deterministic')
    parser.add_argument('--start_step', type=int, default=999)
    parser.add_argument('--with_edge', action='store_true')
    parser.add_argument('--guidance_strength', type=float, default=1e-3)
    args = parser.parse_args()

    args.delta = ('delta' in args.property_set) # temporary

    # # Load configs
    # Load property dataset
    with open(args.property_set, 'rb') as f:
        test_split, descriptor_names = pickle.load(f)
            
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)
    # load ckpt and train config
    ckpt = torch.load(f'{args.outdir}/checkpoints/110000.pt', map_location=args.device)
    train_config = ckpt['config']
    config.dataset.root = '/home/mli/tili/mnt/MolDiffAE/data/geom_drug'

    if 'drug3d' not in args.name:
        config.dataset.name = args.name
        if 'crossdocked' in args.name:
            config.dataset.root = '/home/mli/tili/mnt/MolDiffAE/data/crossdocked'
            config.dataset.split = 'split_by_key.pt'

    # # Transform
    featurizer = FeaturizeMol(config.chem.atomic_numbers, config.chem.mol_bond_types,
                            use_mask_node=config.transform.use_mask_node,
                            use_mask_edge=config.transform.use_mask_edge, 
                          random=False
                            )
    transform = Compose([
        featurizer,
    ])

    
    dataset, subsets = get_dataset(
            config = config.dataset,
            transform = transform,
        )
    train_set, val_set, test_set = subsets['train'], subsets['val'], subsets['test']
    
    # # Model
    if train_config.model.name == 'diffusion':
        model = MolDiff(
                    config=train_config.model,
                    num_node_types=featurizer.num_node_types,
                    num_edge_types=featurizer.num_edge_types
                ).to(args.device)
    else:
        raise NotImplementedError
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # # Property predictor and guidance
    config_cls = load_config('configs/train/train_propertypred.yml')
    config_cls.transform.use_mask_edge = args.with_edge
    featurizer = FeaturizeMol(config_cls.chem.atomic_numbers, config_cls.chem.mol_bond_types,
                                    use_mask_node=config_cls.transform.use_mask_node,
                                    use_mask_edge=config_cls.transform.use_mask_edge
                                    )
    
    if args.with_edge:
        print('Utilizing edge type features')
        predictor = PropertyPredictorWithEdge(
            config=config_cls.model,
            num_node_types=featurizer.num_node_types,
            num_edge_types=featurizer.num_edge_types
        ).to(args.device)    
        #cls_ckpt = f'/home/mli/tili/mnt/MolDiffAE/model/property_predictor/PropertyPred-{idx_pp}-withedge'
        cls_ckpt_suffix = 'withedge'
    else:
        predictor = PropertyPredictor(
            config=config_cls.model,
            num_node_types=featurizer.num_node_types,
            num_edge_types=featurizer.num_edge_types
        ).to(args.device)
        #cls_ckpt = f'/home/mli/tili/mnt/MolDiffAE/model/property_predictor/PropertyPred-{idx_pp}-noedge'
        cls_ckpt_suffix = 'noedge'
    
    guidance = ('mse', args.guidance_strength)

    
    # # generating molecules
    tasks = task_list[args.task_id]
    alpha = 1
    if args.delta:
        idx_test, targets, target = test_split[tasks]
    else:
        idx_test, target = test_split[tasks]

    
    predictors = []
    for i in range(len(tasks)):

        pp = tasks[i][:-1]
        idx_pp = descriptor_names.index(pp)
        cls_ckpt = f'/home/mli/tili/mnt/MolDiffAE/model/property_predictor/PropertyPred-{idx_pp}-{cls_ckpt_suffix}'
        ckpt = torch.load(f'{cls_ckpt}/checkpoints/10000.pt', map_location='cuda')
        predictor.load_state_dict(ckpt['model'], strict=False)
        predictor.cuda()
        predictor.eval()

        prop = torch.tensor([[target[i]]]).to(args.device)
        if 'test' in args.property_set:
            predictors.append([predictor, prop, guidance_strength_dict[pp]])
        else:
            predictors.append([predictor, prop, args.guidance_strength])
            

    if args.noise == 'deterministic':
        save_name = f'property-{args.name}_multiobj-{args.task_id}_ddim-guidance-{args.cls_dataset}-{args.start_step}'
        n_graphs = 1
        
    elif args.noise == 'random':
        save_name = f'property-{args.name}_multiobj-{args.task_id}_ddim-guidance-randnoise-{args.cls_dataset}-{args.start_step}'
        n_graphs = 10

    # generation        
    gen_dict = {}
    for n, i in enumerate(idx_test):
        if args.delta:
            for k in range(len(predictors)):
                predictors[k][1] = torch.tensor([[targets[n][k]]]).to(args.device)
                print(predictors[k][1])
    #for i in range(len(test_set)):
        _ , h_node_pert, pos_pert, batch_node, h_halfedge_pert, halfedge_index, batch_halfedge = reverse_map_noise(test_set[i], model, noise=args.noise, start_step=args.start_step, n_graphs=n_graphs)
        print(h_halfedge_pert.shape)
        mol_list_manipulate = reconstruct_ddim(h_node_pert, pos_pert, batch_node, h_halfedge_pert, halfedge_index, batch_halfedge, predictors=predictors, guidance=guidance, start_step=args.start_step, with_edge=args.with_edge, n_graphs=n_graphs)

        pp_list_manipulate = []

        if len(mol_list_manipulate) > 0:
            for mol in mol_list_manipulate:
                desc = get_descriptors(mol['rdmol'])
                pp_list_manipulate.append(desc)

        gen_dict[i] = (mol_list_manipulate, pp_list_manipulate)
        #if manipulate is not None:
        #    print(i, np.mean([d[args.pp] for d in pp_list_manipulate]))

    #if 'test' in args.property_set:
    #    save_name = f'property-{args.name}_multiobj-{args.task_id}_ddim-guidance-{args.cls_dataset}-{args.start_step}'
    #else: 
    #    save_name = f'property-{args.name}_multiobj-{args.task_id}_ddim-guidance-{args.cls_dataset}-{args.start_step}-{args.guidance_strength}'
    if args.with_edge:
        save_name += '-withedge'
    if args.delta:
        save_name += '-delta'
    if 'test' in args.property_set:
        save_name += '_test'
    with open(f'{args.outdir}/{save_name}.pkl', 'wb') as f:
        pickle.dump(gen_dict, f)
    