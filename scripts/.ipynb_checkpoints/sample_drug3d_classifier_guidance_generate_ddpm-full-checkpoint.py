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
            """
            p_save_traj = np.random.rand()  # save traj
            if p_save_traj <  config.sample.save_traj_prob:
                traj_info = [featurizer.decode_output(
                    pred_node=output_mol['traj'][0][t],
                    pred_pos=output_mol['traj'][1][t],
                    pred_halfedge=output_mol['traj'][2][t],
                    halfedge_index=output_mol['halfedge_index'],
                ) for t in range(len(output_mol['traj'][0]))]
                mol_traj = []
                for t in range(len(traj_info)):
                    try:
                        mol_traj.append(reconstruct_from_generated_with_edges(traj_info[t], False, add_edge=add_edge))
                    except MolReconsError:
                        mol_traj.append(Chem.MolFromSmiles('O'))
                mol_info['traj'] = mol_traj
            """
            gen_list.append(mol_info) 

    return gen_list
    
def sample_from_property(prop, model, mode='template', n_graphs=10, max_size=None, guidance=None, bond_predictor=None, manipulate=None, start_step=None, predictor=None, stride=1, add_edge=None):
    if max_size is None:
        max_size = random.randint(20, 40)
    
    batch_holder = make_data_placeholder(n_graphs=n_graphs, device=device, max_size=max_size)
    batch_node, halfedge_index, batch_halfedge = batch_holder['batch_node'], batch_holder['halfedge_index'], batch_holder['batch_halfedge']
        
    n_nodes_all = len(batch_node)
    n_halfedges_all = len(batch_halfedge)
    
    node_init = model.node_transition.sample_init(n_nodes_all)
    pos_init = model.pos_transition.sample_init([n_nodes_all, 3])
    halfedge_init = model.edge_transition.sample_init(n_halfedges_all)
    
    h_node_init = node_init
    h_halfedge_init = halfedge_init

    h_node_pert = h_node_init
    pos_pert = pos_init
    h_halfedge_pert = h_halfedge_init
    edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
    batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)

    prop = prop.repeat_interleave(n_graphs,0)
    # inference
    time_sequence = list(range(0, model.num_timesteps, stride))

        
    for i, step in tqdm(enumerate(time_sequence[::-1]), total=len(time_sequence)):
       
        time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)
        h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)
        
        # # 1 inference
        preds = model(
            h_node_pert, pos_pert, batch_node,
            h_edge_pert, edge_index, batch_edge, 
            time_step
        )
        pred_node = preds['pred_node'].detach()  # (N, num_node_types)
        pred_pos = preds['pred_pos'].detach()  # (N, 3)
        pred_halfedge = preds['pred_halfedge'].detach()  # (E//2, num_bond_types)
    
        # # 2 get the t + 1 state
        pos_prev = model.pos_transition.get_prev_from_recon(
            x_t=pos_pert, x_recon=pred_pos, t=time_step, batch=batch_node)
        h_node_prev = model.node_transition.get_prev_from_recon(
            x_t=h_node_pert, x_recon=pred_node, t=time_step, batch=batch_node)
        h_halfedge_prev = model.edge_transition.get_prev_from_recon(
            x_t=h_halfedge_pert, x_recon=pred_halfedge, t=time_step, batch=batch_halfedge)
        
        if guidance is not None:
            gui_type, gui_scale = guidance
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
    
        pos_pert.detach().cpu()
        h_node_pert.detach().cpu()
        h_halfedge_pert.detach().cpu()
    
            
        # # 3 update t-1
        pos_pert = pos_prev
        h_node_pert = h_node_prev
        h_halfedge_pert = h_halfedge_prev   
    
    outputs ={
                'pred': [pred_node, pred_pos, pred_halfedge],
            }
    outputs = {key:[v.cpu().numpy() for v in value] for key, value in outputs.items()}

    # decode outputs to molecules
    gen_list = process_outputs(outputs, batch_node, halfedge_index, batch_halfedge, n_graphs=n_graphs, add_edge=add_edge)

    return gen_list
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/sample/sample_MolDiff.yml')
    parser.add_argument('--name', type=str, default='drug3d')
    parser.add_argument('--mode', type=str, default='template')
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--property_set', type=str, default='logs')
    parser.add_argument('--task', type=str, default='qed+')
    parser.add_argument('--target', type=float, default=1.0)
    parser.add_argument('--start_step', type=int, default=999)
    parser.add_argument('--with_edge', action='store_true')
    parser.add_argument('--prop_normalize', action='store_true')
    parser.add_argument('--guidance_strength', type=float, default=1e-3)
    args = parser.parse_args()

    args.delta = ('delta' in args.property_set) # temporary

    # # Load configs    
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)
    # load ckpt and train config
    ckpt = torch.load(f'{args.outdir}/checkpoints/110000.pt', map_location=args.device)
    train_config = ckpt['config']
    config.dataset.root = '/home/mli/tili/mnt/MolDiffAE/data/geom_drug'

    cls_epoch=10000
    cls_prefix = 'full'
    if 'drug3d' in args.name:
        config.dataset.name = 'drug3d'
        if args.name == 'drug3d1000':
            cls_prefix = f'full-drug3d1000'
            cls_epoch=5000

            
    elif 'crossdocked' in args.name:
        config.dataset.root = '/home/mli/tili/mnt/MolDiffAE/data/crossdocked'
        config.dataset.name = 'crossdocked'
        cls_prefix = f'full-crossdocked'

    # # Transform
    featurizer = FeaturizeMol(config.chem.atomic_numbers, config.chem.mol_bond_types,
                            use_mask_node=config.transform.use_mask_node,
                            use_mask_edge=config.transform.use_mask_edge, 
                          random=False
                            )
    transform = Compose([
        featurizer,
    ])

    ## Load property dataset
    with open('property_prediction_data_val.pkl', 'rb') as f:
        idx_train, idx_test, prop_train, prop_test, mol_train, mol_test, descriptor_names = pickle.load(f)

    prop_test_all = np.vstack([prop_train, prop_test])
    idx_test_all = np.concatenate([idx_train, idx_test])
    mol_test_all = mol_train + mol_test
        
    prop_mean = np.mean(prop_train, axis=0)
    prop_std = np.std(prop_train, axis=0)
    
    selected_property_list = []
    prop_min = np.quantile(prop_train, 0.001, axis=0)
    prop_max = np.quantile(prop_train, 0.999, axis=0)
    for pp in descriptor_names:
        if prop_max[descriptor_names.index(pp)] > prop_min[descriptor_names.index(pp)]:
            selected_property_list.append(pp)

    selected_property_idx = [descriptor_names.index(pp) for pp in selected_property_list]
    if args.prop_normalize:
        print('Normalizing targets')
        prop_min = torch.tensor(np.quantile(prop_train, 0.001, axis=0)[selected_property_idx]).float().to(args.device)
        prop_max = torch.tensor(np.quantile(prop_train, 0.999, axis=0)[selected_property_idx]).float().to(args.device)
    else:
        prop_min = 0
        prop_max = 1
    
    ## Model
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
    
    config_cls.model.output_dim = len(selected_property_list)
    if args.with_edge:
        print('Utilizing edge type features')
        predictor = PropertyPredictorWithEdge(
            config=config_cls.model,
            num_node_types=featurizer.num_node_types,
            num_edge_types=featurizer.num_edge_types
        ).to(args.device)    
        cls_ckpt = f'/home/mli/tili/mnt/MolDiffAE/model/property_predictor/PropertyPred-{cls_prefix}-withedge'
    else:
        predictor = PropertyPredictor(
            config=config_cls.model,
            num_node_types=featurizer.num_node_types,
            num_edge_types=featurizer.num_edge_types
        ).to(args.device)
        cls_ckpt = f'/home/mli/tili/mnt/MolDiffAE/model/property_predictor/PropertyPred-{cls_prefix}-noedge'
    
    guidance = ('mse', args.guidance_strength)

    
    # # generating molecules

    ckpt = torch.load(f'{cls_ckpt}/checkpoints/{cls_epoch}.pt', map_location='cuda')
    predictor.load_state_dict(ckpt['model'], strict=False)
    predictor.cuda()
    predictor.eval()


    # generation  
    device=args.device
    n_graphs = 10

    guidance = ('mse', args.guidance_strength)

    if args.name == 'drug3d1000':
        property_fname = f'property_{args.mode}_gen_task_test_drug3d1000.pkl'
    else:
        property_fname = f'property_{args.mode}_gen_task_test.pkl'

    print("Porperty set:", property_fname)
    with open(property_fname, 'rb') as f:
        emb_prop_dict = pickle.load(f)
        
    gen_dict = {}
    for i in emb_prop_dict:
    #for i in range(len(test_set)):
        prop_raw = emb_prop_dict[i]
        prop = torch.tensor([[prop_raw[pp] for pp in selected_property_list]], device=device)
        prop = (prop - prop_min)/(prop_max-prop_min)
        
        max_size = prop_raw['HeavyAtomCount']
        
        mol_list_manipulate = sample_from_property(prop, model, mode='template', n_graphs=n_graphs, max_size=max_size, predictor=predictor, guidance=guidance)

        pp_list_manipulate = []

        if len(mol_list_manipulate) > 0:
            for mol in mol_list_manipulate:
                desc = get_descriptors(mol['rdmol'])
                pp_list_manipulate.append(desc)

        gen_dict[i] = (mol_list_manipulate, pp_list_manipulate)
        #if manipulate is not None:
        #    print(i, np.mean([d[args.pp] for d in pp_list_manipulate]))

    save_name = f'{args.mode}-{args.name}_ddpm-guidance-{args.guidance_strength}'
    if args.with_edge:
        save_name += '-withedge'
    if args.delta:
        save_name += '-delta'
    if 'test' in args.property_set:
        save_name += '_test'
    with open(f'{args.outdir}/{save_name}.pkl', 'wb') as f:
        pickle.dump(gen_dict, f)
    