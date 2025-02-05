import os
import shutil
import argparse
import sys
sys.path.append('.')

from tqdm import tqdm
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
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
from models.bond_predictor import *

task_target_dict = {"qed": [1.0],
                    "SAS": [0.0],
                    "SPS": [30.0],
                    "Asphericity": [1.0, 0.0],
                    "fr_halogen": [2,0, 0.0],
                    #"fr_halogen": [0.0],
                    #"NumAromaticRings": [2,0, 0.0],
                    "NumAromaticRings": [0.0],
                    "MolLogP": [5.0, 0.0],
                    "RadiusOfGyration": [6.0],
                    "PBF": [1.5],
                    "orig": [1.0],
                    "random": [1.0]}


backprop_param_dict = {"qed": (1.0, 50),
                       "SAS":(0.1, 50), 
                       "Asphericity":(1.0, 50),
                       "fr_halogen":(0.1, 50),
                       "MolLogP":(0.1, 50),
                       "RadiusOfGyration":(0.1, 50)}

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')   

from scipy.stats import spearmanr, pearsonr
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

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

def embedding_optimization(emb, aux_cls, manipulate):
    '''
    The manipulate tuple (mode, target, alpha)
    mode:
    - value: the value itself as the objective (gradient ascent/negative alpha to maximize value, descent/positive alpha to minimize value)
    - mse: minimizes MSE to the target value (alpha should always be positive)
    target: ignored when mode is "value"
    alpha: negative for gradient ascent, positive for descent
    idx_pp: the dimension of the property (Notice: wrt the output vector, not the full descriptor vector)
    steps: number of steps to optimize
    '''

    mode, target, alpha, idx_pp, steps = manipulate
    if mode == 'mse':
        target = torch.tensor([target]).float().to(emb.device)

    emb_prev = emb.to(emb.device)
    
    for i in range(steps):
        emb_in = emb_prev.detach().requires_grad_(True)
    
        pred_prop = aux_cls(emb_in)[:,0]
        
        if mode == 'mse':
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(pred_prop, target)
            
        elif mode == 'value':
            loss = pred_prop.mean()
        emb_delta = -torch.autograd.grad(loss, emb_in, retain_graph=True)[0] * alpha
        
        emb_prev = emb_in + emb_delta
    print('Optimized cls: ', aux_cls(emb_prev).item())
    return emb_prev.detach().cpu()
    
def sample_from_template_conditioned_ddim(data, model, mode='template', stride=1, n_graphs=1, start_step=None, max_size=None, aux_cls=None, manipulate=None, add_edge=None):
    if max_size is None:
        max_size = len(data.element)
    if start_step is None:
        time_sequence = list(range(0, model.num_timesteps-stride, stride))
    else:
        time_sequence = list(range(0, start_step, stride))
        
    batch = Batch.from_data_list([data.clone() for _ in range(n_graphs)], follow_batch = ['halfedge_type','node_type']).to(device)

    node_type = batch.node_type 
    node_pos = batch.node_pos
    batch_node = batch.node_type_batch
    halfedge_type = batch.halfedge_type
    halfedge_index = batch.halfedge_index
    batch_halfedge = batch.halfedge_type_batch
    num_mol = batch.num_graphs
    
    emb, _ = model.encode(node_type, node_pos, batch_node,
            halfedge_type, halfedge_index, batch_halfedge,
            num_mol)
    emb = emb.repeat_interleave(n_graphs,0)
    
    edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
    batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
    
    h_node_pert = F.one_hot(batch.node_type, model.num_node_types).float() / model.scaling[1]
    pos_pert = batch.node_pos / model.scaling[0]
    h_halfedge_pert = F.one_hot(batch.halfedge_type, model.num_edge_types).float()/ model.scaling[2]
    
    
    for i, step in tqdm(enumerate(time_sequence), total=len(time_sequence)):
        time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)
        h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)
        
        # # 1 inference
        preds = model(
            h_node_pert, pos_pert, batch_node,
            h_edge_pert, edge_index, batch_edge, 
            time_step, 
            emb
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
    
    pos_init = pos_pert
    h_node_init = h_node_pert
    h_halfedge_init = h_halfedge_pert
    
    h_node_pert = h_node_init
    pos_pert = pos_init
    h_halfedge_pert = h_halfedge_init

    ## Manipulate embedding
    if manipulate is not None:
        emb_new = embedding_optimization(emb.detach()[0:1], aux_cls, manipulate)
        emb = emb.detach().cpu()
        emb_new = emb_new.to(node_pos)
        emb = emb_new
            
        emb = emb.repeat_interleave(n_graphs,0)
    
    
    for i, step in tqdm(enumerate(time_sequence[::-1]), total=len(time_sequence)):
       
        time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)
        h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)
        
        # # 1 inference
        preds = model(
            h_node_pert, pos_pert, batch_node,
            h_edge_pert, edge_index, batch_edge, 
            time_step, 
            emb
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train/train_MolDiffAE_discrete.yml')
    parser.add_argument('--guidance', type=str, default=None)
    parser.add_argument('--mode', type=str, default='mse')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--opt_steps', type=int, default=10)
    parser.add_argument('--name', type=str, default='drug3d')
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--emb_prop_dim', type=int, default=32)
    parser.add_argument('--wass_weight', type=float, default=10.0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--property_set', type=str, default='logs')
    parser.add_argument('--task', type=str, default='qed+')
    parser.add_argument('--target', type=float, default=1.0)
    parser.add_argument('--start_step', type=int, default=1000)
    parser.add_argument('--logdir', type=str, default='logs')
    args = parser.parse_args()
    
    ### 1. Load model and data
    # Load property dataset
    pp = args.task[:-1]
    with open(args.property_set, 'rb') as f:
        test_split, descriptor_names = pickle.load(f)
    idx_test, target = test_split[args.task]
    
    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    data_root = config.dataset.root


    idx_pp = descriptor_names.index(pp)
    model_prefix = args.logdir.split('/')[-1]
    logdir = f'{args.logdir}/{model_prefix}-{idx_pp}-{args.emb_prop_dim}/'
    print(logdir, idx_pp, pp)
    
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]

    ckpt = torch.load(f'{logdir}/checkpoints/20000.pt', map_location=args.device)

    #scaling = config.model.diff.scaling # temporary
    config = ckpt['config']
    
    #config.model.diff.scaling = scaling # temporary
    config.model.encoder.emb_dim = args.emb_dim
    config.model.encoder.wass_weight = args.wass_weight
    config.dataset.root = data_root

    # For earlier versions without residual type specification
    if not 'prop_residual' in config.model.encoder or not config.model.encoder.prop_residual:
        config.model.encoder.prop_residual = 'linear'
        config.model.encoder.prop_mlp_layers = 2 # other values not used
        if not 'prop_hidden_dim' in config.model.encoder or not config.model.encoder.prop_hidden_dim:
            config.model.encoder.prop_hidden_dim = 64 # other values not used
            
    elif 'res-add' in args.ckpt:
        config.model.encoder.prop_residual = 'res-add'
    elif 'res-cat' in args.ckpt:
        config.model.encoder.prop_residual = 'res-cat'
    
    seed_all(config.train.seed)

    # Transforms
    featurizer = FeaturizeMol(config.chem.atomic_numbers, config.chem.mol_bond_types,
                              use_mask_node=config.transform.use_mask_node,
                              use_mask_edge=config.transform.use_mask_edge,
                              random=False
                            )
    transform = Compose([
        featurizer,
    ])

    # Datasets and loaders
    #logger.info('Loading dataset...')
    if 'drug3d' not in args.name:
        config.dataset.name = args.name
        if 'crossdocked' in args.name:
            config.dataset.root = '/home/mli/tili/mnt/MolDiffAE/data/crossdocked'
            #config.dataset.split = 'split_by_key.pt'

    dataset, subsets = get_dataset(
        config = config.dataset,
        transform = transform,
    )
    train_set, val_set, test_set = subsets['train'], subsets['val'], subsets['test']

    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=featurizer.follow_batch, exclude_keys=featurizer.exclude_keys)

    test_loader = DataLoader(test_set, config.train.batch_size, shuffle=False,
                            follow_batch=featurizer.follow_batch, exclude_keys=featurizer.exclude_keys)


    # Model
    #logger.info('Building model...')
    if config.model.name == 'diffusion':
        model = MolDiffAEFT(
            config=config.model,
            num_node_types=featurizer.num_node_types,
            num_edge_types=featurizer.num_edge_types
        ).to(args.device)
    else:
        raise NotImplementedError('Model %s not implemented' % config.model.name)
    print('Num of trainable parameters is', np.sum([p.numel() for p in model.parameters() if p.requires_grad]))

    # Load ckpt
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    
    ### 3. Generation
    device=args.device
    add_edge = None
    n_graphs = 1
    
    if args.guidance is not None:
        ckpt_bond = torch.load(args.guidance, map_location=device)
        bond_predictor = BondPredictor(ckpt_bond['config']['model'],
                featurizer.num_node_types,
                featurizer.num_edge_types-1 # note: bond_predictor not use edge mask
        ).to(device)
        bond_predictor.load_state_dict(ckpt_bond['model'])
        bond_predictor.eval()
        guidance = ('uncertainty', 1.e-4)
    else:
        bond_predictor = None
        guidance = None
        
    # regression    
    if args.task not in ['random','orig']:
        if 'test' in args.property_set:
            alpha, opt_steps = backprop_param_dict[pp][0], args.opt_steps
        else:
            alpha, opt_steps = args.alpha, args.opt_steps
            
        if args.mode == 'value' and args.task[-1] == '+':
            manipulate = (args.mode, target, -alpha, idx_pp, opt_steps) # gradient ascent
        else:
            manipulate = (args.mode, target, alpha, idx_pp, opt_steps)
    else:
        manipulate = None

    if args.task == 'random':
        mode = 'random'
    else:
        mode = 'template'

    # generation        
    gen_dict = {}
    for i in idx_test:
        mol_list_manipulate = sample_from_template_conditioned_ddim(test_set[i], model, mode='template', n_graphs=n_graphs, manipulate=manipulate, start_step=args.start_step, aux_cls=model.encoder.aux_cls)

        pp_list_manipulate = []

        if len(mol_list_manipulate) > 0:
            for mol in mol_list_manipulate:
                desc = get_descriptors(mol['rdmol'])
                pp_list_manipulate.append(desc)

        gen_dict[i] = (mol_list_manipulate, pp_list_manipulate)
        #if manipulate is not None:
        #    print(i, np.mean([d[args.pp] for d in pp_list_manipulate]))

    save_name = f'property-{args.name}_{args.task}_ddim-{args.start_step}-backprop-{args.mode}-{args.alpha}-{args.opt_steps}'
    if args.guidance is not None:
        save_name += '-guided'
    if 'test' in args.property_set:
        save_name += '_test'           
    with open(f'{args.logdir}/{save_name}.pkl', 'wb') as f:
        pickle.dump(gen_dict, f)

    