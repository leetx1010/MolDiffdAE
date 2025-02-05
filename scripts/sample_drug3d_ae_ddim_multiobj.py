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

from multiprocessing import Process, Queue

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

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')   

from scipy.stats import spearmanr, pearsonr
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

def q_reconstruct_from_generated_with_edges(q, mol_info, add_edge):
    q.put(reconstruct_from_generated_with_edges(mol_info, add_edge=add_edge))

def process_outputs(outputs, batch_node_raw, halfedge_index_raw, batch_halfedge_raw, n_graphs=1, add_edge=None):
    batch_node, halfedge_index, batch_halfedge = batch_node_raw.cpu().numpy(), halfedge_index_raw.cpu().numpy(), batch_halfedge_raw.cpu().numpy()
    output_list = seperate_outputs_no_traj(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge)

    gen_list = []
    pool = EasyDict({
        'failed': [],
        'finished': [],
    })
    for i_mol, output_mol in enumerate(output_list):
        try:
            mol_info = featurizer.decode_output(
                pred_node=output_mol['pred'][0],
                pred_pos=output_mol['pred'][1],
                pred_halfedge=output_mol['pred'][2],
                halfedge_index=output_mol['halfedge_index'],
            )  # note: traj is not used
        except MolReconsError:
            pool.failed.append(mol_info)
            continue
        """
        try:
            rdmol = reconstruct_from_generated_with_edges(mol_info, add_edge=add_edge)
        except MolReconsError:
            pool.failed.append(mol_info)
            continue
        """
        # Reconstruct: this is prone to very long processing time
        # Use multiprocess to handle timeout
        q = Queue()
        p = Process(target=q_reconstruct_from_generated_with_edges, args=(q, mol_info, add_edge))
        
        p.start()
        p.join(timeout=15.0)
        if p.exitcode is None:
            p.terminate()
            pool.failed.append(mol_info)
            continue
            print(n, i, 'timeout')
        else:
            rdmol = q.get()
            print(n, i)
        if rdmol is None:
            continue
        # End reconstruction
        
        mol_info['rdmol'] = rdmol
        smiles = Chem.MolToSmiles(rdmol)
        mol_info['smiles'] = smiles
        print(smiles)
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

def sample_from_template_conditioned_multi_ddim(data, model, mode='template', noise='deterministic', stride=1, n_graphs=1, emb_prop_dim=None, start_step=None, max_size=None, guidance=None, bond_predictor=None, manipulate=None):
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
    
    emb = model.encode(node_type, node_pos, batch_node,
            halfedge_type, halfedge_index, batch_halfedge,
            num_mol)
    emb = emb.repeat_interleave(n_graphs,0)
    
    edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
    batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
    
    if noise == 'deterministic':
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
        
    elif noise == 'random':
        n_nodes_all = node_type.shape[0]
        n_halfedges_all = halfedge_type.shape[0]
        h_node_init = model.node_transition.sample_init(n_nodes_all)
        pos_init = model.pos_transition.sample_init([n_nodes_all, 3])
        h_halfedge_init = model.edge_transition.sample_init(n_halfedges_all)

    
    h_node_pert = h_node_init
    pos_pert = pos_init
    h_halfedge_pert = h_halfedge_init

    ## Manipulate embedding
    if mode == 'template':
        if manipulate is not None:
            emb = emb.detach().cpu().numpy()[0:1]
    
            """
            # Independent classifiers
            for manipulate_sub in manipulate:
                reg, target, alpha = manipulate_sub
                w = reg.coef_
                b = reg.intercept_
        
                s = (target - b - (emb*w).sum())/(w*w).sum()
                emb_new = emb + alpha*s*w
        
                print(reg.predict(emb)[0], reg.predict(emb_new)[0])
                emb = emb_new
                  
            emb = torch.tensor(emb).to(node_pos)        
            emb = emb.repeat_interleave(n_graphs,0)
            """
            
            # Simultaenous update using a multi-dimensional classifier
            reg, target, alpha = manipulate
            w = reg.coef_
            b = reg.intercept_
            target = np.array(target)
    
            emb_new = emb + np.matmul(alpha*(target - b - np.matmul(emb, w.transpose())), np.linalg.pinv(w).transpose())
            print(reg.predict(emb_new))
    
            emb = emb_new
                  
            emb = torch.tensor(emb).to(node_pos)        
            emb = emb.repeat_interleave(n_graphs,0)
    elif mode == 'random':
        emb = torch.randn(n_graphs, model.emb_dim, device=device) 
        
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
    
    gen_list = process_outputs(outputs, batch_node, halfedge_index, batch_halfedge)
    return gen_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train/train_MolDiffAE_discrete.yml')
    parser.add_argument('--guidance', type=str, default=None)
    parser.add_argument('--name', type=str, default='drug3d')
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--wass_weight', type=float, default=10.0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--property_set', type=str, default='logs')
    parser.add_argument('--task_id', type=int, default=0)
    parser.add_argument('--template', type=str, default='template')
    parser.add_argument('--noise', type=str, default='deterministic')
    parser.add_argument('--start_step', type=int, default=1000)
    parser.add_argument('--logdir', type=str, default='logs')
    args = parser.parse_args()
    
    args.delta = ('delta' in args.property_set) # temporary

    ### 1. Load model and data
    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]

    ckpt = torch.load(args.ckpt, map_location=args.device)

    #scaling = config.model.diff.scaling # temporary    
    config.model.encoder.emb_dim = args.emb_dim
    config.model.encoder.wass_weight = args.wass_weight
    
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
    if args.name == 'crossdocked':
        config.dataset.name='crossdocked'
        config.dataset.split='split_by_key.pt'
        config.dataset.root='/home/mli/tili/mnt/MolDiffAE/data/crossdocked'
        
    dataset, subsets = get_dataset(
        config = config.dataset,
        transform = transform,
    )
    train_set, test_set = subsets['train'], subsets['test']

    test_loader = DataLoader(test_set, config.train.batch_size, shuffle=False,
                            follow_batch=featurizer.follow_batch, exclude_keys=featurizer.exclude_keys)

    # Load property dataset
    with open(args.property_set, 'rb') as f:
        test_split, descriptor_names = pickle.load(f)

    # Model
    #logger.info('Building model...')
    config = ckpt['config']
    if config.model.name == 'diffusion':
        model = MolDiffAE(
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
    tasks = task_list[args.task_id]
    alpha = 1
    if args.delta:
        idx_test, targets, target = test_split[tasks]
    else:
        idx_test, target = test_split[tasks]

    """
    manipulate = []
    reg_dict = {}
    for i in range(len(tasks)):
        task = tasks[i]
        pp = task[:-1]
        with open(f'{args.logdir}/classifiers/linear_{pp}.pkl', 'rb') as f:
            reg = pickle.load(f)
        manipulate.append([reg, targets[i], alpha])
    """
    with open('property_prediction_data_val.pkl', 'rb') as f:
        idx_train, _, prop_train, _, mol_train, _, descriptor_names = pickle.load(f)
        
    with open(f'{args.logdir}/classifiers/embedings_drug3d.pkl', 'rb') as f:
        emb_val, emb_test = pickle.load(f)
        
    pp_index = [descriptor_names.index(pp[:-1]) for pp in tasks]
    prop = prop_train[:,pp_index]
    reg = LinearRegression().fit(emb_val[idx_train,:], prop)
    manipulate = (reg, np.array(target), alpha)
    

    if args.noise == 'deterministic':
        save_name = f'property-{args.name}_multiobj-{args.task_id}_ddim-{args.start_step}'
        n_graphs = 1
    elif args.noise == 'random':
        save_name = f'property-{args.name}_multiobj-{args.task_id}_ddim-randnoise-{args.start_step}'
        n_graphs = 10

    check_validity = False # temp: only used for tasks that stuck when checking validity.
    # generation     
    #mode = 'template'
    gen_dict = {}
    for n, i in enumerate(idx_test):
        if args.delta:
            manipulate = (reg, np.array(targets[n]), 1)
            print(targets[n])

        mol_list_manipulate = sample_from_template_conditioned_multi_ddim(test_set[i], model, mode=args.template, noise=args.noise, n_graphs=n_graphs, manipulate=manipulate, start_step=args.start_step, bond_predictor=bond_predictor, guidance=guidance)

        pp_list_manipulate = []

        if len(mol_list_manipulate) > 0:
            for mol in mol_list_manipulate:
                desc = get_descriptors(mol['rdmol'])
                pp_list_manipulate.append(desc)

        gen_dict[i] = (mol_list_manipulate, pp_list_manipulate)
        #if manipulate is not None:
        #    print(i, np.mean([d[args.pp] for d in pp_list_manipulate]))

    if args.guidance is not None:
        save_name += '-guided'
    if args.template == 'random':
        save_name += '-random'  
    if args.delta:
        save_name += '-delta'
    if 'test' in args.property_set:
        save_name += '_test'  
    with open(f'{args.logdir}/{save_name}.pkl', 'wb') as f:
        pickle.dump(gen_dict, f)