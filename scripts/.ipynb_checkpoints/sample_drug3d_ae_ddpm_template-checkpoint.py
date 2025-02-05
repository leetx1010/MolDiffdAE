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
                    "NumAromaticRings": [2,0, 0.0],
                    #"NumAromaticRings": [0.0],
                    #"MolLogP": [5.0, 0.0],
                    "MolLogP": [0.0],
                    "RadiusOfGyration": [6.0],
                    "PBF": [1.5],
                    "orig": [1.0],
                    "random": [1.0]}
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')   

from scipy.stats import spearmanr, pearsonr
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

def sample_from_template_conditioned(data, model, mode='template', n_graphs=10, max_size=None, guidance=None, bond_predictor=None, manipulate=None):
    if max_size is None:
        max_size = len(data.element)
    batch = Batch.from_data_list([data.clone() for _ in range(n_graphs)], follow_batch = ['halfedge_type','node_type']).to(device)
    if mode == 'template':
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

        if manipulate is not None:
            emb = emb.detach().cpu().numpy()[0:1]
            reg, target, alpha = manipulate
            w = reg.coef_
            b = reg.intercept_
    
            s = (target - b - (emb*w).sum())/(w*w).sum()
            emb_new = emb + alpha*s*w

            print(reg.predict(emb)[0], reg.predict(emb_new)[0])
            emb_new = torch.tensor(emb_new).to(node_pos)
            emb = emb_new
            
        emb = emb.repeat_interleave(n_graphs,0)
    else:
        emb = None

    batch_holder = make_data_placeholder(n_graphs=n_graphs, device=device, max_size=max_size)
    batch_node, halfedge_index, batch_halfedge = batch_holder['batch_node'], batch_holder['halfedge_index'], batch_holder['batch_halfedge']

    # inference
    outputs = model.sample(
        n_graphs=n_graphs,
        batch_node=batch_node,
        halfedge_index=halfedge_index,
        batch_halfedge=batch_halfedge,
        bond_predictor=bond_predictor,
        guidance=guidance,
        emb=emb
    )
    outputs = {key:[v.cpu().numpy() for v in value] for key, value in outputs.items()}
    
    # decode outputs to molecules
    batch_node, halfedge_index, batch_halfedge = batch_node.cpu().numpy(), halfedge_index.cpu().numpy(), batch_halfedge.cpu().numpy()
    try:
        output_list = seperate_outputs(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge)
    except:
        return None
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
    parser.add_argument('--config', type=str, default='configs/train/train_MolDiffAE_discrete.yml')
    parser.add_argument('--guidance', type=str, default=None)
    parser.add_argument('--name', type=str, default='drug3d')
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--wass_weight', type=float, default=10.0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--property_set', type=str, default='logs')
    parser.add_argument('--task', type=str, default='qed+')
    parser.add_argument('--target', type=float, default=1.0)
    parser.add_argument('--template', type=str, default='template')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='logs')
    args = parser.parse_args()
    
    args.delta = ('delta' in args.property_set) # temporary

    ### 1. Load model and data
    # Load configs
    config = load_config(args.config)

    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]

    ckpt = torch.load(args.ckpt, map_location=args.device)

    #scaling = config.model.diff.scaling # temporary
    config = ckpt['config']
    
    #config.model.diff.scaling = scaling # temporary
    config.model.encoder.emb_dim = args.emb_dim
    config.model.encoder.wass_weight = args.wass_weight

    if args.seed is not None:
        config.train.seed = args.seed
    seed_all(config.train.seed)

    # Logging
    """
    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    """

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
    train_set, test_set = subsets['train'], subsets['test']

    test_loader = DataLoader(test_set, config.train.batch_size, shuffle=False,
                            follow_batch=featurizer.follow_batch, exclude_keys=featurizer.exclude_keys)

    # Load property dataset
    if args.name == 'drug3d':
        with open('property_prediction_data_test.pkl', 'rb') as f:
            idx_train, idx_test, prop_train, prop_test, mol_train, mol_test, descriptor_names = pickle.load(f)    
        prop_test_all = np.vstack([prop_train, prop_test])
        idx_test_all = np.concatenate([idx_train, idx_test])
        mol_test_all = mol_train + mol_test
    elif args.name == 'crossdocked':
        with open('property_prediction_data_crossdocked_test.pkl', 'rb') as f:
            idx_train, idx_test, prop_train, prop_test, mol_train, mol_test, descriptor_names = pickle.load(f)    
        prop_test_all = prop_test
        idx_test_all = idx_test
        mol_test_all = mol_test
        
    idx_test = idx_test_all[:100]

    # Model
    #logger.info('Building model...')
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
    model.load_state_dict(ckpt['model'])
    model.eval()


    ### 3. Generation
    device=args.device
    add_edge = None
    n_graphs = 10
        
    # regression    

    # generation        
    gen_dict = {}
    for n, i in enumerate(idx_test):
        manipulate = None
        mol_list_manipulate = sample_from_template_conditioned(test_set[i], model, mode=args.template, n_graphs=n_graphs, manipulate=manipulate)

        pp_list_manipulate = []

        if len(mol_list_manipulate) > 0:
            for mol in mol_list_manipulate:
                desc = get_descriptors(mol['rdmol'])
                pp_list_manipulate.append(desc)

        gen_dict[i] = (mol_list_manipulate, pp_list_manipulate)
        #if manipulate is not None:
        #    print(i, np.mean([d[args.pp] for d in pp_list_manipulate]))

    save_name = f'template-{args.name}_ddpm'
    if args.guidance is not None:
        save_name += '-guided'
    if args.template == 'random':
        save_name += '-random'
    if 'test' in args.property_set:
        save_name += '_test'    
    with open(f'{args.logdir}/{save_name}.pkl', 'wb') as f:
        pickle.dump(gen_dict, f)
    