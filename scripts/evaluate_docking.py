import os
import sys
import shutil
import argparse
sys.path.append('.')

from utils.evaluation_docking.docking_qvina import QVinaDockingTask
from utils.evaluation_docking.docking_vina import VinaDockingTask

import pickle
import torch
import numpy as np
from rdkit.Chem import Draw, AllChem
import rdkit.Chem as Chem
from copy import deepcopy

from models.model import MolDiff
from utils.dataset import get_dataset
from utils.transforms import FeaturizeMol, Compose
from utils.transforms import *
from utils.misc import *
from utils.reconstruct import *
from utils.sample import seperate_outputs, seperate_outputs_no_traj

from easydict import EasyDict

import umap
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--file_gen', type=str)
#parser.add_argument('--task', type=str, default='qed+')
args = parser.parse_args()

args.task = args.file_gen.split('/')[-1].split('_')[-3]
if 'halogen' in args.task:
    args.task = 'fr_'+args.task

config = load_config('configs/train/train_MolDiffAE_continuous.yml')

config.model.diff.categorical_space='continuous'
config.model.diff.scaling = [1., 4., 8.]
config.dataset.name='crossdocked'
config.dataset.split='split_by_key.pt'
config.dataset.root='/home/mli/tili/mnt/MolDiffAE/data/crossdocked'
config.model.encoder.emb_dim = 32
config.train.batch_size = 16
config.train.num_workers = 1

protein_root='/home/mli/tili/Projects/MolDiffAE/MolInterDiffAE/data/test_set/'

    

multiobj_task_list = [('qed+','SAS-'), 
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
            ]

selected_descrpitors_list = ['qed', 
                             'SAS', 
                             'Asphericity', 
                             'fr_halogen', 
                             'MolLogP',
                             'RadiusOfGyration',
                            ]

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

### Load generated list and property data
print("Loading ", args.file_gen)
with open(f'{args.file_gen}','rb') as f:
    gen_list = pickle.load(f)

if 'vae' in args.file_gen:
    gen_list = {i:[[{'rdmol':gen_list[i][0]}], [gen_list[i][1]]] for i in gen_list}

with open('property_prediction_data_crossdocked_test.pkl', 'rb') as f:
    idx_train, idx_test, prop_train, prop_test, mol_train, mol_test, descriptor_names = pickle.load(f)

if 'multiobj' in args.task:
    with open('property_prediction_test_tasks_crossdocked_multiobj.pkl', 'rb') as f:
        test_split, descriptor_names = pickle.load(f)
    task_id = int(args.task[-1])
    idx_test_all, target = test_split[multiobj_task_list[task_id]]
else:
    with open('property_prediction_test_tasks_crossdocked-delta.pkl', 'rb') as f:
        test_split, descriptor_names = pickle.load(f)

    idx_test_all, targets, target = test_split[args.task]
    
prop_mean = np.mean(prop_train, axis=0)
prop_std = np.std(prop_train, axis=0)

margin = {}

for pp in selected_descrpitors_list:
    margin[pp] = prop_std[descriptor_names.index(pp)] * 0.5

### Load generations

if 'multiobj' in args.task:
    task_id = int(args.task[-1])
    pp_index = [(task[:-1], (task[-1]=='+')*2-1) for task in multiobj_task_list[task_id]]

    print(pp_index)
else:
    pp = args.task[:-1]
    dd = (args.task[-1]=='+')*2-1
    
filtered_id = []
for i in gen_list:
    mol_list = gen_list[i]
        

    if len(mol_list[0]) == 0:
        continue

    #for m in plot_list:
    #    m.Compute2DCoords()
    if 'random' in args.file_gen:
        filtered_id.append(i)
    else:        
        ii = np.where(idx_test==i)[0][0]
        """
        # delta
        #try:
        if 'multiobj' in args.task:
            success = 1
            for pp, dd in pp_index:
                success *= ((prop_test[ii, descriptor_names.index(pp)] - mol_list[1][0][pp])*dd > margin[pp])
            if success:
                filtered_id.append(i)
        else:
            if (mol_list[1][0][pp] - prop_test[ii, descriptor_names.index(pp)])*dd > margin[pp]:
                filtered_id.append(i)
        """
        if 'multiobj' in args.task:
            success = 1
            for pp, dd in pp_index:
                success *= ((mol_list[1][0][pp] - prop_mean[descriptor_names.index(pp)])*dd > 0)
            if success:
                filtered_id.append(i)
        else:
            if ((mol_list[1][0][pp] - prop_mean[descriptor_names.index(pp)])*dd > 0):
                filtered_id.append(i)        
    #except:
    #    continue
        

### Autodock 
def qvina_score(mol, mol_orig, ligand_filename):
    try:
        xyz = []
    
        for i, atom in enumerate(mol_orig.GetAtoms()):
            positions = mol_orig.GetConformer().GetAtomPosition(i)
            xyz.append([positions.x, positions.y, positions.z])
        xyz = np.array(xyz)
    
        vina_task = QVinaDockingTask.from_generated_mol(
                            mol, ligand_filename, protein_root=protein_root, center=xyz.mean(0), conda_env='targetdiff')
        vina_results = vina_task.run_sync()
    
        return vina_results[0]['affinity']
    except:
        return 1
        
split_by_name = torch.load('/home/mli/tili/Projects/MolDiffAE/MolInterDiffAE/data/split_by_name.pt')
test_sdf = [n[1] for n in split_by_name['test']]

qvina_score_res = {}

for i in tqdm(filtered_id):
    ligand_filename = test_sdf[i]


    suppl = Chem.SDMolSupplier(f'{protein_root}/{ligand_filename}')
    mol = gen_list[i][0][0]['rdmol']
    mol_orig = suppl[0]
    
    v0 = qvina_score(mol_orig, mol_orig, ligand_filename)
    v1 = qvina_score(mol, mol_orig, ligand_filename)

    qvina_score_res[i] = (mol, mol_orig, v1, v0)

save_name = '.'.join(args.file_gen.split('.')[:-1]) + '_qvina.pkl'
with open(save_name, 'wb') as f:
    pickle.dump(qvina_score_res, f)