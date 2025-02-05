"""
Converted from https://github.com/mims-harvard/TDC
"""
#!/usr/bin/env python
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn import svm
import pickle
import re
import os.path as op
rdBase.DisableLog('rdApp.error')

"""Scores based on an ECFP classifier for activity."""

gsk3b_model = None
def load_model():
    global gsk3b_model
    name = op.join(op.dirname(__file__), 'gsk3b_current.pkl')
    with open(name, "rb") as f:
        gsk3b_model = pickle.load(f)

def get_score(mol):
    if gsk3b_model is None:
        load_model()

    #mol = Chem.MolFromSmiles(smile)
    if mol:
        fp = fingerprints_from_mol(mol)
        score = gsk3b_model.predict_proba(fp)[:, 1]
        return float(score)
    return 0.0

def fingerprints_from_mol(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, features)
    fp = features.reshape(1, -1)
    
    return fp