# ['MW', 'ALOGP', 'HBA', 'HBD', 'PSA', 'ROTB', 'AROM', 'ALERTS']
# use rdkit to calculate the physicochemical properties

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd

def cal_physicochemical_properties(smi, all_features):
    mol = Chem.MolFromSmiles(smi)
    res = []
    if "MW" in all_features:
        MW = Descriptors.MolWt(mol)
        res.append(MW)
    if "ALOGP" in all_features:
        ALOGP = Descriptors.MolLogP(mol)
        res.append(ALOGP)
    if "HBA" in all_features:
        HBA = Descriptors.NumHAcceptors(mol)
        res.append(HBA)
    if "HBD" in all_features:
        HBD = Descriptors.NumHDonors(mol)
        res.append(HBD)
    if "PSA" in all_features:
        PSA = Descriptors.TPSA(mol)
        res.append(PSA)
    if "ROTB" in all_features:
        ROTB = Descriptors.NumRotatableBonds(mol)
        res.append(ROTB)
    if "AROM" in all_features:
        AROM = Descriptors.NumAromaticRings(mol)
        res.append(AROM)

    return np.array(res)
