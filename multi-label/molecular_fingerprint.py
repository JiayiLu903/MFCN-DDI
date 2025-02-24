import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import rdMolDescriptors
import torch
import torch.nn as nn


input_file = 'drug_smiles.csv'
output_file = 'drug_fingerprints_combined.csv'




def generate_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)


    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    morgan_fp = np.array(morgan_fp)


    maccs_fp = AllChem.GetMACCSKeysFingerprint(mol)  #AllChem.GetMACCSKeysFingerprint(mol)
    maccs_fp = np.array(maccs_fp)


    atom_pair_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=1024)
    atom_pair_fp = np.array(atom_pair_fp)


    erg_fp = AllChem.GetErGFingerprint(mol,fuzzIncrement=0.3,maxPath=21,minPath=1)
    erg_fp = np.array(erg_fp)


    fingerprint_vector = np.concatenate([morgan_fp, maccs_fp, atom_pair_fp, erg_fp])
    return fingerprint_vector



df = pd.read_csv(input_file)


output_data = []


for index, row in df.iterrows():
    drug_id = row['drug_id']
    smiles = row['SMILES']


    fingerprint_vector = generate_fingerprints(smiles)


    fingerprint_tensor = torch.tensor(fingerprint_vector, dtype=torch.float32).unsqueeze(0)


    output_data.append([drug_id, fingerprint_tensor])


output_df = pd.DataFrame(output_data, columns=['drug', 'emb'])


output_df.to_csv(output_file, index=False)
