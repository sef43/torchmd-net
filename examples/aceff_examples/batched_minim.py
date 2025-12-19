# This script demonstrates how to do batched minimization of molecules with AceFF torchmd-net models

import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom
import torch
from torchmdnet.models.model import load_model
from torchmdnet.calculators import optimize_geometries
from torchmdnet.utils import mols_to_batch, batch_to_mols


def batch_minim(mols, model_file_path):

    # load the AceFF model
    model = load_model(model_file_path, derivative=False, check_errors=True, static_shapes=False, max_num_neighbors=64, remove_ref_energy=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.to(device='cuda')

    # convert list of RDKit molecules to a batch
    z, pos, m, batch, q = mols_to_batch(mols)

    # minimize
    minimized_pos, energy_trajectories = optimize_geometries(model, z, pos, batch,q)

    # convert back to RDKit
    mols, energies_per_mol = batch_to_mols(z, minimized_pos, batch, energy_trajectories, mols)

    return mols, energies_per_mol



def rdkit_confgen(smiles, N):
    """Standard RDKit confgen method"""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    conformers = rdDistGeom.EmbedMultipleConfs(mol, useRandomCoords=True, numConfs=N, numThreads=8)
    
    return mol


if __name__ == "__main__":
    N = 10
    smiles_list = [
     'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
     'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5'
    ]
    # 2 molecules each with 10 confs
    mols = [rdkit_confgen(smiles,N) for smiles in smiles_list]

    from huggingface_hub import hf_hub_download

    model_file_path = hf_hub_download(
        repo_id="Acellera/AceFF-1.1",
        filename="aceff_v1.1.ckpt"
    )
    mols, energies_per_mol = batch_minim(mols, model_file_path)


    # plot the energy trajectories to check convergence
    for i,energies_per_mol in enumerate(energies_per_mol):
        plt.figure()
        plt.plot(energies_per_mol)
        plt.savefig(f'energies_{i}.png')
    