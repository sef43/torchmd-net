# This script demonstrates how to do batched MD of molecules with AceFF torchmd-net models

import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom
import torch
from torchmdnet.models.model import load_model
from torchmdnet.calculators import BatchedMLIPIntegrator
from torchmdnet.utils import mols_to_batch, batch_to_mols


def rdkit_confgen(smiles, N):
    """Standard RDKit confgen method"""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    conformers = rdDistGeom.EmbedMultipleConfs(
        mol, useRandomCoords=True, numConfs=N, numThreads=8
    )

    return mol


if __name__ == "__main__":

    # number of conformers of each molecule
    N = 10
    smiles_list = [
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
    ]
    # 2 molecules each with N confs
    mols = [rdkit_confgen(smiles, N) for smiles in smiles_list]

    from huggingface_hub import hf_hub_download

    model_file_path = hf_hub_download(
        repo_id="Acellera/AceFF-1.1", filename="aceff_v1.1.ckpt"
    )

    # convert mols to batch
    z, pos, m, batch, q = mols_to_batch(mols)

    # create the integrator
    langevin_temperature = 300  # K
    langevin_gamma = 1.0  # 1/ps
    timestep = 1  # fs
    integrator = BatchedMLIPIntegrator(
        model_file_path,
        z,
        pos,
        m,
        batch,
        q,
        device="cuda",
        timestep=timestep,
        gamma=langevin_gamma,
        T=langevin_temperature,
    )

    # run the MD, 10 lots of 100 steps
    inner_steps = 100
    for i in range(10):
        t1 = time.perf_counter()
        Ekin, pot, T = integrator.step(inner_steps)
        t2 = time.perf_counter()
        print("step:", (i + 1) * inner_steps)
        print("energies:", pot)
        print("T:", T)
        print(f"time per step: {(t2-t1)/inner_steps*1000} ms")
