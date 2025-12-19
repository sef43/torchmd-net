# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
from torchmdnet.models.model import load_model
import warnings
import numpy as np

# dict of preset transforms
tranforms = {
    "eV/A -> kcal/mol/A": lambda energy, forces: (
        energy * 23.0609,
        forces * 23.0609,
    ),  # eV->kcal/mol, eV/A -> kcal/mol/A
    "Hartree/Bohr -> kcal/mol/A": lambda energy, forces: (
        energy * 627.509,
        forces * 627.509 / 0.529177,
    ),  # Hartree -> kcal/mol, Hartree/Bohr -> kcal/mol/A
    "Hartree/A -> kcal/mol/A": lambda energy, forces: (
        energy * 627.509,
        forces * 627.509,
    ),  # Hartree -> kcal/mol, Hartree/A -> kcal/mol/A
}


class External:
    """This is an adapter to use TorchMD-Net models in TorchMD.

    Parameters
    ----------
    netfile : str or torch.nn.Module
        Path to the checkpoint file of the model or the model itself.
    embeddings : torch.Tensor
        Embeddings of the atoms in the system.
    device : str, optional
        Device on which the model should be run. Default: "cpu"
    output_transform : str or callable, optional
        Transform to apply to the energy and forces.
        If a string is given, it should be a key in the `transforms` dict.
        If a callable is given, it should take two arguments (energy and forces) and return two tensors of the same shape.
        Default: None
    use_cuda_graph : bool, optional
        Whether to use CUDA graphs to speed up the calculation. Default: False
    cuda_graph_warmup_steps : int, optional
        Number of steps to run as warmup before recording the CUDA graph. Default: 12
    dtype : torch.dtype or str, optional
        Cast the input to this dtype if defined. If passed as a string it should be a valid torch dtype. Default: torch.float32
    kwargs : dict, optional
        Extra arguments to pass to the model when loading it.
    """

    def __init__(
        self,
        netfile,
        embeddings,
        device="cpu",
        output_transform=None,
        use_cuda_graph=False,
        cuda_graph_warmup_steps=12,
        dtype=torch.float32,
        **kwargs,
    ):
        if isinstance(netfile, str):
            extra_args = kwargs
            if use_cuda_graph:
                warnings.warn(
                    "CUDA graphs are enabled, setting static_shapes=True and check_errors=False"
                )
                extra_args["static_shapes"] = True
                extra_args["check_errors"] = False
            self.model = load_model(
                netfile,
                device=device,
                derivative=True,
                **extra_args,
            )
        elif isinstance(netfile, torch.nn.Module):
            if kwargs:
                warnings.warn(
                    "Warning: extra arguments are being ignored when passing a torch.nn.Module"
                )
            self.model = netfile
        else:
            raise ValueError(
                f"Expected a path to a checkpoint file or a torch.nn.Module, got {type(netfile)}"
            )
        self.device = device
        self.n_atoms = embeddings.size(1)
        self.embeddings = embeddings.reshape(-1).to(device)
        self.batch = torch.arange(embeddings.size(0), device=device).repeat_interleave(
            embeddings.size(1)
        )
        self.model.eval()

        if not output_transform:
            self.output_transformer = lambda energy, forces: (
                energy,
                forces,
            )  # identity
        elif output_transform in tranforms.keys():
            self.output_transformer = tranforms[output_transform]
        else:
            self.output_transformer = eval(output_transform)
        if not torch.cuda.is_available() and use_cuda_graph:
            raise ValueError("CUDA graphs are only available if CUDA is")
        self.use_cuda_graph = use_cuda_graph
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps
        self.cuda_graph = None
        self.energy = None
        self.forces = None
        self.box = None
        self.pos = None
        if isinstance(dtype, str):
            try:
                dtype = getattr(torch, dtype)
            except AttributeError:
                raise ValueError(f"Unknown torch dtype {dtype}")
        self.dtype = dtype

    def _init_cuda_graph(self):
        stream = torch.cuda.Stream()
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            for _ in range(self.cuda_graph_warmup_steps):
                self.energy, self.forces = self.model(
                    self.embeddings, self.pos, self.batch, self.box
                )
            with torch.cuda.graph(self.cuda_graph):
                self.energy, self.forces = self.model(
                    self.embeddings, self.pos, self.batch, self.box
                )

    def calculate(self, pos, box=None):
        """Calculate the energy and forces of the system.

        Parameters
        ----------
        pos : torch.Tensor
            Positions of the atoms in the system.
        box : torch.Tensor, optional
            Box vectors of the system. Default: None

        Returns
        -------
        energy : torch.Tensor
            Energy of the system.
        forces : torch.Tensor
            Forces on the atoms in the system.
        """
        pos = pos.to(self.device).to(self.dtype).reshape(-1, 3)
        if box is not None:
            box = box.to(self.device).to(self.dtype)
        if self.use_cuda_graph:
            if self.pos is None:
                self.pos = (
                    pos.clone()
                    .to(self.device)
                    .detach()
                    .requires_grad_(pos.requires_grad)
                )
            if self.box is None and box is not None:
                self.box = box.clone().to(self.device).to(self.dtype).detach()
            if self.cuda_graph is None:
                self._init_cuda_graph()
            assert (
                self.cuda_graph is not None
            ), "CUDA graph is not initialized. This should not had happened."
            with torch.no_grad():
                self.pos.copy_(pos)
                if box is not None:
                    self.box.copy_(box)
                self.cuda_graph.replay()
        else:
            self.energy, self.forces = self.model(self.embeddings, pos, self.batch, box)
        assert self.forces is not None, "The model is not returning forces"
        assert self.energy is not None, "The model is not returning energy"
        return self.output_transformer(
            self.energy.clone().detach(),
            self.forces.clone().reshape(-1, self.n_atoms, 3).detach(),
        )


import ase.calculators.calculator as ase_calc


class TMDNETCalculator(ase_calc.Calculator):
    """This is an adapter to use TorchMD-Net models as an ASE Calculator.

    Parameters
    ----------
    model_file : str
        Path to the checkpoint file of the model.
    device : str, optional
        Device on which the model should be run. Default: "cpu"
    dtype : torch.dtype or str, optional
        Cast the input to this dtype if defined. If passed as a string it should be a valid torch dtype. Default: torch.float32
    compile: bool
        Whether to use torch.compile to speed up calcuations, useful for molecular dynamics. Default: False
    kwargs : dict, optional
        Extra arguments to pass to the model when loading it.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model_file,
        device="cpu",
        dtype=torch.float32,
        compile=False,
        **kwargs,
    ):

        ase_calc.Calculator.__init__(self)
        self.device = device
        self.model = load_model(
            model_file,
            derivative=False,
            remove_ref_energy=kwargs.get("remove_ref_energy", True),
            max_num_neighbors=kwargs.get("max_num_neighbors", 64),
            static_shapes=True if compile else False,
            check_errors=False if compile else True,
            **kwargs,
        )
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.to(device=self.device, dtype=dtype)

        if compile:
            self.compile = True
        else:
            self.compile = False

        self.compiled = False

        self.evals = 0

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=ase_calc.all_changes,
    ):

        if not properties:
            properties = ["energy"]

        ase_calc.Calculator.calculate(self, atoms, properties, system_changes)

        numbers = atoms.numbers

        positions = atoms.positions
        total_charge = atoms.info["charge"]
        batch = [0 for _ in range(len(numbers))]

        batch = torch.tensor(batch, device=self.device, dtype=torch.long)
        numbers = torch.tensor(numbers, device=self.device, dtype=torch.long)
        positions = torch.tensor(
            positions, device=self.device, dtype=torch.float32, requires_grad=True
        )
        total_charge = torch.tensor(
            [total_charge], device=self.device, dtype=torch.float32
        )

        if self.compile and "numbers" in system_changes:
            if self.evals == 0:
                print("compiling...")
            else:
                print("atomic numbers changed, re-compiling...")

            self.model.representation_model.setup_for_compile_cudagraphs(batch)
            self.model.output_model.setup_for_compile_cudagraphs(batch)
            self.model.to(self.device)
            self.compiled_model = torch.compile(
                self.model,
                backend="inductor",
                dynamic=False,
                fullgraph=True,
                mode="reduce-overhead",
            )
            self.compiled = True

        if self.compiled:
            energy, _ = self.compiled_model(
                numbers, positions, batch=batch, q=total_charge
            )
        else:
            energy, _ = self.model(numbers, positions, batch=batch, q=total_charge)

        energy.backward()
        forces = -positions.grad

        self.results["energy"] = energy.detach().cpu().item()
        self.results["forces"] = forces.detach().cpu().numpy()
        self.evals += 1


def optimize_geometries(model, z, pos, batch, q, interations=10):
    """function to opmize geometries using torch.optim"""

    pos = pos.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS([pos])

    energies = []

    def closure():
        optimizer.zero_grad()
        energy, _ = model(z, pos, batch, q=q)

        energies.append(energy.detach().cpu().numpy())

        energy = energy.sum()
        energy.backward()

        return energy

    for i in range(interations):
        optimizer.step(closure)

    energies = np.array(energies).squeeze(-1)

    return pos.detach().cpu().numpy(), energies


# Integration functions copied from TorchMD and modified for batch use

TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191
PICOSEC2TIMEU = 1000.0 / TIMEFACTOR


def _first_VV(pos, vel, force, mass, dt):
    accel = force / mass
    pos += vel * dt + 0.5 * accel * dt * dt
    vel += 0.5 * dt * accel


def _second_VV(vel, force, mass, dt):
    accel = force / mass
    vel += 0.5 * dt * accel


def langevin(vel, gamma, coeff, dt, device):
    csi = torch.randn_like(vel, device=device) * coeff
    vel += -gamma * vel * dt + csi


def batched_kinetic_energy(masses, vel, batch):
    v_sq = torch.sum(vel**2, dim=1)
    E_per_node = 0.5 * masses.squeeze(-1) * v_sq
    n_batch = int(torch.max(batch).item() + 1)
    Ekin = torch.zeros(n_batch, device=vel.device, dtype=vel.dtype).index_add(
        0, batch, E_per_node
    )
    return Ekin


def batched_kinetic_to_temp(Ekin, natoms):
    return 2.0 / (3.0 * natoms * BOLTZMAN) * Ekin


def kinetic_to_temp(Ekin, natoms):
    return 2.0 / (3.0 * natoms * BOLTZMAN) * Ekin


class BatchedMLIPIntegrator:
    """Batched Integrator for MLIP systems

    modified from https://github.com/torchmd/torchmd
    """

    def __init__(
        self,
        model_file_path,
        z,
        pos,
        masses,
        batch,
        q,
        timestep,
        device="cuda",
        gamma=None,
        T=None,
    ):

        # load the AceFF model
        self.model = load_model(
            model_file_path,
            derivative=True,
            check_errors=True,
            static_shapes=False,
            max_num_neighbors=64,
            remove_ref_energy=True,
        )
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.to(device)
        self.z = z
        self.M = masses.unsqueeze(-1)
        self.pos = pos
        self.batch = batch
        self.q = q
        self.box = None
        self.device = device
        self.dt = timestep / TIMEFACTOR
        gamma = gamma / PICOSEC2TIMEU
        self.gamma = gamma
        self.T = T
        if T:
            self.vcoeff = torch.sqrt(2.0 * gamma / self.M * BOLTZMAN * T * self.dt).to(
                device
            )
        self.vel = torch.zeros_like(pos)
        self.forces = torch.zeros_like(pos)
        self.n_atoms_per_batch = torch.bincount(batch)
        print(self.n_atoms_per_batch)

    def step(self, niter=1):
        for _ in range(niter):
            _first_VV(self.pos, self.vel, self.forces, self.M, self.dt)

            _pos = self.pos.clone().requires_grad_(True)
            pot, f = self.model(self.z, _pos, self.batch, q=self.q)

            self.forces = f.clone().detach()

            if self.T:
                langevin(self.vel, self.gamma, self.vcoeff, self.dt, self.device)
            _second_VV(self.vel, self.forces, self.M, self.dt)

        Ekin = batched_kinetic_energy(self.M, self.vel, self.batch)
        T = kinetic_to_temp(Ekin, self.n_atoms_per_batch)

        return (
            Ekin.detach().cpu().numpy(),
            pot.detach().squeeze(-1).cpu().numpy(),
            T.detach().cpu().numpy(),
        )
