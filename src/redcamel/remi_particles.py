#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Hannes Lindenblatt
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from typing import Iterable
from chemformula import ChemFormula
import numpy as np
import scipp as sc
from .units import get_mass
from .remi_calculator import RemiCalculator


class Particle:
    momentum_sample: sc.DataArray
    detector_hits: sc.DataArray

    def __init__(
        self,
        formula: ChemFormula,
        charge_count: int,
        remi: RemiCalculator,
        *,
        energy=None,
        color=None,
        name=None,
    ):
        self.formula = formula
        if name is None:
            self.name = str(self.formula)
        self.mass = get_mass(formula)
        self.charge_count = charge_count
        self.charge = charge_count * sc.constants.e
        self.formula.charge = charge_count
        self.color = color
        self.energy = energy
        self.remi = remi
        self.detector_transformation_graph = remi.make_scipp_graph_for_detector(
            self.mass, self.charge
        )

    @property
    def latex(self):
        tex_string = self.formula.latex
        tex_string = tex_string.replace("textnormal", "text")
        return tex_string

    def calculate_detector_hits(self):
        self.detector_hits = self.momentum_sample.transform_coords(
            ["x", "y", "tof", "R"], graph=self.detector_transformation_graph
        )


class Electron(Particle):
    def __init__(self, remi: RemiCalculator, **kwargs):
        super().__init__(ChemFormula("e"), -1, remi, **kwargs)


class Ion(Particle):
    pass


class Coincidence:
    def __init__(self, name, ions: Iterable[Ion], electrons: Iterable[Electron]):
        self.name = name

        self.ions = {}
        self.ion_counter = {}
        for ion in ions:
            if ion.name not in self.ions:
                self.ions[ion.name] = ion
                self.ion_counter[ion.name] = 1
            else:
                new_name = f"{ion.name}_{self.ion_counter[ion.name]}"
                self.ions[new_name] = ion
                self.ion_counter[ion.name] += 1

        self.electrons = {}
        self.electron_counter = {}
        for electron in electrons:
            if electron.name not in self.electrons:
                self.electrons[electron.name] = electron
                self.electron_counter[electron.name] = 1
            else:
                new_name = f"{electron.name}_{self.electron_counter[electron.name]}"
                self.electrons[new_name] = electron
                self.electron_counter[electron.name] += 1

        self.particles = {}
        self.particles.update(self.ions)
        self.particles.update(self.electrons)

    def calculate_detector_hits(self):
        for part in self.particles.values():
            part.calculate_detector_hits()

    @property
    def datagroup(self) -> sc.DataGroup:
        return sc.DataGroup({name: part.detector_hits for name, part in self.particles.items()})


def sample_photoionization(
    atom_formula: ChemFormula,
    binding_energy: sc.Variable,
    photon_energy: sc.Variable,
    energy_width: sc.Variable,
    sizes: dict,
    remi: RemiCalculator,
    name: str = None,
    color=None,
) -> Coincidence:
    # TODO handle higher charge states
    dims, shape = zip(*sizes.items())
    assert "p" in dims
    assert sizes["p"] == 1

    mean_kinetic_energy = photon_energy - binding_energy
    kinetic_energy = (
        sc.array(dims=dims, values=np.random.randn(*shape)) * energy_width + mean_kinetic_energy
    )
    kinetic_energy = sc.where(
        kinetic_energy < sc.scalar(0, unit="eV"),
        sc.scalar(np.nan, unit="eV"),  # those electrons didn't make it out of the atom
        kinetic_energy,
    )

    ion = Ion(atom_formula, charge_count=1, remi=remi, color=color)
    electron = Electron(remi=remi)

    if name is None:
        name = "_".join([ion.name, electron.name])
    sample_two_body_fragmentation(kinetic_energy, ion, electron)

    return Coincidence(name, ions=[ion], electrons=[electron])


def sample_coulomb_explosion(
    fragment_formulas: Iterable[ChemFormula],
    charge_counts: Iterable[int],
    kinetic_energy_release: sc.Variable,
    energy_width: sc.Variable,
    sizes: dict,
    remi: RemiCalculator,
    name: str = None,
    colors=None,
) -> Coincidence:
    # TODO handle higher charge states
    dims, shape = zip(*sizes.items())
    assert "p" in dims
    assert sizes["p"] == 1
    assert len(fragment_formulas) > 1

    if colors is None:
        colors = [None for _ in fragment_formulas]
    assert len(colors) == len(fragment_formulas)

    if len(fragment_formulas) > 2:
        raise NotImplementedError("Default behaviour for many particles is not implemented.")

    kinetic_energy = (
        sc.array(dims=dims, values=np.random.randn(*shape)) * energy_width + kinetic_energy_release
    )
    kinetic_energy = sc.where(
        kinetic_energy < sc.scalar(0, unit="eV"),
        sc.scalar(np.nan, unit="eV"),  # those electrons didn't make it out of the atom
        kinetic_energy,
    )

    ions = [
        Ion(formula, charge_count=charge, remi=remi, color=color)
        for formula, charge, color in zip(fragment_formulas, charge_counts, colors)
    ]

    if name is None:
        name = "_".join(ion.name for ion in ions)
    sample_two_body_fragmentation(kinetic_energy, ions[0], ions[1])

    return Coincidence(name, ions=ions, electrons=[])


def sample_lonely_particle(
    particle: Particle, energy_mean: sc.Variable, energy_width: sc.Variable, sizes: dict
):
    dims, shape = zip(*sizes.items())
    energy = sc.array(dims=dims, values=np.random.randn(*shape) * energy_width + energy_mean)
    absolute_momentum = sc.sqrt(2 * energy * particle.mass)
    momentum_vectors = sample_random_momentum_vectors(absolute_momentum)
    particle.momentum_sample = sc.DataArray(
        sc.ones(dims=dims, shape=shape), coords={"p": momentum_vectors}
    )


def sample_two_body_fragmentation(
    kinetic_energy: sc.Variable, particle_1: Particle, particle_2: Particle
):
    absolute_momentum = sc.sqrt(2 * kinetic_energy / (1 / particle_1.mass + 1 / particle_2.mass))
    momentum_1 = sample_random_momentum_vectors(absolute_momentum)
    momentum_2 = -momentum_1
    particle_1.momentum_sample = sc.DataArray(
        data=sc.ones(sizes=kinetic_energy.sizes), coords={"p": momentum_1}
    )
    particle_2.momentum_sample = sc.DataArray(
        data=sc.ones(sizes=kinetic_energy.sizes), coords={"p": momentum_2}
    )


def sample_random_momentum_vectors(absolute_momentum: sc.Variable) -> sc.Variable:
    dims = absolute_momentum.dims
    shape = absolute_momentum.shape
    phi = sc.array(dims=dims, values=np.random.rand(*shape) * 2 * np.pi, unit="rad")
    cos_theta = sc.array(dims=dims, values=np.random.rand(*shape) * 2 - 1)
    theta = sc.acos(cos_theta)

    x = absolute_momentum * sc.sin(theta) * sc.cos(phi)
    y = absolute_momentum * sc.sin(theta) * sc.sin(phi)
    z = absolute_momentum * cos_theta
    return sc.spatial.as_vectors(x, y, z).to(unit="au momentum")
