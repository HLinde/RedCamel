#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Hannes Lindenblatt
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from chemformula import ChemFormula
import scipp as sc
import numpy as np
from .units import get_mass
from .remi_scipp_helpers import make_scipp_detector_converters


class Particle:
    def __init__(self, formula: ChemFormula, charge: int, *, energy=None, color=None, name=None):
        self.formula = formula
        if name is None:
            self.name = str(self.formula)
        self.mass = get_mass(formula)
        self.charge = charge * sc.constants.e
        self.formula.charge = charge
        self.color = color
        self.energy = energy

    def make_detector_converter(
        self, length_acceleration, length_drift, electric_field, magnetic_field
    ):
        calc_tof, calc_xyR = make_scipp_detector_converters(
            length_acceleration,
            length_drift,
            electric_field,
            magnetic_field,
            self.mass.to(unit="kg"),
            self.charge.to(unit="C"),
        )
        self.converter_graph = {"tof": calc_tof, ("x", "y", "R"): calc_xyR}

    @property
    def latex(self):
        tex_string = self.formula.latex
        tex_string = tex_string.replace("textnormal", "text")
        return tex_string

    def sample_momentum(self, sizes):
        assert self.energy is not None
        mean_energy = self.energy
        dims, shape = zip(sizes.items())
        energy_width = mean_energy / 10
        energy_sample = sc.array(
            dims=dims, values=np.random.randn(*shape) * energy_width + mean_energy
        )
        phi = sc.array(dims=dims, values=np.random.rand(*shape) * 2 * np.pi, unit="rad")
        cos_theta = sc.array(dims=dims, values=np.random.rand(*shape) * 2 - 1)

        r_mom = sc.sqrt(energy_sample * 2 * self.mass)
        theta = sc.acos(cos_theta)

        x = r_mom * sc.sin(theta) * sc.cos(phi)
        y = r_mom * sc.sin(theta) * sc.sin(phi)
        z = r_mom * cos_theta
        self.momentum = sc.spatial.as_vectors(x, y, z)


class Electron(Particle):
    def __init__(self, **kwargs):
        super().__init__("e", -1, **kwargs)


class Coincidence:
    def __init__(
        self,
        name,
        ion_formulas,
        charges,
        *,
        nuclear_energy=None,
        electron_energies=None,
        colors=None,
    ):
        self.name = name
        assert len(ion_formulas) == len(charges)
        if len(ion_formulas > 2):
            raise NotImplementedError("Don't now how to distribute energy.")

        if nuclear_energy is not None:
            nuclear_energy = sc.scalar(nuclear_energy, unit="eV")
        self.nuclear_energy = nuclear_energy

        if electron_energies is None:
            electron_energies = electron_energies
        elif isinstance(electron_energies, sc.Variable):
            electron_energies = electron_energies
        else:
            electron_energies = sc.array(dims=["E"], values=electron_energies, unit="eV")

        if colors is None:
            colors = [None for _ in charges]
        self.colors = colors

        if electron_energies is None:
            self.electrons = []
        else:
            n_electrons = sum(charges)
            assert len(electron_energies) == n_electrons
            self.electrons = [
                Electron(name=f"e_{i}", energy=energy) for i, energy in enumerate(electron_energies)
            ]

        unique_formulas, formula_counts = np.unique(ion_formulas, return_counts=True)
        if np.any(formula_counts > 1):
            multiple_ions = unique_formulas[formula_counts > 1]
            previous_counts = {formula: 0 for formula in multiple_ions}
            ion_names = []
            for formula in ion_formulas:
                if formula in multiple_ions:
                    i = previous_counts[formula] + 1
                    ion_names.append(f"{formula}_{i}")
                    previous_counts[formula] = i
                else:
                    ion_names.append(formula)
        else:
            ion_names = ion_formulas

        ion_chemformulas = map(ChemFormula, ion_formulas)

        self.ions = [
            Particle(formula=formula, name=name, charge=charge, color=color)
            for formula, name, charge, color in zip(ion_chemformulas, ion_names, charges, colors)
        ]

    def make_detector_converters(
        self, length_acceleration, length_drift, electric_field, magnetic_field
    ):
        for particle in self.particles:
            particle.make_detector_converter(
                length_acceleration, length_drift, electric_field, magnetic_field
            )

    def sample(self, sizes: dict, *, v_jet: sc.vector):
        assert self.nuclear_energy is not None
        # sample electrons without correlation
        if self.electrons:
            for ele in self.electrons:
                ele.sample_momentum(sizes)
        # sample correlated momenta
        dims, shape = zip(sizes.items())
        energy_width = self.nuclear_energy / 10
        kinetic_energy = sc.array(
            dims=dims, values=np.random.randn(*shape) * energy_width + self.nuclear_energy
        )
        self.sample_two_body_fragmentation(kinetic_energy, self.ions[0], self.ions[1])
        self.dataset = sc.Dataset()

    def sample_two_body_fragmentation(energy, particle_1, particle_2):
        pass
