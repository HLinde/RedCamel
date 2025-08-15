#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Hannes Lindenblatt
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from chemformula import ChemFormula
import scipp as sc
from .units import get_mass


class Particle:
    momentum_sample: sc.DataArray
    detector_hits: sc.DataArray
    detector_transformation_graph: dict

    def __init__(self, formula: ChemFormula, charge: int, *, energy=None, color=None, name=None):
        self.formula = formula
        if name is None:
            self.name = str(self.formula)
        self.mass = get_mass(formula)
        self.charge = charge * sc.constants.e
        self.formula.charge = charge
        self.color = color
        self.energy = energy

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
    def __init__(self, **kwargs):
        super().__init__(ChemFormula("e"), -1, **kwargs)


class Ion(Particle):
    pass
