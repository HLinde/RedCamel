#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Hannes Lindenblatt
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from chemformula import ChemFormula
import scipp as sc
from scipp import constants

m_e = constants.m_e.value  # electron_mass
q_e = constants.e.value  # elementary_charge
amu = 1.66053906660e-27  # atomic mass unit
sc.units.aliases["au momentum"] = constants.physical_constants("atomic unit of momentum")
sc.units.aliases["au energy"] = constants.physical_constants("atomic unit of energy")
sc.units.aliases["au mass"] = constants.m_e


def get_mass(formula: ChemFormula) -> sc.Variable:
    if formula.formula == "e":
        mass_amu = (constants.m_e).to(unit="u")
    else:
        mass_amu = formula.formula_weight * sc.Unit("u")
    return mass_amu
