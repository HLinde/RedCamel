# SPDX-FileCopyrightText: 2025 Hannes Lindenblatt
#
# SPDX-License-Identifier: GPL-3.0-or-later
import chemformula
import pytest

from redcamel import get_mass


def test_get_mass_amu():
    """Test the electron mass."""
    assert get_mass(chemformula.ChemFormula("e")).to(unit="au mass").value == pytest.approx(1)
    """ Test the hydrogen mass."""
    assert get_mass(chemformula.ChemFormula("H")).to(unit="amu").value == pytest.approx(1.008)
