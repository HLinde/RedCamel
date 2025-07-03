# SPDX-FileCopyrightText: 2025 Hannes Lindenblatt
#
# SPDX-License-Identifier: GPL-3.0-or-later
import pytest
import chemformula
from redcamel import get_mass_amu


def test_get_mass_amu():
    """Test the electron mass."""
    assert get_mass_amu(chemformula.ChemFormula("e")) == pytest.approx(1 / 1822.888)
    """ Test the hydrogen mass."""
    assert get_mass_amu(chemformula.ChemFormula("H")) == pytest.approx(1.008)
