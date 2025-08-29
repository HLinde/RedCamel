# SPDX-FileCopyrightText: 2025 Hannes Lindenblatt
#
# SPDX-License-Identifier: GPL-3.0-or-later
import pytest
import scipp as sc
from chemformula import ChemFormula
from scipp.testing import assert_allclose

from redcamel.remi_calculator import RemiCalculator
from redcamel.remi_particles import sample_coulomb_explosion


@pytest.fixture(
    params=[0.0, -1.23, +4.56], ids=["no magnet", "neg. magnetic field", "pos. magnetic field"]
)
def remi_settings(request):
    return {"magnetic_field": (request.param, "G")}


@pytest.fixture
def remi(remi_settings):
    kwargs = {}
    for key, (value, unit) in remi_settings.items():
        kwargs[key] = sc.scalar(value=value, unit=unit)
    return RemiCalculator(
        length_acceleration_ion=sc.scalar(0.1, unit="m"),
        length_drift_ion=sc.scalar(0.0, unit="m"),
        voltage_ion=sc.scalar(200.0, unit="V"),
        length_acceleration_electron=sc.scalar(0.2, unit="m"),
        length_drift_electron=sc.scalar(0.0, unit="m"),
        voltage_electron=sc.scalar(400.0, unit="V"),
        v_jet=sc.scalar(1000.0, unit="m/s"),
        jet_direction="+x",
        field_direction="+z",
        **kwargs,
    )


@pytest.fixture
def nitrogen_coincidence(remi):
    return sample_coulomb_explosion(
        fragment_formulas=[ChemFormula("N"), ChemFormula("N")],
        charge_counts=[1, 2],
        kinetic_energy_release=sc.scalar(3.456, unit="eV"),
        energy_width=sc.scalar(0.5, unit="eV"),
        sizes={"events": 1000, "someparameter": 5, "p": 1},
        remi=remi,
    )


def test_unity_transform_p_z(nitrogen_coincidence):
    nitrogen_coincidence.calculate_detector_hits()
    data = nitrogen_coincidence.datagroup
    ion_name = list(data.keys())[0]
    ion = nitrogen_coincidence.ions[ion_name]
    hits = data[ion_name]
    remi = ion.remi
    converter_graph = remi.make_graph_for_momentum_calculation(ion.mass, ion.charge)
    hits = hits.transform_coords(["p_z"], graph=converter_graph)
    assert_allclose(hits.coords["p_z"], hits.coords["p_long"])


def test_unity_transform_p_xy(nitrogen_coincidence):
    nitrogen_coincidence.calculate_detector_hits()
    data = nitrogen_coincidence.datagroup
    ion_name = list(data.keys())[0]
    ion = nitrogen_coincidence.ions[ion_name]
    hits = data[ion_name]
    remi = ion.remi

    converter_graph = remi.make_graph_for_momentum_calculation(ion.mass, ion.charge)
    hits = hits.transform_coords(["energy"], graph=converter_graph)
    print(hits)
    assert_allclose(hits.coords["p_x"], hits.coords["p_jet"])
    assert_allclose(hits.coords["p_y"], hits.coords["p_trans"])
