# SPDX-FileCopyrightText: 2025 Hannes Lindenblatt
#
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import pytest
import scipp as sc
from chemformula import ChemFormula
from scipp import constants
from scipp.testing import assert_allclose

from redcamel.remi_calculator import RemiCalculator
from redcamel.remi_particles import (
    Electron,
    Ion,
    sample_coulomb_explosion,
    sample_lonely_particle,
    sample_photoionization,
)


@pytest.fixture(
    params=[0.0, -100.0, +200.0], ids=["no magnet", "neg. magnetic field", "pos. magnetic field"]
)
def magnet_settings(request):
    return {"magnetic_field": (request.param, "G")}


@pytest.fixture
def remi(magnet_settings):
    kwargs = {}
    for key, (value, unit) in magnet_settings.items():
        kwargs[key] = sc.scalar(value=value, unit=unit)
    return RemiCalculator(
        length_acceleration_ion=sc.scalar(1.0, unit="m"),
        length_drift_ion=sc.scalar(0.0, unit="m"),
        voltage_ion=sc.scalar(-20.0, unit="V"),
        length_acceleration_electron=sc.scalar(1.0, unit="m"),
        length_drift_electron=sc.scalar(0.0, unit="m"),
        voltage_electron=sc.scalar(+20.0, unit="V"),
        v_jet=sc.scalar(1.0, unit="m/s"),
        jet_direction="+x",
        field_direction="+z",
        **kwargs,
    )


@pytest.fixture
def nitrogen_coincidence(remi):
    coulomb_explosion = sample_coulomb_explosion(
        fragment_formulas=[ChemFormula("N"), ChemFormula("N")],
        charge_counts=[1, 2],
        kinetic_energy_release=sc.scalar(3.456, unit="eV"),
        energy_width=sc.scalar(0.5, unit="eV"),
        sizes={"events": 1000, "someparameter": 5, "p": 1},
        remi=remi,
    )
    coulomb_explosion.calculate_detector_hits()
    return coulomb_explosion


@pytest.fixture
def photoionization_coincidence(remi):
    coin = sample_photoionization(
        atom_formula=ChemFormula("He"),
        binding_energy=sc.scalar(10.0, unit="eV"),
        photon_energy=sc.scalar(15.0, unit="eV"),
        energy_width=sc.scalar(0.15, unit="eV"),
        sizes={"events": 1000, "someparameter": 5, "p": 1},
        remi=remi,
    )
    coin.calculate_detector_hits()
    return coin


def test_float_mass_ion(remi):
    testparticle = Ion(formula=123.456, charge_count=1, remi=remi)
    sample_lonely_particle(
        testparticle, sc.scalar(10.0, unit="eV"), sc.scalar(10.0, unit="eV"), sizes={"stuff": 20}
    )
    testparticle.calculate_detector_hits()


@pytest.fixture
def coincidence_test_cases(photoionization_coincidence, nitrogen_coincidence):
    return [photoionization_coincidence, nitrogen_coincidence]


def test_unity_transform_p_xyz(coincidence_test_cases):
    for coin in coincidence_test_cases:
        for particle_name, particle in coin.particles.items():
            hits = coin.datagroup[particle_name]
            remi = particle.remi
            converter_graph = remi.make_graph_for_momentum_calculation(
                particle.mass, particle.charge
            )
            hits = hits.transform_coords(["energy"], graph=converter_graph)
            assert_allclose(hits.coords["p_z"], hits.coords["p_long"])
            assert_allclose(hits.coords["p_x"], hits.coords["p_jet"])
            assert_allclose(hits.coords["p_y"], hits.coords["p_trans"])


def test_particles_at_detector(coincidence_test_cases):
    for coin in coincidence_test_cases:
        for particle_name, particle in coin.particles.items():
            hits = coin.datagroup[particle_name]
            remi = particle.remi
            hits = hits.transform_coords(["z"], graph=particle.detector_transformation_graph)
            z = hits.coords["z"]
            if particle.charge_count < 0:
                detector_position = -(
                    remi.length_acceleration_electron + remi.length_drift_electron
                )
            else:
                detector_position = remi.length_acceleration_ion + remi.length_drift_ion
            assert_allclose(z, sc.zeros_like(z) + detector_position)


def test_drifting_electron():
    remi = RemiCalculator(
        length_acceleration_ion=sc.scalar(1.0, unit="m"),
        length_drift_ion=sc.scalar(0.0, unit="m"),
        voltage_ion=sc.scalar(-1e-30, unit="V"),
        length_acceleration_electron=sc.scalar(1.0, unit="m"),
        length_drift_electron=sc.scalar(0.0, unit="m"),
        voltage_electron=sc.scalar(+1e-30, unit="V"),
        v_jet=sc.scalar(0.0, unit="m/s"),
        magnetic_field=sc.scalar(0.0, unit="G"),
        jet_direction="+x",
        field_direction="+z",
    )
    tester = Electron(remi=remi)
    v_vec = sc.vectors(dims=["p"], values=[[1.0, 1.0, -1.0]], unit="m/s")
    p_vec = v_vec * tester.mass
    p_vec = p_vec.to(unit="au momentum")
    tester.momentum_sample = sc.DataArray(data=sc.ones(sizes=p_vec.sizes), coords={"p": p_vec})
    tester.calculate_detector_hits()
    hits = tester.detector_hits
    tof = hits.coords["tof"].to(unit="s")
    x = hits.coords["x"].to(unit="m")
    y = hits.coords["y"].to(unit="m")
    for coord in [x, y, tof]:
        assert_allclose(coord, sc.ones_like(coord))


def test_drifting_ion():
    remi = RemiCalculator(
        length_acceleration_ion=sc.scalar(1.0, unit="m"),
        length_drift_ion=sc.scalar(0.0, unit="m"),
        voltage_ion=sc.scalar(-1e-30, unit="V"),
        length_acceleration_electron=sc.scalar(1.0, unit="m"),
        length_drift_electron=sc.scalar(0.0, unit="m"),
        voltage_electron=sc.scalar(+1e-30, unit="V"),
        v_jet=sc.scalar(0.0, unit="m/s"),
        magnetic_field=sc.scalar(0.0, unit="G"),
        jet_direction="+x",
        field_direction="+z",
    )
    tester = Ion(ChemFormula("He"), +1, remi=remi)
    v_vec = sc.vectors(dims=["p"], values=[[1.0, 1.0, 1.0]], unit="m/s")
    p_vec = v_vec * tester.mass
    p_vec = p_vec.to(unit="au momentum")
    tester.momentum_sample = sc.DataArray(data=sc.ones(sizes=p_vec.sizes), coords={"p": p_vec})
    tester.calculate_detector_hits()
    hits = tester.detector_hits
    tof = hits.coords["tof"].to(unit="s")
    x = hits.coords["x"].to(unit="m")
    y = hits.coords["y"].to(unit="m")
    for coord in [x, y, tof]:
        assert_allclose(coord, sc.ones_like(coord))


def test_full_rotating_electron():
    cyclotron_period = sc.scalar(1.0, unit="s")
    magnetic_field = 2 * np.pi * constants.m_e / (constants.e * cyclotron_period)
    remi = RemiCalculator(
        length_acceleration_ion=sc.scalar(1.0, unit="m"),
        length_drift_ion=sc.scalar(0.0, unit="m"),
        voltage_ion=sc.scalar(-1e-30, unit="V"),
        length_acceleration_electron=sc.scalar(1.0, unit="m"),
        length_drift_electron=sc.scalar(0.0, unit="m"),
        voltage_electron=sc.scalar(+1e-30, unit="V"),
        v_jet=sc.scalar(0.0, unit="m/s"),
        magnetic_field=magnetic_field.to(unit="G"),
        jet_direction="+x",
        field_direction="+z",
    )
    tester = Electron(remi=remi)
    v_vec = sc.vectors(dims=["p"], values=[[1.0, 1.0, -1.0]], unit="m/s")
    p_vec = v_vec * tester.mass
    p_vec = p_vec.to(unit="au momentum")
    tester.momentum_sample = sc.DataArray(data=sc.ones(sizes=p_vec.sizes), coords={"p": p_vec})
    tester.calculate_detector_hits()
    hits = tester.detector_hits
    tof = hits.coords["tof"].to(unit="s")
    x = hits.coords["x"].to(unit="m")
    y = hits.coords["y"].to(unit="m")
    assert_allclose(tof, sc.ones_like(tof))
    for coord in [x, y]:
        assert_allclose(coord, sc.zeros_like(coord), atol=sc.scalar(1e-15, unit="m"))


def test_half_rotating_electron():
    cyclotron_period = sc.scalar(2.0, unit="s")
    magnetic_field = 2 * np.pi * constants.m_e / (constants.e * cyclotron_period)
    remi = RemiCalculator(
        length_acceleration_ion=sc.scalar(1.0, unit="m"),
        length_drift_ion=sc.scalar(0.0, unit="m"),
        voltage_ion=sc.scalar(-1e-30, unit="V"),
        length_acceleration_electron=sc.scalar(1.0, unit="m"),
        length_drift_electron=sc.scalar(0.0, unit="m"),
        voltage_electron=sc.scalar(+1e-30, unit="V"),
        v_jet=sc.scalar(0.0, unit="m/s"),
        magnetic_field=magnetic_field.to(unit="G"),
        jet_direction="+x",
        field_direction="+z",
    )
    tester = Electron(remi=remi)
    v_vec = sc.vectors(dims=["p"], values=[[1.0, 0.0, -1.0]], unit="m/s")
    p_vec = v_vec * tester.mass
    p_vec = p_vec.to(unit="au momentum")
    tester.momentum_sample = sc.DataArray(data=sc.ones(sizes=p_vec.sizes), coords={"p": p_vec})
    tester.calculate_detector_hits()
    hits = tester.detector_hits
    tof = hits.coords["tof"].to(unit="s")
    x = hits.coords["x"].to(unit="m")
    y = hits.coords["y"].to(unit="m")

    assert_allclose(tof, sc.ones_like(tof))
    assert_allclose(x, sc.zeros_like(x), atol=sc.scalar(1e-15, unit="m"))
    cyclotron_radius = sc.sqrt(p_vec.fields.x**2 + p_vec.fields.y**2) / (
        constants.e * magnetic_field
    )
    # Looking from above, negative charges go clockwise at positive magnetic field
    assert_allclose(y, -2 * cyclotron_radius.to(unit="m"))
