#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Hannes Lindenblatt
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from typing import Iterable, Literal
from chemformula import ChemFormula
import scipp as sc
import numpy as np
from .remi_particles import Ion, Electron, Particle

CoordinateDirection = Literal["+x", "-x", "+y", "-y", "+z", "-z"]
axis_vectors = {
    "+x": sc.vector([+1, 0, 0]),
    "-x": sc.vector([-1, 0, 0]),
    "+y": sc.vector([0, +1, 0]),
    "-y": sc.vector([0, -1, 0]),
    "+z": sc.vector([0, 0, +1]),
    "-z": sc.vector([0, 0, -1]),
}


class RemiCalculator:
    def __init__(
        self,
        length_acceleration_ion: sc.Variable,
        length_drift_ion: sc.Variable,
        voltage_ion: sc.Variable,
        length_acceleration_electron: sc.Variable,
        length_drift_electron: sc.Variable,
        voltage_electron: sc.Variable,
        magnetic_field: sc.Variable,
        v_jet: sc.Variable,
        jet_direction: CoordinateDirection = "+x",
        field_direction: CoordinateDirection = "+z",
    ):
        self.length_acceleration_ion = length_acceleration_ion
        self.length_drift_ion = length_drift_ion
        self.voltage_ion = voltage_ion
        self.length_acceleration_electron = length_acceleration_electron
        self.length_drift_electron = length_drift_electron
        self.voltage_electron = voltage_electron
        self.magnetic_field = magnetic_field.to(unit="T")
        self.v_jet = v_jet
        self.jet_direction = jet_direction
        self.field_direction = field_direction

    @property
    def jet_unitvector(self):
        return axis_vectors[self.jet_direction]

    @property
    def field_unitvector(self):
        return axis_vectors[self.field_direction]

    @property
    def transverse_unitvector(self):
        return sc.cross(self.field_unitvector, self.jet_unitvector)

    @property
    def voltage_difference(self):
        return self.voltage_ion - self.voltage_electron

    @property
    def length_acceleration_total(self):
        return self.length_acceleration_electron + self.length_acceleration_ion

    @property
    def electric_field(self):
        return self.voltage_difference / self.length_acceleration_total

    def make_scipp_graph_for_detector(self, mass: sc.Variable, charge: sc.Variable):
        graph = {
            "p_jet": lambda p: self.jet_momentum(p),
            "p_trans": lambda p: self.transverse_momentum(p),
            "p_long": lambda p: self.longitudinal_momentum(p),
            "tof": lambda p_long: self.tof(p_long, mass, charge),
            ("x", "y", "R"): lambda tof, p_jet, p_trans: {
                label: func
                for label, func in zip(
                    ("x", "y", "R"), self.hit_position_xyR(tof, p_jet, p_trans, mass, charge)
                )
            },
            "z": lambda p_long, tof: self.position_longitudinal(p_long, tof, mass, charge),
        }
        return graph

    def set_particle_converter_graph(self, particle: Particle):
        particle.detector_transformation_graph = self.make_scipp_graph_for_detector(
            particle.mass, particle.charge
        )

    def longitudinal_momentum(self, momentum: sc.Variable):
        return sc.dot(momentum, self.field_unitvector)

    def jet_momentum(self, momentum: sc.Variable):
        return sc.dot(momentum, self.jet_unitvector)

    def transverse_momentum(self, momentum: sc.Variable):
        return sc.dot(momentum, self.transverse_unitvector)

    def tof(self, momentum_longitudinal: sc.Variable, mass: sc.Variable, charge: sc.Variable):
        momentum_longitudinal = momentum_longitudinal.to(unit="N*s")
        mass = mass.to(unit="kg")
        charge = charge.to(unit="C")
        acceleration_direction = np.sign(charge.value * self.electric_field.value)
        if acceleration_direction > 0:
            length_acceleration = self.length_acceleration_electron
            length_drift = self.length_drift_electron
        else:
            length_acceleration = self.length_acceleration_ion
            length_drift = self.length_drift_ion

        voltage = -acceleration_direction * length_acceleration * self.electric_field
        # TODO add case where particle overcomes opposite acceleration step
        D = momentum_longitudinal * momentum_longitudinal - 2 * charge * voltage * mass
        rootD = sc.sqrt(D)
        tof = sc.where(
            D < 0 * sc.Unit("J*kg"),
            sc.scalar(np.nan, unit="s"),
            mass
            * (
                2 * length_acceleration / (rootD + acceleration_direction * momentum_longitudinal)
                + length_drift / rootD
            ),
        )
        return tof.to(unit="ns")

    def hit_position_xyR(
        self,
        tof: sc.Variable,
        momentum_jet: sc.Variable,
        momentum_transverse: sc.Variable,
        mass: sc.Variable,
        charge: sc.Variable,
    ):
        p_x = momentum_jet + (self.v_jet * mass).to(unit="au momentum")
        p_y = momentum_transverse
        assert p_x.dims == p_y.dims
        dims = p_x.dims

        # cyclotron motion or linear motion?
        if sc.abs(self.magnetic_field) > 0 * sc.Unit("T"):
            p_xy = sc.sqrt(p_x**2 + p_y**2)
            phi = sc.atan2(x=p_x, y=p_y)  # angle in xy-plane towards jet-direction
            omega = self.calc_omega(mass, charge)

            # alpha/2 has to be periodic in 1*pi!
            # sign of alpha is important as it gives the direction of deflection
            # The sign has to be included also in the modulo operation!
            alpha = (omega.to(unit="1/s") * tof.to(unit="s")).values
            alpha = alpha % (np.sign(alpha) * 2 * np.pi)
            alpha = sc.array(dims=dims, values=alpha, unit="rad")

            theta = phi + (alpha / 2)
            # Here the signs of alpha, charge and magnetic_field cancel out so R is positive :)
            R = (2 * p_xy * sc.sin(alpha / 2)) / (charge * self.magnetic_field)
            x = R * sc.cos(theta)
            y = R * sc.sin(theta)
        else:  # for small magnetic field it reduces to this linear motion:
            v_x = p_x / mass
            v_y = p_y / mass
            x = v_x * tof
            y = v_y * tof
            R = sc.sqrt(x**2 + y**2)
        return x.to(unit="mm"), y.to(unit="mm"), R.to(unit="mm")

    def tof_in_acceleration_part(
        self, momentum_longitudinal: sc.Variable, mass: sc.Variable, charge: sc.Variable
    ):
        momentum_longitudinal = momentum_longitudinal.to(unit="N*s")
        mass = mass.to(unit="kg")
        charge = charge.to(unit="C")
        acceleration_direction = np.sign(charge.value * self.electric_field.value)
        if acceleration_direction > 0:
            length_acceleration = self.length_acceleration_electron
        else:
            length_acceleration = self.length_acceleration_ion

        voltage = -acceleration_direction * length_acceleration * self.electric_field
        # TODO add case where particle overcomes opposite acceleration step
        D = momentum_longitudinal * momentum_longitudinal - 2 * charge * voltage * mass
        rootD = sc.sqrt(D)
        tof = sc.where(
            D < 0 * sc.Unit("J*kg"),
            sc.scalar(np.nan, unit="s"),
            mass
            * (2 * length_acceleration / (rootD + acceleration_direction * momentum_longitudinal)),
        )
        return tof.to(unit="ns")

    def position_longitudinal(
        self,
        momentum_longitudinal: sc.Variable,
        tof: sc.Variable,
        mass: sc.Variable,
        charge: sc.Variable,
    ):
        # TODO add case where particle overcomes opposite acceleration step
        tof = tof.to(unit="s")
        v_0 = (momentum_longitudinal / mass).to(unit="m/s")
        tof_acceleration = self.tof_in_acceleration_part(momentum_longitudinal, mass, charge).to(
            unit="s"
        )
        acceleration = (charge * self.electric_field / mass).to(
            unit="m/s**2"
        )
        tof_drift = tof - tof_acceleration
        final_velocity = tof_acceleration * acceleration + v_0
        z = sc.where(
            tof_drift < sc.scalar(0, unit="s"),
            acceleration * tof**2 / 2 + v_0 * tof,
            acceleration * tof_acceleration**2 / 2
            + v_0 * tof_acceleration
            + final_velocity * tof_drift,
        )
        return z

    def calc_omega(self, mass: sc.Variable, charge: sc.Variable):
        return (charge * self.magnetic_field / mass).to(unit="1/s")


class Coincidence:
    def __init__(
        self,
        name,
        ions: Iterable[Ion],
        electrons: Iterable[Electron],
        remi: RemiCalculator,
        *,
        colors=None,
    ):
        self.name = name
        self.remi = remi

        if colors is None:
            colors = [None for _ in ions]
        assert len(colors) == len(ions)
        self.colors = colors

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
            self.remi.set_particle_converter_graph(part)
            part.calculate_detector_hits()

    @property
    def datagroup(self) -> sc.DataGroup:
        return sc.DataGroup(({name: part.momentum_sample} for name, part in self.particles.items()))


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

    ion = Ion(atom_formula, charge=1)
    electron = Electron()

    if name is None:
        name = "_".join([ion.name, electron.name])
    sample_two_body_fragmentation(kinetic_energy, ion, electron)

    return Coincidence(name, ions=[ion], electrons=[electron], colors=[color], remi=remi)


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
