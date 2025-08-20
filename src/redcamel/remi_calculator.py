#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Hannes Lindenblatt
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from typing import Literal
import scipp as sc
import numpy as np

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
        acceleration = (charge * self.electric_field / mass).to(unit="m/s**2")
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
