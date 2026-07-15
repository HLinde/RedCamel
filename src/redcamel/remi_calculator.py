#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Hannes Lindenblatt
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
from typing import Literal

import numpy as np
import scipp as sc

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
        resolution_x: sc.Variable,
        resolution_y: sc.Variable,
        resolution_tof: sc.Variable,
        jet_direction: CoordinateDirection = "+x",
        field_direction: CoordinateDirection = "+z",
        projectile_direction: CoordinateDirection = "-y",
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
        self.projectile_direction = projectile_direction
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_tof = resolution_tof

    @property
    def jet_unitvector(self):
        return axis_vectors[self.jet_direction]

    @property
    def field_unitvector(self):
        return axis_vectors[self.field_direction]

    @property
    def projectile_unitvector(self):
        return axis_vectors[self.projectile_direction]

    @property
    def transverse_unitvector(self):
        return sc.cross(self.field_unitvector, self.jet_unitvector)

    @property
    def voltage_difference(self):
        return self.voltage_electron - self.voltage_ion

    @property
    def length_acceleration_total(self):
        return self.length_acceleration_electron + self.length_acceleration_ion

    @property
    def electric_field(self):
        return self.voltage_difference / self.length_acceleration_total

    def make_scipp_graph_for_detector(self, mass: sc.Variable, charge: sc.Variable):
        graph = {
            "p_jet": self.jet_momentum,
            "p_trans": self.transverse_momentum,
            "p_long": self.longitudinal_momentum,
            # calculate true hit positions
            ("tof_accel", "tof_drift"): lambda p_long: self.tof_in_parts(p_long, mass, charge),
            "tof_true": lambda tof_accel, tof_drift: tof_accel + tof_drift,
            ("x_true", "y_true", "R_true", "alpha_true"): lambda tof_true, p_jet, p_trans: {
                label: func
                for label, func in zip(
                    ("x_true", "y_true", "R_true", "alpha_true"),
                    self.hit_position_xyRa(tof_true, p_jet, p_trans, mass, charge),
                    strict=True,
                )
            },
            # add noise to simulate finite detector resolution:
            "tof": lambda tof_true: tof_true + self.sample_resolution_tof(tof_true),
            ("x", "y", "R"): lambda x_true, y_true: {
                label: func
                for label, func in zip(
                    ("x", "y", "R"), self.sample_resolution_pos(x_true, y_true), strict=True
                )
            },
            "alpha": lambda tof: self.calc_alpha(tof=tof, mass=mass, charge=charge),
            # helper to calculate position along the time of flight:
            "z": lambda p_long, tof_true, tof_accel, tof_drift: self.position_longitudinal(
                p_long, tof_true, tof_accel, tof_drift, mass, charge
            ),
        }
        return graph

    def make_graph_for_momentum_calculation(self, mass: sc.Variable, charge: sc.Variable):
        graph = {
            "p_z": lambda tof: self.p_z(tof, mass, charge),
            ("p_x", "p_y"): lambda tof, x, y: {
                label: func
                for label, func in zip(
                    ("p_x", "p_y"), self.p_xy(tof, x, y, mass, charge), strict=True
                )
            },
            "p_obs": lambda p_x, p_y, p_z: sc.spatial.as_vectors(x=p_x, y=p_y, z=p_z),
            "energy": lambda p_obs: (0.5 * sc.norm(p_obs) ** 2 / mass).to(unit="eV"),
        }
        return graph

    def longitudinal_momentum(self, p: sc.Variable):
        return sc.dot(p, self.field_unitvector)

    def jet_momentum(self, p: sc.Variable):
        return sc.dot(p, self.jet_unitvector)

    def transverse_momentum(self, p: sc.Variable):
        return sc.dot(p, self.transverse_unitvector)

    def tof_in_parts(
        self, momentum_longitudinal: sc.Variable, mass: sc.Variable, charge: sc.Variable
    ):
        momentum_longitudinal = momentum_longitudinal.to(unit="N*s")
        mass = mass.to(unit="kg")
        charge = charge.to(unit="C")
        acceleration_direction = np.sign(charge.value * self.electric_field.value)
        if acceleration_direction > 0:
            length_acceleration = self.length_acceleration_ion
            length_drift = self.length_drift_ion
        elif acceleration_direction < 0:
            length_acceleration = self.length_acceleration_electron
            length_drift = self.length_drift_electron
        else:
            raise NotImplementedError("Calculation without electric field not implemented")
        voltage = -acceleration_direction * length_acceleration * self.electric_field
        potential_energy = -voltage * charge
        # TODO add case where particle overcomes opposite acceleration step
        d = momentum_longitudinal**2 + 2 * potential_energy * mass
        root_d = sc.sqrt(d)

        tof_accel = sc.where(
            d < 0 * sc.Unit("J*kg"),
            sc.scalar(np.nan, unit="ns"),
            (
                mass
                * 2
                * length_acceleration
                / (root_d + acceleration_direction * momentum_longitudinal)
            ).to(unit="ns"),
        )
        tof_drift = sc.where(
            d < 0 * sc.Unit("J*kg"),
            sc.scalar(np.nan, unit="ns"),
            (mass * length_drift / root_d).to(unit="ns"),
        )
        return {"tof_accel": tof_accel, "tof_drift": tof_drift}

    def hit_position_xyRa(
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

        # Cyclotron motion or linear motion?
        if sc.abs(self.magnetic_field) > 0 * sc.Unit("T"):
            p_xy = sc.sqrt(p_x**2 + p_y**2)
            phi = sc.atan2(x=p_x, y=p_y)  # angle in xy-plane towards jet-direction

            alpha = self.calc_alpha(tof=tof, mass=mass, charge=charge)
            theta = phi + (alpha / 2)

            R = sc.abs((2 * p_xy * sc.sin(alpha / 2)) / (charge * self.magnetic_field))
            x = R * sc.cos(theta)
            y = R * sc.sin(theta)
        else:  # For small magnetic field it reduces to this linear motion:
            v_x = p_x / mass
            v_y = p_y / mass
            x = v_x * tof
            y = v_y * tof
            alpha = sc.ones_like(tof, unit="rad")
        R = sc.sqrt(x**2 + y**2)
        return x.to(unit="mm"), y.to(unit="mm"), R.to(unit="mm"), alpha.to(unit="rad")

    def sample_resolution_pos(self, x_true, y_true):
        x = x_true + sc.array(
            dims=x_true.dims, values=np.random.randn(*x_true.shape)
        ) * self.resolution_x.to(unit=x_true.unit)
        y = y_true + sc.array(
            dims=y_true.dims, values=np.random.randn(*y_true.shape)
        ) * self.resolution_y.to(unit=y_true.unit)
        R = sc.sqrt(x**2 + y**2)
        return x.to(unit="mm"), y.to(unit="mm"), R.to(unit="mm")

    def position_longitudinal(
        self,
        momentum_longitudinal: sc.Variable,
        tof: sc.Variable,
        tof_accel: sc.Variable,
        tof_drift: sc.Variable,
        mass: sc.Variable,
        charge: sc.Variable,
    ):
        # TODO add case where particle overcomes opposite acceleration step
        tof = tof.to(unit="s")
        tof_accel = tof_accel.to(unit="s")
        tof_drift = tof_drift.to(unit="s")
        v_0 = (momentum_longitudinal / mass).to(unit="m/s")
        electric_force = charge * self.electric_field
        acceleration = (electric_force / mass).to(unit="m/s**2")
        final_velocity = tof_accel.to(unit="s") * acceleration + v_0
        z = sc.where(
            tof_drift < sc.scalar(0, unit="s"),
            acceleration * tof**2 / 2 + v_0 * tof,
            acceleration * tof_accel**2 / 2 + v_0 * tof_accel + final_velocity * tof_drift,
        )
        return z

    def calc_omega(self, mass: sc.Variable, charge: sc.Variable):
        return (charge * self.magnetic_field / mass).to(unit="1/s")

    def calc_alpha(self, tof: sc.Variable, mass: sc.Variable, charge: sc.Variable):
        omega = self.calc_omega(mass, charge)
        alpha = (tof * omega).to(unit="dimensionless") * sc.scalar(1, unit="rad")
        return alpha

    def p_xy(
        self,
        tof: sc.Variable,
        x: sc.Variable,
        y: sc.Variable,
        mass: sc.Variable,
        charge: sc.Variable,
    ):
        if sc.abs(self.magnetic_field) > 0 * sc.Unit("T"):
            alpha = self.calc_alpha(tof=tof, mass=mass, charge=charge)
            radius = sc.sqrt(x**2 + y**2)
            p_r = sc.abs(charge * self.magnetic_field * radius / sc.sin(alpha / 2) / 2)
            phi = sc.atan2(x=x, y=y) - alpha / 2
            p_x_lab = p_r * sc.cos(phi)
            p_y_lab = p_r * sc.sin(phi)
            v_x = (p_x_lab / mass).to(unit="m/s") - self.v_jet
            p_x = v_x * mass
            p_y = p_y_lab
        else:
            x = x.to(unit="mm")
            y = y.to(unit="mm")
            tof = tof.to(unit="ns")
            mass = mass.to(unit="kg")
            charge = charge.to(unit="C")
            v_x = (x / tof).to(unit="m/s") - self.v_jet
            v_y = y / tof
            p_x = mass * v_x
            p_y = mass * v_y
        return p_x.to(unit="au momentum"), p_y.to(unit="au momentum")

    def p_z(self, tof: sc.Variable, mass: sc.Variable, charge: sc.Variable):
        tof = tof.to(unit="ns")
        mass = mass.to(unit="kg")
        charge = charge.to(unit="C")
        acceleration_direction = np.sign(charge.value * self.electric_field.value)
        if acceleration_direction > 0:
            length_acceleration = self.length_acceleration_ion
            length_drift = self.length_drift_ion
        else:
            length_acceleration = self.length_acceleration_electron
            length_drift = self.length_drift_electron

        # TODO add case where particle overcomes opposite acceleration step
        # TODO add drift case
        if length_drift > sc.scalar(0, unit="m"):
            raise NotImplementedError()
        else:
            pz = (acceleration_direction * mass * length_acceleration / tof).to(
                unit="au momentum"
            ) - (tof * charge * self.electric_field / 2).to(unit="au momentum")
        return pz.to(unit="au momentum")

    def sample_resolution_tof(self, tof_accel):
        return (
            sc.array(dims=tof_accel.dims, values=np.random.randn(*tof_accel.shape))
            * self.resolution_tof
        )
