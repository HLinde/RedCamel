#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Hannes Lindenblatt
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
import scipp as sc
import numpy as np
from .units import m_e, q_e


def make_scipp_detector_converters(
    length_acceleration, length_drift, electric_field, magnetic_field, mass, charge
):
    voltage_difference = electric_field * length_acceleration
    acceleration_direction = np.sign(charge.value * electric_field.value)

    def calc_tof(p):
        p_z = p.fields.z
        D = p_z * p_z - 2 * charge * voltage_difference * mass
        rootD = sc.sqrt(D)
        tof = sc.where(
            D < 0 * sc.Unit("J*kg"),
            sc.scalar(np.nan, unit="s"),
            mass
            * (
                2 * length_acceleration / (rootD + acceleration_direction * p_z)
                + length_drift / rootD
            ),
        )
        return {"tof": tof.to(unit="ns")}


    def calc_xyR(p, tof):
        p_x = p.fields.x
        p_y = p.fields.y

        # cyclotron motion or linear motion?
        if sc.abs(magnetic_field) > 0 * sc.Unit("T"):
            p_xy = sc.sqrt(p_x**2 + p_y**2)
            phi = sc.atan2(x=p_x, y=p_y)
            omega = calc_omega(magnetic_field, charge, mass)

            # alpha/2 has to be periodic in 1*pi!
            # sign of alpha is important as it gives the direction of deflection
            # The sign has to be included also in the modulo operation!
            alpha = (omega.to(unit="1/s") * tof.to(unit="s")).values
            alpha = alpha % (np.sign(alpha) * 2 * np.pi)
            alpha = sc.array(dims=p.dims, values=alpha, unit="rad")

            theta = phi + (alpha / 2)
            # Here the signs of alpha, charge and magnetic_field cancel out so R is positive :)
            R = (2 * p_xy * sc.sin(alpha / 2)) / (charge * magnetic_field)
            x = R * sc.cos(theta)
            y = R * sc.sin(theta)
        else:  # for small magnetic field it reduces to this linear motion:
            v_x = p_x / mass
            v_y = p_y / mass
            x = v_x * tof
            y = v_y * tof
            R = sc.sqrt(x**2 + y**2)
        return {"x": x.to(unit="mm"), "y": y.to(unit="mm"), "R": R.to(unit="mm")}

    def calc_tof_in_acceleration_part(p):
        p_z = p.fields.z
        D = p_z * p_z - 2 * charge * voltage_difference * mass
        rootD = sc.sqrt(D)
        tof = sc.where(
            D < 0 * sc.Unit("J*kg"),
            sc.scalar(np.nan, unit="s"),
            mass * (2 * length_acceleration / (rootD + acceleration_direction * p_z)),
        )
        return tof

    def calc_z(p, tof):
        tof = tof.to(unit="s")
        p_z = p.fields.z
        v_0 = (p_z / mass).to(unit="m/s")
        tof_acceleration = calc_tof_in_acceleration_part(p).to(unit="s")
        acceleration = (charge * electric_field / mass).to(unit="m/s**2")
        tof_drift = tof - tof_acceleration
        final_velocity = tof_acceleration * acceleration + v_0
        z = sc.where(
            tof_drift < sc.scalar(0, unit="s"),
            acceleration * tof**2 / 2 + v_0 * tof,
            acceleration * tof_acceleration**2 / 2
            + v_0 * tof_acceleration
            + final_velocity * tof_drift,
        )
        return {"z": z.to(unit="m")}

    return calc_tof, calc_xyR, calc_z


def calc_omega(B, q=-q_e, m=m_e):
    return q * B / m
