#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Patrizia Schoch
# SPDX-FileContributor: Hannes Lindenblatt
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 10:58:39 2020

@author: patrizia
"""

#############################
#### imports ################
############################
from tkinter import Tk, IntVar, DoubleVar, HORIZONTAL, VERTICAL, Canvas
from tkinter.ttk import (
    Style,
    Button,
    Checkbutton,
    Entry,
    Frame,
    Label,
    LabelFrame,
    Radiobutton,
    Scale,
    Notebook,
    Scrollbar,
    Sizegrip,
)
import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import scipp as sc
from scipp import constants
from chemformula import ChemFormula

#############################
#### constants ##############
############################

# SI
m_e = constants.m_e.value  # electron_mass
q_e = constants.e.value  # elementary_charge
amu = 1.66053906660e-27  # atomic mass unit
sc.units.aliases["au momentum"] = constants.physical_constants("atomic unit of momentum")
sc.units.aliases["au mass"] = constants.m_e

#############################
#### functions ##############
############################


def get_mass(formula: ChemFormula) -> sc.Variable:
    if formula.formula == "e":
        mass_amu = (constants.m_e).to(unit="u")
    else:
        mass_amu = formula.formula_weight * sc.Unit("u")
    return mass_amu


def make_gaussian_momentum_distribution(number_of_particles=1000):
    """
    Parameters
    ----------
    number_of_particles : int
        Sets the number of events that will be generated

    Returns
    -------
    momentum : ndarray
        Array with x-, y-  and z-momenta
    """

    momentum = np.random.randn(number_of_particles, 3) * 2e-24
    return momentum


def make_gaussian_energy_distribution(energy_mean, width, mass, number_of_particles=1000):
    """
    Parameters
    ----------
    energy_mean : float
        Mean energy in eV
    width : float
        width of energy distribution
    number_of_particles : int
        Sets the number of events that will be generated

    Returns
    -------
    momentum : ndarray
        Array with x-, y-  and z-momenta
    """

    r = (np.random.randn(number_of_particles) * width + energy_mean) * q_e
    phi = np.random.rand(number_of_particles) * 2 * np.pi
    cos_theta = np.random.rand(number_of_particles) * 2 - 1

    r_mom = np.sqrt(r * 2 * mass)

    theta = np.arccos(cos_theta)

    x = r_mom * np.sin(theta) * np.cos(phi)
    y = r_mom * np.sin(theta) * np.sin(phi)
    z = r_mom * cos_theta

    momentum = np.stack([x, y, z], axis=-1)

    return momentum


def make_momentum_ion_dis(KER, mass_i1, mass_i2, number_of_particles=1000, v_jet=0):
    # mean_momentum
    momentum_mean = np.sqrt(2 * KER / (1 / mass_i1 + 1 / mass_i2))
    # first ion
    energy_mean = momentum_mean**2 / (2 * mass_i1)
    width = energy_mean / 10
    momentum_i1 = make_gaussian_energy_distribution(
        energy_mean, width, mass_i1, number_of_particles
    )
    # second ion
    momentum_i2 = -momentum_i1
    # add initial momentum from gas-jet
    momentum_i1[:, 0] += v_jet * mass_i1
    momentum_i2[:, 0] += v_jet * mass_i2
    return momentum_i1, momentum_i2


def calc_tof(
    momentum, electric_field, length_acceleration, length_drift, particle_params=(m_e, -q_e)
):
    """
    Parameters
    ----------
    momentum : ndarray
        momentum for tof calculation

    Returns
    -------
    tof : array
        Time of flight for each particle
    """
    m, q = particle_params
    p_z = momentum[:, 2]
    voltage_difference = electric_field * length_acceleration
    D = np.abs(p_z**2 + 2 * q * voltage_difference * m)
    rootD = np.sqrt(D)
    # tof = ((p_z) - np.sqrt(D))/(-q*U)*l_a
    tof = m * (2 * length_acceleration / (rootD + p_z) + length_drift / rootD)
    return tof


def calc_xytof(
    momentum,
    electric_field,
    magnetic_field,
    length_acceleration,
    length_drift,
    particle_params=(m_e, -q_e),
):
    """
    Parameters
    ----------
    momentum : ndarray
        momentum for radius calculation

    Returns
    -------
    x : array
        x coordinate for each particle
    y : array
        x coordinate for each particle
    """
    m, q = particle_params
    p_x = momentum[..., 0]
    p_y = momentum[..., 1]
    tof = calc_tof(momentum, electric_field, length_acceleration, length_drift, particle_params)

    if magnetic_field:  # cyclotron motion during the time-of-flight
        p_xy = np.sqrt(p_x**2 + p_y**2)
        phi = np.atan2(p_x, p_y)
        omega = calc_omega(magnetic_field, q, m)
        alpha = omega * tof
        theta = phi + alpha / 2
        R = (2 * p_xy * np.abs(np.sin(alpha / 2))) / (q * magnetic_field)
        x = R * np.sin(theta)
        y = R * np.cos(theta)
    else:  # for small magnetic field it reduces to this linear motion:
        v_x = p_x / m
        v_y = p_y / m
        x = v_x * tof
        y = v_y * tof
    return x, y, tof


def calc_R(
    momentum,
    electric_field,
    magnetic_field,
    length_acceleration,
    length_drift,
    particle_params=(m_e, -q_e),
):
    """
    Parameters
    ----------
    momentum : ndarray
        momentum for radius calculation

    Returns
    -------
    R : array
        Distance from reaction point to detection point in xy for each particle
    """
    m, q = particle_params
    x, y, tof = calc_xytof(
        momentum, electric_field, magnetic_field, length_acceleration, length_drift, particle_params
    )
    R = np.sqrt(x**2 + y**2)
    return R


def calc_omega(B, q=-q_e, m=m_e):
    return q * B / m


def calc_R_fit(K, tof, voltage, magnetic_field, length_acceleration, particle_params):
    m, q = particle_params
    D = K**2 - (m * length_acceleration / tof - voltage * q * tof / (2 * length_acceleration)) ** 2
    R = (
        2
        / (m * calc_omega(magnetic_field, q, m))
        * np.sqrt(D)
        * np.abs(np.sin(calc_omega(magnetic_field, q, m) * tof / 2))
    )
    return R


def calc_R_fit_ion(K, tof, voltage, magnetic_field, length_acceleration, particle_params):
    m, q = particle_params
    D = K**2 - (m * length_acceleration / tof - voltage * q * tof / (2 * length_acceleration)) ** 2
    R = 2 / m * np.sqrt(D)
    R = (
        2
        / (m * calc_omega(magnetic_field, q, m))
        * np.sqrt(D)
        * np.abs(np.sin(calc_omega(magnetic_field, q, m) * tof / 2))
    )
    return R


########### IONS ###########################################################
def calc_ion_momenta(KER, m_1, m_2):
    p = np.sqrt(2 * KER / (1 / m_1 + 1 / m_2))
    return p


#############################
#### GUI ####################
############################
canvas_background_color = "mintcream"


class mclass:
    def __init__(self, window):
        self.window = window
        window.title("Red Camel")
        style = Style()
        for labeler in [
            "TFrame",
            "TLabel",
            "TLabelframe",
            "TRadiobutton",
            "TLabelframe.Label",
            "TCheckbutton",
            "Canvas",
        ]:
            style.configure(labeler, background=canvas_background_color)
        style.map("TButton", background=[("active", "aliceblue")])
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        tabControl = Notebook(window)
        tabControl.grid(column=0, row=0, sticky="nsew")
        self.tabs = tabs = {}
        labels = [
            "Electrons",
            "Ions",
            "Spectrometer",
            "Target",
            "Background",
            "Load Data",
            "Export Data",
            "Detector Calibration",
        ]
        # create scrollable tabs based on https://web.archive.org/web/20170514022131id_/http://tkinter.unpythonic.net/wiki/VerticalScrolledFrame
        for label in labels:
            tab_frame = Frame(tabControl)

            # Only enable tabs, that are implemented:
            tab_implemented = label in ["Electrons", "Ions", "Export Data"]
            tab_state = "normal" if tab_implemented else "disabled"
            tabControl.add(tab_frame, text=label, state=tab_state)

            tab_frame.columnconfigure(0, weight=1)
            tab_frame.rowconfigure(0, weight=1)
            h = Scrollbar(tab_frame, orient=HORIZONTAL)
            v = Scrollbar(tab_frame, orient=VERTICAL)

            tab_canvas = Canvas(
                tab_frame,
                yscrollcommand=v.set,
                xscrollcommand=h.set,
                scrollregion=(0, 0, 1900, 1600),
                width=1900,
                height=1600,
                background=canvas_background_color,
            )
            h["command"] = tab_canvas.xview
            v["command"] = tab_canvas.yview
            tab_canvas.grid(column=0, row=0, sticky="nsew")
            h.grid(column=0, row=1, sticky="ew")
            v.grid(column=1, row=0, sticky="ns")
            sizegrip = Sizegrip(tab_frame)
            sizegrip.grid(column=1, row=1, sticky="nsew")

            # reset the view
            tab_canvas.xview_moveto(0)
            tab_canvas.yview_moveto(0)

            # Create a frame inside the canvas which will be scrolled with it.
            tab = Frame(tab_canvas)
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)

            tab_canvas.create_window(0, 0, window=tab, anchor="nw")

            # Track changes to the canvas and frame width and sync them,
            # also updating the scrollbar.
            def _configure_tab(event):
                # update the scrollbars to match the size of the inner frame
                new_width = tab.winfo_reqwidth()
                new_height = tab.winfo_reqheight()
                tab_canvas.config(scrollregion="0 0 %s %s" % (new_width, new_height))
                tab_canvas.config(width=new_width)
                tab_canvas.config(height=new_height)

            tab.bind("<Configure>", _configure_tab)

            tabs[label] = tab

        ######## global Remi variables ####################
        self.length_accel_ion = DoubleVar(value=0.11)
        self.length_drift_ion = DoubleVar(value=0.0)
        self.length_accel_electron = DoubleVar(value=0.18)
        self.length_drift_electron = DoubleVar(value=0.0)
        self.voltage_electron = DoubleVar(value=+500.0)
        self.voltage_ion = DoubleVar()  # gets initialized by ratio of distances
        self.magnetic_field_gauss = DoubleVar(value=5.0)
        self.velocity_jet = DoubleVar(value=1.0)

        self.number_of_particles = IntVar(value=1000)
        self.electron_params = (sc.constants.m_e, -sc.constants.e)
        self.particle_params = (sc.constants.m_e.value, -sc.constants.e.value)

        self.fixed_center_potential = IntVar()
        self.fixed_center_potential.set(1)

        self.v = IntVar()
        self.v.set(2)

        ######## variable callback actions ####################

        def write_callback_voltage_electron(var, index, mode):
            if self.fixed_center_potential.get():
                new_voltage = self.voltage_electron.get()
                distance_electron = self.length_accel_electron.get()
                distance_ion = self.length_accel_ion.get()
                new_voltage_ion = -new_voltage * distance_ion / distance_electron
                self.voltage_ion.set(new_voltage_ion)
            self.update_electron_positions()
            self.update_ion_positions()

        self.voltage_electron.trace("w", write_callback_voltage_electron)
        self.length_accel_electron.trace("w", write_callback_voltage_electron)
        self.fixed_center_potential.trace("w", write_callback_voltage_electron)

        def write_callback_voltage_ion(var, index, mode):
            if self.fixed_center_potential.get():
                new_voltage = self.voltage_ion.get()
                distance_electron = self.length_accel_electron.get()
                distance_ion = self.length_accel_ion.get()
                new_voltage_ion = -new_voltage * distance_electron / distance_ion
                self.voltage_electron.set(new_voltage_ion)
            self.update_electron_positions()
            self.update_ion_positions()

        self.voltage_ion.trace("w", write_callback_voltage_ion)
        self.length_accel_ion.trace("w", write_callback_voltage_ion)

        def write_callback_spectrometer_both(var, index, mode):
            self.update_electron_positions()
            self.update_ion_positions()

        self.magnetic_field_gauss.trace("w", write_callback_spectrometer_both)
        self.velocity_jet.trace("w", write_callback_spectrometer_both)

        def write_callback_spectrometer_electron(var, index, mode):
            self.update_electron_positions()

        self.length_drift_electron.trace("w", write_callback_spectrometer_electron)

        def write_callback_spectrometer_ion(var, index, mode):
            self.update_electron_positions()

        self.length_drift_ion.trace("w", write_callback_spectrometer_ion)

        def write_callback_momentum_electrons(var, index, mode):
            self.update_electron_momenta()

        self.v.trace("w", write_callback_momentum_electrons)

        #################################################################################
        #############################      Electrons      ###############################
        #################################################################################

        ######## higher groups ####################
        left_bar_group = LabelFrame(tabs["Electrons"], text="", padding=(5, 5, 5, 5))
        left_bar_group.grid(row=0, column=0, columnspan=1, rowspan=5, padx=5, pady=5, sticky="nsew")

        top_bar_group = LabelFrame(tabs["Electrons"], text="", padding=(5, 5, 5, 5))
        top_bar_group.grid(row=0, column=1, columnspan=5, rowspan=1, padx=5, pady=5, sticky="nsew")

        self.R_tof_plot_group = LabelFrame(tabs["Electrons"], text="Electron plots")
        self.R_tof_plot_group.grid(
            row=1, column=1, columnspan=5, rowspan=4, padx="5", pady="5", sticky="nsew"
        )

        ######## REMI configurations ##############
        remi_conf_group = LabelFrame(left_bar_group, text="REMI Configuration for Electrons")
        remi_conf_group.grid(row=100, column=100, columnspan=2, padx="5", pady="5", sticky="nsew")

        self.LABEL_SET_B = Label(remi_conf_group, text="B[Gauss]:")
        self.LABEL_SET_B.grid(row=103, column=101, padx="5", pady="5", sticky="ew")
        self.ENTRY_SET_B = Entry(remi_conf_group, textvariable=self.magnetic_field_gauss)
        self.ENTRY_SET_B.grid(row=103, column=102, padx="5", pady="5", sticky="ew")

        self.LABEL_SET_l_d = Label(remi_conf_group, text="drift length[m]:")
        self.LABEL_SET_l_d.grid(row=104, column=101, padx="5", pady="5", sticky="ew")
        self.ENTRY_SET_l_d = Entry(remi_conf_group, textvariable=self.length_drift_electron)
        self.ENTRY_SET_l_d.grid(row=104, column=102, padx="5", pady="5", sticky="ew")

        self.LABEL_SET_l_a = Label(remi_conf_group, text="acc length[m]:")
        self.LABEL_SET_l_a.grid(row=105, column=101, padx="5", pady="5", sticky="ew")
        self.ENTRY_SET_l_a = Entry(remi_conf_group, textvariable=self.length_accel_electron)
        self.ENTRY_SET_l_a.grid(row=105, column=102, padx="5", pady="5", sticky="ew")

        self.LABEL_SET_U = Label(remi_conf_group, text="U[V]:")
        self.LABEL_SET_U.grid(row=106, column=101, padx="5", pady="5", sticky="ew")
        self.ENTRY_SET_U = Entry(remi_conf_group, textvariable=self.voltage_electron)
        self.ENTRY_SET_U.grid(row=106, column=102, padx="5", pady="5", sticky="ew")

        self.CHECK_fixed_potential_ele = Checkbutton(
            remi_conf_group,
            text="Interaction region on Ground potential",
            variable=self.fixed_center_potential,
            onvalue=1,
        )
        self.CHECK_fixed_potential_ele.grid(
            row=107, column=101, columnspan=2, padx="5", pady="5", sticky="ew"
        )

        ######## momentum, R, tof calculation #############
        self.R_tof_group = LabelFrame(left_bar_group, text="Electron Momentum Distribution")
        self.R_tof_group.grid(row=102, column=100, columnspan=2, rowspan=5, sticky="nswe")

        self.CHOOSE_MOMENTUM = Radiobutton(
            self.R_tof_group, command=self.check, text="Momentum", variable=self.v, value=1
        )
        self.CHOOSE_ENERGY = Radiobutton(
            self.R_tof_group, command=self.check, text="Energy", variable=self.v, value=2
        )
        self.CHOOSE_MOMENTUM.grid(row=99, column=110, padx="5", pady="5", sticky="w")
        self.CHOOSE_ENERGY.grid(row=100, column=110, padx="5", pady="5", sticky="w")
        self.CHOOSE_ENERGY_MULTI = Radiobutton(
            self.R_tof_group,
            command=self.check,
            text="Multiple Particles",
            variable=self.v,
            value=3,
        )
        self.CHOOSE_ENERGY_MULTI.grid(row=101, column=110, padx="5", pady="5", sticky="w")

        self.LABEL_NUMBER_PART = Label(self.R_tof_group, text="number of Particles:")
        self.LABEL_PART_MASS = Label(self.R_tof_group, text="Particle mass:")
        self.LABEL_PART_CHARGE = Label(self.R_tof_group, text="Particle charge:")
        self.ENTRY_NUMBER_PART = Entry(self.R_tof_group, textvariable=self.number_of_particles)
        self.ENTRY_PART_MASS = Entry(self.R_tof_group)
        self.ENTRY_PART_CHARGE = Entry(self.R_tof_group)

        # if selecting calculation with energy
        self.LABEL_MEAN_ENERGY = Label(self.R_tof_group, text="Mean Energy:")
        self.LABEL_WIDTH = Label(self.R_tof_group, text="Width:")
        self.ENTRY_MEAN_ENERGY = Entry(self.R_tof_group)
        self.ENTRY_WIDTH = Entry(self.R_tof_group)

        self.LABEL_MEAN_ENERGY.grid(row=105, column=110, padx="5", pady="5", sticky="w")
        self.LABEL_WIDTH.grid(row=106, column=110, padx="5", pady="5", sticky="w")
        self.ENTRY_MEAN_ENERGY.grid(row=105, column=111, padx="5", pady="5", sticky="w")
        self.ENTRY_WIDTH.grid(row=106, column=111, padx="5", pady="5", sticky="w")
        self.ENTRY_MEAN_ENERGY.insert(0, 10)
        self.ENTRY_WIDTH.insert(0, 0.1)

        self.ENTRY_PART_MASS.insert(0, 1)
        self.ENTRY_PART_CHARGE.insert(0, -1)

        self.LABEL_NUMBER_PART.grid(row=102, column=110, padx="5", pady="5", sticky="w")
        self.LABEL_PART_MASS.grid(row=103, column=110, padx="5", pady="5", sticky="w")
        self.LABEL_PART_CHARGE.grid(row=104, column=110, padx="5", pady="5", sticky="w")
        self.ENTRY_NUMBER_PART.grid(row=102, column=111, padx="5", pady="5", sticky="w")
        self.ENTRY_PART_MASS.grid(row=103, column=111, padx="5", pady="5", sticky="w")
        self.ENTRY_PART_CHARGE.grid(row=104, column=111, padx="5", pady="5", sticky="w")

        # if multiple particles
        self.LABEL_MULTI_PART_ENERGY_STEP = Label(self.R_tof_group, text="Energy Step:")
        self.LABEL_MULTI_PART_NUMBER = Label(self.R_tof_group, text="Number of Particles")
        self.ENTRY_MULTI_PART_ENERGY_STEP = Entry(self.R_tof_group)
        self.ENTRY_MULTI_PART_ENERGY_STEP.insert(0, 1.5)
        self.ENTRY_MULTI_PART_NUMBER = Entry(self.R_tof_group)
        self.ENTRY_MULTI_PART_NUMBER.insert(0, 3)
        self.LABEL_MULTI_PART_ENERGY_STEP.grid(row=107, column=110, padx="5", pady="5", sticky="w")
        self.LABEL_MULTI_PART_NUMBER.grid(row=108, column=110, padx="5", pady="5", sticky="w")
        self.ENTRY_MULTI_PART_ENERGY_STEP.grid(row=107, column=111, padx="5", pady="5", sticky="w")
        self.ENTRY_MULTI_PART_NUMBER.grid(row=108, column=111, padx="5", pady="5", sticky="w")

        self.LABEL_MULTI_PART_NUMBER.grid_remove()
        self.LABEL_MULTI_PART_ENERGY_STEP.grid_remove()
        self.ENTRY_MULTI_PART_ENERGY_STEP.grid_remove()
        self.ENTRY_MULTI_PART_NUMBER.grid_remove()

        ######## R tof simulation ##########################
        self.R_tof_sim_group = LabelFrame(top_bar_group, text="R-tof simulation")
        self.R_tof_sim_group.grid(
            row=100, column=110, columnspan=4, rowspan=6, padx="5", pady="5", sticky="nswe"
        )

        self.LABEL_KIN_ENERGY = Label(self.R_tof_sim_group, text="Kinetic Energy [EV]:")
        self.ENTRY_KIN_ENERGY_1 = Entry(self.R_tof_sim_group)  # , fg="firebrick")
        self.ENTRY_KIN_ENERGY_2 = Entry(self.R_tof_sim_group)  # , fg="deepskyblue")
        self.ENTRY_KIN_ENERGY_3 = Entry(self.R_tof_sim_group)  # , fg="darkorange")
        self.LABEL_MASS = Label(self.R_tof_sim_group, text="Mass [a.u.]:")
        self.ENTRY_MASS_1 = Entry(self.R_tof_sim_group)  # , fg="firebrick")
        self.ENTRY_MASS_2 = Entry(self.R_tof_sim_group)  # , fg="deepskyblue")
        self.ENTRY_MASS_3 = Entry(self.R_tof_sim_group)  # , fg="darkorange")
        self.LABEL_CHARGE = Label(self.R_tof_sim_group, text="Charge [a.u.]:")
        self.ENTRY_CHARGE_1 = Entry(self.R_tof_sim_group)  # , fg="firebrick")
        self.ENTRY_CHARGE_2 = Entry(self.R_tof_sim_group)  # , fg="deepskyblue")
        self.ENTRY_CHARGE_3 = Entry(self.R_tof_sim_group)  # , fg="darkorange")
        self.LABEL_TOF = Label(self.R_tof_sim_group, text="Time of Flight maximum [ns]:")
        self.ENTRY_TOF = Entry(self.R_tof_sim_group)
        self.BUTTON_R_TOF_SIM = Button(
            self.R_tof_sim_group, text="Simulate Particle", command=self.R_tof_sim
        )

        self.ENTRY_KIN_ENERGY_1.insert(0, 1)
        self.ENTRY_KIN_ENERGY_2.insert(0, 2)
        self.ENTRY_KIN_ENERGY_3.insert(0, 3)
        self.ENTRY_MASS_1.insert(0, 1)
        self.ENTRY_MASS_2.insert(0, 1)
        self.ENTRY_MASS_3.insert(0, 1)
        self.ENTRY_CHARGE_1.insert(0, 1)
        self.ENTRY_CHARGE_2.insert(0, 1)
        self.ENTRY_CHARGE_3.insert(0, 1)
        self.ENTRY_TOF.insert(0, 1000)

        self.LABEL_KIN_ENERGY.grid(row=106, column=110, padx="5", pady="5", sticky="w")
        self.ENTRY_KIN_ENERGY_1.grid(row=107, column=110, padx="5", pady="5", sticky="w")
        self.ENTRY_KIN_ENERGY_2.grid(row=108, column=110, padx="5", pady="5", sticky="w")
        self.ENTRY_KIN_ENERGY_3.grid(row=109, column=110, padx="5", pady="5", sticky="w")
        self.LABEL_MASS.grid(row=106, column=111, padx="5", pady="5", sticky="w")
        self.ENTRY_MASS_1.grid(row=107, column=111, padx="5", pady="5", sticky="w")
        self.ENTRY_MASS_2.grid(row=108, column=111, padx="5", pady="5", sticky="w")
        self.ENTRY_MASS_3.grid(row=109, column=111, padx="5", pady="5", sticky="w")
        self.LABEL_CHARGE.grid(row=106, column=112, padx="5", pady="5", sticky="w")
        self.ENTRY_CHARGE_1.grid(row=107, column=112, padx="5", pady="5", sticky="w")
        self.ENTRY_CHARGE_2.grid(row=108, column=112, padx="5", pady="5", sticky="w")
        self.ENTRY_CHARGE_3.grid(row=109, column=112, padx="5", pady="5", sticky="w")
        self.ENTRY_TOF.grid(row=110, column=111, padx="5", pady="5", sticky="w")
        self.LABEL_TOF.grid(row=110, column=110, padx="5", pady="5", sticky="w")
        self.BUTTON_R_TOF_SIM.grid(
            row=110, column=112, columnspan=1, rowspan=5, padx="5", pady="5", sticky="ns"
        )

        #### Plots and Slidebars ##############

        self.v_ir = IntVar()
        self.v_ir.set(0)
        self.CHECK_IR_PLOT = Checkbutton(
            self.R_tof_plot_group, text="Enable IR plot Mode", variable=self.v_ir, onvalue=1
        )
        self.CHECK_IR_PLOT.grid(row=105, column=100, columnspan=2, padx="5", pady="5", sticky="ew")

        self.LABEL_SLIDE_U = Label(self.R_tof_plot_group, text="Voltage electron side [V]")
        self.LABEL_SLIDE_U.grid(row=106, column=100, padx="5", pady="5", sticky="ew")
        self.LABEL_VALUE_SLIDE_U = Label(self.R_tof_plot_group, textvariable=self.voltage_electron)
        self.LABEL_VALUE_SLIDE_U.grid(row=106, column=101, padx="5", pady="5", sticky="w")

        self.SLIDE_U = Scale(
            self.R_tof_plot_group,
            from_=0,
            to=500,
            # resolution=0.1,
            orient=HORIZONTAL,
            variable=self.voltage_electron,
        )
        self.SLIDE_U.grid(row=107, column=100, columnspan=2, padx="5", pady="5", sticky="ew")

        self.LABEL_SLIDE_B = Label(self.R_tof_plot_group, text="Magnetic Field [G]")
        self.LABEL_SLIDE_B.grid(row=106, column=110, padx="5", pady="5", sticky="w")
        self.LABEL_VALUE_SLIDE_B = Label(
            self.R_tof_plot_group, textvariable=self.magnetic_field_gauss
        )
        self.LABEL_VALUE_SLIDE_B.grid(row=106, column=111, padx="5", pady="5", sticky="w")

        self.SLIDE_B = Scale(
            self.R_tof_plot_group,
            from_=0,
            to=100,
            # resolution=0.1,
            orient=HORIZONTAL,
            variable=self.magnetic_field_gauss,
        )
        self.SLIDE_B.grid(row=107, column=110, columnspan=2, padx="5", pady="5", sticky="ew")

        #### IR mode #####
        self.ir_mode_group = LabelFrame(top_bar_group, text="IR-Mode")
        self.ir_mode_group.grid(
            row=100, column=120, columnspan=2, padx="5", pady="5", sticky="nswe"
        )

        self.LABEL_KIN_ENERGY_START = Label(self.ir_mode_group, text="First Kin Energy [eV]")
        self.LABEL_KIN_ENERGY_STEP = Label(self.ir_mode_group, text="Kin Energy Stepsize [eV]")
        self.LABEL_NUMBER_OF_PART = Label(self.ir_mode_group, text="Numer of particles")
        self.LABEL_MASS_IR = Label(self.ir_mode_group, text="Mass")
        self.LABEL_CHARGE_IR = Label(self.ir_mode_group, text="Charge")

        self.ENTRY_KIN_ENERGY_START = Entry(self.ir_mode_group)
        self.ENTRY_KIN_ENERGY_STEP = Entry(self.ir_mode_group)
        self.ENTRY_NUMBER_OF_PART = Entry(self.ir_mode_group)
        self.ENTRY_MASS_IR = Entry(self.ir_mode_group)
        self.ENTRY_CHARGE_IR = Entry(self.ir_mode_group)

        self.ENTRY_KIN_ENERGY_START.insert(0, 1.3)
        self.ENTRY_KIN_ENERGY_STEP.insert(0, 1.55)
        self.ENTRY_NUMBER_OF_PART.insert(0, 10)
        self.ENTRY_MASS_IR.insert(0, 1)
        self.ENTRY_CHARGE_IR.insert(0, 1)

        self.BUTTON_SIM_IR_MODE = Button(
            self.ir_mode_group, text="Simulate Particle IR Mode", command=self.R_tof_sim_ir
        )
        self.BUTTON_SIM_IR_MODE.grid(row=4, column=4, padx="5", pady="5", sticky="ns")

        self.LABEL_KIN_ENERGY_START.grid(
            row=6, column=2, columnspan=1, padx="5", pady="5", sticky="w"
        )
        self.LABEL_KIN_ENERGY_STEP.grid(
            row=7, column=2, columnspan=1, padx="5", pady="5", sticky="w"
        )
        self.LABEL_NUMBER_OF_PART.grid(
            row=8, column=2, columnspan=1, padx="5", pady="5", sticky="w"
        )
        self.LABEL_MASS_IR.grid(row=4, column=2, columnspan=1, padx="5", pady="5", sticky="w")
        self.LABEL_CHARGE_IR.grid(row=5, column=2, columnspan=1, padx="5", pady="5", sticky="w")

        self.ENTRY_KIN_ENERGY_START.grid(
            row=6, column=3, columnspan=1, padx="5", pady="5", sticky="w"
        )
        self.ENTRY_KIN_ENERGY_STEP.grid(
            row=7, column=3, columnspan=1, padx="5", pady="5", sticky="w"
        )
        self.ENTRY_NUMBER_OF_PART.grid(
            row=8, column=3, columnspan=1, padx="5", pady="5", sticky="w"
        )
        self.ENTRY_MASS_IR.grid(row=4, column=3, columnspan=1, padx="5", pady="5", sticky="w")
        self.ENTRY_CHARGE_IR.grid(row=5, column=3, columnspan=1, padx="5", pady="5", sticky="w")

        #################################################################################
        ###############################      Ions      ##################################
        #################################################################################

        ######## higher groups ####################
        left_tab2_group = Frame(tabs["Ions"])
        left_tab2_group.grid(row=0, column=0, columnspan=2, sticky="nsew")

        ker_group = LabelFrame(left_tab2_group, text="Calculate KER")
        ker_group.grid(row=0, column=0, padx="5", pady="5", sticky="nsew")

        remi_ion_conf_group = LabelFrame(left_tab2_group, text="REMI Configuration for Ion")
        remi_ion_conf_group.grid(row=1, column=0, padx="5", pady="5", sticky="nsew")

        self.ion_generation_group = LabelFrame(left_tab2_group, text="Ion generation")
        self.ion_generation_group.grid(row=2, column=0, padx="5", pady="5", sticky="nsew")

        self.pipico_plot_group = LabelFrame(tabs["Ions"], text="PIPICO")
        self.pipico_plot_group.grid(
            row=0, column=2, columnspan=3, padx="5", pady="5", sticky="nsew"
        )

        ######## KER ##############################
        self.LABEL_DISTANCE = Label(ker_group, text="internuclear distance R [Å]:")
        self.LABEL_CHARGE_ION_1 = Label(ker_group, text="Charge Ion 1:")
        self.LABEL_CHARGE_ION_2 = Label(ker_group, text="Charge Ion 2:")
        self.BUTTON_CALC_KER = Button(
            ker_group, command=self.calc_ker, text="Kinetic Energy Release:"
        )

        self.ENTRY_DISTANCE = Entry(ker_group)
        self.ENTRY_CHARGE_ION_1 = Entry(ker_group)
        self.ENTRY_CHARGE_ION_2 = Entry(ker_group)
        self.LABEL_KER = Label(ker_group, text="")

        self.LABEL_DISTANCE.grid(row=1, column=1, padx="5", pady="5", sticky="w")
        self.LABEL_CHARGE_ION_1.grid(row=2, column=1, padx="5", pady="5", sticky="w")
        self.LABEL_CHARGE_ION_2.grid(row=3, column=1, padx="5", pady="5", sticky="w")
        self.BUTTON_CALC_KER.grid(row=4, column=1, padx="5", pady="5", sticky="w")

        self.ENTRY_DISTANCE.grid(row=1, column=2, padx="5", pady="5", sticky="w")
        self.ENTRY_CHARGE_ION_1.grid(row=2, column=2, padx="5", pady="5", sticky="w")
        self.ENTRY_CHARGE_ION_2.grid(row=3, column=2, padx="5", pady="5", sticky="w")
        self.LABEL_KER.grid(row=4, column=2, padx="5", pady="5", sticky="w")

        self.ENTRY_DISTANCE.insert(0, 2.52)
        self.ENTRY_CHARGE_ION_1.insert(0, 1)
        self.ENTRY_CHARGE_ION_2.insert(0, 1)

        #### REMI parameter for Ion ####
        self.LABEL_SET_v_jet = Label(remi_ion_conf_group, text="v jet[mm/µs]:")
        self.LABEL_SET_v_jet.grid(row=101, column=101, padx="5", pady="5", sticky="w")
        self.ENTRY_SET_v_jet = Entry(remi_ion_conf_group, textvariable=self.velocity_jet)
        self.ENTRY_SET_v_jet.grid(row=101, column=102, padx="5", pady="5", sticky="w")

        self.LABEL_SET_B_ion = Label(remi_ion_conf_group, text="B[G]:")
        self.LABEL_SET_B_ion.grid(row=102, column=101, padx="5", pady="5", sticky="w")
        self.ENTRY_SET_B_ion = Entry(remi_ion_conf_group, textvariable=self.magnetic_field_gauss)
        self.ENTRY_SET_B_ion.grid(row=102, column=102, padx="5", pady="5", sticky="w")

        self.LABEL_SET_bunch_modulo = Label(remi_ion_conf_group, text="bunch modulo [ns]:")
        self.LABEL_SET_bunch_modulo.grid(row=103, column=101, padx="5", pady="5", sticky="w")
        self.ENTRY_SET_bunch_modulo = Entry(remi_ion_conf_group)
        self.ENTRY_SET_bunch_modulo.grid(row=103, column=102, padx="5", pady="5", sticky="w")

        self.LABEL_SET_detector_diameter = Label(
            remi_ion_conf_group, text="detector diameter [mm]:"
        )
        self.LABEL_SET_detector_diameter.grid(row=104, column=101, padx="5", pady="5", sticky="w")
        self.ENTRY_SET_detector_diameter = Entry(remi_ion_conf_group)
        self.ENTRY_SET_detector_diameter.grid(row=104, column=102, padx="5", pady="5", sticky="w")

        self.LABEL_SET_l_d_ion = Label(remi_ion_conf_group, text="drift length[m]:")
        self.LABEL_SET_l_d_ion.grid(row=105, column=101, padx="5", pady="5", sticky="w")
        self.ENTRY_SET_l_d_ion = Entry(remi_ion_conf_group, textvariable=self.length_drift_ion)
        self.ENTRY_SET_l_d_ion.grid(row=105, column=102, padx="5", pady="5", sticky="w")

        self.LABEL_SET_l_a_ion = Label(remi_ion_conf_group, text="acc length[m]:")
        self.LABEL_SET_l_a_ion.grid(row=106, column=101, padx="5", pady="5", sticky="w")
        self.ENTRY_SET_l_a_ion = Entry(remi_ion_conf_group, textvariable=self.length_accel_ion)
        self.ENTRY_SET_l_a_ion.grid(row=106, column=102, padx="5", pady="5", sticky="w")

        self.LABEL_SET_U_ion = Label(remi_ion_conf_group, text="U[V]:")
        self.LABEL_SET_U_ion.grid(row=107, column=101, padx="5", pady="5", sticky="w")
        self.ENTRY_SET_U_ion = Entry(remi_ion_conf_group, textvariable=self.voltage_ion)
        self.ENTRY_SET_U_ion.grid(row=107, column=102, padx="5", pady="5", sticky="w")

        self.CHECK_fixed_potential_ion = Checkbutton(
            remi_ion_conf_group,
            text="Interaction region on Ground potential",
            variable=self.fixed_center_potential,
            onvalue=1,
        )
        self.CHECK_fixed_potential_ion.grid(
            row=108, column=101, columnspan=2, padx="5", pady="5", sticky="w"
        )

        self.ENTRY_SET_bunch_modulo.insert(0, 5316.9231)
        self.ENTRY_SET_detector_diameter.insert(0, 120)

        self.LABEL_SLIDE_U_pipco = Label(self.pipico_plot_group, text="Voltage ion side [V]")
        self.LABEL_SLIDE_U_pipco.grid(
            row=3, column=1, columnspan=2, padx="5", pady="5", sticky="ew"
        )

        self.SLIDE_U_pipco = Scale(
            self.pipico_plot_group,
            from_=0,
            to=-3000,
            orient=HORIZONTAL,
            # resolution=0.1,
            variable=self.voltage_ion,
        )
        self.SLIDE_U_pipco.grid(row=4, column=1, columnspan=2, padx="5", pady="5", sticky="ew")

        self.LABEL_SLIDE_B_pipco = Label(self.pipico_plot_group, text="Magnetig Field [G]")
        self.LABEL_SLIDE_B_pipco.grid(
            row=5, column=1, columnspan=2, padx="5", pady="5", sticky="ew"
        )

        self.SLIDE_B_pipco = Scale(
            self.pipico_plot_group,
            from_=0,
            to=200,
            orient=HORIZONTAL,
            # resolution=0.1,
            variable=self.magnetic_field_gauss,
        )
        self.SLIDE_B_pipco.grid(row=6, column=1, columnspan=2, padx="5", pady="5", sticky="ew")

        ### ion generator ###################

        self.LABEL_FORMULA_IONS = Label(self.ion_generation_group, text="ChemFormula:")
        self.LABEL_MASS_IONS = Label(self.ion_generation_group, text="Mass [amu]:")
        self.LABEL_CHARGE_IONS = Label(self.ion_generation_group, text="Charge [au]:")
        self.LABEL_KER_IONS = Label(self.ion_generation_group, text="KER [eV]:")
        self.LABEL_TOF_IONS = Label(self.ion_generation_group, text="TOF [ns]:")

        self.ENTRY_NUMBER_IONS = Entry(self.ion_generation_group)
        self.ENTRY_NUMBER_IONS.grid(row=0, column=2, padx="5", pady="5", sticky="w")
        self.ENTRY_NUMBER_IONS.insert(0, 5)

        self.LABEL_FORMULA_IONS.grid(row=1, column=1, padx="5", pady="5", sticky="w")
        self.LABEL_MASS_IONS.grid(row=1, column=2, padx="5", pady="5", sticky="w")
        self.LABEL_CHARGE_IONS.grid(row=1, column=3, padx="5", pady="5", sticky="w")
        self.LABEL_KER_IONS.grid(row=1, column=4, padx="5", pady="5", sticky="w")
        self.LABEL_TOF_IONS.grid(row=1, column=5, padx="5", pady="5", sticky="w")
        self.LABEL_FORMULA_IONS.grid_remove()
        self.LABEL_MASS_IONS.grid_remove()
        self.LABEL_CHARGE_IONS.grid_remove()
        self.LABEL_KER_IONS.grid_remove()
        self.LABEL_TOF_IONS.grid_remove()

        self.BUTTON_GENERATE_IONS = Button(
            self.ion_generation_group, command=self.generate_entrys, text="Make Ion Couples"
        )
        self.BUTTON_GENERATE_IONS.grid(row=0, column=1, padx="5", pady="5", sticky="w")
        self.last_ion_number = 0
        self.labels_ion_tof = []
        self.entries_ker = []

        self.BUTTON_CALC_ION_TOF = Button(
            self.ion_generation_group, command=self.calc_ion_tof, text="Update"
        )
        self.BUTTON_CALC_ION_TOF.grid(row=0, column=5, padx="5", pady="5", sticky="w")

        fig, axes = plt.subplot_mosaic(
            [
                ["xtof"] * 5,
                ["ytof"] * 5,
                ["pipico"] * 3 + ["XY"] * 2,
                ["pipico"] * 3 + ["XY"] * 2,
                ["pipico"] * 3 + ["."] * 2,
            ],
            figsize=(8, 8),
            facecolor=canvas_background_color,
            layout="constrained",
        )
        self.pipico_fig = fig
        self.pipico_XY_ax = axes["XY"]
        self.pipico_XY_ax.set_aspect("equal")
        self.pipico_xtof_ax = axes["xtof"]
        self.pipico_ytof_ax = axes["ytof"]
        self.pipico_xtof_ax.sharex(self.pipico_ytof_ax)
        self.pipico_ax = axes["pipico"]

        self.pipico_ax.set_aspect("equal")
        self.pipico_canvas = FigureCanvasTkAgg(self.pipico_fig, master=self.pipico_plot_group)
        self.pipico_canvas.get_tk_widget().grid(
            row=1, column=1, rowspan=1, columnspan=1, padx="5", pady="5", sticky="ew"
        )
        self.pipico_toolbar = NavigationToolbar2Tk(
            canvas=self.pipico_canvas, window=self.pipico_plot_group, pack_toolbar=False
        )
        self.pipico_toolbar.grid(
            row=2, column=1, rowspan=1, columnspan=1, padx="5", pady="5", sticky="ew"
        )
        self.pipico_toolbar.update()

        ### initialize data and plots ###################
        self.make_R_tof_figure()
        self.init_dataset()
        write_callback_voltage_electron(None, None, None)  # initalize ion voltages
        self.calc_ker()
        self.generate_entrys()
        self.calc_ion_tof()
        self.make_export_tab()

    def make_export_tab(self):
        self.valid_group = LabelFrame(self.tabs["Export Data"], text="Save Electron Data")
        self.valid_group.grid(
            row=115, column=100, columnspan=4, rowspan=6, padx="5", pady="5", sticky="nsew"
        )

        self.BUTTON_SAVE_MOM = Button(
            self.valid_group, text="Save Momentum Data", command=self.export_momenta
        )
        self.BUTTON_SAVE_MOM.grid(row=10, column=100, columnspan=2, padx="5", pady="5", sticky="w")

        self.BUTTON_CALC_MCP_TIMES = Button(
            self.valid_group, text="Save MCP times", command=self.calc_mcp
        )
        self.BUTTON_CALC_MCP_TIMES.grid(
            row=11, column=100, columnspan=2, padx="5", pady="5", sticky="w"
        )

        self.BUTTON_EXPORT_DATA = Button(
            self.valid_group, text="Save Electron Position", command=self.export_data
        )
        self.BUTTON_EXPORT_DATA.grid(
            row=12, column=100, columnspan=2, padx="5", pady="5", sticky="w"
        )

    def make_plot(
        self, row, column, master, rowspan=1, columnspan=1, figsize=(5, 5), withcax=False
    ):
        """
        Initializes a figure canvas with plot axis

        Returns
        -------
        fig : Figure
        a : axis
        canvas : canvas
        """
        fig = plt.figure(figsize=figsize, facecolor=canvas_background_color, layout="constrained")
        canvas = FigureCanvasTkAgg(fig, master=master)
        canvas.get_tk_widget().grid(
            row=row,
            column=column,
            rowspan=rowspan,
            columnspan=columnspan,
            padx="5",
            pady="5",
            sticky="nsew",
        )
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas=canvas, window=master, pack_toolbar=False)
        toolbar.grid(
            row=row + 1,
            column=column,
            rowspan=1,
            columnspan=columnspan,
            padx="5",
            pady="5",
            sticky="nsew",
        )
        toolbar.update()

        if withcax:
            axes = fig.subplots(1, 2, width_ratios=[0.93, 0.07])
        else:
            axes = fig.subplots(1, 1)
        return fig, axes, canvas, toolbar

    @property
    def magnetic_field_si(self):
        return self.magnetic_field_gauss.get() * 1e-4

    @property
    def velocity_jet_si(self):
        return self.velocity_jet.get() * 1e3

    @property
    def electric_field(self):
        voltage_difference = self.voltage_electron.get() - self.voltage_ion.get()
        voltage_distance = self.length_accel_electron.get() + self.length_accel_ion.get()
        return voltage_difference / voltage_distance

    def check(self):
        if self.v.get() == 1:
            self.LABEL_MEAN_ENERGY.grid_remove()
            self.LABEL_WIDTH.grid_remove()
            self.ENTRY_MEAN_ENERGY.grid_remove()
            self.ENTRY_WIDTH.grid_remove()
            self.LABEL_MULTI_PART_NUMBER.grid_remove()
            self.LABEL_MULTI_PART_ENERGY_STEP.grid_remove()
            self.ENTRY_MULTI_PART_ENERGY_STEP.grid_remove()
            self.ENTRY_MULTI_PART_NUMBER.grid_remove()
        elif self.v.get() == 2:
            self.LABEL_MEAN_ENERGY.grid()
            self.LABEL_WIDTH.grid()
            self.ENTRY_MEAN_ENERGY.grid()
            self.ENTRY_WIDTH.grid()
            self.LABEL_MULTI_PART_NUMBER.grid_remove()
            self.LABEL_MULTI_PART_ENERGY_STEP.grid_remove()
            self.ENTRY_MULTI_PART_ENERGY_STEP.grid_remove()
            self.ENTRY_MULTI_PART_NUMBER.grid_remove()
        elif self.v.get() == 3:
            self.LABEL_MEAN_ENERGY.grid()
            self.LABEL_WIDTH.grid()
            self.ENTRY_MEAN_ENERGY.grid()
            self.ENTRY_WIDTH.grid()
            self.LABEL_MULTI_PART_NUMBER.grid()
            self.LABEL_MULTI_PART_ENERGY_STEP.grid()
            self.ENTRY_MULTI_PART_ENERGY_STEP.grid()
            self.ENTRY_MULTI_PART_NUMBER.grid()

    def make_R_tof_figure(self):
        """
        Generates the R vs tof plot and the electron position plot with random data points
        """
        self.fig_R_tof, self.ax_R_tof, self.canvas_R_tof, self.toolbar_R_tof = self.make_plot(
            100, 100, self.R_tof_plot_group, figsize=(5, 5), columnspan=2
        )
        self.ele_pos_fig, self.ele_pos_a, self.ele_pos_canvas, self.ele_pos_toolbar = (
            self.make_plot(100, 110, self.R_tof_plot_group, figsize=(5, 5), columnspan=2)
        )

    def update_R_tof(self):
        """
        Updates the R vs tof and the position plot, while moving the sliders for B and U
        """
        ax = self.ax_R_tof
        ax.cla()

        max_tof = sc.scalar(self.calc_max_tof(), unit="s").to(unit="ns")
        tof_limit = max(max_tof * 1.2, 1e-9 * sc.Unit("ns"))
        R_limit = sc.scalar(0.07, unit="m").to(unit="mm")
        pos_bins = 100
        tof_bins = 200

        self.x_bins_electrons = sc.linspace("x", -R_limit, R_limit, pos_bins)
        self.y_bins_electrons = sc.linspace("y", -R_limit, R_limit, pos_bins)
        self.tof_bins_electrons = sc.linspace("tof", sc.scalar(0, unit="ns"), tof_limit, tof_bins)
        self.R_bins_electrons = sc.linspace("R", sc.scalar(0, unit="mm"), R_limit, pos_bins)

        R_tof_hist = self.electron_hits.hist(
            R=self.R_bins_electrons, tof=self.tof_bins_electrons, dim=("p", "pulses", "HitNr")
        )
        R_tof_hist.plot(ax=ax, cbar=False, norm="log", cmap="PuBuGn")

        if self.v_ir.get() == 1:
            self.R_tof_sim_ir()

        ax.axvline(max_tof.value, color="darkgrey")

        no_mom_tof = (self.calc_no_momentum_tof() * sc.Unit("s")).to(unit="ns")

        m, q = self.particle_params
        omega = calc_omega(self.magnetic_field_si, q, m)
        if omega == 0:
            cyclotron_period = np.inf
        else:
            cyclotron_period = np.abs(2 * np.pi / omega)
        cyclotron_period = (cyclotron_period * sc.Unit("s")).to(unit="ns")

        ax.axvline(no_mom_tof.value, ls="--", color="darkgrey")
        if sc.isfinite(cyclotron_period) and max_tof > cyclotron_period:
            for node_tof in np.arange(0.0, max_tof.value, cyclotron_period.value):
                ax.axvline(node_tof, ls=":", color="darkgrey")

        self.canvas_R_tof.draw()

        ax = self.ele_pos_a
        ax.cla()

        x_y_hist = self.electron_hits.hist(
            x=self.x_bins_electrons, y=self.y_bins_electrons, dim=("p", "pulses", "HitNr")
        )
        x_y_hist.plot(ax=ax, cbar=False, norm="log", cmap="PuBuGn")

        detector = plt.Circle((0, 0), 60, color="cadetblue", fill=False, figure=self.ele_pos_fig)
        ax.add_artist(detector)

        ax.set_xlim(-70, 70)
        ax.set_ylim(-70, 70)

        self.ele_pos_canvas.draw()

    def R_tof_sim(self):
        """
        Generates a R vs tof plot
        """
        tof_max = float(self.ENTRY_TOF.get()) * 1e-9
        tof = np.linspace(0, tof_max, int(tof_max * 1000e9))
        while len(self.ax_R_tof.lines) > 1:
            self.ax_R_tof.lines[-1].remove()

        if len(self.ENTRY_KIN_ENERGY_1.get()) != 0:
            energy_1 = float(self.ENTRY_KIN_ENERGY_1.get()) * q_e
            mass_1 = float(self.ENTRY_MASS_1.get()) * m_e
            charge_1 = float(self.ENTRY_CHARGE_1.get()) * q_e
            particle_params_1 = (mass_1, charge_1)
            K_1 = np.sqrt(2 * mass_1 * energy_1)
            R_1 = calc_R_fit(
                K_1,
                tof,
                self.voltage_electron.get(),
                self.magnetic_field_si,
                self.length_accel_electron.get(),
                particle_params_1,
            )
            self.ax_R_tof.plot(tof, R_1, color="firebrick")

        if len(self.ENTRY_KIN_ENERGY_2.get()) != 0:
            energy_2 = float(self.ENTRY_KIN_ENERGY_2.get()) * q_e
            mass_2 = float(self.ENTRY_MASS_2.get()) * m_e
            charge_2 = float(self.ENTRY_CHARGE_2.get()) * q_e
            particle_params_2 = (mass_2, charge_2)
            K_2 = np.sqrt(2 * mass_2 * energy_2)
            R_2 = calc_R_fit(
                K_2,
                tof,
                self.voltage_electron.get(),
                self.magnetic_field_si,
                self.length_accel_electron.get(),
                particle_params_2,
            )
            self.ax_R_tof.plot(tof, R_2, color="deepskyblue")

        if len(self.ENTRY_KIN_ENERGY_3.get()) != 0:
            energy_3 = float(self.ENTRY_KIN_ENERGY_3.get()) * q_e
            mass_3 = float(self.ENTRY_MASS_3.get()) * m_e
            charge_3 = float(self.ENTRY_CHARGE_3.get()) * q_e
            particle_params_3 = (mass_3, charge_3)
            K_3 = np.sqrt(2 * mass_3 * energy_3)
            R_3 = calc_R_fit(
                K_3,
                tof,
                self.voltage_electron.get(),
                self.magnetic_field_si,
                self.length_accel_electron.get(),
                particle_params_3,
            )
            self.ax_R_tof.plot(tof, R_3, color="darkorange")

        max_tof = self.calc_max_tof()
        self.ax_R_tof.axvline(max_tof, 0, 1, color="darkgrey")
        no_mom_tof = self.calc_no_momentum_tof()
        self.ax_R_tof.axvline(no_mom_tof, 0, 1, ls="--", color="darkgrey")
        self.canvas_R_tof.draw()

    def calc_max_tof(self):
        """
        calculates the maximal tof for the electron to not fly into the ion detector
        """
        m, q = self.particle_params

        l_a_ion = self.length_accel_ion.get()
        l_a_el = self.length_accel_electron.get()
        l_d_el = self.length_drift_electron.get()
        U_ion = self.voltage_ion.get()
        U_el = self.voltage_electron.get()
        E = self.electric_field
        if (
            np.abs(E) < 4 * np.finfo(E).eps
        ):  # check if field magnitude is above numerical resolution
            # No acceleration.. anything towards ion detector is lost
            # assume a tiny kinetic energy towards electron detector..
            E_kin = 1e-1 * q_e
            momentum = np.array([[0, 0, np.sqrt(2 * E_kin * m)]])
            tof_max = calc_tof(momentum, E, l_a_el, l_d_el, particle_params=self.particle_params)[0]
        elif E > 0:
            # Start with kinetic energy towards ion detector equal to the potential difference
            # time from reaction point to end of ion acceleration
            time_1 = np.sqrt(2 * l_a_ion * m / (-E * q))
            # time from ion acceleration end to electron acceleration end
            time_2 = np.sqrt(2 * (l_a_ion + l_a_el) * m / (-E * q))
            # now kinetic energy is exactly the total potential for the edge case
            E_kin = np.abs((U_ion + U_el) * q)
            v_drift = np.sqrt(2 * E_kin / m)
            time_3 = l_d_el / v_drift
            tof_max = time_1 + time_2 + time_3
        else:
            # Field is deccelerating!
            # Start with kinetic energy towards electron detector equal to the potential difference
            # time from reaction point to end of ion acceleration
            # now kinetic energy is exactly the total potential for the edge case
            E_kin = E * l_a_el
            momentum = np.array([[0, 0, np.sqrt(2 * E_kin * m)]])
            tof_max = calc_tof(momentum, E, l_a_el, l_d_el, particle_params=self.particle_params)[0]
        return tof_max

    def calc_no_momentum_tof(self):
        """
        calculates the time of flight for a paticle with no z-momentum
        """
        no_momentum = np.zeros((1, 3))
        zero_momentum_tof = calc_tof(
            no_momentum,
            self.electric_field,
            self.length_accel_electron.get(),
            self.length_drift_electron.get(),
            particle_params=self.particle_params,
        )[0]
        return zero_momentum_tof

    def export_pipico(self):
        pass

    @property
    def electron_x(self):
        return self.electron_hits.coords["x"]

    @property
    def electron_y(self):
        return self.electron_hits.coords["y"]

    @property
    def electron_tof(self):
        return self.electron_hits.coords["tof"]

    def export_data(self):
        """
        writes electron position data to a file
        """
        x, y, tof = (
            self.electron_x.values.flatten(),
            self.electron_y.values.flatten(),
            self.electron_tof.values.flatten(),
        )
        data = np.array([x, y, tof])
        data = data.T
        with open("pos_data.txt", "w") as datafile:
            np.savetxt(datafile, data, fmt=["%.3E", "%.3E", "%.3E"])

    def export_momenta(self):
        """
        writes electron momentum data to a file
        """
        px, py, pz = self.electron_momenta.coords["p"].fields.values()
        mom = np.array([px.values.flatten(), py.values.flatten(), pz.values.flatten()])
        mom = mom.T
        with open("mom_data.txt", "w") as datafile:
            np.savetxt(datafile, mom, fmt=["%.3E", "%.3E", "%.3E"])

    def calc_mcp(self):
        """
        calculates the mcp times and write them to a file
        """
        x, y, tof = self.electron_x, self.electron_y, self.electron_tof

        x = x.to(unit="m").values.flatten()
        y = y.to(unit="m").values.flatten()
        tof = tof.to(unit="s").values.flatten()

        t_mcp = tof
        v_delayline = 0.857 * 1e6
        timesum_x = 143.8 * 1e-9
        timesum_y = 141.2 * 1e-9

        def calc_t(time_sum, v, x):
            t = -x / v + time_sum / 2
            return t

        t2_x = calc_t(timesum_x, v_delayline, x)
        t2_y = calc_t(timesum_y, v_delayline, y)
        t1_x = timesum_x - t2_x
        t1_y = timesum_y - t2_y

        times = np.array([t_mcp, t1_x, t2_x, t1_y, t2_y])
        times = times.T
        with open("mcp_data.txt", "w") as datafile:
            np.savetxt(datafile, times, fmt=["%.3E", "%.3E", "%.3E", "%.3E", "%.3E"])

    def calc_ion_position(self):
        """
        calculates the ion position of
        """
        p_x = -self.momenta[:, 0]
        p_y = -self.momenta[:, 1]
        v_jet = self.velocity_jet_si
        ion_formula = ChemFormula(self.ENTRY_ION_MASS.get())

        ion_mass_amu = get_mass(ion_formula)
        ion_mass = ion_mass_amu * m_e
        ion_params = (ion_mass, float(self.ENTRY_ION_CHARGE.get()) * q_e)
        tof = calc_tof(
            -self.momenta,
            self.electric_field,
            length_acceleration=self.length_accel_ion.get(),
            length_drift=self.length_drift_ion.get(),
            particle_params=ion_params,
        )
        x_pos_ion = (p_x / ion_mass + v_jet) * tof
        y_pos_ion = (p_y / ion_mass) * tof
        self.make_plot(x_pos_ion, y_pos_ion, 120, 100, self.ion_pos_group, columnspan=2)
        return (x_pos_ion, y_pos_ion)

    def R_tof_sim_ir(self):
        tof_max = float(self.ENTRY_TOF.get()) * 1e-9
        tof = np.linspace(0, tof_max, int(tof_max * 100e9))
        start_energy = float(self.ENTRY_KIN_ENERGY_START.get()) * q_e
        step_energy = float(self.ENTRY_KIN_ENERGY_STEP.get()) * q_e
        number_sim = int(self.ENTRY_NUMBER_OF_PART.get())
        energys = np.linspace(
            start_energy, (number_sim * step_energy) + start_energy, number_sim + 1
        )
        mass = float(self.ENTRY_MASS_IR.get()) * m_e
        charge = float(self.ENTRY_CHARGE_IR.get()) * q_e
        particle_params = (mass, charge)

        while len(self.ax_R_tof.lines) > 1:
            self.ax_R_tof.lines[-1].remove()

        for i in range(number_sim):
            K = np.sqrt(2 * mass * energys[i])
            R = calc_R_fit(
                K,
                tof,
                self.voltage_electron.get(),
                self.magnetic_field_si,
                self.length_accel_electron.get(),
                particle_params,
            )
            self.ax_R_tof.plot(tof, R, color="firebrick")
        self.canvas_R_tof.draw()

    ##########################################################################
    ###############   TAB 2 ##################################################
    ##########################################################################

    def calc_ker(self):
        dis_R = float(self.ENTRY_DISTANCE.get()) * 1e-10  # angström
        charge_1 = float(self.ENTRY_CHARGE_ION_1.get()) * q_e
        charge_2 = float(self.ENTRY_CHARGE_ION_2.get()) * q_e
        ele_const = 8.854187e-12

        ### einheiten
        factor = 4.3597447e-18

        ker = 1 / (4 * np.pi * ele_const) * (charge_1 * charge_2 / dis_R) / factor * 27.211
        self.LABEL_KER.config(text="{:.2f} eV".format(ker))

        return ker

    def generate_entrys(self):
        self.LABEL_FORMULA_IONS.grid()
        self.LABEL_MASS_IONS.grid()
        self.LABEL_CHARGE_IONS.grid()
        self.LABEL_KER_IONS.grid()
        ion_number = int(self.ENTRY_NUMBER_IONS.get()) * 2

        colors = np.array(matplotlib.color_sequences["tab20"])
        self.ion_color = colors[np.arange(ion_number) % len(colors)]

        # saving last entrys
        empty_length = max(self.last_ion_number, ion_number)
        masses = np.zeros(empty_length)
        charges = np.zeros(empty_length)
        formulas = ["" for i in range(empty_length)]
        ker_length = max(len(self.entries_ker), ion_number // 2)
        kers = np.zeros(ker_length)

        for n in range(self.last_ion_number):
            try:
                formulas[n] = ChemFormula(self.entries_formula[n].get())
            except IndexError:
                formulas[n] = ChemFormula("")
            masses[n] = get_mass(formulas[n]).value
            try:
                charges[n] = float(self.entries_charge[n].get())
            except IndexError:
                charges[n] = 0
            self.entries_formula[n].grid_remove()
            self.labels_mass[n].grid_remove()
            self.entries_charge[n].grid_remove()
            self.ion_labels[n].grid_remove()
            self.labels_ion_tof[n].grid_remove()

        for n in range(self.last_ion_number, ion_number):
            charges[n] = 1
            match n:
                case 0:
                    formulas[n] = ChemFormula("C4H8O2")
                case 1:
                    formulas[n] = ChemFormula("S2")
                case 2:
                    formulas[n] = ChemFormula("C4H8SO2")
                case 3:
                    formulas[n] = ChemFormula("S")
                case 4:
                    formulas[n] = ChemFormula("C4H7S2O2")
                case 5:
                    formulas[n] = ChemFormula("H")
                case 6:
                    formulas[n] = ChemFormula("H2")
                case 7:
                    formulas[n] = ChemFormula("H")
                case 8:
                    formulas[n] = ChemFormula("C4H8S2O2")
                case 9:
                    formulas[n] = ChemFormula("C4H8S2O2")
                case 10:
                    formulas[n] = ChemFormula("C8H16S3O4")
                case 11:
                    formulas[n] = ChemFormula("S")
                case 12:
                    formulas[n] = ChemFormula("S")
                    charges[n] = 3
                case 13:
                    formulas[n] = ChemFormula("S")
                    charges[n] = 4
                case _:
                    formulas[n] = ChemFormula("H")
            masses[n] = get_mass(formulas[n]).value

        for i in range(ker_length):
            kers[i] = 15
            try:
                kers[i] = float(self.entries_ker[i].get())
                self.entries_ker[i].grid_remove()
            except IndexError:
                pass

        self.entries_formula = []
        self.labels_mass = []
        self.entries_charge = []
        self.ion_labels = []
        for n in range(ion_number):
            self.ion_labels.append(Label(self.ion_generation_group, text="Ion " + str(n + 1)))
            self.ion_labels[n].grid(row=n + 3, column=0)

            this_ion_color = matplotlib.colors.to_hex(self.ion_color[n])
            self.entries_formula.append(Entry(self.ion_generation_group, foreground=this_ion_color))
            self.entries_charge.append(Entry(self.ion_generation_group, foreground=this_ion_color))

            self.entries_formula[n].grid(row=n + 3, column=1)
            self.entries_charge[n].grid(row=n + 3, column=3)
            self.labels_mass.append(
                Label(
                    self.ion_generation_group,
                    text="{:.5g}".format(masses[n]),
                    foreground=this_ion_color,
                )
            )
            self.labels_mass[n].grid(row=n + 3, column=2)
            self.labels_ion_tof.append(
                Label(self.ion_generation_group, text="", foreground=this_ion_color)
            )
            self.labels_ion_tof[n].grid(row=n + 3, column=5)

            self.entries_formula[n].insert(0, formulas[n])
            self.entries_charge[n].insert(0, charges[n])

        self.entries_ker = []
        for n in range(ion_number // 2):
            this_pair_color = matplotlib.colors.to_hex(
                (self.ion_color[2 * n] + self.ion_color[2 * n + 1]) / 2
            )
            self.entries_ker.append(Entry(self.ion_generation_group, foreground=this_pair_color))
            self.entries_ker[n].grid(row=(n * 2) + 3, column=4, rowspan=2, sticky="ns")
            self.entries_ker[n].insert(0, kers[n])

        self.last_ion_number = ion_number
        self.calc_ion_tof()

    def calc_ion_tof(self):
        self.LABEL_TOF_IONS.grid()
        formulas = ["" for n in range(self.last_ion_number)]
        masses = np.zeros(self.last_ion_number)
        charges = np.zeros(self.last_ion_number)
        for n in range(self.last_ion_number):
            try:
                formulas[n] = ChemFormula(self.entries_formula[n].get())
            except IndexError:
                formulas[n] = ChemFormula("")
            mass_amu = get_mass(formulas[n]).value
            masses[n] = mass_amu * amu
            self.labels_mass[n]["text"] = "{:.4g}".format(mass_amu)
            try:
                charges[n] = float(self.entries_charge[n].get()) * q_e
            except IndexError:
                charges[n] = 0
        for n in range(self.last_ion_number):
            this_ion_tof = calc_tof(
                np.zeros((1, 3)),
                self.electric_field,
                self.length_accel_ion.get(),
                self.length_drift_ion.get(),
                (masses[n], charges[n]),
            )[0]
            self.labels_ion_tof[n]["text"] = "{:.4g}".format(this_ion_tof * 1e9)
        self.make_ion_pipico_plot()

    def make_ion_pipico_plot(self):
        # read in charge, mass, and KER
        ion_formula_1 = []
        ion_formula_2 = []
        ion_mass_1 = []
        ion_mass_2 = []
        ion_charge_1 = []
        ion_charge_2 = []
        ion_ker = []
        for n in range(self.last_ion_number):
            if n % 2 == 0:
                try:
                    formula = ChemFormula(self.entries_formula[n].get())
                    mass = get_mass(formula).value
                    charge = float(self.entries_charge[n].get())
                    ker = float(self.entries_ker[n // 2].get())
                    ion_formula_1.append(formula)
                    ion_mass_1.append(mass)
                    ion_charge_1.append(charge)
                    ion_ker.append(ker)
                except IndexError:
                    ion_formula_1.append(ChemFormula(""))
                    ion_mass_1.append(0)
                    ion_charge_1.append(0)
                    ion_ker.append(0)
            elif n % 2 == 1:
                try:
                    formula = ChemFormula(self.entries_formula[n].get())
                    mass = get_mass(formula).value
                    charge = float(self.entries_charge[n].get())
                    ion_formula_2.append(formula)
                    ion_mass_2.append(mass)
                    ion_charge_2.append(charge)
                except IndexError:
                    ion_formula_2.append(ChemFormula(""))
                    ion_mass_2.append(0)
                    ion_charge_2.append(0)
        ion_mass_1 = np.array(ion_mass_1) * amu
        ion_charge_1 = np.array(ion_charge_1) * q_e
        ion_mass_2 = np.array(ion_mass_2) * amu
        ion_charge_2 = np.array(ion_charge_2) * q_e
        ion_ker_eV = np.array(ion_ker)

        # calc R tof for ions

        v_jet = self.velocity_jet_si
        ion_tof_1 = []
        ion_tof_2 = []
        ion_X_1 = []
        ion_X_2 = []
        ion_Y_1 = []
        ion_Y_2 = []
        electric_field = self.electric_field
        magnetic_field = self.magnetic_field_si
        length_acceleration = self.length_accel_ion.get()
        length_drift = self.length_drift_ion.get()
        for mass_1, mass_2, charge_1, charge_2, ker in zip(
            ion_mass_1, ion_mass_2, ion_charge_1, ion_charge_2, ion_ker_eV
        ):
            p_ion_1, p_ion_2 = make_momentum_ion_dis(
                ker, mass_1, mass_2, v_jet=v_jet, number_of_particles=1000
            )
            X_1, Y_1, tof_1 = calc_xytof(
                p_ion_1,
                electric_field,
                magnetic_field,
                length_acceleration,
                length_drift,
                particle_params=(mass_1, charge_1),
            )
            X_2, Y_2, tof_2 = calc_xytof(
                p_ion_2,
                electric_field,
                magnetic_field,
                length_acceleration,
                length_drift,
                particle_params=(mass_2, charge_2),
            )
            ion_tof_1.append(tof_1 * 1e9)
            ion_tof_2.append(tof_2 * 1e9)
            ion_X_1.append(X_1 * 1e3)
            ion_X_2.append(X_2 * 1e3)
            ion_Y_1.append(Y_1 * 1e3)
            ion_Y_2.append(Y_2 * 1e3)
        ion_tof_1 = np.array(ion_tof_1)
        ion_tof_2 = np.array(ion_tof_2)
        ion_X_1 = np.array(ion_X_1)
        ion_X_2 = np.array(ion_X_2)
        ion_Y_1 = np.array(ion_Y_1)
        ion_Y_2 = np.array(ion_Y_2)

        # cleanup plot
        for ax in self.pipico_fig.axes:
            ax.cla()
        for legend_object in self.pipico_fig.legends:
            legend_object.remove()

        # do new plots
        ax_x_tof = self.pipico_xtof_ax
        ax_y_tof = self.pipico_ytof_ax
        modulo = float(self.ENTRY_SET_bunch_modulo.get())
        detector_diameter = float(self.ENTRY_SET_detector_diameter.get())
        ax_x_tof.set_ylim(-1.2 * detector_diameter, 1.2 * detector_diameter)
        ax_y_tof.set_ylim(-1.2 * detector_diameter, 1.2 * detector_diameter)
        x_edges = y_edges = np.linspace(-detector_diameter * 0.55, detector_diameter * 0.55, 250)

        counts = 0
        legend_handles_even = []
        legend_labels_even = []
        legend_handles_odd = []
        legend_labels_odd = []
        for n in range(self.last_ion_number // 2):
            dots = ax_x_tof.scatter(
                ion_tof_1[n] % modulo,
                ion_X_1[n],
                color=self.ion_color[2 * n],
                alpha=0.2,
                edgecolors="none",
            )
            legend_handles_even.append(dots)
            legend_labels_even.append(f"{ion_formula_1[n]}$^{{{ion_charge_1[n] / q_e:.1g}+}}$")
            dots = ax_x_tof.scatter(
                ion_tof_2[n] % modulo,
                ion_X_2[n],
                color=self.ion_color[2 * n + 1],
                alpha=0.2,
                edgecolors="none",
            )
            legend_handles_odd.append(dots)
            legend_labels_odd.append(f"{ion_formula_2[n]}$^{{{ion_charge_2[n] / q_e:.1g}+}}$")
            ax_y_tof.scatter(
                ion_tof_1[n] % modulo,
                ion_Y_1[n],
                color=self.ion_color[2 * n],
                alpha=0.2,
                edgecolors="none",
            )
            ax_y_tof.scatter(
                ion_tof_2[n] % modulo,
                ion_Y_2[n],
                color=self.ion_color[2 * n + 1],
                alpha=0.2,
                edgecolors="none",
            )
            new_counts, _, _ = np.histogram2d(ion_X_1[n], ion_Y_1[n], bins=(x_edges, y_edges))
            counts += new_counts
            new_counts, _, _ = np.histogram2d(ion_X_2[n], ion_Y_2[n], bins=(x_edges, y_edges))
            counts += new_counts
        ax_xy = self.pipico_XY_ax
        counts[counts < 1] = np.nan
        ax_xy.pcolormesh(x_edges, y_edges, counts.T)
        ax_xy.add_artist(
            plt.Circle(
                (0, 0), detector_diameter / 2, color="cadetblue", fill=False, figure=self.pipico_fig
            )
        )
        ax_xy.set_xlabel("X [mm]")
        ax_xy.yaxis.set_label_position("right")
        ax_xy.yaxis.tick_right()
        ax_xy.set_ylabel("Y [mm]")
        ax_xy.grid()

        ax_x_tof.xaxis.tick_top()
        ax_y_tof.xaxis.tick_top()
        ax_x_tof.xaxis.set_label_position("top")
        ax_y_tof.xaxis.set_tick_params(
            "both", labelbottom=False, labeltop=False, bottom=False, top=False
        )
        ax_x_tof.set_xlabel("tof [ns]")
        ax_x_tof.set_ylabel("X [mm]")
        ax_y_tof.set_ylabel("Y [mm]")

        for i in range(5):
            jettof = np.linspace(modulo * i, modulo * (i + 1), 2)
            label = "Jet" if i == 0 else None
            ax_x_tof.plot([0, modulo], jettof * v_jet / 1e6, label=label, color="k", alpha=0.3)
        for ax in [ax_x_tof, ax_y_tof]:
            ax.axhline(detector_diameter / 2, color="red")
            ax.axhline(-detector_diameter / 2, color="red")
            ax.set_xlim(0, modulo)
            ax.grid()

        a = self.pipico_ax

        modulo = float(self.ENTRY_SET_bunch_modulo.get())
        for n in range(self.last_ion_number // 2):
            a.scatter(
                ion_tof_1[n] % modulo,
                ion_tof_2[n] % modulo,
                color=self.ion_color[2 * n],
                alpha=0.1,
                edgecolors="none",
            )
            a.scatter(
                ion_tof_2[n] % modulo,
                ion_tof_1[n] % modulo,
                color=self.ion_color[2 * n],
                alpha=0.1,
                edgecolors="none",
            )
        a.grid()
        a.set_xlabel("tof 1 [ns]")
        a.set_ylabel("tof 2 [ns]")
        a.set_xlim(0, modulo)
        a.set_ylim(0, modulo)
        legend_handles = legend_handles_even + legend_handles_odd
        legend_labels = legend_labels_even + legend_labels_odd
        legend = self.pipico_fig.legend(legend_handles, legend_labels, loc=4, ncols=2)
        for artist in legend.legend_handles:
            artist.set_alpha(1)
        self.pipico_canvas.draw()

    def update_electron_momenta(self):
        mass, charge = self.electron_params
        number_of_particles = len(self.pulses)
        if self.v.get() == 1:
            momentum = sc.vectors(
                dims=["pulses", "HitNr"],
                values=np.random.randn(number_of_particles, 1, 3),
                unit="au momentum",
            )
            dataarray = sc.DataArray(
                data=sc.ones(
                    sizes={"pulses": number_of_particles, "HitNr": 1, "p": 1}, dtype="int32"
                ),
                coords={
                    "pulses": self.pulses,
                    "HitNr": sc.array(dims=["HitNr"], values=np.arange(1)),
                    "p": momentum.to(unit="N*s"),
                },
            )

        elif self.v.get() == 2:
            energy_mean = sc.scalar(float(self.ENTRY_MEAN_ENERGY.get()), unit="eV")
            width = sc.scalar(float(self.ENTRY_WIDTH.get()), unit="eV")
            energy = (
                sc.array(dims=["pulses", "HitNr"], values=np.random.randn(number_of_particles, 1))
                * width
                + energy_mean
            )
            r_mom = sc.sqrt(energy * 2 * mass)

            phi = sc.array(
                dims=["pulses", "HitNr"],
                values=np.random.rand(number_of_particles, 1) * 2 * np.pi,
                unit="rad",
            )

            cos_theta = sc.array(
                dims=["pulses", "HitNr"], values=np.random.rand(number_of_particles, 1) * 2 - 1
            )
            theta = sc.acos(cos_theta)

            x = r_mom * sc.sin(theta) * sc.cos(phi)
            y = r_mom * sc.sin(theta) * sc.sin(phi)
            z = r_mom * cos_theta
            dataarray = sc.DataArray(
                data=sc.ones(
                    sizes={"pulses": number_of_particles, "HitNr": 1, "p": 1}, dtype="int32"
                ),
                coords={
                    "pulses": self.pulses,
                    "HitNr": sc.array(dims=["HitNr"], values=np.arange(1)),
                    "p": sc.spatial.as_vectors(x, y, z).to(unit="N*s"),
                },
            )
        elif self.v.get() == 3:
            energy_mean = sc.scalar(float(self.ENTRY_MEAN_ENERGY.get()), unit="eV")
            width = sc.scalar(float(self.ENTRY_WIDTH.get()), unit="eV")
            energy_step = sc.scalar(float(self.ENTRY_MULTI_PART_ENERGY_STEP.get()), unit="eV")
            part_num = int(self.ENTRY_MULTI_PART_NUMBER.get())
            particles = []
            for i in range(part_num):
                energy = (
                    sc.array(
                        dims=["pulses", "HitNr"], values=np.random.randn(number_of_particles, 1)
                    )
                    * width
                    + energy_mean
                    + i * energy_step
                )
                r_mom = sc.sqrt(energy * 2 * mass)

                phi = sc.array(
                    dims=["pulses", "HitNr"],
                    values=np.random.rand(number_of_particles, 1) * 2 * np.pi,
                    unit="rad",
                )

                cos_theta = sc.array(
                    dims=["pulses", "HitNr"], values=np.random.rand(number_of_particles, 1) * 2 - 1
                )
                theta = sc.acos(cos_theta)

                x = r_mom * sc.sin(theta) * sc.cos(phi)
                y = r_mom * sc.sin(theta) * sc.sin(phi)
                z = r_mom * cos_theta
                momentum = sc.spatial.as_vectors(x, y, z)

                particles.append(momentum)
            dataarray = sc.DataArray(
                data=sc.ones(
                    sizes={"pulses": number_of_particles, "HitNr": part_num, "p": 1}, dtype="int32"
                ),
                coords={
                    "pulses": self.pulses,
                    "HitNr": sc.array(dims=["HitNr"], values=np.arange(part_num)),
                    "p": sc.concat(particles, "HitNr").to(unit="N*s"),
                },
            )
        self.electron_momenta = dataarray
        self.update_electron_positions()

    def update_electron_positions(self):
        mass, charge = self.electron_params

        length_acceleration = sc.scalar(self.length_accel_electron.get(), unit="m")
        length_drift = sc.scalar(self.length_drift_electron.get(), unit="m")
        electric_field = sc.scalar(self.electric_field, unit="V/m")
        magnetic_field = sc.scalar(self.magnetic_field_si, unit="T")
        voltage_difference = electric_field * length_acceleration

        def calc_tof(p):
            p_z = p.fields.z
            D = p_z * p_z - 2 * charge * voltage_difference * mass
            rootD = sc.sqrt(D)
            tof = sc.where(
                D < 0 * sc.Unit("J*kg"),
                sc.scalar(np.nan, unit="s"),
                mass * (2 * length_acceleration / (rootD + p_z) + length_drift / rootD),
            )
            return {"tof": tof.to(unit="ns")}

        def calc_xy(p, tof):
            p_x = p.fields.x
            p_y = p.fields.y
            if magnetic_field > 0 * sc.Unit("T"):  # cyclotron motion during the time-of-flight
                p_xy = sc.sqrt(p_x**2 + p_y**2)
                phi = sc.atan2(x=p_x, y=p_y)
                omega = calc_omega(magnetic_field, charge, mass)
                alpha = omega * tof.to(unit="s") * sc.scalar(1, unit="rad")
                theta = phi + alpha / 2
                R = (2 * p_xy * sc.abs(sc.sin(alpha / 2))) / (charge * magnetic_field)
                x = R * sc.sin(theta)
                y = R * sc.cos(theta)
            else:  # for small magnetic field it reduces to this linear motion:
                v_x = p_x / mass
                v_y = p_y / mass
                x = v_x * tof
                y = v_y * tof
            return {"x": x.to(unit="mm"), "y": y.to(unit="mm")}

        def calc_R(x, y):
            return sc.sqrt(x**2 + y**2)

        graph = {"tof": calc_tof, ("x", "y"): calc_xy, "R": calc_R}

        dataarray = self.electron_momenta.transform_coords(["x", "y", "tof", "R"], graph=graph)
        self.electron_hits = dataarray
        self.update_R_tof()
        self.update_electron_detector_signals()

    def update_electron_detector_signals(self):
        hits = self.electron_hits
        self.datagroup["electrons"] = hits
        pass

    def update_ion_momenta(self):
        self.update_ion_positions()

    def update_ion_positions(self):
        self.update_ion_detector_signals()

    def update_ion_detector_signals(self):
        pass

    def init_dataset(self):
        n_samples = self.number_of_particles.get()
        self.pulses = sc.arange("pulses", n_samples)
        self.datagroup = sc.DataGroup(pulses=self.pulses)
        self.update_electron_momenta()
        self.update_ion_momenta()


def main():
    window = Tk()
    window.configure(background=canvas_background_color)
    mclass(window)
    window.mainloop()


if __name__ == "__main__":
    main()
