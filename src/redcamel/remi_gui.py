#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Patrizia Schoch
# SPDX-FileContributor: Hannes Lindenblatt
# SPDX-FileContributor: Magdalena Orlowska
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
from collections import defaultdict
from pathlib import Path
from tkinter import (
    HORIZONTAL,
    BooleanVar,
    DoubleVar,
    IntVar,
    PhotoImage,
    StringVar,
    Tk,
    filedialog,
    font,
)
from tkinter.ttk import (
    Button,
    Checkbutton,
    Entry,
    Frame,
    Label,
    LabelFrame,
    Notebook,
    Radiobutton,
    Scale,
    Style,
)

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import scipp as sc
from chemformula import ChemFormula
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from .remi_calculator import RemiCalculator
from .remi_particles import Electron, Particle, sample_coulomb_explosion, sample_photoionization
from .units import q_e

#############################
#### GUI ####################
############################
canvas_background_color = "mintcream"


class RemiCalculatorTk(RemiCalculator):
    def __init__(
        self,
        length_acceleration_ion_tkvariable: DoubleVar,
        length_drift_ion_tkvariable: DoubleVar,
        voltage_ion_tkvariable: DoubleVar,
        length_acceleration_electron_tkvariable: DoubleVar,
        length_drift_electron_tkvariable: DoubleVar,
        voltage_electron_tkvariable: DoubleVar,
        magnetic_field_tkvariable: DoubleVar,
        v_jet_tkvariable: DoubleVar,
    ):
        self.length_acceleration_ion_tkvariable = length_acceleration_ion_tkvariable
        self.length_drift_ion_tkvariable = length_drift_ion_tkvariable
        self.voltage_ion_tkvariable = voltage_ion_tkvariable
        self.length_acceleration_electron_tkvariable = length_acceleration_electron_tkvariable
        self.length_drift_electron_tkvariable = length_drift_electron_tkvariable
        self.voltage_electron_tkvariable = voltage_electron_tkvariable
        self.magnetic_field_tkvariable = magnetic_field_tkvariable
        self.v_jet_tkvariable = v_jet_tkvariable
        self.jet_direction = "+x"
        self.field_direction = "+z"

    @property
    def length_acceleration_ion(self):
        return sc.scalar(self.length_acceleration_ion_tkvariable.get(), unit="m")

    @property
    def length_drift_ion(self):
        return sc.scalar(self.length_drift_ion_tkvariable.get(), unit="m")

    @property
    def voltage_ion(self):
        return sc.scalar(self.voltage_ion_tkvariable.get(), unit="V")

    @property
    def length_acceleration_electron(self):
        return sc.scalar(self.length_acceleration_electron_tkvariable.get(), unit="m")

    @property
    def length_drift_electron(self):
        return sc.scalar(self.length_drift_electron_tkvariable.get(), unit="m")

    @property
    def voltage_electron(self):
        return sc.scalar(self.voltage_electron_tkvariable.get(), unit="V")

    @property
    def magnetic_field(self):
        return sc.scalar(self.magnetic_field_tkvariable.get(), unit="G").to(unit="T")

    @property
    def v_jet(self):
        return sc.scalar(self.v_jet_tkvariable.get(), unit="mm/us")


class mclass:
    def __init__(self, window):
        self.window = window
        window.title("Red Camel")
        photo = PhotoImage(file=Path(__file__).parent / "icon.png")
        window.wm_iconphoto(False, photo)

        self.scale_UI()

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
        for label in labels:
            tab_frame = Frame(tabControl)

            # Only enable tabs, that are implemented:
            tab_implemented = label in ["Electrons", "Ions", "Export Data", "Spectrometer"]
            tab_state = "normal" if tab_implemented else "disabled"
            tabControl.add(tab_frame, text=label, state=tab_state)

            tab_frame.columnconfigure(0, weight=1)
            tab_frame.rowconfigure(0, weight=1)
            tab = Frame(tab_frame)
            tab.grid(column=0, row=0, sticky="nsew")

            tabs[label] = tab

        ######## global Remi variables ####################
        self.length_accel_ion = DoubleVar(value=0.115)
        self.length_drift_ion = DoubleVar(value=0.0)
        self.length_accel_electron = DoubleVar(value=0.2)
        self.length_drift_electron = DoubleVar(value=0.0)
        self.voltage_electron = DoubleVar(value=+469.5652173913043)
        self.voltage_ion = DoubleVar()  # gets initialized by ratio of distances
        self.magnetic_field_gauss = DoubleVar(value=6.0)
        self.velocity_jet = DoubleVar(value=5.0)

        self.remicalculator = RemiCalculatorTk(
            self.length_accel_ion,
            self.length_drift_ion,
            self.voltage_ion,
            self.length_accel_electron,
            self.length_drift_electron,
            self.voltage_electron,
            self.magnetic_field_gauss,
            self.velocity_jet,
        )

        self.detector_diameter_ions = DoubleVar(value=120)
        self.detector_diameter_electrons = DoubleVar(value=80)

        self.number_of_particles = IntVar(value=1000)
        self.electron_params = (sc.constants.m_e, -sc.constants.e)
        self.bunch_modulo = DoubleVar(value=5000)

        self.fixed_center_potential = IntVar()
        self.fixed_center_potential.set(1)

        self.v = IntVar()
        self.v.set(2)

        ######## variable callback actions ####################

        job_wait_ms = 20
        self.callback_job_id = None

        def update_callback_spectrometer():
            self.update_electron_positions()
            self.update_ion_positions()
            self.update_spectrometer_tab()
            self.callback_job_id = None

        def write_callback_voltage_electron(var, index, mode):
            if self.fixed_center_potential.get():
                new_voltage = self.voltage_electron.get()
                distance_electron = self.length_accel_electron.get()
                distance_ion = self.length_accel_ion.get()
                new_voltage_ion = -new_voltage * distance_ion / distance_electron
                self.voltage_ion.set(new_voltage_ion)

            # cancel previous update if triggered too fast:
            if self.callback_job_id is not None:
                self.window.after_cancel(self.callback_job_id)
            # schedule update for later:
            self.callback_job_id = self.window.after(job_wait_ms, update_callback_spectrometer)

        for variable in [
            self.voltage_electron,
            self.length_accel_electron,
            self.fixed_center_potential,
        ]:
            variable.trace("w", write_callback_voltage_electron)

        self.callback_job_id_ion = None

        def write_callback_voltage_ion(var, index, mode):
            if self.fixed_center_potential.get():
                new_voltage = self.voltage_ion.get()
                distance_electron = self.length_accel_electron.get()
                distance_ion = self.length_accel_ion.get()
                new_voltage_ion = -new_voltage * distance_electron / distance_ion
                self.voltage_electron.set(new_voltage_ion)

            # cancel previous update if triggered too fast:
            if self.callback_job_id is not None:
                self.window.after_cancel(self.callback_job_id)
            # schedule update for later:
            self.callback_job_id = self.window.after(job_wait_ms, update_callback_spectrometer)

        for variable in [self.voltage_ion, self.length_accel_ion]:
            variable.trace("w", write_callback_voltage_ion)

        def write_callback_spectrometer(var, index, mode):
            # cancel previous update if triggered too fast:
            if self.callback_job_id is not None:
                self.window.after_cancel(self.callback_job_id)
            # schedule update for later:
            self.callback_job_id = self.window.after(job_wait_ms, update_callback_spectrometer)

        for variable in [
            self.magnetic_field_gauss,
            self.velocity_jet,
            self.length_drift_electron,
            self.detector_diameter_electrons,
            self.length_drift_ion,
            self.bunch_modulo,
            self.detector_diameter_ions,
        ]:
            variable.trace("w", write_callback_spectrometer)

        def write_callback_momentum_electrons(var, index, mode):
            self.update_electron_momenta()

        self.v.trace("w", write_callback_momentum_electrons)

        #################################################################################
        #############################      Electrons      ###############################
        #################################################################################

        ######## higher groups ####################
        tabs["Electrons"].rowconfigure(0, weight=1, minsize=200)
        for row in range(1, 6):
            tabs["Electrons"].rowconfigure(row, weight=1, minsize=50)
        tabs["Electrons"].columnconfigure(0, weight=1, minsize=280)
        for col in range(1, 6):
            tabs["Electrons"].columnconfigure(col, weight=1, minsize=100)
        left_bar_group = Frame(tabs["Electrons"], padding=(5, 5, 5, 5))
        left_bar_group.grid(row=0, column=0, columnspan=1, rowspan=6, sticky="nsew")

        top_bar_group = Frame(tabs["Electrons"], padding=(5, 5, 5, 0))
        top_bar_group.grid(row=0, column=1, columnspan=5, rowspan=1, sticky="nsew")

        self.R_tof_plot_group_frame = Frame(tabs["Electrons"], padding=(5, 0, 5, 5))
        self.R_tof_plot_group_frame.grid(row=1, column=1, columnspan=5, rowspan=5, sticky="nsew")
        self.R_tof_plot_group_frame.rowconfigure(0, weight=1, minsize=0)
        self.R_tof_plot_group_frame.columnconfigure(0, weight=1, minsize=0)
        self.R_tof_plot_group = LabelFrame(self.R_tof_plot_group_frame, text="Electron plots")
        self.R_tof_plot_group.grid(row=0, column=0, sticky="nsew")

        ######## REMI configurations ##############
        left_bar_group.columnconfigure(0, weight=1, minsize=280)
        left_bar_group.rowconfigure(0, weight=1)
        left_bar_group.rowconfigure(1, weight=1)

        remi_conf_group = LabelFrame(left_bar_group, text="REMI Configuration for Electrons")
        remi_conf_group.grid(row=0, column=0, rowspan=1, sticky="nsew")
        remi_conf_group.columnconfigure(1, weight=1)

        self.LABEL_SET_B = Label(remi_conf_group, text="B[Gauss]:")
        self.LABEL_SET_B.grid(row=0, column=0, padx="5", pady="5", sticky="ew")
        self.ENTRY_SET_B = Entry(remi_conf_group, textvariable=self.magnetic_field_gauss)
        self.ENTRY_SET_B.grid(row=0, column=1, padx="5", pady="5", sticky="ew")

        self.LABEL_SET_l_d = Label(remi_conf_group, text="drift length[m]:")
        self.LABEL_SET_l_d.grid(row=1, column=0, padx="5", pady="5", sticky="ew")
        self.ENTRY_SET_l_d = Entry(remi_conf_group, textvariable=self.length_drift_electron)
        self.ENTRY_SET_l_d.grid(row=1, column=1, padx="5", pady="5", sticky="ew")

        self.LABEL_SET_l_a = Label(remi_conf_group, text="acc length[m]:")
        self.LABEL_SET_l_a.grid(row=2, column=0, padx="5", pady="5", sticky="ew")
        self.ENTRY_SET_l_a = Entry(remi_conf_group, textvariable=self.length_accel_electron)
        self.ENTRY_SET_l_a.grid(row=2, column=1, padx="5", pady="5", sticky="ew")

        self.LABEL_SET_U = Label(remi_conf_group, text="U[V]:")
        self.LABEL_SET_U.grid(row=3, column=0, padx="5", pady="5", sticky="ew")
        self.ENTRY_SET_U = Entry(remi_conf_group, textvariable=self.voltage_electron)
        self.ENTRY_SET_U.grid(row=3, column=1, padx="5", pady="5", sticky="ew")

        self.LABEL_SET_electron_detector_diameter = Label(
            remi_conf_group, text="detector diameter [mm]:"
        )
        self.LABEL_SET_electron_detector_diameter.grid(
            row=4, column=0, padx="5", pady="5", sticky="w"
        )
        self.ENTRY_SET_electron_detector_diameter = Entry(
            remi_conf_group, textvariable=self.detector_diameter_electrons
        )
        self.ENTRY_SET_electron_detector_diameter.grid(
            row=4, column=1, padx="5", pady="5", sticky="w"
        )

        self.CHECK_fixed_potential_ele = Checkbutton(
            remi_conf_group,
            text="Interaction region on Ground potential",
            variable=self.fixed_center_potential,
        )
        self.CHECK_fixed_potential_ele.grid(
            row=5, column=0, columnspan=2, padx="5", pady="5", sticky="ew"
        )

        ######## momentum, R, tof calculation #############
        self.R_tof_group = LabelFrame(left_bar_group, text="Electron Momentum Distribution")
        self.R_tof_group.grid(row=1, column=0, rowspan=5, sticky="nswe")
        self.R_tof_group.columnconfigure(1, weight=1, minsize=50)

        self.CHOOSE_MOMENTUM = Radiobutton(
            self.R_tof_group, command=self.check, text="Momentum", variable=self.v, value=1
        )
        self.CHOOSE_ENERGY = Radiobutton(
            self.R_tof_group, command=self.check, text="Energy", variable=self.v, value=2
        )
        self.CHOOSE_MOMENTUM.grid(row=0, column=0, padx="5", pady="5", sticky="w")
        self.CHOOSE_ENERGY.grid(row=1, column=0, padx="5", pady="5", sticky="w")
        self.CHOOSE_ENERGY_MULTI = Radiobutton(
            self.R_tof_group, command=self.check, text="Multiple Energies", variable=self.v, value=3
        )
        self.CHOOSE_ENERGY_MULTI.grid(row=2, column=0, padx="5", pady="5", sticky="w")

        self.LABEL_NUMBER_PART = Label(self.R_tof_group, text="number of Particles:")
        self.ENTRY_NUMBER_PART = Entry(self.R_tof_group, textvariable=self.number_of_particles)

        # if selecting calculation with energy
        self.LABEL_MEAN_ENERGY = Label(self.R_tof_group, text="Mean Energy:")
        self.LABEL_WIDTH = Label(self.R_tof_group, text="Width:")
        self.ENTRY_MEAN_ENERGY = Entry(self.R_tof_group)
        self.ENTRY_WIDTH = Entry(self.R_tof_group)

        # if multiple particles
        self.LABEL_MULTI_PART_ENERGY_STEP = Label(self.R_tof_group, text="Energy Step:")
        self.ENTRY_MULTI_PART_ENERGY_STEP = Entry(self.R_tof_group)
        self.LABEL_MULTI_PART_NUMBER = Label(self.R_tof_group, text="Number of Particles")
        self.ENTRY_MULTI_PART_NUMBER = Entry(self.R_tof_group)

        self.LABEL_NUMBER_PART.grid(row=3, column=0, padx="5", pady="5", sticky="w")
        self.ENTRY_NUMBER_PART.grid(row=3, column=1, padx="5", pady="5", sticky="ew")
        self.LABEL_MEAN_ENERGY.grid(row=4, column=0, padx="5", pady="5", sticky="w")
        self.ENTRY_MEAN_ENERGY.grid(row=4, column=1, padx="5", pady="5", sticky="ew")
        self.LABEL_WIDTH.grid(row=5, column=0, padx="5", pady="5", sticky="w")
        self.ENTRY_WIDTH.grid(row=5, column=1, padx="5", pady="5", sticky="ew")
        self.LABEL_MULTI_PART_ENERGY_STEP.grid(row=6, column=0, padx="5", pady="5", sticky="w")
        self.ENTRY_MULTI_PART_ENERGY_STEP.grid(row=6, column=1, padx="5", pady="5", sticky="ew")
        self.LABEL_MULTI_PART_NUMBER.grid(row=7, column=0, padx="5", pady="5", sticky="w")
        self.ENTRY_MULTI_PART_NUMBER.grid(row=7, column=1, padx="5", pady="5", sticky="ew")

        self.ENTRY_MEAN_ENERGY.insert(0, 10)
        self.ENTRY_WIDTH.insert(0, 0.1)
        self.ENTRY_MULTI_PART_ENERGY_STEP.insert(0, 1.5)
        self.ENTRY_MULTI_PART_NUMBER.insert(0, 3)

        self.LABEL_MULTI_PART_ENERGY_STEP.grid_remove()
        self.ENTRY_MULTI_PART_ENERGY_STEP.grid_remove()
        self.LABEL_MULTI_PART_NUMBER.grid_remove()
        self.ENTRY_MULTI_PART_NUMBER.grid_remove()

        ######## R tof simulation ##########################
        top_bar_group.rowconfigure(0, weight=1)
        for col in range(2):
            top_bar_group.columnconfigure(col, weight=1)

        self.R_tof_sim_group = LabelFrame(top_bar_group, text="R-tof lines")
        self.R_tof_sim_group.grid(row=0, column=0, sticky="nswe", padx=(0, 5))
        self.R_tof_sim_group.columnconfigure(1, weight=1)

        self.LABEL_KIN_ENERGY = Label(self.R_tof_sim_group, text="Kinetic Energy [eV]:")
        ele_1_color = "firebrick"
        ele_2_color = "deepskyblue"
        ele_3_color = "darkorange"
        self.ENTRY_KIN_ENERGY_1 = Entry(self.R_tof_sim_group, foreground=ele_1_color)
        self.ENTRY_KIN_ENERGY_2 = Entry(self.R_tof_sim_group, foreground=ele_2_color)
        self.ENTRY_KIN_ENERGY_3 = Entry(self.R_tof_sim_group, foreground=ele_3_color)
        self.BUTTON_R_TOF_SIM = Button(
            self.R_tof_sim_group, text="Draw lines", command=self.R_tof_sim
        )

        self.ENTRY_KIN_ENERGY_1.insert(0, 5)
        self.ENTRY_KIN_ENERGY_2.insert(0, 10)
        self.ENTRY_KIN_ENERGY_3.insert(0, 15)

        self.LABEL_KIN_ENERGY.grid(row=0, column=0, padx="5", pady="5", sticky="w")
        self.BUTTON_R_TOF_SIM.grid(row=2, column=0, padx="5", pady="5", sticky="w")
        self.ENTRY_KIN_ENERGY_1.grid(row=0, column=1, padx="5", pady="5", sticky="ew")
        self.ENTRY_KIN_ENERGY_2.grid(row=1, column=1, padx="5", pady="5", sticky="ew")
        self.ENTRY_KIN_ENERGY_3.grid(row=2, column=1, padx="5", pady="5", sticky="ew")

        #### Plots and Slidebars ##############
        for row in range(4):
            self.R_tof_plot_group.rowconfigure(row, weight=1, minsize=0)
        for col in range(4):
            self.R_tof_plot_group.columnconfigure(col, weight=1, minsize=0)
        self.v_ir = IntVar()
        self.v_ir.set(0)
        self.CHECK_IR_PLOT = Checkbutton(
            self.R_tof_plot_group, text="Enable IR plot Mode", variable=self.v_ir
        )
        self.CHECK_IR_PLOT.grid(row=5, column=0, columnspan=2, padx="5", pady="5", sticky="ew")

        self.LABEL_SLIDE_U = Label(self.R_tof_plot_group, text="Voltage electron side [V]")
        self.LABEL_SLIDE_U.grid(row=6, column=0, padx="5", pady="5", sticky="ew")
        self.LABEL_VALUE_SLIDE_U = Label(self.R_tof_plot_group, textvariable=self.voltage_electron)
        self.LABEL_VALUE_SLIDE_U.grid(row=6, column=1, padx="5", pady="5", sticky="w")

        self.SLIDE_U = Scale(
            self.R_tof_plot_group,
            from_=0,
            to=1000,
            orient=HORIZONTAL,
            variable=self.voltage_electron,
        )
        self.SLIDE_U.grid(row=7, column=0, columnspan=2, padx="5", pady="5", sticky="ew")

        self.LABEL_SLIDE_B = Label(self.R_tof_plot_group, text="Magnetic Field [G]")
        self.LABEL_SLIDE_B.grid(row=6, column=2, padx="5", pady="5", sticky="w")
        self.LABEL_VALUE_SLIDE_B = Label(
            self.R_tof_plot_group, textvariable=self.magnetic_field_gauss
        )
        self.LABEL_VALUE_SLIDE_B.grid(row=6, column=3, padx="5", pady="5", sticky="w")

        self.SLIDE_B = Scale(
            self.R_tof_plot_group,
            from_=0,
            to=50,
            orient=HORIZONTAL,
            variable=self.magnetic_field_gauss,
        )
        self.SLIDE_B.grid(row=7, column=2, columnspan=2, padx="5", pady="5", sticky="ew")

        #### IR mode #####
        self.ir_mode_group = LabelFrame(top_bar_group, text="IR-Mode")
        self.ir_mode_group.grid(row=0, column=1, columnspan=1, sticky="nswe")
        self.ir_mode_group.columnconfigure(1, weight=1)

        self.LABEL_KIN_ENERGY_START = Label(self.ir_mode_group, text="First Kin Energy [eV]")
        self.LABEL_KIN_ENERGY_STEP = Label(self.ir_mode_group, text="Kin Energy Stepsize [eV]")
        self.LABEL_NUMBER_OF_PART = Label(self.ir_mode_group, text="Number of particles")

        self.ENTRY_KIN_ENERGY_START = Entry(self.ir_mode_group, foreground=ele_1_color)
        self.ENTRY_KIN_ENERGY_STEP = Entry(self.ir_mode_group, foreground=ele_1_color)
        self.ENTRY_NUMBER_OF_PART = Entry(self.ir_mode_group, foreground=ele_1_color)

        self.ENTRY_KIN_ENERGY_START.insert(0, 1.3)
        self.ENTRY_KIN_ENERGY_STEP.insert(0, 1.55)
        self.ENTRY_NUMBER_OF_PART.insert(0, 10)

        self.BUTTON_SIM_IR_MODE = Button(
            self.ir_mode_group, text="Draw lines", command=self.R_tof_sim_ir
        )
        self.BUTTON_SIM_IR_MODE.grid(row=2, column=2, padx="5", pady="5", sticky="e")

        self.LABEL_KIN_ENERGY_START.grid(row=0, column=0, padx="5", pady="5", sticky="w")
        self.LABEL_KIN_ENERGY_STEP.grid(row=1, column=0, padx="5", pady="5", sticky="w")
        self.LABEL_NUMBER_OF_PART.grid(row=2, column=0, padx="5", pady="5", sticky="w")

        self.ENTRY_KIN_ENERGY_START.grid(row=0, column=1, padx="5", pady="5", sticky="ew")
        self.ENTRY_KIN_ENERGY_STEP.grid(row=1, column=1, padx="5", pady="5", sticky="ew")
        self.ENTRY_NUMBER_OF_PART.grid(row=2, column=1, padx="5", pady="5", sticky="ew")

        #################################################################################
        ###############################      Ions      ##################################
        #################################################################################
        tabs["Ions"].rowconfigure(0, weight=1, minsize=50)
        for col in range(2):
            tabs["Ions"].columnconfigure(col, weight=1, minsize=100)
        for col in range(2, 5):
            tabs["Ions"].columnconfigure(col, weight=2, minsize=100)

        ######## higher groups ####################
        left_tab2_group = Frame(tabs["Ions"])
        left_tab2_group.grid(row=0, column=0, columnspan=2, sticky="nsew")
        left_tab2_group.columnconfigure(0, weight=1, minsize=500)
        left_tab2_group.rowconfigure(2, weight=1, minsize=500)

        ker_group = LabelFrame(left_tab2_group, text="Calculate KER")
        ker_group.grid(row=0, column=0, padx="5", pady="5", sticky="nsew")

        remi_ion_conf_group = LabelFrame(left_tab2_group, text="REMI Configuration for Ion")
        remi_ion_conf_group.grid(row=1, column=0, padx="5", pady="5", sticky="nsew")

        self.ion_generation_group = LabelFrame(left_tab2_group, text="Ion generation")
        self.ion_generation_group.grid(row=2, column=0, padx="5", pady="5", sticky="nsew")
        for col in range(1, 6):
            self.ion_generation_group.columnconfigure(col, weight=1, minsize=50)

        self.pipico_plot_group = LabelFrame(tabs["Ions"], text="PIPICO")
        self.pipico_plot_group.grid(
            row=0, column=2, columnspan=3, padx="5", pady="5", sticky="nsew"
        )
        self.pipico_plot_group.rowconfigure(0, weight=1, minsize=100)
        self.pipico_plot_group.columnconfigure(0, weight=1, minsize=300)

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
        self.ENTRY_SET_bunch_modulo = Entry(remi_ion_conf_group, textvariable=self.bunch_modulo)
        self.ENTRY_SET_bunch_modulo.grid(row=103, column=102, padx="5", pady="5", sticky="w")

        self.LABEL_SET_ion_detector_diameter = Label(
            remi_ion_conf_group, text="detector diameter [mm]:"
        )
        self.LABEL_SET_ion_detector_diameter.grid(
            row=104, column=101, padx="5", pady="5", sticky="w"
        )
        self.ENTRY_SET_ion_detector_diameter = Entry(
            remi_ion_conf_group, textvariable=self.detector_diameter_ions
        )
        self.ENTRY_SET_ion_detector_diameter.grid(
            row=104, column=102, padx="5", pady="5", sticky="w"
        )

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
        )
        self.CHECK_fixed_potential_ion.grid(
            row=108, column=101, columnspan=2, padx="5", pady="5", sticky="w"
        )

        self.LABEL_SLIDE_U_pipco = Label(self.pipico_plot_group, text="Voltage ion side [V]")
        self.LABEL_SLIDE_U_pipco.grid(row=2, column=0, padx="5", pady="5", sticky="w")

        self.SLIDE_U_pipco = Scale(
            self.pipico_plot_group,
            from_=0,
            to=-3000,
            orient=HORIZONTAL,
            # resolution=0.1,
            variable=self.voltage_ion,
        )
        self.SLIDE_U_pipco.grid(row=3, column=0, padx="5", pady="5", sticky="ew")

        self.LABEL_SLIDE_B_pipco = Label(self.pipico_plot_group, text="Magnetig Field [G]")
        self.LABEL_SLIDE_B_pipco.grid(row=4, column=0, padx="5", pady="5", sticky="w")

        self.SLIDE_B_pipco = Scale(
            self.pipico_plot_group,
            from_=0,
            to=50,
            orient=HORIZONTAL,
            # resolution=0.1,
            variable=self.magnetic_field_gauss,
        )
        self.SLIDE_B_pipco.grid(row=5, column=0, padx="5", pady="5", sticky="ew")

        ### ion generator ###################

        self.LABEL_FORMULA_IONS = Label(self.ion_generation_group, text="ChemFormula:")
        self.LABEL_MASS_IONS = Label(self.ion_generation_group, text="Mass [amu]:")
        self.LABEL_CHARGE_IONS = Label(self.ion_generation_group, text="Charge [au]:")
        self.LABEL_KER_IONS = Label(self.ion_generation_group, text="KER [eV]:")
        self.LABEL_TOF_IONS = Label(self.ion_generation_group, text="TOF [ns]:")
        self.LABEL_CHECK = Label(self.ion_generation_group, text="Plot?")

        self.ENTRY_NUMBER_IONS = Entry(self.ion_generation_group)
        self.ENTRY_NUMBER_IONS.grid(row=0, column=2, padx="5", pady="5", sticky="w")
        self.ENTRY_NUMBER_IONS.insert(0, 6)

        self.LABEL_FORMULA_IONS.grid(row=1, column=1, padx="5", pady="5", sticky="w")
        self.LABEL_MASS_IONS.grid(row=1, column=2, padx="5", pady="5", sticky="w")
        self.LABEL_CHARGE_IONS.grid(row=1, column=3, padx="5", pady="5", sticky="w")
        self.LABEL_TOF_IONS.grid(row=1, column=4, padx="5", pady="5", sticky="w")
        self.LABEL_KER_IONS.grid(row=1, column=5, padx="5", pady="5", sticky="w")
        self.LABEL_CHECK.grid(row=1, column=6, padx="5", pady="5", sticky="w")

        self.ion_labels = []
        self.entries_formula = []
        self.formula_variables = []
        self.labels_mass = []
        self.entries_charge = []
        self.charge_variables = []
        self.labels_ion_tof = []
        self.entries_ker = []
        self.ker_variables = []
        self.active_check_variables = []
        self.active_channels = 0

        self.BUTTON_GENERATE_IONS = Button(
            self.ion_generation_group,
            command=self.generate_ion_pair_entries,
            text="Make Ion Couples",
        )
        self.BUTTON_GENERATE_IONS.grid(row=0, column=1, padx="5", pady="5", sticky="w")

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
        self.pipico_canvas.get_tk_widget().grid(row=0, column=0, padx="5", pady="5", sticky="ew")
        self.pipico_toolbar = NavigationToolbar2Tk(
            canvas=self.pipico_canvas, window=self.pipico_plot_group, pack_toolbar=False
        )
        self.pipico_toolbar.grid(row=1, column=0, padx="5", pady="5", sticky="ew")
        self.pipico_toolbar.update()

        ### initialize plots ###################
        self.make_R_tof_figure()
        self.make_export_tab()
        self.make_spectrometer_tab()

        ### initialize data and update plots ###################
        self.init_dataset()
        write_callback_voltage_electron(None, None, None)  # initalize ion voltages
        self.calc_ker()

    def scale_UI(self):
        # screen size in pixels:
        width = self.window.winfo_screenwidth()
        height = self.window.winfo_screenheight()
        # (primary) screen resolution
        reported_width = width / self.window.winfo_fpixels("1m")
        reported_height = height / self.window.winfo_fpixels("1m")
        # print("screen width [mm]: ", reported_width)
        # print("screen height [mm]: ", reported_height)

        # The UI-scaling should be a setting somewhere in the GUI
        # replace with measurement here:
        measured_width = reported_width
        measured_height = reported_height

        dpi = self.window.winfo_fpixels("1i")  # returns how many pixels is 1 inch
        tk_assumed_dpi = 72.0
        current_scaling = self.window.tk.call("tk", "scaling")
        np.testing.assert_almost_equal(dpi / current_scaling, tk_assumed_dpi, decimal=1)
        rescaling = (measured_width / reported_width + measured_height / reported_height) / 2
        new_scaling = current_scaling * rescaling
        self.window.tk.call("tk", "scaling", new_scaling)  # conserves fontsize by scaling it down

        for name in font.names(self.window):
            tkfont = font.Font(root=self.window, name=name, exists=True)
            size = int(tkfont["size"])
            tkfont["size"] = int(np.ceil(size * rescaling))

    def make_spectrometer_tab(self):
        self.tabs["Spectrometer"].columnconfigure(0, weight=1)
        self.tabs["Spectrometer"].rowconfigure(0, weight=1)
        (
            self.fig_spectrometer,
            self.ax_spectrometer,
            self.canvas_spectrometer,
            self.toolbar_spectrometer,
        ) = self.make_plot(0, 0, self.tabs["Spectrometer"], figsize=(5, 5), columnspan=1, rowspan=2)
        (
            self.fig_trajectory,
            self.ax_trajectory,
            self.canvas_trajectory,
            self.toolbar_trajectory,
        ) = self.make_plot(
            2,
            0,
            self.tabs["Spectrometer"],
            figsize=(5, 5),
            columnspan=1,
            rowspan=2,
            subplot_kw={"projection": "3d"},
        )

    def update_spectrometer_tab(self):
        self.ax_spectrometer.clear()
        self.ax_trajectory.clear()

        Ld_i = self.length_drift_ion.get()
        La_i = self.length_accel_ion.get()
        La_e = self.length_accel_electron.get()
        Ld_e = self.length_drift_electron.get()
        U_i = self.voltage_ion.get()
        U_e = self.voltage_electron.get()

        z = np.cumsum([0, Ld_e, La_e, La_i, Ld_i])
        z -= z[2]  # shift center to zero

        if self.fixed_center_potential.get():
            U_0 = 0
        else:
            E = (U_e - U_i) / (La_i + La_e)
            U_0 = U_i + E * La_i

        U = [U_e, U_e, U_0, U_i, U_i]

        self.ax_spectrometer.text(
            0.2,
            1.05,
            "Electrons",
            transform=self.ax_spectrometer.transAxes,
            ha="center",
            fontsize=15,
            color="black",
        )

        self.ax_spectrometer.text(
            0.8,
            1.05,
            "Ions",
            transform=self.ax_spectrometer.transAxes,
            ha="center",
            fontsize=15,
            color="black",
        )

        self.ax_spectrometer.plot(z, U, color="pink", linewidth=2)
        self.ax_spectrometer.set_xlabel("z [m]")
        self.ax_spectrometer.set_ylabel("U [V]")

        total_distance = Ld_e + La_e + La_i + Ld_i
        y_arrow_li = U_i * 1.05
        y_text_li = U_i * 1.1
        y_arrow_le = U_e * 1.05
        y_text_le = U_e * 1.1
        self.ax_spectrometer.set_ylim(U_i * 1.3, U_e * 1.25)
        offset_behind = total_distance / 50
        self.ax_spectrometer.set_xlim(-(La_e + Ld_e + offset_behind), La_i + Ld_i + offset_behind)

        if Ld_i > 0.0:
            self.ax_spectrometer.annotate(
                "",
                xy=((Ld_i + La_i), y_arrow_li),
                xytext=(La_i, y_arrow_li),
                arrowprops=dict(arrowstyle="<->", color="green"),
            )
            self.ax_spectrometer.text(
                La_i + Ld_i * 4 / 5,
                y_text_li,
                rf"$L_d = {Ld_i:.3g}\,\mathrm{{m}}$",
                va="top",
                ha="right",
                fontsize=12,
                color="green",
            )

        self.ax_spectrometer.annotate(
            "",
            xy=(0, y_arrow_li),
            xytext=(La_i, y_arrow_li),
            arrowprops=dict(arrowstyle="<->", color="green"),
        )
        self.ax_spectrometer.text(
            La_i / 2,
            y_text_li,
            rf"$L_a={La_i:.3g}\,\mathrm{{m}} $",
            va="top",
            ha="center",
            fontsize=12,
            color="green",
        )

        self.ax_spectrometer.annotate(
            "",
            xy=(0, y_arrow_le),
            xytext=(-La_e, y_arrow_le),
            arrowprops=dict(arrowstyle="<->", color="green"),
        )
        self.ax_spectrometer.text(
            -La_e / 2,
            y_text_le,
            rf"$L_a={La_e:.3g}\,\mathrm{{m}} $",
            va="bottom",
            ha="center",
            fontsize=12,
            color="green",
        )

        if Ld_e > 0.0:
            self.ax_spectrometer.annotate(
                "",
                xy=(-La_e, y_arrow_le),
                xytext=(-(La_e + Ld_e), y_arrow_le),
                arrowprops=dict(arrowstyle="<->", color="green"),
            )
            self.ax_spectrometer.text(
                -(La_e + Ld_e * 4 / 5),
                y_text_le,
                rf"$L_d = {Ld_e:.3g}\,\mathrm{{m}}$",
                va="bottom",
                ha="left",
                fontsize=12,
                color="green",
            )

        offset_voltage = (U_e - U_i) / 20
        self.ax_spectrometer.annotate(
            rf"{self.electric_field:.4g} V/cm",
            xy=(-La_e / 3, U_e * 1 / 3 + offset_voltage),
            xytext=(La_i / 3, U_i * 1 / 3 + offset_voltage),
            arrowprops=dict(arrowstyle="->", color="green"),
            color="green",
        )

        offset_distance_arrow = total_distance / 50
        offset_distance_text = total_distance / 30
        self.ax_spectrometer.annotate(
            "",
            xy=(-offset_distance_arrow, 0),
            xytext=(-offset_distance_arrow, U_i),
            arrowprops=dict(arrowstyle="<->", color="darkblue", linewidth=2),
        )
        self.ax_spectrometer.text(
            -offset_distance_text,
            U_i / 2,
            rf"$U_i = {U_i:.3g}\,\mathrm{{V}}$",
            ha="right",
            va="center",
            fontsize=12,
            color="darkblue",
        )

        self.ax_spectrometer.annotate(
            "",
            xy=(offset_distance_arrow, 0),
            xytext=(offset_distance_arrow, U_e),
            arrowprops=dict(arrowstyle="<->", color="darkblue", linewidth=2),
        )
        self.ax_spectrometer.text(
            offset_distance_text,
            U_e / 2,
            rf"$U_e = {U_e:.3g}\,\mathrm{{V}}$",
            ha="left",
            va="center",
            fontsize=12,
            color="darkblue",
        )

        self.ax_spectrometer.axvline(Ld_i + La_i, color="black", linewidth=2, zorder=0)
        self.ax_spectrometer.axvline(-(Ld_e + La_e), color="black", linewidth=2, zorder=0)
        self.ax_spectrometer.axvline(0, color="black", linewidth=1, zorder=0)
        self.ax_spectrometer.axhline(0, color="black", linewidth=1, zorder=0)

        self.ax_spectrometer.grid(axis="both", linestyle="--", color="gray")
        self.canvas_spectrometer.draw()

        n_trajectories = 20
        coin = self.electron_tab_coincidences[0]
        colors = np.array(matplotlib.color_sequences["tab20"])
        colors = colors[~np.isin(np.arange(len(colors)), [6, 7])]  # Skip red for color blind
        channel_colors = colors[np.arange(n_trajectories) % len(colors)]
        channel_colors = channel_colors.reshape((-1, 2, 3))
        for in_channel_colors in channel_colors:
            for particle, color in zip(coin.particles.values(), in_channel_colors):
                trajectory = self.get_random_trajectory(particle)
                if trajectory is not None:
                    self.ax_trajectory.plot(
                        trajectory.coords["x"].to(unit="mm").values,
                        trajectory.coords["y"].to(unit="mm").values,
                        trajectory.coords["z"].to(unit="mm").values,
                        color=color,
                    )

        detector_points = np.linspace(0, 2 * np.pi, 100)
        for detector_radius, detector_offset in [
            (
                self.detector_diameter_electrons.get() / 2,
                -(self.length_accel_electron.get() + self.length_drift_electron.get()) * 1e3,
            ),
            (
                self.detector_diameter_ions.get() / 2,
                +(self.length_accel_ion.get() + self.length_drift_ion.get()) * 1e3,
            ),
        ]:
            detector_x = np.sin(detector_points) * detector_radius
            detector_y = np.cos(detector_points) * detector_radius
            detector_z = np.ones_like(detector_x) * detector_offset
            self.ax_trajectory.plot(detector_x, detector_y, detector_z)

        self.ax_trajectory.set_xlabel("x [mm]")
        self.ax_trajectory.set_ylabel("y [mm]")
        self.ax_trajectory.set_zlabel("z [mm]")
        ion_tex, ele_tex = [part.latex for part in coin.particles.values()]
        self.ax_trajectory.set_title(rf"Some Coincidences with ${ele_tex}$ and ${ion_tex}$")
        self.canvas_trajectory.draw()

    def get_random_trajectory(self, particle: Particle):
        rng = np.random.default_rng()
        hits = particle.detector_hits
        slicers = zip(hits.dims, rng.integers(hits.shape))
        some_hit = hits
        for dim, index in slicers:
            some_hit = some_hit[dim, index]

        tof_value = some_hit.coords["tof"]
        tof_accel_value = some_hit.coords["tof_accel"]
        if sc.isnan(tof_value):
            return None
        n_steps = 200
        tof_range = sc.linspace("tof", start=tof_value / n_steps, stop=tof_value, num=n_steps)
        tof_accel_range = sc.where(tof_range > tof_accel_value, tof_accel_value, tof_range)
        tof_drift_range = sc.where(
            tof_range > tof_accel_value, tof_range - tof_accel_value, sc.zeros_like(tof_range)
        )
        starting_momentum = sc.broadcast(some_hit.coords["p"], dims=["tof"], shape=tof_range.shape)
        trajectory = sc.DataArray(
            data=sc.ones_like(tof_range),
            coords={
                "p": starting_momentum,
                "tof": tof_range,
                "tof_accel": tof_accel_range,
                "tof_drift": tof_drift_range,
            },
        )
        trajectory = trajectory.transform_coords(
            ["x", "y", "z"], graph=particle.detector_transformation_graph
        )
        return trajectory

    def make_export_tab(self):
        self.tabs["Export Data"].columnconfigure(0, weight=1)
        self.tabs["Export Data"].rowconfigure(0, weight=1)
        self.valid_group = LabelFrame(self.tabs["Export Data"], text="Save Electron Data")
        self.valid_group.grid(
            row=0, column=0, columnspan=1, rowspan=1, padx="5", pady="5", sticky="nsew"
        )

        self.BUTTON_SAVE_MOM = Button(
            self.valid_group, text="Save Momentum Data", command=self.export_momenta
        )
        self.BUTTON_SAVE_MOM.grid(row=0, column=0, columnspan=1, padx="5", pady="5", sticky="w")

        self.BUTTON_CALC_MCP_TIMES = Button(
            self.valid_group, text="Save MCP times", command=self.calc_mcp
        )
        self.BUTTON_CALC_MCP_TIMES.grid(
            row=1, column=0, columnspan=1, padx="5", pady="5", sticky="w"
        )

        self.BUTTON_EXPORT_DATA = Button(
            self.valid_group, text="Save Electron Position", command=self.export_data
        )
        self.BUTTON_EXPORT_DATA.grid(row=2, column=0, columnspan=1, padx="5", pady="5", sticky="w")

        self.BUTTON_EXPORT_HDF5 = Button(
            self.valid_group, text="Save Scipp DataGroup to hdf5", command=self.export_scipp_hdf5
        )
        self.BUTTON_EXPORT_HDF5.grid(row=3, column=0, columnspan=1, padx="5", pady="5", sticky="w")

    def make_plot(
        self,
        row,
        column,
        master,
        rowspan=2,
        columnspan=1,
        figsize=(5, 5),
        withcax=False,
        subplot_kw=None,
    ):
        """
        Initializes a figure canvas with plot axis

        Returns
        -------
        fig : Figure
        a : axis
        canvas : canvas
        """
        assert rowspan >= 2
        fig = plt.figure(figsize=figsize, facecolor=canvas_background_color, layout="constrained")
        canvas = FigureCanvasTkAgg(fig, master=master)
        canvas.get_tk_widget().grid(
            row=row,
            column=column,
            rowspan=rowspan - 1,
            columnspan=columnspan,
            padx="5",
            pady="5",
            sticky="nsew",
        )
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas=canvas, window=master, pack_toolbar=False)
        toolbar.grid(
            row=row + rowspan - 1,
            column=column,
            rowspan=1,
            columnspan=columnspan,
            padx="5",
            pady="5",
            sticky="nsew",
        )
        toolbar.update()

        if withcax:
            axes = fig.subplots(1, 2, width_ratios=[0.93, 0.07], subplot_kw=subplot_kw)
        else:
            axes = fig.subplots(1, 1, subplot_kw=subplot_kw)
        return fig, axes, canvas, toolbar

    @property
    def magnetic_field_si(self):
        return self.magnetic_field_gauss.get() * 1e-4

    @property
    def electric_field_si(self):
        return self.electric_field / 100

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
            0, 0, self.R_tof_plot_group, figsize=(5, 5), columnspan=2, rowspan=5
        )
        self.ele_pos_fig, self.ele_pos_a, self.ele_pos_canvas, self.ele_pos_toolbar = (
            self.make_plot(0, 2, self.R_tof_plot_group, figsize=(5, 5), columnspan=2, rowspan=5)
        )
        self.ele_pos_a.set_aspect("equal")
        self.electron_special_lines = []

    def update_R_tof(self):
        """
        Updates the R vs tof and the position plot, while moving the sliders for B and U
        """
        ax = self.ax_R_tof
        self.electron_special_lines = []
        ax.cla()

        detector_radius = sc.scalar(self.detector_diameter_electrons.get() / 2, unit="mm")

        max_tof = self.calc_max_tof().to(unit="ns")
        tof_limit = max(max_tof * 1.2, 1e-9 * sc.Unit("ns"))
        R_limit = detector_radius * 1.2
        pos_bins = 100
        tof_bins = 200
        self.x_bins_electrons = sc.linspace("x", -R_limit, R_limit, pos_bins)
        self.y_bins_electrons = sc.linspace("y", -R_limit, R_limit, pos_bins)
        self.tof_bins_electrons = sc.linspace("tof", sc.scalar(0.0, unit="ns"), tof_limit, tof_bins)
        self.R_bins_electrons = sc.linspace("R", sc.scalar(0.0, unit="mm"), R_limit, pos_bins)

        R_tof_hists = [
            electron.detector_hits.hist(
                R=self.R_bins_electrons, tof=self.tof_bins_electrons, dim=("p", "pulses")
            )
            for electron in self.electrons
        ]
        R_tof_hist = sum(R_tof_hists)

        R_tof_hist.plot(ax=ax, cbar=False, norm="log", cmap="PuBuGn")

        if self.v_ir.get() == 1:
            self.R_tof_sim_ir()

        ax.axvline(max_tof.value, color="darkgrey")
        ax.axhline(detector_radius.value, lw=1)

        no_mom_tof = self.calc_no_momentum_tof(Electron(self.remicalculator)).to(unit="ns")

        omega = self.remicalculator.calc_omega(*self.electron_params)
        if omega.to(unit="Hz") == sc.scalar(0, unit="Hz"):
            cyclotron_period = sc.scalar(np.inf, unit="ns")
        else:
            cyclotron_period = sc.abs(2 * np.pi / omega).to(unit="ns")

        ax.axvline(no_mom_tof.value, ls="--", color="darkgrey")
        if sc.isfinite(cyclotron_period) and max_tof > cyclotron_period:
            for node_tof in np.arange(0.0, max_tof.value, cyclotron_period.value):
                ax.axvline(node_tof, ls=":", color="darkgrey")

        self.canvas_R_tof.draw()

        ax = self.ele_pos_a
        ax.cla()

        x_y_hists = [
            electron.detector_hits.hist(
                y=self.y_bins_electrons, x=self.x_bins_electrons, dim=("p", "pulses")
            )
            for electron in self.electrons
        ]
        x_y_hist = sum(x_y_hists)
        x_y_hist.plot(ax=ax, cbar=False, norm="log", cmap="PuBuGn")

        detector = plt.Circle(
            (0, 0), detector_radius.value, color="cadetblue", fill=False, figure=self.ele_pos_fig
        )
        ax.add_artist(detector)

        ax.set_xlim(-70, 70)
        ax.set_ylim(-70, 70)

        self.ele_pos_canvas.draw()

    def R_tof_sim(self):
        for line in self.electron_special_lines:
            line.remove()
        self.electron_special_lines = []

        energies = []
        colors = []

        for entry in [self.ENTRY_KIN_ENERGY_1, self.ENTRY_KIN_ENERGY_2, self.ENTRY_KIN_ENERGY_3]:
            if len(entry.get()) != 0:
                energies.append(float(entry.get()))
                colors.append(str(entry["foreground"]))
        energies = sc.array(dims=["p"], values=energies, unit="eV")

        for e, c in zip(energies, colors):
            p_max = sc.sqrt(2 * e * sc.constants.m_e).to(unit="N*s")
            p_z = sc.linspace("p", -p_max, p_max, 100)
            p_x = sc.sqrt(p_max**2 - p_z**2)
            p_y = sc.zeros_like(p_z)
            momentum = sc.spatial.as_vectors(p_x, p_y, p_z)
            da = sc.DataArray(sc.ones_like(p_z), coords={"p": momentum})
            transformed = da.transform_coords(
                ["tof", "x", "y", "R"], graph=self.electron_scipp_graph
            )
            new_lines = self.ax_R_tof.plot(
                transformed.coords["tof"], transformed.coords["R"], color=c
            )
            new_lines += self.ele_pos_a.plot(
                transformed.coords["x"], transformed.coords["y"], color=c
            )
            self.electron_special_lines += new_lines
        self.canvas_R_tof.draw()
        self.ele_pos_canvas.draw()

    def calc_max_tof(self):
        """
        calculates the maximal tof for the electron to not fly into the ion detector
        """
        remi = self.remicalculator
        test_electron = Electron(remi)
        opposite_voltage = remi.electric_field * remi.length_acceleration_ion
        energy = -opposite_voltage * test_electron.charge
        absolute_momentum = sc.sqrt(2 * energy * test_electron.mass).to(unit="au momentum")
        momentum_vector = remi.field_unitvector * absolute_momentum
        test_electron.momentum_sample = sc.DataArray(
            sc.ones(dims=["p"], shape=(1,)), coords={"p": momentum_vector}
        )
        test_electron.calculate_detector_hits()
        tof_max = test_electron.detector_hits.coords["tof"]
        return tof_max

    def calc_no_momentum_tof(self, test_particle: Particle):
        """
        calculates the time of flight for a paticle with no z-momentum
        """
        momentum_vector = sc.vector([0.0, 0.0, 0.0], unit="au momentum")
        test_particle.momentum_sample = sc.DataArray(
            sc.ones(dims=["p"], shape=(1,)), coords={"p": momentum_vector}
        )
        test_particle.calculate_detector_hits()
        zero_momentum_tof = test_particle.detector_hits.coords["tof"]
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

    def export_scipp_hdf5(self):
        """
        Write dataset to hdf5
        """
        filename = filedialog.asksaveasfilename(
            initialfile="examples/redcamel_data.h5", filetypes=[("hdf5", ".h5")]
        )
        if isinstance(filename, str):
            self.datagroup.save_hdf5(filename)

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

    def R_tof_sim_ir(self):
        start_energy = float(self.ENTRY_KIN_ENERGY_START.get())
        step_energy = float(self.ENTRY_KIN_ENERGY_STEP.get())
        number_sim = int(self.ENTRY_NUMBER_OF_PART.get())
        color = str(self.ENTRY_KIN_ENERGY_START["foreground"])
        energies = sc.linspace(
            "p",
            start_energy,
            ((number_sim - 1) * step_energy) + start_energy,
            number_sim,
            unit="eV",
        )

        for line in self.electron_special_lines:
            line.remove()
        self.electron_special_lines = []
        for i, e in enumerate(energies):
            p_max = sc.sqrt(2 * e * sc.constants.m_e).to(unit="N*s")
            p_z = sc.linspace("p", -p_max, p_max, 100)
            if number_sim == 2:
                p_x = sc.sqrt(p_max**2 - p_z**2) * i
                p_y = sc.sqrt(p_max**2 - p_z**2) * (1 - i)
            else:
                p_x = sc.sqrt(p_max**2 - p_z**2)
                p_y = sc.zeros_like(p_z)
            momentum = sc.spatial.as_vectors(p_x, p_y, p_z)
            da = sc.DataArray(sc.ones_like(p_z), coords={"p": momentum})
            transformed = da.transform_coords(
                ["tof", "x", "y", "R"], graph=self.electron_scipp_graph
            )
            new_lines = self.ax_R_tof.plot(
                transformed.coords["tof"], transformed.coords["R"], color=color
            )
            new_lines += self.ele_pos_a.plot(
                transformed.coords["x"], transformed.coords["y"], color=color
            )
            self.electron_special_lines += new_lines
        self.canvas_R_tof.draw()
        self.ele_pos_canvas.draw()

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
        self.LABEL_KER.config(text=f"{ker:.2f} eV")

        return ker

    def generate_ion_pair_entries(self):
        ions_per_channel = 2
        previous_active_channels = self.active_channels
        self.active_channels = int(self.ENTRY_NUMBER_IONS.get())
        active_ions = self.active_channels * ions_per_channel
        previous_max_channel_number = len(self.entries_ker)

        colors = np.array(matplotlib.color_sequences["tab20"])
        colors = colors[~np.isin(np.arange(len(colors)), [6, 7])]  # Skip red for color blind
        channel_colors = colors[np.arange(active_ions) % len(colors)]
        self.channel_colors = channel_colors.reshape((-1, 2, 3))

        predefined_ions = defaultdict(lambda: [(1, ChemFormula("H")), (1, ChemFormula("H"))])
        predefined_ions[0] = [(1, ChemFormula("H")), (2, ChemFormula("OH"))]
        predefined_ions[1] = [(1, ChemFormula("N")), (3, ChemFormula("N"))]
        predefined_ions[2] = [(1, ChemFormula("Ar")), (1, ChemFormula("Ar"))]
        predefined_ions[3] = [(2, ChemFormula("S")), (4, ChemFormula("CO"))]
        predefined_ions[4] = [(1, ChemFormula("H")), (1, ChemFormula("CH3"))]
        predefined_ions[5] = [(1, ChemFormula("H")), (2, ChemFormula("CH2I"))]

        predefined_kers = defaultdict(lambda: 3.5)
        for i, ker in enumerate([6.0, 6.0, 10.0, 20.0, 4.0]):
            predefined_kers[i] = ker

        def write_callback_ion_channels(var, index, mode):
            self.generate_ion_channels()

        def write_callback_ion_plot(var, index, mode):
            self.make_ion_pipico_plot()

        for i in range(previous_max_channel_number, self.active_channels):
            channel_color = matplotlib.colors.to_hex(
                matplotlib.colors.to_hex(self.channel_colors[i].mean(axis=0))
            )
            channel_row_index = (i * 2) + 3

            self.ion_labels.append([])
            self.formula_variables.append([])
            self.entries_formula.append([])
            self.labels_mass.append([])
            self.charge_variables.append([])
            self.entries_charge.append([])
            self.labels_ion_tof.append([])

            self.ker_variables.append(DoubleVar(value=predefined_kers[i]))
            self.ker_variables[i].trace("w", write_callback_ion_channels)
            self.entries_ker.append(
                Entry(
                    self.ion_generation_group,
                    foreground=channel_color,
                    textvariable=self.ker_variables[i],
                )
            )
            self.entries_ker[i].grid(row=channel_row_index, column=5, rowspan=2, sticky="nsew")

            variable = BooleanVar(value=True)
            variable.trace("w", write_callback_ion_plot)
            checkbox = Checkbutton(self.ion_generation_group, variable=variable, text="")
            checkbox.grid(row=channel_row_index, column=6, rowspan=2)
            self.active_check_variables.append((variable, checkbox))

            for j, (charge, formula) in enumerate(predefined_ions[i]):
                ion_index = ions_per_channel * i + j
                ion_row_index = ion_index + 3
                self.ion_labels[i].append(
                    Label(self.ion_generation_group, text="Ion " + str(ion_index + 1))
                )
                self.ion_labels[i][j].grid(row=ion_row_index, column=0, sticky="nsew")

                this_ion_color = matplotlib.colors.to_hex(self.channel_colors[i, j])
                self.formula_variables[i].append(StringVar(value=formula))
                self.formula_variables[i][j].trace("w", write_callback_ion_channels)
                self.entries_formula[i].append(
                    Entry(
                        self.ion_generation_group,
                        foreground=this_ion_color,
                        textvariable=self.formula_variables[i][j],
                    )
                )
                self.entries_formula[i][j].grid(row=ion_row_index, column=1, sticky="nsew")

                self.labels_mass[i].append(
                    Label(self.ion_generation_group, foreground=this_ion_color)
                )
                self.labels_mass[i][j].grid(row=ion_row_index, column=2)

                self.charge_variables[i].append(IntVar(value=charge))
                self.charge_variables[i][j].trace("w", write_callback_ion_channels)
                self.entries_charge[i].append(
                    Entry(
                        self.ion_generation_group,
                        foreground=this_ion_color,
                        textvariable=self.charge_variables[i][j],
                    )
                )
                self.entries_charge[i][j].grid(row=ion_row_index, column=3, sticky="nsew")

                self.labels_ion_tof[i].append(
                    Label(self.ion_generation_group, foreground=this_ion_color)
                )
                self.labels_ion_tof[i][j].grid(row=ion_row_index, column=4)

        # show activated channels
        for i in range(previous_active_channels, self.active_channels):
            var, checkbox = self.active_check_variables[i]
            checkbox.grid()
            self.entries_ker[i].grid()
            for j in range(ions_per_channel):
                self.entries_formula[i][j].grid()
                self.labels_mass[i][j].grid()
                self.entries_charge[i][j].grid()
                self.ion_labels[i][j].grid()
                self.labels_ion_tof[i][j].grid()
        # hide deactivated channels
        for i in range(self.active_channels, previous_active_channels):
            var, checkbox = self.active_check_variables[i]
            checkbox.grid_remove()
            self.entries_ker[i].grid_remove()
            for j in range(ions_per_channel):
                self.entries_formula[i][j].grid_remove()
                self.labels_mass[i][j].grid_remove()
                self.entries_charge[i][j].grid_remove()
                self.ion_labels[i][j].grid_remove()
                self.labels_ion_tof[i][j].grid_remove()
        self.update_ion_momenta()

    def make_ion_pipico_plot(self):
        plotted_channels = []
        for i in range(self.active_channels):
            variable, checkbox = self.active_check_variables[i]
            if variable.get():
                plotted_channels.append(self.ion_channels[i])

        # cleanup plot
        for ax in self.pipico_fig.axes:
            ax.cla()
        for legend_object in self.pipico_fig.legends:
            legend_object.remove()

        # do new plots
        ax_x_tof = self.pipico_xtof_ax
        ax_y_tof = self.pipico_ytof_ax
        modulo = self.bunch_modulo.get() * sc.Unit("ns")
        detector_radius = self.detector_diameter_ions.get() / 2
        space_lim = 1.2 * detector_radius
        ax_x_tof.set_ylim(-space_lim, space_lim)
        ax_y_tof.set_ylim(-space_lim, space_lim)
        x_edges = y_edges = np.linspace(-space_lim, space_lim, 250)

        counts, _, _ = np.histogram2d([], [], bins=(x_edges, y_edges))
        legend_handles_even = []
        legend_labels_even = []
        legend_handles_odd = []
        legend_labels_odd = []
        for channel in plotted_channels:
            ion_1, ion_2 = channel.particles.values()

            dots = ax_x_tof.scatter(
                (ion_1.detector_hits.coords["tof"] % modulo).values,
                (ion_1.detector_hits.coords["x"]).values,
                color=ion_1.color,
                alpha=0.2,
                edgecolors="none",
            )
            legend_handles_even.append(dots)
            legend_labels_even.append(f"${ion_1.latex}$")
            dots = ax_x_tof.scatter(
                (ion_2.detector_hits.coords["tof"] % modulo).values,
                (ion_2.detector_hits.coords["x"]).values,
                color=ion_2.color,
                alpha=0.2,
                edgecolors="none",
            )
            legend_handles_odd.append(dots)
            legend_labels_odd.append(f"${ion_2.latex}$")
            ax_y_tof.scatter(
                (ion_1.detector_hits.coords["tof"] % modulo).values,
                (ion_1.detector_hits.coords["y"]).values,
                color=ion_1.color,
                alpha=0.2,
                edgecolors="none",
            )
            ax_y_tof.scatter(
                (ion_2.detector_hits.coords["tof"] % modulo).values,
                (ion_2.detector_hits.coords["y"]).values,
                color=ion_2.color,
                alpha=0.2,
                edgecolors="none",
            )

            self.pipico_ax.scatter(
                (ion_1.detector_hits.coords["tof"] % modulo).values,
                (ion_2.detector_hits.coords["tof"] % modulo).values,
                color=ion_1.color,
                alpha=0.1,
                edgecolors="none",
            )
            self.pipico_ax.scatter(
                (ion_2.detector_hits.coords["tof"] % modulo).values,
                (ion_1.detector_hits.coords["tof"] % modulo).values,
                color=ion_2.color,
                alpha=0.1,
                edgecolors="none",
            )

            for h in [ion_1.detector_hits, ion_2.detector_hits]:
                new_counts, _, _ = np.histogram2d(
                    h.coords["x"].values.flatten(),
                    h.coords["y"].values.flatten(),
                    bins=(x_edges, y_edges),
                )
                counts += new_counts
        ax_xy = self.pipico_XY_ax
        counts[counts < 1] = np.nan
        ax_xy.pcolormesh(x_edges, y_edges, counts.T)
        ax_xy.add_artist(
            plt.Circle(
                (0, 0), detector_radius, color="cadetblue", fill=False, figure=self.pipico_fig
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

        v_jet = self.velocity_jet.get() * sc.Unit("mm/µs")
        for i in range(5):
            jet_tof = sc.linspace("tof", modulo * i, modulo * (i + 1), 2)
            jet_offset = (jet_tof * v_jet).to(unit="mm")
            label = "Jet" if i == 0 else None
            ax_x_tof.plot([0, modulo.value], jet_offset.values, label=label, color="k", alpha=0.3)
        for ax in [ax_x_tof, ax_y_tof]:
            ax.axhline(detector_radius, color="red")
            ax.axhline(-detector_radius, color="red")
            ax.set_xlim(0, modulo.value)
            ax.grid()

        self.pipico_ax.grid()
        self.pipico_ax.set_xlabel("tof 1 [ns]")
        self.pipico_ax.set_ylabel("tof 2 [ns]")
        self.pipico_ax.set_xlim(0, modulo.value)
        self.pipico_ax.set_ylim(0, modulo.value)

        legend_handles = legend_handles_even + legend_handles_odd
        legend_labels = legend_labels_even + legend_labels_odd
        legend = self.pipico_fig.legend(legend_handles, legend_labels, loc=4, ncols=2)
        for artist in legend.legend_handles:
            artist.set_alpha(1)
        self.pipico_canvas.draw()

        make_icon = False
        if make_icon:
            icon_fig, icon_ax = plt.subplots(figsize=(5, 5), layout="tight")
            for channel in plotted_channels:
                ion_1, ion_2 = channel.particles
                ion_1.hits = self.ion_hits[channel.name][ion_1.name]
                ion_2.hits = self.ion_hits[channel.name][ion_2.name]
                icon_ax.scatter(
                    ion_1.hits.coords["tof"] % modulo,
                    ion_2.hits.coords["tof"] % modulo,
                    color=ion_1.color,
                    alpha=0.1,
                    edgecolors="none",
                )
                icon_ax.scatter(
                    ion_2.hits.coords["tof"] % modulo,
                    ion_1.hits.coords["tof"] % modulo,
                    color=ion_2.color,
                    alpha=0.1,
                    edgecolors="none",
                )
            icon_ax.axis("off")
            icon_fig.savefig("icon.png", dpi=12.8)
            plt.close(icon_fig)

    def update_electron_momenta(self):
        mass, charge = self.electron_params
        self.electron_tab_coincidences = []
        default_atom = ChemFormula("He")
        he_binding_energy = sc.scalar(24.587389011, unit="eV")

        if self.v.get() == 1:
            self.electron_tab_coincidences.append(
                sample_photoionization(
                    default_atom,
                    binding_energy=he_binding_energy,
                    photon_energy=he_binding_energy,
                    energy_width=sc.scalar(1, unit="au energy").to(unit="eV"),
                    sizes=self.sample_sizes,
                    remi=self.remicalculator,
                )
            )
        elif self.v.get() == 2:
            energy_mean = sc.scalar(float(self.ENTRY_MEAN_ENERGY.get()), unit="eV")
            width = sc.scalar(float(self.ENTRY_WIDTH.get()), unit="eV")
            self.electron_tab_coincidences.append(
                sample_photoionization(
                    default_atom,
                    binding_energy=he_binding_energy,
                    photon_energy=he_binding_energy + energy_mean,
                    energy_width=width,
                    sizes=self.sample_sizes,
                    remi=self.remicalculator,
                )
            )
        elif self.v.get() == 3:
            energy_mean = sc.scalar(float(self.ENTRY_MEAN_ENERGY.get()), unit="eV")
            width = sc.scalar(float(self.ENTRY_WIDTH.get()), unit="eV")
            energy_step = sc.scalar(float(self.ENTRY_MULTI_PART_ENERGY_STEP.get()), unit="eV")
            part_num = int(self.ENTRY_MULTI_PART_NUMBER.get())
            for i in range(part_num):
                self.electron_tab_coincidences.append(
                    sample_photoionization(
                        default_atom,
                        binding_energy=he_binding_energy,
                        photon_energy=he_binding_energy + energy_mean + i * energy_step,
                        energy_width=width,
                        sizes=self.sample_sizes,
                        remi=self.remicalculator,
                    )
                )
        self.electrons = [coin.electrons["e"] for coin in self.electron_tab_coincidences]
        self.update_electron_positions()

    def update_electron_positions(self):
        mass, charge = self.electron_params
        for coin in self.electron_tab_coincidences:
            coin.calculate_detector_hits()
        self.update_R_tof()
        self.update_electron_detector_signals()

    def update_electron_detector_signals(self):
        groupgroup = self.datagroup["electronsim"] = sc.DataGroup()
        for coin in self.electron_tab_coincidences:
            groupgroup[coin.name] = coin.datagroup

    def update_ion_momenta(self):
        self.ion_channels = []
        for i in range(self.active_channels):
            formulas = []
            for f in self.formula_variables[i]:
                formula_string = f.get()
                try:
                    formula_float = float(formula_string)
                    formulas.append(formula_float)
                except ValueError:
                    formulas.append(ChemFormula(formula_string))

            def get_formula_name(formula):
                if isinstance(formula, ChemFormula):
                    return formula.formula
                return f"M{formula:.2f}"

            name = " + ".join(get_formula_name(f) for f in formulas)
            ker = sc.scalar(float(self.entries_ker[i].get()), unit="eV")
            coin = sample_coulomb_explosion(
                fragment_formulas=formulas,
                charge_counts=[c.get() for c in self.charge_variables[i]],
                kinetic_energy_release=ker,
                energy_width=ker / 10,
                sizes=self.sample_sizes,
                remi=self.remicalculator,
                name=name,
                colors=self.channel_colors[i],
            )
            self.ion_channels.append(coin)
            for j, particle in enumerate(coin.particles.values()):
                self.labels_mass[i][j]["text"] = f"{particle.mass.value:.4g}"
        self.update_ion_positions()

    def update_ion_positions(self):
        for coin in self.ion_channels:
            coin.calculate_detector_hits()

        self.calc_ion_tof()
        self.make_ion_pipico_plot()
        self.update_ion_detector_signals()

    def calc_ion_tof(self):
        for i, coin in enumerate(self.ion_channels):
            for j, particle in enumerate(coin.particles.values()):
                test_particle = Particle(particle.formula, particle.charge_count, particle.remi)
                no_mom_tof = self.calc_no_momentum_tof(test_particle)
                self.labels_ion_tof[i][j]["text"] = f"{no_mom_tof.value:.4g}"

    def update_ion_detector_signals(self):
        coins = self.ion_channels
        groupgroup = self.datagroup["coulombexplosions"] = sc.DataGroup()
        for coin in coins:
            groupgroup[coin.name] = coin.datagroup

    def init_dataset(self):
        n_samples = self.number_of_particles.get()
        self.pulses = sc.arange("pulses", n_samples)
        self.sample_sizes = {"pulses": n_samples, "p": 1}
        self.datagroup = sc.DataGroup(pulses=self.pulses)
        self.update_electron_momenta()
        self.generate_ion_pair_entries()


def main():
    window = Tk()
    window.configure(background=canvas_background_color)
    mclass(window)
    window.mainloop()


if __name__ == "__main__":
    main()
