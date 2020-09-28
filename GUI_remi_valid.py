#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 10:58:39 2020

@author: patrizia
"""

#############################
#### imports ################
############################

import matplotlib
#matplotlib.use('TkAgg')
import numpy as np

try:
    import ttk
    from Tkinter import Tk, Button, Entry, Label, Listbox, END, LabelFrame, Radiobutton, IntVar, Scale, HORIZONTAL, Checkbutton
except:
    from tkinter import Tk, Button, Entry, Label, Listbox, END, LabelFrame, ttk , Radiobutton, IntVar, Scale, HORIZONTAL, Checkbutton

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import xarray as xr



#############################
#### constants ##############
############################

# SI
m_e = 9.10938356e-31          # electron_mass
q_e = 1.6021766208e-19           # electron_charge


#############################
#### functions ##############
############################

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
    
    momentum = np.random.randn(number_of_particles, 1, 3)*2e-24
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
    
    r = (np.random.randn(number_of_particles, 1) * width + energy_mean)*q_e
    phi = np.random.rand(number_of_particles, 1)*2*np.pi
    cos_theta = np.random.rand(number_of_particles, 1)*2-1
    
    r_mom = np.sqrt(r*2*mass)
    
    theta = np.arccos(cos_theta)
    
    x= r_mom * np.sin(theta) * np.cos(phi)
    y= r_mom * np.sin(theta) * np.sin(phi)
    z= r_mom * cos_theta
    
    energy = np.stack([x, y, z], axis=-1)
    
    return energy

def calc_tof(momentum, remi_params, particle_params=(m_e, q_e)):
    """
    Parameters
    ----------
    momentum : ndarray
        momentum for tof calculation
    remi_params : array
        configuration of REMI with values for U, B and l_a
        
    Returns
    -------
    tof : array
        Time of flight for each particle
    """
    U, B, l_a = remi_params
    m, q = particle_params
    p_z = momentum[:,0,2]
    D = p_z**2 + 2 * q * U * m
    tof = ((p_z) - np.sqrt(D))/(-q*U)*l_a
    return tof

def calc_R(momentum, remi_params, particle_params=(m_e, q_e)):
    """
    Parameters
    ----------
    momentum : ndarray
        momentum for radius calculation
    remi_params : array
        configuration of REMI with values for U, B and l_a
        
    Returns
    -------
    R : array
        Distance from reaction point to detection point in xy for each particle
    """
    U, B, l_a = remi_params
    m, q = particle_params
    p_x = momentum[:,0,0]
    p_y = momentum[:,0,1]
    p_xy = np.sqrt(p_x**2 + p_y**2)
    tof = calc_tof(momentum, remi_params, particle_params)
    
    R = (2*p_xy*np.abs(np.sin(calc_omega(B, q, m)*tof/2)))/(q*B)
    return R

def make_R_tof_array(momentum, remi_params, particle_params=(m_e, q_e)):
    tof = calc_tof(momentum, remi_params, particle_params)
    R = calc_R(momentum, remi_params, particle_params)
    ar = xr.DataArray(R, coords=[tof], dims=["time"])
    return ar

def calc_omega(B, q=q_e, m=m_e):
    return q * B / m

def calc_R_fit(K, tof, remi_params, particle_params):
    U, B, l_a = remi_params
    m, q = particle_params
    D = K**2 - (m*l_a/tof - U*q*tof/(2*l_a))**2
    R = 2/(m*calc_omega(B, q, m)) * np.sqrt(D) * np.abs(np.sin(calc_omega(B, q, m)*tof/2))
    return R

#############################
#### GUI ####################
############################
    
class mclass:

    def __init__(self,  window):
        self.window = window
        window.title('REMI Analysis Validation')
        style = ttk.Style()
        style.configure('BW.TLabel', background="whitesmoke")
        tabControl = ttk.Notebook(window)
        tab1 = ttk.Frame(tabControl, width=300, height=300, style='BW.TLabel')
        tab2 = ttk.Frame(tabControl, width=300, height=300, style='BW.TLabel')
        tab3 = ttk.Frame(tabControl, width=300, height=300, style='BW.TLabel')
        tabControl.add(tab1, text='R vs TOF')
        tabControl.add(tab2, text='?')
        tabControl.add(tab3, text='Coincidences')
        tabControl.grid(column=0)

        button_color = 'aliceblue'
        frame_color = 'mintcream'


        self.l_a = 0.188         # acc_length 
        self.U = 190            # electic_field 
        self.B = 5*1e-4           # magnetic_field 
        self.omega = q_e * self.B / m_e
        
    ######## higher groups ####################
        left_bar_group = LabelFrame(tab1, text="", padx=5, pady=5, bd=3, background=frame_color)
        left_bar_group.grid(row=100, column=100, columnspan=2, rowspan=20, padx='5', pady='5', sticky='new')
        
        top_bar_group = LabelFrame(tab1, text="", padx=5, pady=5, bd=3, background=frame_color)
        top_bar_group.grid(row=100, column=103, columnspan=20, rowspan=2, padx='5', pady='5', sticky='new')
        
    ######## REMI configurations ##############
        remi_conf_group = LabelFrame(left_bar_group, text="REMI Configuration for Electrons", padx=5, pady=5, bd=3, background=frame_color)
        remi_conf_group.grid(row=100, column=100, columnspan=2, padx='5', pady='5', sticky='new')
        
        self.LABEL_SET_U = Label(remi_conf_group, text='U[V]:', background=frame_color)
        self.LABEL_SET_B = Label(remi_conf_group, text='B[Gauss]:', background=frame_color)
        self.LABEL_SET_l_a = Label(remi_conf_group, text='acc length[m]:', background=frame_color)
        
        self.LABEL_SET_U.grid(row=103, column=101, padx='5', pady='5', sticky='w')
        self.LABEL_SET_B.grid(row=104, column=101, padx='5', pady='5', sticky='w')
        self.LABEL_SET_l_a.grid(row=105, column=101, padx='5', pady='5', sticky='w')
        
        self.ENTRY_SET_U = Entry(remi_conf_group)
        self.ENTRY_SET_B = Entry(remi_conf_group)
        self.ENTRY_SET_l_a = Entry(remi_conf_group)
        
        self.ENTRY_SET_U.grid(row=103, column=102, padx='5', pady='5', sticky='w')
        self.ENTRY_SET_B.grid(row=104, column=102, padx='5', pady='5', sticky='w')
        self.ENTRY_SET_l_a.grid(row=105, column=102, padx='5', pady='5', sticky='w')
        
        self.ENTRY_SET_U.insert(0, 190)
        self.ENTRY_SET_B.insert(0, 5)
        self.ENTRY_SET_l_a.insert(0, 0.188)
        
        self.BUTTON_CHANGE_REMI_CONF = Button(remi_conf_group, text="Change configuration", command=self.change_remi_conf, activebackground = button_color)
        self.BUTTON_CHANGE_REMI_CONF.grid(row=106, column=101, columnspan=2, padx='5', pady='5', sticky='w')
        
    ######## momentum, R, tof calculation #############
        self.R_tof_group = LabelFrame(left_bar_group, text="R-tof calculation", padx=5, pady=5, bd=3, background=frame_color)
        self.R_tof_group.grid(row=102, column=100, columnspan=2, rowspan=5, padx='5', pady='5', sticky='nwe')
        
        self.v = IntVar()
        self.v.set(2)
        self.CHOOSE_MOMENTUM = Radiobutton(self.R_tof_group, command=self.check, text="Momentum", variable=self.v, value=1, background=frame_color)
        self.CHOOSE_ENERGY = Radiobutton(self.R_tof_group, command=self.check, text="Energy", variable=self.v, value=2, background=frame_color)
        self.CHOOSE_ENERGY.select()
        self.CHOOSE_MOMENTUM.grid(row=99, column=110, padx='5', pady='5', sticky='w')
        self.CHOOSE_ENERGY.grid(row=100, column=110, padx='5', pady='5', sticky='w')
        self.CHOOSE_ENERGY_MULTI = Radiobutton(self.R_tof_group, command=self.check, text="Multiple Prticle", variable=self.v, value=3, background=frame_color)
        self.CHOOSE_ENERGY_MULTI.grid(row=101, column=110, padx='5', pady='5', sticky='w')
        
        self.LABEL_NUMBER_PART = Label(self.R_tof_group, text='number of Particles:', background=frame_color)
        self.LABEL_PART_MASS = Label(self.R_tof_group, text='Particle mass:', background=frame_color)
        self.LABEL_PART_CHARGE = Label(self.R_tof_group, text='Particle charge:', background=frame_color)
        self.ENTRY_NUMBER_PART = Entry(self.R_tof_group)
        self.ENTRY_PART_MASS = Entry(self.R_tof_group)
        self.ENTRY_PART_CHARGE = Entry(self.R_tof_group)
        self.BUTTON_R_TOF = Button(self.R_tof_group, text="Calculate radius and tof", command=self.make_R_tof, activebackground = button_color)
        
        
        #if selecting calculation with energy
        self.LABEL_MEAN_ENERGY = Label(self.R_tof_group, text='Mean Energy:', background=frame_color)
        self.LABEL_WIDTH = Label(self.R_tof_group, text='Width:', background=frame_color)
        self.ENTRY_MEAN_ENERGY = Entry(self.R_tof_group)
        self.ENTRY_WIDTH = Entry(self.R_tof_group)
        
        self.LABEL_MEAN_ENERGY.grid(row=105, column=110, padx='5', pady='5', sticky='w')
        self.LABEL_WIDTH.grid(row=106, column=110, padx='5', pady='5', sticky='w')
        self.ENTRY_MEAN_ENERGY.grid(row=105, column=111, padx='5', pady='5', sticky='w')
        self.ENTRY_WIDTH.grid(row=106, column=111, padx='5', pady='5', sticky='w')
        self.ENTRY_MEAN_ENERGY.insert(0,1)
        self.ENTRY_WIDTH.insert(0,0.1)
        
        self.ENTRY_PART_MASS.insert(0,1)
        self.ENTRY_PART_CHARGE.insert(0,1)
        
        self.LABEL_NUMBER_PART.grid(row=102, column=110, padx='5', pady='5', sticky='w')
        self.LABEL_PART_MASS.grid(row=103, column=110, padx='5', pady='5', sticky='w')
        self.LABEL_PART_CHARGE.grid(row=104, column=110, padx='5', pady='5', sticky='w')
        self.ENTRY_NUMBER_PART.grid(row=102, column=111, padx='5', pady='5', sticky='w')
        self.ENTRY_PART_MASS.grid(row=103, column=111, padx='5', pady='5', sticky='w')
        self.ENTRY_PART_CHARGE.grid(row=104, column=111, padx='5', pady='5', sticky='w')
        self.BUTTON_R_TOF.grid(row=109, column=110, columnspan=2, padx='5', pady='5', sticky='w')
       
        
        self.ENTRY_NUMBER_PART.insert(0, 1000)
        
        #if multiple particles
        self.LABEL_MULTI_PART_ENERGY_STEP = Label(self.R_tof_group, text='Energy Step:', background=frame_color)
        self.LABEL_MULTI_PART_NUMBER = Label(self.R_tof_group, text='Number of Particles', background=frame_color)
        self.ENTRY_MULTI_PART_ENERGY_STEP = Entry(self.R_tof_group)
        self.ENTRY_MULTI_PART_NUMBER = Entry(self.R_tof_group)
        self.LABEL_MULTI_PART_ENERGY_STEP.grid(row=107, column=110, padx='5', pady='5', sticky='w')
        self.LABEL_MULTI_PART_NUMBER.grid(row=108, column=110, padx='5', pady='5', sticky='w')
        self.ENTRY_MULTI_PART_ENERGY_STEP.grid(row=107, column=111, padx='5', pady='5', sticky='w')
        self.ENTRY_MULTI_PART_NUMBER.grid(row=108, column=111, padx='5', pady='5', sticky='w')
        
        self.LABEL_MULTI_PART_NUMBER.grid_remove()
        self.LABEL_MULTI_PART_ENERGY_STEP.grid_remove()
        self.ENTRY_MULTI_PART_ENERGY_STEP.grid_remove()
        self.ENTRY_MULTI_PART_NUMBER.grid_remove()
        
    ######### SAVES FOR VALIDATION ########################
        self.valid_group = LabelFrame(left_bar_group, text="Save Data for validation", padx=5, pady=5, bd=3, background=frame_color)
        self.valid_group.grid(row=115, column=100, columnspan=4, rowspan=6, padx='5', pady='5', sticky='new')
        
        self.BUTTON_SAVE_MOM = Button(self.valid_group, text="Save Momentum Data", command=self.export_momenta, activebackground = button_color)
        self.BUTTON_SAVE_MOM.grid(row=10, column=100, columnspan=2, padx='5', pady='5', sticky='w')
        
        self.BUTTON_CALC_MCP_TIMES = Button(self.valid_group, text='Save MCP times', command=self.calc_mcp, activebackground = button_color)
        self.BUTTON_CALC_MCP_TIMES.grid(row=11, column=100, columnspan=2, padx='5', pady='5', sticky='w')
        
        self.BUTTON_EXPORT_DATA = Button(self.valid_group, text='Save Electron Position', command=self.export_data, activebackground = button_color)
        self.BUTTON_EXPORT_DATA.grid(row=12, column=100, columnspan=2, padx='5', pady='5', sticky='w')
        
    ######## R tof simulation ##########################
        self.R_tof_sim_group = LabelFrame(top_bar_group, text="R-tof simulation", padx=5, pady=5, bd=3, background=frame_color)
        self.R_tof_sim_group.grid(row=100, column=110, columnspan=4, rowspan=6, padx='5', pady='5', sticky='nwe')
        
        self.LABEL_KIN_ENERGY = Label(self.R_tof_sim_group, text="Kinetic Energy [EV]:", background=frame_color)
        self.ENTRY_KIN_ENERGY_1 = Entry(self.R_tof_sim_group, fg='firebrick')
        self.ENTRY_KIN_ENERGY_2 = Entry(self.R_tof_sim_group, fg='deepskyblue')
        self.ENTRY_KIN_ENERGY_3 = Entry(self.R_tof_sim_group, fg='darkorange')
        self.LABEL_MASS = Label(self.R_tof_sim_group, text="Mass [a.u.]:", background=frame_color)
        self.ENTRY_MASS_1 = Entry(self.R_tof_sim_group, fg='firebrick')
        self.ENTRY_MASS_2 = Entry(self.R_tof_sim_group, fg='deepskyblue')
        self.ENTRY_MASS_3 = Entry(self.R_tof_sim_group, fg='darkorange')
        self.LABEL_CHARGE = Label(self.R_tof_sim_group, text="Charge [a.u.]:", background=frame_color)
        self.ENTRY_CHARGE_1 = Entry(self.R_tof_sim_group, fg='firebrick')
        self.ENTRY_CHARGE_2 = Entry(self.R_tof_sim_group, fg='deepskyblue')
        self.ENTRY_CHARGE_3 = Entry(self.R_tof_sim_group, fg='darkorange')
        self.LABEL_TOF = Label(self.R_tof_sim_group, text="Time of Flight maximum [ns]:", background=frame_color)
        self.ENTRY_TOF = Entry(self.R_tof_sim_group)
        self.BUTTON_R_TOF_SIM = Button(self.R_tof_sim_group, text="Simulate Particle", command=self.R_tof_sim, activebackground = button_color)
        
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
        
        self.LABEL_KIN_ENERGY.grid(row=106, column=110, padx='5', pady='5', sticky='w')
        self.ENTRY_KIN_ENERGY_1.grid(row=107, column=110, padx='5', pady='5', sticky='w')
        self.ENTRY_KIN_ENERGY_2.grid(row=108, column=110, padx='5', pady='5', sticky='w')
        self.ENTRY_KIN_ENERGY_3.grid(row=109, column=110, padx='5', pady='5', sticky='w')
        self.LABEL_MASS.grid(row=106, column=111, padx='5', pady='5', sticky='w')
        self.ENTRY_MASS_1.grid(row=107, column=111, padx='5', pady='5', sticky='w')
        self.ENTRY_MASS_2.grid(row=108, column=111, padx='5', pady='5', sticky='w')
        self.ENTRY_MASS_3.grid(row=109, column=111, padx='5', pady='5', sticky='w')
        self.LABEL_CHARGE.grid(row=106, column=112, padx='5', pady='5', sticky='w')
        self.ENTRY_CHARGE_1.grid(row=107, column=112, padx='5', pady='5', sticky='w')
        self.ENTRY_CHARGE_2.grid(row=108, column=112, padx='5', pady='5', sticky='w')
        self.ENTRY_CHARGE_3.grid(row=109, column=112, padx='5', pady='5', sticky='w')
        self.ENTRY_TOF.grid(row=110, column=111, padx='5', pady='5', sticky='w')
        self.LABEL_TOF.grid(row=110, column=110, padx='5', pady='5', sticky='w')
        self.BUTTON_R_TOF_SIM.grid(row=110, column=112, columnspan=1, rowspan=5, padx='5', pady='5', sticky='ns')
        
        #### Plots and Slidebars ##############
        
        self.R_tof_plot_group = LabelFrame(tab1, text="Electron plots", padx=5, pady=5, bd=3, background=frame_color)
        self.R_tof_plot_group.grid(row=110, column=110, columnspan=2, rowspan=40, padx='5', pady='5', sticky='nwe')
        self.R_tof_plot_group.grid_remove()
        
        self.v_ir = IntVar()
        self.v_ir.set(0)
        self.CHECK_IR_PLOT = Checkbutton(self.R_tof_plot_group, text="Enable IR plot Mode", variable=self.v_ir, onvalue=1, background=frame_color)
        self.CHECK_IR_PLOT.grid(row=105, column=100, columnspan=2, padx='5', pady='5', sticky='ew')
        
        self.LABEL_SLIDE_U = Label(self.R_tof_plot_group, text="Voltage", background=frame_color)
        self.LABEL_SLIDE_U.grid(row=106, column=100, columnspan=2, padx='5', pady='5', sticky='ew')
        
        self.SLIDE_U = Scale(self.R_tof_plot_group, from_=0, to=200, orient=HORIZONTAL, command=self.set_new_u)
        self.SLIDE_U.grid(row=107, column=100, columnspan=2, padx='5', pady='5', sticky='ew')
        
        self.LABEL_SLIDE_B = Label(self.R_tof_plot_group, text="Magnetic Field", background=frame_color)
        self.LABEL_SLIDE_B.grid(row=108, column=100, columnspan=2, padx='5', pady='5', sticky='ew')
        
        self.SLIDE_B = Scale(self.R_tof_plot_group, from_=0, to=100, orient=HORIZONTAL, command=self.set_new_b)
        self.SLIDE_B.grid(row=109, column=100, columnspan=2, padx='5', pady='5', sticky='ew')
        
        
        #### IR mode #####
        self.ir_mode_group = LabelFrame(top_bar_group, text="IR-Mode", padx=5, pady=5, bd=3, background=frame_color)
        self.ir_mode_group.grid(row=100, column=120, columnspan=2, padx='5', pady='5', sticky='nwe')
        
        self.LABEL_KIN_ENERGY_START = Label(self.ir_mode_group, text="First Kin Energy [eV]", background=frame_color)
        self.LABEL_KIN_ENERGY_STEP = Label(self.ir_mode_group, text="Kin Energy Stepsize [eV]", background=frame_color)
        self.LABEL_NUMBER_OF_PART = Label(self.ir_mode_group, text="Numer of particles", background=frame_color)
        self.LABEL_MASS_IR = Label(self.ir_mode_group, text="Mass", background=frame_color)
        self.LABEL_CHARGE_IR = Label(self.ir_mode_group, text="Charge", background=frame_color)
        
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
        
        self.BUTTON_SIM_IR_MODE = Button(self.ir_mode_group, text="Simulate Particle IR Mode", command=self.R_tof_sim_ir, activebackground = button_color)
        self.BUTTON_SIM_IR_MODE.grid(row=4, column=4, padx='5', pady='5', sticky='ns')
        
        self.LABEL_KIN_ENERGY_START.grid(row=6, column=2, columnspan=1, padx='5', pady='5', sticky='w')
        self.LABEL_KIN_ENERGY_STEP.grid(row=7, column=2, columnspan=1, padx='5', pady='5', sticky='w')
        self.LABEL_NUMBER_OF_PART.grid(row=8, column=2, columnspan=1, padx='5', pady='5', sticky='w')
        self.LABEL_MASS_IR.grid(row=4, column=2, columnspan=1, padx='5', pady='5', sticky='w')
        self.LABEL_CHARGE_IR.grid(row=5, column=2, columnspan=1, padx='5', pady='5', sticky='w')
        
        self.ENTRY_KIN_ENERGY_START.grid(row=6, column=3, columnspan=1, padx='5', pady='5', sticky='w')
        self.ENTRY_KIN_ENERGY_STEP.grid(row=7, column=3, columnspan=1, padx='5', pady='5', sticky='w')
        self.ENTRY_NUMBER_OF_PART.grid(row=8, column=3, columnspan=1, padx='5', pady='5', sticky='w')
        self.ENTRY_MASS_IR.grid(row=4, column=3, columnspan=1, padx='5', pady='5', sticky='w')
        self.ENTRY_CHARGE_IR.grid(row=5, column=3, columnspan=1, padx='5', pady='5', sticky='w')
        
 
    ######## Coincidences ##################################
        #### REMI parameter for Ion ####
        remi_ion_conf_group = LabelFrame(tab3, text="REMI Configuration for Ion", padx=5, pady=5, bd=3, background=frame_color)
        remi_ion_conf_group.grid(row=100, column=100, columnspan=2, rowspan=6, padx='5', pady='5', sticky='nw')
        
        self.LABEL_SET_U_ion = Label(remi_ion_conf_group, text='U[V]:', background=frame_color)
        self.LABEL_SET_B_ion = Label(remi_ion_conf_group, text='B[Gauss]:', background=frame_color)
        self.LABEL_SET_l_a_ion = Label(remi_ion_conf_group, text='acc length[m]:', background=frame_color)
        self.LABEL_SET_v_jet = Label(remi_ion_conf_group, text='v jet[mm/ns]:', background=frame_color)
        
        self.LABEL_SET_U_ion.grid(row=103, column=101, padx='5', pady='5', sticky='w')
        self.LABEL_SET_B_ion.grid(row=104, column=101, padx='5', pady='5', sticky='w')
        self.LABEL_SET_l_a_ion.grid(row=105, column=101, padx='5', pady='5', sticky='w')
        self.LABEL_SET_v_jet.grid(row=106, column=101, padx='5', pady='5', sticky='w')
        
        self.ENTRY_SET_U_ion = Entry(remi_ion_conf_group)
        self.ENTRY_SET_B_ion = Entry(remi_ion_conf_group)
        self.ENTRY_SET_l_a_ion = Entry(remi_ion_conf_group)
        self.ENTRY_SET_v_jet = Entry(remi_ion_conf_group)
        
        self.ENTRY_SET_U_ion.grid(row=103, column=102, padx='5', pady='5', sticky='w')
        self.ENTRY_SET_B_ion.grid(row=104, column=102, padx='5', pady='5', sticky='w')
        self.ENTRY_SET_l_a_ion.grid(row=105, column=102, padx='5', pady='5', sticky='w')
        self.ENTRY_SET_v_jet.grid(row=106, column=102, padx='5', pady='5', sticky='w')
        
        self.ENTRY_SET_U_ion.insert(0, 190)
        self.ENTRY_SET_B_ion.insert(0, 5)
        self.ENTRY_SET_l_a_ion.insert(0, 0.188)
        self.ENTRY_SET_v_jet.insert(0, 0.001)
        
        #### Ion configuration ####
        ion_conf_group = LabelFrame(tab3, text="REMI Configuration for Ion", padx=5, pady=5, bd=3, background=frame_color)
        ion_conf_group.grid(row=110, column=100, columnspan=2, rowspan=6, padx='5', pady='5', sticky='nw')
        
        self.LABEL_ION_MASS = Label(ion_conf_group, text='Ion Mass [a.u.]:', background=frame_color)
        self.LABEL_ION_CHARGE = Label(ion_conf_group, text="Ion Charge [a.u.]:", background=frame_color)
        
        self.LABEL_ION_MASS.grid(row=110, column=100, padx='5', pady='5', sticky='w')
        self.LABEL_ION_CHARGE.grid(row=111, column=100, padx='5', pady='5', sticky='w')
        
        self.ENTRY_ION_MASS = Entry(ion_conf_group)
        self.ENTRY_ION_CHARGE = Entry(ion_conf_group)
        
        self.ENTRY_ION_MASS.grid(row=110, column=101, padx='5', pady='5', sticky='w')
        self.ENTRY_ION_CHARGE.grid(row=111, column=101, padx='5', pady='5', sticky='w')
        
    
        self.ion_pos_group = LabelFrame(tab3, text="Ion Postitions", padx=5, pady=5, bd=3, background=frame_color)
        self.ion_pos_group.grid(row=120, column=100, columnspan=4, rowspan=6, padx='5', pady='5', sticky='nw')
        
        self.BUTTON_ION_POSITION = Button(self.ion_pos_group, text='Calculate Ion Positions', command=self.calc_ion_position, activebackground = button_color)
        self.BUTTON_ION_POSITION.grid(row=110, column=100, padx='5', pady='5', sticky='w')
   
    def make_plot_xarray(self, data, row, column, master, sorting=False, sort='time', rowspan=1, columnspan=1, figsize=(4,4), color='blue', marker='.', ls='', title=''):
        """
        Plots the data at the given position
        
        Parameters
        ----------
        data: xarray
            data to be plottet
        row: int
            row to place the plot
        column: int
            column to place the plot
        master: Frame
        sorting: Bool, optional
        sort: string, optional
        rowspan: int, optional
        columnspan: int, optional
        figsize: tuple, optional
        color: string, optional
        marker: string, optional
        ls: string, optional
        title: string, optional
        
        Returns
        -------
        fig : Figure
        a : axis
        canvas : canvas
        """
        
        fig = Figure(figsize=figsize, facecolor='whitesmoke')
        a = fig.add_subplot(111)
        if sorting==False:
            data.plot(ax=a, marker=marker, ls=ls, color=color)
        else:
            time = data.sortby(sort).time
            rad = data.sortby(sort).values
            a.hexbin(time, rad, mincnt=1, edgecolors='face', gridsize=50, cmap='PuBuGn')
            a.set_title(title)
            
        a.autoscale(tight=True)
        canvas = FigureCanvasTkAgg(fig, master=master)
        canvas.get_tk_widget().grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, padx='5', pady='5', sticky='ew')
        canvas.draw()
        return fig, a, canvas
    
    def make_plot(self, x, y, row, column, master, rowspan=1, columnspan=1, figsize=(4,4), title='', xlim=None, ylim=None, extent=None):
        """
        Makes a hexbin-plot of x and y
        
        Returns
        -------
        fig : Figure
        a : axis
        canvas : canvas
        """
        fig = Figure(figsize=figsize, facecolor='whitesmoke')
        a = fig.add_subplot(111)
        a.hexbin(x, y, mincnt=1, edgecolors='face', gridsize=100, cmap='PuBuGn', extent=extent)
        a.set_title(title)
        a.set_xlim(xlim)
        a.set_ylim(ylim)
        a.autoscale(tight=True)
        canvas = FigureCanvasTkAgg(fig, master=master)
        canvas.get_tk_widget().grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, padx='5', pady='5', sticky='ew')
        canvas.draw()
        return fig, a, canvas
        
    def change_remi_conf(self):
        """
        Changes the setings of the Remi (Voltage, Magnetic-field, acceleration length)
        """
        U = float(self.ENTRY_SET_U.get())
        B = float(self.ENTRY_SET_B.get())*1e-4 
        l_a = float(self.ENTRY_SET_l_a.get())
        self.remi_params = np.array([U, B, l_a])
        return self.remi_params
    
    def check(self):
        if self.v.get()==1:
            self.LABEL_MEAN_ENERGY.grid_remove()
            self.LABEL_WIDTH.grid_remove()
            self.ENTRY_MEAN_ENERGY.grid_remove()
            self.ENTRY_WIDTH.grid_remove()
            self.LABEL_MULTI_PART_NUMBER.grid_remove()
            self.LABEL_MULTI_PART_ENERGY_STEP.grid_remove()
            self.ENTRY_MULTI_PART_ENERGY_STEP.grid_remove()
            self.ENTRY_MULTI_PART_NUMBER.grid_remove()
        elif self.v.get()==2:
            self.LABEL_MEAN_ENERGY.grid()
            self.LABEL_WIDTH.grid()
            self.ENTRY_MEAN_ENERGY.grid()
            self.ENTRY_WIDTH.grid()
            self.LABEL_MULTI_PART_NUMBER.grid_remove()
            self.LABEL_MULTI_PART_ENERGY_STEP.grid_remove()
            self.ENTRY_MULTI_PART_ENERGY_STEP.grid_remove()
            self.ENTRY_MULTI_PART_NUMBER.grid_remove()
        elif self.v.get()==3:
            self.LABEL_MEAN_ENERGY.grid()
            self.LABEL_WIDTH.grid()
            self.ENTRY_MEAN_ENERGY.grid()
            self.ENTRY_WIDTH.grid()
            self.LABEL_MULTI_PART_NUMBER.grid()
            self.LABEL_MULTI_PART_ENERGY_STEP.grid()
            self.ENTRY_MULTI_PART_ENERGY_STEP.grid()
            self.ENTRY_MULTI_PART_NUMBER.grid()
            
    def make_R_tof(self):
        """
        Generates the R vs tof plot and the electron position plot with random data points
        """
        self.R_tof_plot_group.grid()
        self.particle_params=(float(self.ENTRY_PART_MASS.get())*m_e, float(self.ENTRY_PART_CHARGE.get())*q_e)
        if self.v.get()==1:
            self.momenta = make_gaussian_momentum_distribution(int(self.ENTRY_NUMBER_PART.get()))
        elif self.v.get()==2:
            energy_mean = float(self.ENTRY_MEAN_ENERGY.get())
            width = float(self.ENTRY_WIDTH.get())
            self.momenta = make_gaussian_energy_distribution(energy_mean, width, self.particle_params[0], number_of_particles=int(self.ENTRY_NUMBER_PART.get()))
        elif self.v.get()==3:
            energy_step = float(self.ENTRY_MULTI_PART_ENERGY_STEP.get())
            energy_mean = float(self.ENTRY_MEAN_ENERGY.get())
            width = float(self.ENTRY_WIDTH.get())
            part_num = int(self.ENTRY_MULTI_PART_NUMBER.get())
            self.momenta = make_gaussian_energy_distribution(energy_mean, width, self.particle_params[0], number_of_particles=int(self.ENTRY_NUMBER_PART.get()))
            for i in range(1, part_num):
                self.momenta = np.concatenate([self.momenta, make_gaussian_energy_distribution(energy_mean+(i*energy_step), width, self.particle_params[0], number_of_particles=int(self.ENTRY_NUMBER_PART.get()))])
           
        self.R_tof = make_R_tof_array(self.momenta, self.remi_params, self.particle_params)
        self.fig_R_tof, self.ax_R_tof, self.canvas_R_tof = self.make_plot_xarray(self.R_tof, 100, 100, self.R_tof_plot_group, sorting=True, sort='time', columnspan=2, color='powderblue', figsize=(6,6), title='Rad vs Time') 
        self.plot_position()
        
        max_tof = self.calc_max_tof()
        self.ax_R_tof.axvline(max_tof, 0, 1, color='darkgrey')
        no_mom_tof = self.calc_no_momentum_tof()
        self.ax_R_tof.axvline(no_mom_tof, 0, 1, ls='--', color='darkgrey')
        self.canvas_R_tof.draw()
        
    def update_R_tof(self):
        """
        Updates the R vs tof and the position plot, while moving the sliders for B and U
        """
        self.particle_params=(float(self.ENTRY_PART_MASS.get())*m_e, float(self.ENTRY_PART_CHARGE.get())*q_e)
        if self.v.get()==1:
            self.momenta = make_gaussian_momentum_distribution(int(self.ENTRY_NUMBER_PART.get()))
        elif self.v.get()==2:
            energy_mean = float(self.ENTRY_MEAN_ENERGY.get())
            width = float(self.ENTRY_WIDTH.get())
            self.momenta = make_gaussian_energy_distribution(energy_mean, width, self.particle_params[0], number_of_particles=int(self.ENTRY_NUMBER_PART.get()))
        elif self.v.get()==3:
            energy_step = float(self.ENTRY_MULTI_PART_ENERGY_STEP.get())
            energy_mean = float(self.ENTRY_MEAN_ENERGY.get())
            width = float(self.ENTRY_WIDTH.get())
            part_num = int(self.ENTRY_MULTI_PART_NUMBER.get())
            self.momenta = make_gaussian_energy_distribution(energy_mean, width, self.particle_params[0], number_of_particles=int(self.ENTRY_NUMBER_PART.get()))
            for i in range(1, part_num):
                self.momenta = np.concatenate([self.momenta, make_gaussian_energy_distribution(energy_mean+(i*energy_step), width, self.particle_params[0], number_of_particles=int(self.ENTRY_NUMBER_PART.get()))])
        self.R_tof = make_R_tof_array(self.momenta, self.remi_params, self.particle_params)
        
        self.ax_R_tof.cla()
        self.ax_R_tof.hexbin(self.R_tof.time, self.R_tof.values, mincnt=1, edgecolors='face', gridsize=50, cmap='PuBuGn')
        if self.v_ir.get()==1:
            self.R_tof_sim_ir()

        max_tof = self.calc_max_tof()
        self.ax_R_tof.axvline(max_tof, 0, 1, color='darkgrey')
        
        no_mom_tof = self.calc_no_momentum_tof()
        self.ax_R_tof.axvline(no_mom_tof, 0, 1, ls='--', color='darkgrey')
        self.canvas_R_tof.draw()

        
        self.ele_pos_a.cla()
        x,y = self.calc_position()

        self.ele_pos_a.hexbin(x, y, mincnt=1, edgecolors='face', gridsize=100, cmap='PuBuGn', extent=(-0.1,0.1,-0.1,0.1))
        self.ele_pos_a.set_xlim(-0.1,0.1)
        self.ele_pos_a.set_ylim(-0.1,0.1)
        detector = plt.Circle((0, 0), 0.04, color='cadetblue', fill=False, figure=self.ele_pos_fig)
        self.ele_pos_a.add_artist(detector)
        self.ele_pos_canvas.draw()

        
    def R_tof_sim(self):
        """
        Generates a R vs tof plot
        """
        tof_max = float(self.ENTRY_TOF.get())*1e-9
        tof = np.linspace(0, tof_max, int(tof_max*1000e9))
        while len(self.ax_R_tof.lines)>1:
            self.ax_R_tof.lines[-1].remove()

        if len(self.ENTRY_KIN_ENERGY_1.get())!=0:
            energy_1 = float(self.ENTRY_KIN_ENERGY_1.get())*1.6e-19
            mass_1 = float(self.ENTRY_MASS_1.get())*m_e
            charge_1 = float(self.ENTRY_CHARGE_1.get())*q_e
            particle_params_1 = (mass_1, charge_1)
            K_1 = np.sqrt(2*mass_1*energy_1)
            R_1 = calc_R_fit(K_1, tof, self.remi_params, particle_params_1)
            self.ax_R_tof.plot(tof, R_1, color='firebrick')
            
        if len(self.ENTRY_KIN_ENERGY_2.get())!=0:
            energy_2 = float(self.ENTRY_KIN_ENERGY_2.get())*1.6e-19
            mass_2 = float(self.ENTRY_MASS_2.get())*m_e
            charge_2 = float(self.ENTRY_CHARGE_2.get())*q_e
            particle_params_2 = (mass_2, charge_2)
            K_2 = np.sqrt(2*mass_2*energy_2)
            R_2 = calc_R_fit(K_2, tof, self.remi_params, particle_params_2)
            self.ax_R_tof.plot(tof, R_2, color='deepskyblue')

        if len(self.ENTRY_KIN_ENERGY_3.get())!=0:
            energy_3 = float(self.ENTRY_KIN_ENERGY_3.get())*1.6e-19
            mass_3 = float(self.ENTRY_MASS_3.get())*m_e
            charge_3 = float(self.ENTRY_CHARGE_3.get())*q_e
            particle_params_3 = (mass_3, charge_3)
            K_3 = np.sqrt(2*mass_3*energy_3)
            R_3 = calc_R_fit(K_3, tof, self.remi_params, particle_params_3)
            self.ax_R_tof.plot(tof, R_3, color='darkorange')
            
        
        max_tof = self.calc_max_tof()
        self.ax_R_tof.axvline(max_tof, 0, 1, color='darkgrey')
        no_mom_tof = self.calc_no_momentum_tof()
        self.ax_R_tof.axvline(no_mom_tof, 0, 1, ls='--', color='darkgrey')
        self.canvas_R_tof.draw()
        
    def calc_position(self):
        """
        calculates the electron positions (x,y)
        """
        U, B, l_a = self.remi_params
        m, q = self.particle_params
        tof = self.R_tof.time
        R = self.R_tof.values
        
        alpha = calc_omega(B, q, m)*tof
        alpha2 = 180-np.abs(180-alpha)
        beta = (180-alpha2)/2
        
        p_x = self.momenta[:,0,0]
        p_y = self.momenta[:,0,1]
        phi = np.arctan2(p_y,p_x)
        
        theta = phi + 90 + beta
        
        x = R*np.sin(180-theta)
        y = R*np.cos(180-theta)
        
        return x,y
    
    def plot_position(self):
        """
        generates a hex-plot of the electron positions with random distribution
        """
        x,y = self.calc_position()
        detector_radius = 0.04
        self.ele_pos_fig, self.ele_pos_a, self.ele_pos_canvas = self.make_plot(x, y, 100, 110, self.R_tof_plot_group, figsize=(6,6), title='Electron Positions', extent=(-0.1,0.1,-0.1,0.1))
        self.ele_pos_a.set_xlim(-0.1,0.1)
        self.ele_pos_a.set_ylim(-0.1,0.1)
        detector = plt.Circle((0, 0), detector_radius, color='cadetblue', fill=False, figure=self.ele_pos_fig)
        self.ele_pos_a.add_artist(detector)
        self.ele_pos_canvas.draw()
        
    def calc_max_tof(self):
        U, B, l_a = self.remi_params
        m, q = self.particle_params
        l_ion = 0.0945
        
        E = U/l_a
        time_1 = np.sqrt(2*l_ion*m/(E*q)) # time from reaction point to ion detector
        time_2 = np.sqrt(2*(l_a+l_ion)*m/(E*q)) # time from ion detector to electron detector
        tof_max = time_1+time_2
        return tof_max
    
    def calc_no_momentum_tof(self):
        """
        calculates the time of flight for a paticle with no z-momentum
        """
        U, B, l_a = self.remi_params
        m, q = self.particle_params
        E = U/l_a
        tof_no_mom = np.sqrt(2*l_a*m/(E*q))
        return tof_no_mom
    
    def export_data(self):
        """
        writes electron position data to a file
        """
        x, y = self.calc_position()
        tof = self.R_tof.time
        data = np.array([x, y, tof])
        data = data.T
        with open('pos_data.txt', 'w') as datafile:
            np.savetxt(datafile, data, fmt=['%.3E','%.3E','%.3E'])
            
    def export_momenta(self):
        """
        writes electron momentum data to a file
        """
        p_x = self.momenta[:,0,0]
        p_y = self.momenta[:,0,1]
        p_z = self.momenta[:,0,2]
        mom = np.array([p_x, p_y, p_z])
        mom = mom.T
        with open('mom_data.txt', 'w') as datafile:
            np.savetxt(datafile, mom, fmt=['%.3E','%.3E','%.3E'])
            
    def calc_mcp(self):
        """
        calculates the mcp times and write them to a file
        """
        x, y = self.calc_position()
        tof = self.R_tof.time
        t_mcp = tof
        v = 0.857*1e6
        sum_x = 143.8*1e-9
        sum_y = 141.2*1e-9
        def calc_t(time_sum, v, x):
            t = -x/v + time_sum/2
            return t
        t2_x = calc_t(sum_x, v, x)
        t2_y = calc_t(sum_y, v, y)
        t1_x = sum_x - t2_x
        t1_y = sum_y - t2_y
        
        times = np.array([t_mcp, t1_x, t2_x, t1_y, t2_y])
        times = times.T
        with open('mcp_data.txt', 'w') as datafile:
            np.savetxt(datafile, times, fmt=['%.3E','%.3E','%.3E','%.3E','%.3E'])
            
    def calc_ion_position(self):
        """
        calculates the ion position of 
        """
        p_x = -self.momenta[:,0,0]
        p_y = -self.momenta[:,0,1]
        p_z = -self.momenta[:,0,2]
        v_jet = float(self.ENTRY_SET_v_jet.get())*1e6
        ion_mass = float(self.ENTRY_ION_MASS.get())*m_e
        ion_remi_params = (float(self.ENTRY_SET_U_ion.get()), float(self.ENTRY_SET_B_ion.get())*1e-4 , float(self.ENTRY_SET_l_a_ion.get()))
        ion_params = (ion_mass, float(self.ENTRY_ION_CHARGE.get())*1.6e-19)
        tof = calc_tof(-self.momenta, ion_remi_params, ion_params)
        x_pos_ion = (p_x/ion_mass + v_jet)*tof
        y_pos_ion = (p_y/ion_mass)*tof
        self.make_plot(x_pos_ion, y_pos_ion, 120, 100, self.ion_pos_group, columnspan=2)
        return(x_pos_ion, y_pos_ion)
    
    def R_tof_sim_ir(self):
        tof_max = float(self.ENTRY_TOF.get())*1e-9
        tof = np.linspace(0, tof_max, int(tof_max*100e9))
        start_energy = float(self.ENTRY_KIN_ENERGY_START.get())*1.6e-19
        step_energy = float(self.ENTRY_KIN_ENERGY_STEP.get())*1.6e-19
        number_sim = int(self.ENTRY_NUMBER_OF_PART.get())
        energys = np.linspace(start_energy, (number_sim*step_energy)+start_energy, number_sim+1)
        mass = float(self.ENTRY_MASS_IR.get())*m_e
        charge = float(self.ENTRY_CHARGE_IR.get())*q_e
        particle_params = (mass, charge)
        
        while len(self.ax_R_tof.lines)>1:
            self.ax_R_tof.lines[-1].remove()
        
        for i in range(number_sim):
            K = np.sqrt(2*mass*energys[i])
            R = calc_R_fit(K, tof, self.remi_params, particle_params)
            self.ax_R_tof.plot(tof, R, color='firebrick')
        self.canvas_R_tof.draw()
        
    def set_new_u(self, U):
        U = int(self.SLIDE_U.get())
        B = float(self.ENTRY_SET_B.get())*1e-4 
        l_a = float(self.ENTRY_SET_l_a.get())
        self.ENTRY_SET_U.delete(0, END)
        self.ENTRY_SET_U.insert(0, str(U))
        self.remi_params = np.array([U, B, l_a])
        self.update_R_tof()
        return self.remi_params
    
    def set_new_b(self, B):
        U = float(self.ENTRY_SET_U.get())
        B = int(self.SLIDE_B.get())
        self.ENTRY_SET_B.delete(0, END)
        self.ENTRY_SET_B.insert(0, str(B))
        B = B*1e-4 
        l_a = float(self.ENTRY_SET_l_a.get())
        self.remi_params = np.array([U, B, l_a])
        self.update_R_tof()
        return self.remi_params
        
    
window = Tk()
window.configure(background = 'whitesmoke')
start = mclass(window)
window.mainloop()
