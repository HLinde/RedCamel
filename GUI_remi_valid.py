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
    from Tkinter import Tk, Button, Entry, Label, Listbox, END, LabelFrame, Radiobutton, IntVar
except:
    from tkinter import Tk, Button, Entry, Label, Listbox, END, LabelFrame, ttk , Radiobutton, IntVar

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
    
    r = (np.random.randn(number_of_particles, 1) * width + energy_mean)*1.6e-19
    phi = np.random.rand(number_of_particles, 1)*2*np.pi
    cos_theta = np.random.rand(number_of_particles, 1)*2-1
    
    r_mom = np.sqrt(r*2*mass)
    
    theta = np.arccos(cos_theta)
    
    x= r_mom * np.sin(theta) * np.cos(phi)
    y= r_mom * np.sin(theta) * np.sin(phi)
    z= r_mom * cos_theta
    
    energy = np.stack([x, y, z], axis=-1)
    print(energy.shape)
    
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
        style = ttk.Style()
        style.configure('BW.TLabel', background="whitesmoke")
        tabControl = ttk.Notebook(window)
        tab1 = ttk.Frame(tabControl, width=300, height=300, style='BW.TLabel')
        tab2 = ttk.Frame(tabControl, width=300, height=300, style='BW.TLabel')
        tab3 = ttk.Frame(tabControl, width=300, height=300, style='BW.TLabel')
        tabControl.add(tab1, text='R vs TOF')
        tabControl.add(tab2, text='Position')
        tabControl.add(tab3, text='y')
        tabControl.grid(column=0)
            
        back_color = 'whitesmoke'
        button_color = 'aliceblue'
        frame_color = 'mintcream'
        
        self.l_a = 0.188         # acc_length 
        self.U = 190            # electic_field 
        self.B = 5*1e-4           # magnetic_field 
        self.omega = q_e * self.B / m_e
        
    ######## REMI configurations ##############
        remi_conf_group = LabelFrame(tab1, text="REMI Configuration", padx=5, pady=5, bd=3, background=frame_color)
        remi_conf_group.grid(row=100, column=100, columnspan=2, rowspan=6, padx='5', pady='5', sticky='nw')
        
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
        self.R_tof_group = LabelFrame(tab1, text="R-tof calculation", padx=5, pady=5, bd=3, background=frame_color)
        self.R_tof_group.grid(row=100, column=110, columnspan=2, rowspan=5, padx='5', pady='5', sticky='nw')
        
        self.v = IntVar()
        self.v.set(1)
        self.CHOOSE_MOMENTUM = Radiobutton(self.R_tof_group, command=self.check, text="Momentum", variable=self.v, value=1, background=frame_color)
        self.CHOOSE_ENERGY = Radiobutton(self.R_tof_group, command=self.check, text="Energy", variable=self.v, value=2, background=frame_color)
        self.CHOOSE_MOMENTUM.select()
        self.CHOOSE_MOMENTUM.grid(row=100, column=110, padx='5', pady='5', sticky='w')
        self.CHOOSE_ENERGY.grid(row=101, column=110, padx='5', pady='5', sticky='w')
        
        self.LABEL_NUMBER_PART = Label(self.R_tof_group, text='number of Particles:', background=frame_color)
        self.LABEL_PART_MASS = Label(self.R_tof_group, text='Particle mass:', background=frame_color)
        self.LABEL_PART_CHARGE = Label(self.R_tof_group, text='Particle charge:', background=frame_color)
        self.ENTRY_NUMBER_PART = Entry(self.R_tof_group)
        self.ENTRY_PART_MASS = Entry(self.R_tof_group)
        self.ENTRY_PART_CHARGE = Entry(self.R_tof_group)
        self.BUTTON_R_TOF = Button(self.R_tof_group, text="Calculate radius and tof", command=self.make_R_tof, activebackground = button_color)
        self.BUTTON_SAVE_MOM = Button(tab1, text="Save Momentum Data", command=self.export_momenta, activebackground = button_color)
        
        #if selecting calculation with energy
        self.LABEL_MEAN_ENERGY = Label(self.R_tof_group, text='Mean Energy:', background=frame_color)
        self.LABEL_WIDTH = Label(self.R_tof_group, text='Width:', background=frame_color)
        self.ENTRY_MEAN_ENERGY = Entry(self.R_tof_group)
        self.ENTRY_WIDTH = Entry(self.R_tof_group)
        
        self.LABEL_MEAN_ENERGY.grid(row=103, column=112, padx='5', pady='5', sticky='w')
        self.LABEL_WIDTH.grid(row=104, column=112, padx='5', pady='5', sticky='w')
        self.ENTRY_MEAN_ENERGY.grid(row=103, column=113, padx='5', pady='5', sticky='w')
        self.ENTRY_WIDTH.grid(row=104, column=113, padx='5', pady='5', sticky='w')
        self.LABEL_MEAN_ENERGY.grid_remove()
        self.LABEL_WIDTH.grid_remove()
        self.ENTRY_MEAN_ENERGY.grid_remove()
        self.ENTRY_WIDTH.grid_remove()
        
        self.ENTRY_PART_MASS.insert(0,1)
        self.ENTRY_PART_CHARGE.insert(0,1)
        
        self.LABEL_NUMBER_PART.grid(row=102, column=110, padx='5', pady='5', sticky='w')
        self.LABEL_PART_MASS.grid(row=103, column=110, padx='5', pady='5', sticky='w')
        self.LABEL_PART_CHARGE.grid(row=104, column=110, padx='5', pady='5', sticky='w')
        self.ENTRY_NUMBER_PART.grid(row=102, column=111, padx='5', pady='5', sticky='w')
        self.ENTRY_PART_MASS.grid(row=103, column=111, padx='5', pady='5', sticky='w')
        self.ENTRY_PART_CHARGE.grid(row=104, column=111, padx='5', pady='5', sticky='w')
        self.BUTTON_R_TOF.grid(row=105, column=110, columnspan=2, padx='5', pady='5', sticky='w')
        self.BUTTON_SAVE_MOM.grid(row=110, column=100, columnspan=2, padx='5', pady='5', sticky='w')
        
        self.ENTRY_NUMBER_PART.insert(0, 1000)
        
    ######## R tof simulation ##########################
        self.R_tof_sim_group = LabelFrame(tab1, text="R-tof simulation", padx=5, pady=5, bd=3, background=frame_color)
        self.R_tof_sim_group.grid(row=105, column=110, columnspan=4, rowspan=6, padx='5', pady='5', sticky='nw')
        
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
        
        self.ENTRY_KIN_ENERGY_1.insert(0, 10)
        self.ENTRY_KIN_ENERGY_2.insert(0, 20)
        self.ENTRY_KIN_ENERGY_3.insert(0, 30)
        self.ENTRY_MASS_1.insert(0, 1)
        self.ENTRY_MASS_2.insert(0, 1)
        self.ENTRY_MASS_3.insert(0, 1)
        self.ENTRY_CHARGE_1.insert(0, 1)
        self.ENTRY_CHARGE_2.insert(0, 1)
        self.ENTRY_CHARGE_3.insert(0, 1)
        self.ENTRY_TOF.insert(0, 100)
        
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
        self.BUTTON_R_TOF_SIM.grid(row=111, column=110, columnspan=2, padx='5', pady='5', sticky='w')
        
        self.R_tof_plot_group = LabelFrame(tab1, text="R-tof", padx=5, pady=5, bd=3, background=frame_color)
        self.R_tof_plot_group.grid(row=112, column=110, columnspan=2, rowspan=6, padx='5', pady='5', sticky='nw')
        
        
    ######### Position reconstruction ######################
        self.position_group = LabelFrame(tab2, text="Calculate Position", padx=5, pady=5, bd=3, background=frame_color)
        self.position_group.grid(row=100, column=100, columnspan=4, rowspan=6, padx='5', pady='5', sticky='nw')
        
        self.BUTTON_PLOT_POSITION = Button(self.position_group, text='Plot Postitions', command=self.plot_position, activebackground = button_color)
        self.BUTTON_EXPORT_DATA = Button(self.position_group, text='Export Data', command=self.export_data, activebackground = button_color)
        
        self.BUTTON_PLOT_POSITION.grid(row=105, column=105, columnspan=2, padx='5', pady='5', sticky='w')
        self.BUTTON_EXPORT_DATA.grid(row=107, column=105, columnspan=2, padx='5', pady='5', sticky='w')
        
    ######### MCP TIMES CALCULATION ########################
        self.mcp_group = LabelFrame(tab3, text="Calculate MCP times", padx=5, pady=5, bd=3, background=frame_color)
        self.mcp_group.grid(row=100, column=100, columnspan=4, rowspan=6, padx='5', pady='5', sticky='nw')
        
        
        self.BUTTON_CALC_MCP_TIMES = Button(self.mcp_group, text='Calculate MCP times', command=self.calc_mcp, activebackground = button_color)
        self.BUTTON_CALC_MCP_TIMES.grid(row=105, column=105, columnspan=2, padx='5', pady='5', sticky='w')
        
    def make_plot_xarray(self, data, row, column, master, sorting=False, sort='time', rowspan=1, columnspan=1, figsize=(4,4), color='blue', marker='.', ls=''):
        fig = Figure(figsize=figsize, facecolor='whitesmoke')
        a = fig.add_subplot(111)
        if sorting==False:
            data.plot(ax=a, marker=marker, ls=ls, color=color)
        else:
            data.sortby(sort).plot.line(ax=a, marker=marker, ls=ls, color=color)
            
        a.autoscale(tight=True)
        canvas = FigureCanvasTkAgg(fig, master=master)
        canvas.get_tk_widget().grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, padx='5', pady='5', sticky='ew')
        canvas.draw()
        return fig, a, canvas
    
    def make_plot(self, x, y, row, column, master, rowspan=1, columnspan=1, figsize=(4,4), color='blue', marker='.', ls=''):
        fig = Figure(figsize=figsize, facecolor='whitesmoke')
        a = fig.add_subplot(111)
        a.plot(x, y, color=color, marker=marker, ls=ls)
        a.autoscale(tight=True)
        canvas = FigureCanvasTkAgg(fig, master=master)
        canvas.get_tk_widget().grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, padx='5', pady='5', sticky='ew')
        canvas.draw()
        
    def change_remi_conf(self):
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
        elif self.v.get()==2:
            self.LABEL_MEAN_ENERGY.grid()
            self.LABEL_WIDTH.grid()
            self.ENTRY_MEAN_ENERGY.grid()
            self.ENTRY_WIDTH.grid()
    
    def make_R_tof(self):
        self.particle_params=(float(self.ENTRY_PART_MASS.get())*m_e, float(self.ENTRY_PART_CHARGE.get())*q_e)
        if self.v.get()==1:
            self.momenta = make_gaussian_momentum_distribution(int(self.ENTRY_NUMBER_PART.get()))
        elif self.v.get()==2:
            energy_mean = float(self.ENTRY_MEAN_ENERGY.get())
            width = float(self.ENTRY_WIDTH.get())
            self.momenta = make_gaussian_energy_distribution(energy_mean, width, self.particle_params[0], number_of_particles=int(self.ENTRY_NUMBER_PART.get()))
        self.R_tof = make_R_tof_array(self.momenta, self.remi_params, self.particle_params)
        self.fig_R_tof, self.ax_R_tof, self.canvas_R_tof = self.make_plot_xarray(self.R_tof, 104, 110, self.R_tof_plot_group, sorting=True, sort='time', columnspan=2, color='powderblue')  
        
    def R_tof_sim(self):
        tof_max = float(self.ENTRY_TOF.get())*1e-9
        tof = np.linspace(0, tof_max, int(tof_max*100e9))
        print(tof)
        print(len(self.ax_R_tof.lines))
        while len(self.ax_R_tof.lines)>1:
            self.ax_R_tof.lines[-1].remove()

        if len(self.ENTRY_KIN_ENERGY_1.get())!=0:
            energy_1 = float(self.ENTRY_KIN_ENERGY_1.get())*1.6e-19
            mass_1 = float(self.ENTRY_MASS_1.get())*m_e
            charge_1 = float(self.ENTRY_CHARGE_1.get())*q_e
            particle_params_1 = (mass_1, charge_1)
            K_1 = np.sqrt(2*m_e*energy_1)
            R_1 = calc_R_fit(K_1, tof, self.remi_params, particle_params_1)
            self.ax_R_tof.plot(tof, R_1, color='firebrick')
            
        if len(self.ENTRY_KIN_ENERGY_2.get())!=0:
            energy_2 = float(self.ENTRY_KIN_ENERGY_2.get())*1.6e-19
            mass_2 = float(self.ENTRY_MASS_2.get())*m_e
            charge_2 = float(self.ENTRY_CHARGE_2.get())*q_e
            particle_params_2 = (mass_2, charge_2)
            K_2 = np.sqrt(2*m_e*energy_2)
            R_2 = calc_R_fit(K_2, tof, self.remi_params, particle_params_2)
            self.ax_R_tof.plot(tof, R_2, color='deepskyblue')

        if len(self.ENTRY_KIN_ENERGY_3.get())!=0:
            energy_3 = float(self.ENTRY_KIN_ENERGY_3.get())*1.6e-19
            mass_3 = float(self.ENTRY_MASS_3.get())*m_e
            charge_3 = float(self.ENTRY_CHARGE_3.get())*q_e
            particle_params_3 = (mass_3, charge_3)
            K_3 = np.sqrt(2*m_e*energy_3)
            R_3 = calc_R_fit(K_3, tof, self.remi_params, particle_params_3)
            self.ax_R_tof.plot(tof, R_3, color='darkorange')
        self.canvas_R_tof.draw()
        
    def calc_position(self):
        U, B, l_a = self.remi_params
        q, m = self.particle_params
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
        x,y = self.calc_position()
        self.make_plot(x, y, 106, 105, self.position_group)
    
    def export_data(self):
        x, y = self.calc_position()
        tof = self.R_tof.time
        data = np.array([x, y, tof])
        data = data.T
        with open('pos_data.txt', 'w') as datafile:
            np.savetxt(datafile, data, fmt=['%.3E','%.3E','%.3E'])
            
    def export_momenta(self):
        p_x = self.momenta[:,0,0]
        p_y = self.momenta[:,0,1]
        p_z = self.momenta[:,0,2]
        mom = np.array([p_x, p_y, p_z])
        mom = mom.T
        with open('mom_data.txt', 'w') as datafile:
            np.savetxt(datafile, mom, fmt=['%.3E','%.3E','%.3E'])
            
    def calc_mcp(self):
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
        
    
window = Tk()
window.configure(background = 'whitesmoke')
start = mclass(window)
window.mainloop()
