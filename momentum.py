# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

#############################
#### constants ##############
############################

# SI
m_e = 9.10938356e-31          # electron_mass
q_e = 1.6021766208e-19           # electron_charge
l_a = 0.188         # acc_length 
U = 190            # electic_field 
B = 5*1e-4           # magnetic_field 
omega = q_e * B / m_e

# atomic units
#
#m_e = 1          # electron_mass
#q_e = 1          # electron_charge
#l_a = 0.188 / 5.29e-11       # acc_length 
#U = 190 / 27.2         # electic_field 
#B = 0.5 / 2.35e9       # magnetic_field 
#omega = q_e * B / m_e

def make_momentum(number_of_momenta=1000):
    momentum = np.random.randn(number_of_momenta, 1, 3)*2e-24
    return momentum

def calc_tof(momentum):
    p_z = momentum[:,0,2]
    D = p_z**2 + 2 * q_e * U * m_e
    tof = ((p_z) - np.sqrt(D))/(-q_e*U)*l_a
    return tof

def calc_R(momentum):
    p_x = momentum[:,0,0]
    p_y = momentum[:,0,1]
    p_xy = np.sqrt(p_x**2 + p_y**2)
    tof = calc_tof(momentum)
    R = (2*p_xy*np.abs(np.sin(omega*tof/2)))/(q_e*B)
    return R

def make_data_array(momentum):
    tof = calc_tof(momentum)
    R = calc_R(momentum)
    ar = xr.DataArray(R, coords=[tof], dims=["time"])
    return ar

def calc_R_fit(K, tof):
    D = K**2 - (m_e*l_a/tof - U*q_e*tof/(2*l_a))**2
    R = 2/(m_e*omega) * np.sqrt(D) * np.abs(np.sin(omega*tof/2))
    return R

#### EXAMPLE ##########
mom = make_momentum(10000)
print(mom)
tof = calc_tof(mom)
R = calc_R(mom)

ar = make_data_array(mom)
ar.sortby('time').plot.line('b.')

t = (ar.sortby('time').time)
rad = calc_R_fit(2e-24*np.sqrt(3),t)
print(rad)
plt.plot(t, rad, 'ro')

