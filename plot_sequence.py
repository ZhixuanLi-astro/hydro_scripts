"""
This file plot the time evoution
"""
import os
import sys
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
# import dynamo as dyn
from scipy.integrate import odeint,ode,quad
from scipy import optimize
import astropy.constants as cons
#import plot_mesh
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.colors import LogNorm,Normalize, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

#ys.path.insert(0, '/home/yu/Programs/Athena/athena-df_20230314/vis/python')
sys.path.insert(0, '/home/izx/athena_sublimation/vis/python')
import athena_read
import re
from scipy.integrate import solve_ivp, odeint
import pickle
import random as rd
from scipy.stats import gaussian_kde, norm
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ArrowStyle
from matplotlib.patches import FancyArrowPatch, Arrow
from copy import deepcopy

from preplot import pol2car, car2pol, dfdx_2pts, dfdx_5pts, dfdx_7pts, curl_in_polar_rlog,v_Intpl_Sph2car,scaler_Intpl_Sph2car
plt.rcParams.update({'font.size': 15})
AU = cons.au.cgs.value
YR = (365.2425*24*3600)
M_sun = cons.M_sun.cgs.value
M_e = cons.M_earth.cgs.value
M_j = cons.M_jup.cgs.value
GM_sun = cons.GM_sun.cgs.value
GM_e = cons.GM_earth.cgs.value
L_sun = cons.L_sun.cgs.value
R_sun = cons.R_sun.cgs.value
sigma_sb = cons.sigma_sb.cgs.value

# disk slope
T_slope = -0.5
Cs_slope = T_slope/2
H_slope = Cs_slope + 1.5
sigma_slope = -(Cs_slope + H_slope)

rho_slope = sigma_slope - H_slope
p_slope = T_slope + rho_slope

# disk parameter
M_star = 1.0 # Msun
# M_p = 2.4*M_j/M_e # M_e
a_semi = 3.0 # au, semi-major axis
a0 = 3.0 # r0, reference position of disk temperature/density profile
T_profile = lambda r: 150.0*(r/3.0)**(T_slope)
# H_profile = lambda r: 0.033*AU*(r/1.0)**(H_slope)
# default value
T0 = T_profile(a0) # Temperature at planet position
Mdot_gas = 1.e-8*M_sun/YR
alpha = 3.e-3

Cs0 = sqrt(cons.k_B.cgs.value*T0/(2.34*cons.m_p.cgs.value))
UNIT_T = (365.2425*24*3600)/(2*pi)*(a0)**(1.5)*M_star**(-0.5) # 1/omega at planet position
Sigma0 = Mdot_gas/(3.0*pi*alpha*Cs0**2*UNIT_T) # gas surface density at planet position
sigma_profile = lambda r: Sigma0*(r/a0)**(sigma_slope)

# print("d_slope=",rho_slope) # midplane gas density slope
# print("p_over_d_slope=",p_slope-rho_slope)
# print("sigma0=",Sigma0)

# disk slope
T_slope = -0.5
Cs_slope = T_slope/2
H_slope = Cs_slope + 1.5
sigma_slope = -(Cs_slope + H_slope)

rho_slope = sigma_slope - H_slope
p_slope = T_slope + rho_slope

# disk parameter
M_star = 1.0 # Msun
# M_p = 2.4*M_j/M_e # M_e
a_semi = 3.0 # au, semi-major axis
a0 = 3.0 # r0, reference position of disk temperature/density profile
T_profile = lambda r: 150.0*(r/3.0)**(T_slope)
# H_profile = lambda r: 0.033*AU*(r/1.0)**(H_slope)
# default value
T0 = T_profile(a0) # Temperature at planet position
Mdot_gas = 1.e-8*M_sun/YR
alpha = 3.e-3

Cs0 = sqrt(cons.k_B.cgs.value*T0/(2.34*cons.m_p.cgs.value))
UNIT_T = (365.2425*24*3600)/(2*pi)*(a0)**(1.5)*M_star**(-0.5) # 1/omega at planet position
Sigma0 = Mdot_gas/(3.0*pi*alpha*Cs0**2*UNIT_T) # gas surface density at planet position
sigma_profile = lambda r: Sigma0*(r/a0)**(sigma_slope)

# print("d_slope=",rho_slope) # midplane gas density slope
# print("p_over_d_slope=",p_slope-rho_slope)
# print("sigma0=",Sigma0)

# global dimensionless quantity
mu_He = 4
mu_H2 = 2
mu_xy = 2.34

UNIT_V = sqrt(cons.k_B.cgs.value*T0/(mu_xy*cons.m_p.cgs.value))
UNIT_L = UNIT_V*UNIT_T  # scale height at reference poistion
UNIT_DEN = Sigma0/(sqrt(2*pi)*UNIT_L)
UNIT_Fm = (UNIT_L**3*UNIT_DEN/UNIT_T)/(M_sun/YR)
UNIT_PRS = UNIT_DEN*UNIT_V**2
kB_mp_cgs = cons.k_B.cgs.value/cons.m_p.cgs.value
kB_mp = cons.k_B.cgs.value/cons.m_p.cgs.value/(UNIT_V**2)

class chem:
    name = ''

    def __init__(self,name,mu,T_a,P_eq,L_heat):
        self.name = name
        self.mu = mu
        self.T_a = T_a
        self.P_eq = P_eq
        self.R = kB_mp_cgs/mu
        self.L_heat = L_heat

# gas property
# water:
mu_water = 18
P_eq_water = 1.14e13
L_heat_water = 2.75e10
R_water = kB_mp_cgs/mu_water
T_a_water = 6062
chem_H2O = chem('H2O',18,T_a_water,P_eq_water,L_heat_water)

mu_z = chem_H2O.mu
P_eq0 = chem_H2O.P_eq / UNIT_PRS
L_heat = chem_H2O.L_heat / UNIT_V**2
T_a = chem_H2O.T_a
# print("P_eq0=%.10e"%(P_eq0))
# print("L_heat=",L_heat)
# print("T_a=",T_a)

# dimensionless quantity used in intial set-up.
L_norm = (AU/UNIT_L)
r0 = a0*L_norm
# required resolution
rin = 1.0*L_norm
rout = 4.0*L_norm
Nrad = 300

GM = (r0)**3
tlim = 2e5*YR/UNIT_T
dt = 1e3*YR/UNIT_T

def face_f_2_cos(x2min,x2max,cell_width_ratio,num_face):
    x = linspace(0,1,num_face)
    w = arccos(1-x)/(pi/2)
    tmp = w*(x2max-x2min) + x2min
    
    return tmp


def face_f_2_power(x2min,x2max,cell_width_ratio,num_face):
    x = linspace(0,1,num_face)
    w = (x)**(1/3)
    tmp = w*(x2max-x2min) + x2min
    
    return tmp

def Get_kappa(kappa0, d2g, fv):
    return kappa0*(1.0-fv)*UNIT_DEN*UNIT_L


def formatnum(x,pos):
    return '$10^{%.0f}$' % (log10(x))

# Set global font properties
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'  # Replace with your chosen font
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 15})
from copy import deepcopy
half = False

idxs = 0  
idxe = 139 

temL = []
timeL = []
q_diffL = []
q_zL = []
r_want = 150 
phi_want = -1   

try : 
    read = sys.argv[2]
except:
    read = False

if not read:
    print ("r_want= ", r_want)
    print ("phi_want= ", phi_want)
    from tqdm import tqdm
    for nstep in tqdm(range(idxs, idxe)):
        if nstep == 1000: 
            continue
        dT = 1
        # read the data
        try: 
            filename = sys.argv[1]
            DIR = './../' + filename +'/'
        except:
            DIR = '/home/izx/athena_works/snowline_test/'
        # DIR = '/home/yu/Programs/Athena/work/output/snowline_2D/output4/'
        # DIR = '/mnt/disk1/dataYu/output/snowline_2D/output37/'
        #----------------------------------------
        # primitive data read
        #----------------------------------------
        filename = DIR+'iceline.out1.'+str(nstep).rjust(5,'0')+'.athdf'
        # print("\rReading file: ", filename, end='')
        data_prim= athena_read.athdf(filename,face_func_2=face_f_2_power, num_ghost=0)
        if nstep == idxs:
            rad = data_prim['x1v']/ L_norm
            theta = data_prim['x2v']
            phi = data_prim['x3v']

        simu_time = data_prim['Time']

        data_uov= athena_read.athdf(DIR+'iceline.out2.'+str(nstep).rjust(5,'0')+'.athdf',face_func_2=face_f_2_power, num_ghost=0)
        tem = data_uov['Tem']
        tem_xz = tem[0,:,:].T
        q_diff = data_uov['q_diff']
        q_diff_xz = q_diff[0,:,:].T
        q_z = data_uov['q_z']
        q_z_xz = q_z[0,:,:].T

        if any(q_diff_xz)<0 :
            import pdb; pdb.set_trace()
            print('q_diff < 0 is detected at {}'.format(argwhere(q_diff_xz<0)))

        if any (q_z_xz) <0: 
            import pdb; pdb.set_trace()

        #get the temperature at ~3AU and midplane 
        temL.append(tem_xz[r_want, phi_want])
        timeL.append(simu_time)
        q_diffL.append(q_diff_xz[r_want, phi_want])
        q_zL.append(q_z_xz[r_want, phi_want])

    temL = array(temL)
    timeL = array(timeL)
    q_diffL = array(q_diffL)
    q_zL = array(q_zL)
    savez("./tem_time/tem_time_{}_{}.npz".format(r_want, phi_want), timeL=timeL, temL=temL, q_diffL=q_diffL, q_zL = q_zL)
else:
    pass
    # data = load("./tem_time/tem_time.npz")
    # timeL = data['timeL']
    # temL = data['temL']
    # q_diffL = data['q_diffL']
    # print("data loaded from tem_time.npz")


#plot the temperature evolution
import cgs
fg, axL = plt.subplots(1,2, figsize=(12,4))
# axL[0].set_xlim(9000, 13850)
# axL[1].set_xlim(9000, 13850)
# axL[1].axvline(10000)
colL = ['r', 'g', 'b', 'k', 'gray', 'orange']
axL[1].set_ylim(-0.03, 0.05)
if read: 
    for i, r in enumerate([150, 151, 152, 153,154,155]):
        data = load("./tem_time/tem_time_{}_{}.npz".format(r,phi_want))
        timeL = data['timeL']
        temL = data['temL']
        q_diffL = data['q_diffL']
        q_zL = data['q_zL']
        axL[0].plot(timeL, temL,c=colL[i] ,label = "r= "+ str(r))
        axL[1].plot(timeL, q_diffL,c=colL[i])
        axL[1].plot(timeL, q_zL, c=colL[i], alpha=0.3)
        import pdb; pdb.set_trace()


    axL[0].legend()
    axL[1].legend()
    plt.savefig("Tem_time.png", dpi=300)
    plt.close()


# plt.figure(figsize=(9,6))
# plt.savefig("qdiff_time.png", dpi=300)
# plt.close()
#
import pdb; pdb.set_trace()
