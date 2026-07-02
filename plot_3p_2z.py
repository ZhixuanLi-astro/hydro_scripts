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
from preplot import mDiv
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
import sys
import re

def parse_athena_input(filename):
    """
    Parse a typical Athena++ input file (sections in <...>) and return
    a dictionary where keys are section names and values are dictionaries
    of key=value pairs within that section.
    """
    data = {}
    current_section = None
    comment_pattern = re.compile(r'^\s*#')  # lines starting with optional spaces and '#'

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or comment_pattern.match(line):
                continue

            # Check for section header like <section>
            if line.startswith('<') and line.endswith('>'):
                current_section = line[1:-1].strip()
                data[current_section] = {}
                continue

            # Parse key = value within a section
            if current_section is not None and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                data[current_section][key] = value

    return data

#get the file number by args 

try: 
    filenum = sys.argv[1]
except:
    print ("please specify a filenumber")
    sys.exit()

dT = 1
# read the data
try: 
    filename = sys.argv[2]
    DIR = './../' + filename +'/'
except:
    DIR = '/home/izx/athena_works/snowline_test/'


infile = DIR + 'athinput.iceline' 
try:
    parsed = parse_athena_input(infile)
except FileNotFoundError:
    print("Error: File '{}' not found.".format(infile))
    sys.exit(1)
except Exception as err:
    print("Error parsing file: {}".format(err))
    sys.exit(1)

problem_data = parsed.get('problem', {}) 
units = parsed.get('units', {})
dust = parsed.get('dust', {}) 

dust_fluids = [key for key in dust.keys() if key.startswith('initial_D2G_')]
n_df = len(dust_fluids)

#units 
UNIT_M = float(units['mass_cgs'])
UNIT_L = float(units['length_cgs'])
UNIT_T = float(units['time_cgs'])

# disk slope
T_slope = float(problem_data['Tslope'])
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
H_profile = lambda r: 0.033*AU*(r/1.0)**(H_slope)
# default value
T0 = T_profile(a0) # Temperature at planet position
Mdot_gas = 1.e-8*M_sun/YR
alpha = 3.e-3

Cs0 = sqrt(cons.k_B.cgs.value*T0/(2.34*cons.m_p.cgs.value))
# UNIT_T = (365.2425*24*3600)/(2*pi)*(a0)**(1.5)*M_star**(-0.5) # 1/omega at planet position
Sigma0 = Mdot_gas/(3.0*pi*alpha*Cs0**2*UNIT_T) # gas surface density at planet position
sigma_profile = lambda r: Sigma0*(r/a0)**(sigma_slope)

# print("d_slope=",rho_slope) # midplane gas density slope
# print("p_over_d_slope=",p_slope-rho_slope)
# print("sigma0=",Sigma0)

# global dimensionless quantity
mu_He = 4
mu_H2 = 2
mu_xy = 2.34

UNIT_V = UNIT_L/UNIT_T 
UNIT_Sigma = Sigma0 
UNIT_DEN = UNIT_M/(UNIT_L**3)
# UNIT_V = sqrt(cons.k_B.cgs.value*T0/(mu_xy*cons.m_p.cgs.value))
# UNIT_L = UNIT_V*UNIT_T  # scale height at reference poistion
# UNIT_DEN = Sigma0  #in 1d simulation the density is surface density
# UNIT_M = UNIT_DEN*UNIT_L**2/sqrt(2*pi)
UNIT_Fm = (UNIT_M/UNIT_T)/(M_sun/YR)
UNIT_PRS = UNIT_Sigma*UNIT_V**2
kB_mp_cgs = cons.k_B.cgs.value/cons.m_p.cgs.value
kB_mp = cons.k_B.cgs.value/cons.m_p.cgs.value/(UNIT_V**2)

# print("UNIT_T=%.10e"%(UNIT_T))
# print("UNIT_V=%.10e"%(UNIT_V))
# print("UNIT_L=%.10e"%(UNIT_L))
# print("UNIT_DEN=%.10e"%(UNIT_DEN))
# print("UNIT_Fm=%.10e"%(UNIT_Fm))
# print("UNIT_PRS=%.10e"%(UNIT_PRS))
# print("KELVIN=%.10e"%(1/kB_mp))


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
    return kappa0*(1.0-fv)*UNIT_Sigma*UNIT_L

def get_mu(fv):
    return 1.0/(1.0/mu_xy*(1.0-fv)+fv/mu_z)

def formatnum(x,pos):
    return '$10^{%.0f}$' % (log10(x))

# Set global font properties
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.serif'] = 'Times New Roman'  # Replace with your chosen font
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 15})
from copy import deepcopy
half = False

N_P = 2 
N_Z = 2
# DIR = '/home/yu/Programs/Athena/work/output/snowline_2D/output4/'
# DIR = '/mnt/disk1/dataYu/output/snowline_2D/output37/'
#----------------------------------------
# primitive data read
#----------------------------------------
filename = DIR+'iceline.out1.'+str(filenum).rjust(5,'0')+'.athdf'
print("Reading file: ", filename)
data_prim= athena_read.athdf(filename,face_func_2=face_f_2_power, num_ghost=2)
rad = data_prim['x1v']/ L_norm
theta = data_prim['x2v']
phi = data_prim['x3v']

phi_f = data_prim['x3f']
phi_f[-1] = phi_f[0]
phi[-1] = phi[0] = 0.0
theta_f = data_prim['x2f']
rad_f = data_prim['x1f']/ L_norm

simu_time = data_prim['Time']

#get dt 
hstname = DIR+'iceline.hst'
# data_hst = athena_read.hst(hstname)
# dt = data_hst['dt'][int(filenum)-1]

# filename = DIR+'iceline.out5.'+str(nstep).rjust(5,'0')+'.athdf'
# data_cons= athena_read.athdf(filename,face_func_2=face_f_2_power, num_ghost=0)
# rho = data_cons['dens']
## (phi*theta*R)
rho = data_prim['rho']
prs = data_prim['press']
vx1 = data_prim['vel1']
vx2 = data_prim['vel2']
vx3 = data_prim['vel3']

rhod = {}
v1d = {}
v2d = {}
v3d = {}


dust_1_rho = data_prim['dust_1_rho']
dust_1_vx1 = data_prim['dust_1_vel1']
dust_1_vx2 = data_prim['dust_1_vel2']
dust_1_vx3 = data_prim['dust_1_vel3']

dust_2_rho = data_prim['dust_2_rho']
dust_2_vx1 = data_prim['dust_2_vel1']
dust_2_vx2 = data_prim['dust_2_vel2']
dust_2_vx3 = data_prim['dust_2_vel3']

dust_3_rho = data_prim['dust_3_rho']
dust_3_vx1 = data_prim['dust_3_vel1']
dust_3_vx2 = data_prim['dust_3_vel2']
dust_3_vx3 = data_prim['dust_3_vel3']

dust_4_rho = data_prim['dust_4_rho']
dust_4_vx1 = data_prim['dust_4_vel1']
dust_4_vx2 = data_prim['dust_4_vel2']
dust_4_vx3 = data_prim['dust_4_vel3']
dust_5_rho = data_prim['dust_5_rho']
dust_5_vx1 = data_prim['dust_5_vel1']
dust_5_vx2 = data_prim['dust_5_vel2']
dust_5_vx3 = data_prim['dust_5_vel3']

dust_6_rho = data_prim['dust_6_rho']
dust_6_vx1 = data_prim['dust_6_vel1']
dust_6_vx2 = data_prim['dust_6_vel2']
dust_6_vx3 = data_prim['dust_6_vel3']
dust_7_rho = data_prim['dust_7_rho']
dust_7_vx1 = data_prim['dust_7_vel1']
dust_7_vx2 = data_prim['dust_7_vel2']
dust_7_vx3 = data_prim['dust_7_vel3']

dust_8_rho = data_prim['dust_8_rho']
dust_8_vx1 = data_prim['dust_8_vel1']
dust_8_vx2 = data_prim['dust_8_vel2']
dust_8_vx3 = data_prim['dust_8_vel3']

dust_9_rho = data_prim['dust_9_rho']
dust_9_vx1 = data_prim['dust_9_vel1']
dust_9_vx2 = data_prim['dust_9_vel2']
dust_9_vx3 = data_prim['dust_9_vel3']

dust_10_rho = data_prim['dust_10_rho']
dust_10_vx1 = data_prim['dust_10_vel1']
dust_10_vx2 = data_prim['dust_10_vel2']
dust_10_vx3 = data_prim['dust_10_vel3']

#-----------------------------------------
# user defined variable read
# #---------------------------------------
data_uov= athena_read.athdf(DIR+'iceline.out2.'+str(filenum).rjust(5,'0')+'.athdf',face_func_2=face_f_2_power, num_ghost=2)
tem = data_uov['Tem']
# dif = data_uov['dif']
gas_nu = data_uov['dif']

try: 
    drhodt_0 = data_uov['drho_i_dt']
    drhodt_1 = data_uov['drho_i1_dt']
    drhodt_v = data_uov['drho_v_dt']
except:
    print("no drho found")
    drhodt_0 = zeros_like(rho)
    drhodt_1 = zeros_like(rho)
    drhodt_v = zeros_like(rho)

st = data_uov['st_1']
st1 = data_uov['st_2']
st2 = data_uov['st_3']

# tem_equi = data_uov['st']
m_p = data_uov['m_p_1']
m_p1 = data_uov['m_p_2']
m_p2 = data_uov['m_p_3']
s_p = data_uov['s_p_1']
s_p1 = data_uov['s_p_2']
s_p2 = data_uov['s_p_3']
# dfvdt = data_uov['dfvdt']
flx_vap_x1 = data_uov['flx_vap_x1']
flx_x1 = data_uov['flx_x1']

# for multiple dustfluids
flx_sil_x1 = data_uov['flx_sil_x1_1']
flx_sil1_x1 = data_uov['flx_sil_x1_2']

try: 
    mmax = data_uov['mmax']
except:
    mmax = zeros_like(rho) 

try:
    dif_0 = data_uov['dif_sil_1']
    dif_1 = data_uov['dif_sil_2']
except:
    dif_0 = None
    dif_1 = None

try: 
    pres_eq = data_uov['pres_vap']
except:
    pres_eq = None

#get the density change by phase_change process 
drho_exist = True
# gamma = data_uov['gamma']

# face coordinate
index_phi = 0
THETA, PHI, R = meshgrid(theta_f,phi_f,rad_f)
x = R* sin(THETA) * cos(PHI)
y = R* sin(THETA) * sin(PHI)
z = R* cos(THETA)
x_xz = x[index_phi,:,:].T
y_xz = z[index_phi,:,:].T

# cell center coordinate
THETA, PHI, R = meshgrid(theta,phi,rad)
x = R* sin(THETA) * cos(PHI)
y = R* sin(THETA) * sin(PHI)
z = R* cos(THETA)
# x_xz_c = x[index_phi,:,:].T
# y_xz_c = z[index_phi,:,:].T
x_xz_c = x_xz[1:,1:]
y_xz_c = y_xz[1:,1:]

# cell area
dR = data_prim['x1f'][1:]-data_prim['x1f'][0:-1]
dtheta = data_prim['x2f'][1:]-data_prim['x2f'][0:-1]
dphi = array([2.0*pi])
dtheta_3D, dphi_3D, dR_3D = meshgrid(dtheta,dphi, dR)
theta_3D, phi_3D, R_3D = meshgrid(data_prim['x2v'],array([pi]),data_prim['x1v'])

dS_R = R_3D**2 *sin(theta_3D) * dtheta_3D* dphi_3D
dS_theta = R_3D*sin(theta_3D) * dR_3D* dphi_3D
dS_phi = R_3D*dR_3D*dtheta_3D

# unit of flux
flx_vap_x1 *= 2*pi*rad_f[:-1]* UNIT_Fm * L_norm
flx_x1 *= 2*pi*rad_f[:-1]* UNIT_Fm * L_norm 

flx_sil_x1 *= 2*pi*rad_f[:-1]* UNIT_Fm *L_norm 
flx_sil1_x1 *= 2*pi*rad_f[:-1]* UNIT_Fm *L_norm
# slices
index_phi = 0
rho_xz = rho[index_phi,:,:].T
dust_1_rho_xz = dust_1_rho[index_phi,:,:].T
dust_2_rho_xz = dust_2_rho[index_phi,:,:].T
dust_3_rho_xz = dust_3_rho[index_phi,:,:].T
# prs_xz = prs[index_phi,:,:].T
tem_xz = tem[index_phi,:,:].T
# tem_equi_xz = tem_equi[index_phi,:,:].T
st_xz = st[index_phi,:,:].T
# dif_xz = dif[index_phi,:,:].T
# dfvdt_xz = dfvdt[index_phi,:,:].T
m_p_xz = m_p[index_phi,:,:].T
m_p1_xz = m_p1[index_phi,:,:].T
s_p_xz = s_p[index_phi,:,:].T


#the relaxation timescale 
try:
    t_relax = data_uov['t_relax']
except:
    t_relax = None

if all(tem == 0.):
    tem[0,0] = T_profile(rad)

#knowing the P_eq and Tem, we can get the saturation vapor pressure:
P_e = P_eq0*UNIT_PRS*exp(-T_a/tem[0,0])/UNIT_DEN*UNIT_Sigma  # this is very strange, since in athena we also use the UNIT_PRS = UNIT_DEN*UNIT_V**2
#then get the density 
omega_K_cgs = sqrt(GM_sun/(rad*AU)**3)
cs2_cgs = kB_mp_cgs*tem[0,0,:]/mu_z

H_gas_cgs = cs2_cgs**0.5/omega_K_cgs 

rho_e = P_e/cs2_cgs
sigma_e = rho_e*H_gas_cgs*sqrt(2*pi)

#let's unify the color scheme:
colD = {'ga':'black', 'ss':'tab:orange', 'ms':'tab:orange', 'ls':'tab:orange', 'si':'tab:blue', 'mi':"tab:blue" ,'li':'tab:cyan', 'va':'tab:purple'}
lwD  = {'ga':2, 'ss':1, 'ms':3, 'ls':4, 'si':2, 'mi':3, 'li':5, 'va':2}  
alpD = {'ga':1.0, 'ss':1., 'ms': 0.8, 'ls':0.5, 'si':1., 'mi':0.8, 'li':0.5, 'va':1.}

legend_handles = [
    Line2D([0], [0], color=colD['ga'], lw=lwD['ga'], alpha=alpD['ga'], label='gas') ,
    Line2D([0], [0], color=colD['va'], lw=lwD['va'], alpha=alpD['va'], label='vapor'),
    Line2D([0], [0], color=colD['ss'], lw=lwD['ss'], alpha=alpD['ss'], label='sil0') ,
    Line2D([0], [0], color=colD['ms'], lw=lwD['ms'], alpha=alpD['ms'], label='sil1') ,
    Line2D([0], [0], color=colD['ls'], lw=lwD['ls'], alpha=alpD['ls'], label='sil2') ,
    Line2D([0], [0], color=colD['si'], lw=lwD['si'], alpha=alpD['si'], label='ice0') ,
    Line2D([0], [0], color=colD['mi'], lw=lwD['mi'], alpha=alpD['mi'], label='ice1') ,
    Line2D([0], [0], color=colD['li'], lw=lwD['li'], alpha=alpD['li'], label='ice2') ,
]

rad_noghost = rad[2:-2]

rad_ticks = [1, 2, 3, 4]
rad_ticklabels = ['1', '2', '3', '4']

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0,1].set_title("time: {:.2f} yr".format(simu_time*UNIT_T/YR),loc= 'right', y=1.1)
axs[0, 0].set_xscale('log')
axs[0, 0].set_ylim(1.e-3, 2.e3)
axs[0, 0].plot(rad, rho_xz[:, 0]*UNIT_Sigma,        c = colD['ga'], lw=lwD['ga'], alpha = alpD['ga'], label='gas')
axs[0, 0].plot(rad, dust_2_rho[0, 0]*UNIT_Sigma, c = colD['ss'], lw=lwD['ss'], alpha = alpD['ss'], label='sil0')
axs[0, 0].plot(rad, dust_1_rho[0, 0]*UNIT_Sigma, c = colD['si'], lw=lwD['si'], alpha = alpD['si'], label='sil1')
axs[0, 0].plot(rad, dust_4_rho[0, 0]*UNIT_Sigma,    c = colD['ms'], lw=lwD['ms'], alpha = alpD['ms'], label='sil2')
axs[0, 0].plot(rad, dust_3_rho[0, 0]*UNIT_Sigma,    c = colD['mi'], lw=lwD['mi'], alpha = alpD['mi'], label='vapor')
axs[0, 0].plot(rad, dust_6_rho[0, 0]*UNIT_Sigma,    c = colD['ls'], lw=lwD['ls'], alpha = alpD['ls'], label='sil2')
axs[0, 0].plot(rad, dust_5_rho[0, 0]*UNIT_Sigma,    c = colD['li'], lw=lwD['li'], alpha = alpD['li'], label='vapor')
axs[0, 0].plot(rad, dust_7_rho[0, 0]*UNIT_Sigma,    c = colD['va'], lw=lwD['va'], alpha = alpD['va'], label='vapor')
axs[0, 0].set_yscale('log')
axs[0, 0].set_xlabel('r (au)')
axs[0, 0].set_ylabel(r'$\Sigma $(g/cm$^2$)')
axn = axs[0, 0].twinx()
axn.set_yscale('log')
axn.set_ylabel('number density (1/cm$^2$)')
axn.plot(rad, dust_8_rho[0,0]/UNIT_L**2, '--', lw=lwD['ss'], alpha = alpD['ss'], c = 'gray') 
axn.plot(rad, dust_9_rho[0,0]/UNIT_L**2, '--', lw=lwD['ms'], alpha = alpD['ms'], c = 'gray')
axn.plot(rad, dust_10_rho[0,0]/UNIT_L**2, '--', lw=lwD['ls'], alpha = alpD['ls'], c = 'gray')

axs[0, 1].set_xscale('log')
axs[0, 1].set_yscale('symlog', linthresh=1e-8)
axs[0, 1].set_ylim(-200, -1)
axs[0, 1].plot(rad, vx1[0, 0, :]*UNIT_V,             c = colD['ga'],  label='gas v2')
axs[0, 1].plot(rad, dust_2_vx1[0, 0, :]*UNIT_V, '-', lw=lwD['ss'], alpha=alpD['ss'], c = 'gray')
axs[0, 1].plot(rad, dust_4_vx1[0, 0, :]*UNIT_V, '-', lw=lwD['ms'], alpha=alpD['ms'], c = 'gray')
axs[0, 1].plot(rad, dust_6_vx1[0, 0, :]*UNIT_V, '-', lw=lwD['ls'], alpha=alpD['ls'], c = 'gray')
axs[0, 1].set_xlabel('r (au)')
axs[0, 1].set_ylabel('v_r (cm/s)')

omega_K_cgs = sqrt(GM_sun/(rad*AU)**3)
omega_K = omega_K_cgs*UNIT_T
vK_cgs = sqrt(GM_sun/(rad*AU))

#calulate the analitical raidial velocity
cs_cgs = sqrt(kB_mp_cgs*tem[0,0,:]/2.34)
H_gas_cgs = cs_cgs/omega_K_cgs 
eta_cgs = 0.5*(H_gas_cgs/rad/AU)**2 * (-0.5-1)
gas_nu_cgs = gas_nu[0,0,:]*UNIT_V*UNIT_L
gas_acc_vel_cgs = -1.5*gas_nu_cgs/rad/AU 

tau_0 = st[0,0,:]*omega_K 
v0_ana = (vx1[0,0,:]*UNIT_V+ 2.0*tau_0*eta_cgs*vK_cgs)/(1.0 + tau_0**2) 

tau_1 = st1[0,0,:]*omega_K 
v1_ana = (vx1[0,0,:]*UNIT_V+ 2.0*tau_1*eta_cgs*vK_cgs)/(1.0 + tau_1**2)

tau_2 = st2[0,0,:]*omega_K 
v2_ana = (vx1[0,0,:]*UNIT_V+ 2.0*tau_2*eta_cgs*vK_cgs)/(1.0 + tau_2**2)

axs[0, 1].plot(rad, v0_ana, 'k--', label='analytical')
axs[0, 1].plot(rad, v1_ana, 'k--')
axs[0, 1].plot(rad, v2_ana, 'k--')


axs[1, 0].set_yscale('log')
axs[1, 0].set_xscale('log')
axs[1, 0].plot(rad, st[0, 0, :]*omega_K, '-', c = 'gray', lw=lwD['ss'], alpha=alpD['ss'], label='small')
axs[1, 0].plot(rad, st1[0, 0, :]*omega_K,'-', c = 'gray', lw=lwD['ms'], alpha=alpD['ms'], label='mid')
axs[1, 0].plot(rad, st2[0, 0, :]*omega_K,'-', c = 'gray', lw=lwD['ls'], alpha=alpD['ls'], label='large')
# axs[1, 0].set_yscale('log')
axs[1, 0].set_xlabel('r (au)')
axs[1, 0].set_ylabel('Stokes number')

axm = axs[1, 0].twinx() 
axm.set_yscale('log')   
axm.set_ylabel('pebble mass (g)')
axm.plot(rad, m_p_xz.T[0], '--', c = 'gray', lw=lwD['ss'], alpha=alpD['ss'])
axm.plot(rad, m_p1_xz.T[0],'--', c = 'gray', lw=lwD['ms'], alpha=alpD['ms'])
axm.plot(rad, m_p2[0,0,:], '--', c = 'gray', lw=lwD['ls'], alpha=alpD['ls'])

axs[1, 1].set_xscale('log')
# axs[1, 1].plot(rad, drhodt_0[0,0]*UNIT_T/UNIT_Sigma, '-', c = colD['si'], lw=lwD['si'], alpha=alpD['si'])
# axs[1, 1].plot(rad, drhodt_1[0,0]*UNIT_T/UNIT_Sigma, '-', c = colD['li'], lw=lwD['li'], alpha=alpD['li'])
# axs[1, 1].plot(rad, drhodt_v[0,0]*UNIT_T/UNIT_Sigma, '-', c = colD['va'], lw=lwD['va'], alpha=alpD['va'])
# axs[1, 1].plot(rad, drhodt_0[0,0]+drhodt_1[0,0]+drhodt_v[0,0], 'k--', label='total')
#
# axs[1, 1].set_xlabel('r (au)')
# axs[1, 1].set_ylabel(r'$\dot{\rho}$ (g/cm$^3$/s)')

legend_handles_panel4 = [
    Line2D([0], [0], color='k', label=r'$\dot{\rho}$'),
    Line2D([0], [0], ls='--', color = 'grey', label='total'),
    Line2D([0], [0], color='grey', lw=5, alpha=0.5, label='ratio'),
]

# norm_drho_0 = drhodt_0[0,0]/dust_6_rho[0,0]/s_p[0,0]**2*UNIT_T/UNIT_Sigma*UNIT_L**2
# norm_drho_1 = drhodt_1[0,0]/dust_7_rho[0,0]/s_p1[0,0]**2*UNIT_T/UNIT_Sigma*UNIT_L**2
# axnorm = axs[1, 1].twinx()
# axnorm.plot(rad, norm_drho_0/norm_drho_1, '-', c = 'grey', lw=5, alpha=0.5, label='large ice')
# axnorm.set_yticks([0.0,0.5,  1, 1.5, 2.0])
# axnorm.set_ylim(0, 2)

axs[1, 1].legend(handles=legend_handles_panel4, loc='upper left', frameon=True, fontsize=12)

legend_handles_panel1 = [
    Line2D([0], [0], color='black', lw=2, label='surface density'),
    Line2D([0], [0], color='black', lw=2, linestyle='--', label='number density')
]

# Add legend to the first subplot
axs[0,0].legend(handles=legend_handles_panel1, loc='upper right', frameon=True, fontsize=12)


legend_handles_panel3 = [
    Line2D([0], [0], color='black', lw=2, label='Stokes number'),
    Line2D([0], [0], color='black', lw=2, linestyle='--', label='pebble mass')
]
axs[1, 0].legend(handles=legend_handles_panel3, loc='upper right', frameon=True, fontsize=12)
# axs[1, 1].legend(loc='upper right')

legend_handles_panel2 = [
    Line2D([0], [0], color='black', lw=5, alpha=0.5, label='Simulation'),
    Line2D([0], [0], color='black', lw=2, linestyle='--', label='Analytical')
]
axs[0, 1].legend(handles=legend_handles_panel2, loc='upper right', frameon=True, fontsize=12)
# axs[0, 1].legend(loc='upper right')
# After all plotting is done, create legend handles manually

for ax in axs.flatten():
    ax.set_xticks(rad_ticks)
    ax.set_xticklabels(rad_ticklabels)

# Create a single figure legend
fig.legend(handles=legend_handles, loc='upper left', ncol=4, frameon=False, fontsize=15, bbox_to_anchor=(0.05,1.00))

plt.tight_layout()
fig.subplots_adjust(top=0.9)

plt.savefig('plots/1dprop_{:05d}.png'.format(int(filenum)), dpi=300)
plt.close()


#==============================================================================================
#==============================================================================================

#make a figure to check dust properties, including the density, St, flux and power law index 
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0,1].set_title("time: {:.2f} yr".format(simu_time*UNIT_T/YR),loc= 'right', y=1.4)
axs[0, 0].set_xscale('log')
axs[0, 0].set_ylim(1.e-3, 2.e3)
axs[0, 0].plot(rad, rho_xz[:, 0]*UNIT_Sigma,        c = colD['ga'], lw=lwD['ga'], alpha = alpD['ga'], label='gas')
axs[0, 0].plot(rad, dust_2_rho[0, 0]*UNIT_Sigma, c = colD['ss'], lw=lwD['ss'], alpha = alpD['ss'], label='sil0')
axs[0, 0].plot(rad, dust_1_rho[0, 0]*UNIT_Sigma, c = colD['si'], lw=lwD['si'], alpha = alpD['si'], label='sil1')
axs[0, 0].plot(rad, dust_4_rho[0, 0]*UNIT_Sigma,    c = colD['ms'], lw=lwD['ms'], alpha = alpD['ms'], label='sil2')
axs[0, 0].plot(rad, dust_3_rho[0, 0]*UNIT_Sigma,    c = colD['mi'], lw=lwD['mi'], alpha = alpD['mi'], label='vapor')
axs[0, 0].plot(rad, dust_6_rho[0, 0]*UNIT_Sigma,    c = colD['ls'], lw=lwD['ls'], alpha = alpD['ls'], label='sil2')
axs[0, 0].plot(rad, dust_5_rho[0, 0]*UNIT_Sigma,    c = colD['li'], lw=lwD['li'], alpha = alpD['li'], label='vapor')
axs[0, 0].plot(rad, dust_7_rho[0, 0]*UNIT_Sigma,    c = colD['va'], lw=lwD['va'], alpha = alpD['va'], label='vapor')
axs[0, 0].set_yscale('log')
axs[0, 0].set_xlabel('r (au)')
axs[0, 0].set_ylabel(r'$\Sigma $(g/cm$^2$)')
axn = axs[0, 0].twinx()
axn.set_yscale('log')
axn.set_ylabel('number density (1/cm$^2$)')
axn.plot(rad, dust_8_rho[0,0]/UNIT_L**2, '--', lw=lwD['ss'], alpha = alpD['ss'], c = 'gray') 
axn.plot(rad, dust_9_rho[0,0]/UNIT_L**2, '--', lw=lwD['ms'], alpha = alpD['ms'], c = 'gray')
axn.plot(rad, dust_10_rho[0,0]/UNIT_L**2, '--', lw=lwD['ls'], alpha = alpD['ls'], c = 'gray')

#drift timescale 
t_drift_0 = (rad*AU)/(dust_1_vx1[0, 0, :]*UNIT_V)/YR
t_drift_1 = (rad*AU)/(dust_3_vx1[0, 0, :]*UNIT_V)/YR
t_drift_2 = (rad*AU)/(dust_5_vx1[0, 0, :]*UNIT_V)/YR 

N1_column = dust_10_rho[0,0]/UNIT_L**2 
cs = sqrt(kB_mp_cgs*tem/2.34)
H_gas = cs/(omega_K/UNIT_T)
N1_volumn = N1_column/sqrt(2*pi)/H_gas
tau1 = st2*omega_K  #both st1 and omega_K are in code unit
delV = sqrt(3*alpha*tau1)*cs
t_col_1 = 1/(4*N1_volumn*pi*(s_p2[0,0])**2*delV)/YR *100

axs[1, 0].set_xscale('log')
axs[1, 0].set_yscale('log')
axs[1, 0].plot(rad, abs(t_drift_0), c ='gray', lw=lwD['ss'], alpha = alpD['ss'])
axs[1, 0].plot(rad, abs(t_drift_1), c ='gray', lw=lwD['ms'], alpha = alpD['ms'])
axs[1, 0].plot(rad, abs(t_drift_2), c ='gray', lw=lwD['ls'], alpha = alpD['ls'])
#hint the points that the drift velocity is positive 
rad_po0 = rad[dust_1_vx1[0, 0, :]>0]
rad_po1 = rad[dust_3_vx1[0, 0, :]>0] 
rad_po2 = rad[dust_5_vx1[0, 0, :]>0]

t_drift_0_po = t_drift_0[dust_1_vx1[0, 0, :]>0]
t_drift_1_po = t_drift_1[dust_3_vx1[0, 0, :]>0]
t_drift_2_po = t_drift_2[dust_5_vx1[0, 0, :]>0]

axs[1, 0].scatter(rad_po0, t_drift_0_po, marker='o', facecolors='red', edgecolors='none')
axs[1, 0].scatter(rad_po1, t_drift_1_po, marker='o', facecolors='red', edgecolors='none')
axs[1, 0].scatter(rad_po2, t_drift_2_po, marker='o', facecolors='red', edgecolors='none')

axs[1, 0].plot(rad, t_col_1[0,0],'k--', label='T_col_1')
# axs[1, 0].axhline(y= dt*UNIT_T/YR, color='grey', ls='--', label='dt')
axs[1, 0].set_xlabel('r (au)')
axs[1, 0].set_ylabel('timescale (yr)')

pp = zeros((2, len(rad)))

# simplest but not exact way: 
pp[0] = log10((dust_1_rho_xz[:, 0]+dust_2_rho_xz[:, 0])/(dust_3_rho[0, 0] + dust_4_rho[0, 0])
              )/log10(m_p[0,0]/m_p1[0,0])
pp[1] = log10((dust_3_rho[0, 0]+dust_4_rho[0, 0])/(dust_5_rho[0, 0]+dust_6_rho[0, 0])
               )/log10(m_p1[0,0]/m_p2[0,0])

mmin = 1.e-12
mDivL = []
# step = log(mmax[0,0]/mmin)/3 
# mDivL.append(mmin*exp(step ))
# mDivL.append(mmin*exp(2*step))
m_mid = sqrt(mmin*mmax[0,0])
mDivL.append(sqrt(mmin*m_mid))
mDivL.append(m_mid)
def ff_broken (pwl, prec, ml, mu):
    """integrate f(m)=prec*m**-pwl from ml to mu"""
    return prec/(2-pwl)*(mu**(2-pwl)-ml**(2-pwl))

# reconstruct from the moments: 
def ff(p, ml, mu, mch):
    M1oc = 1/(2-p)*(mu**(2-p)-ml**(2-p))
    M2oc = 1/(3-p)*(mu**(3-p)-ml**(3-p))
    return M2oc/M1oc - mch 

p0 = 0.5 
from scipy.optimize import root_scalar 
from scipy.optimize import fsolve 
pp_intra = zeros((3, len(rad)))
cc_intra = zeros((3, len(rad)))
for i in range(len(rad)):
    ms = array([mmin, m_p[0,0,i],mDivL[0][i], m_p1[0,0,i], mDivL[1][i], m_p2[0,0,i], mmax[0,0,i]])

    if all(ms == array(sorted(ms))):
        res = fsolve(ff, x0=p0, args=(mmin, mDivL[0][i], m_p[0,0,i])) 
        pp_intra[0,i] = res[0]
        M1 = ff_broken (pp_intra[0,i], 1, mmin, mDivL[0][i])
        cc_intra[0,i] = (dust_1_rho_xz[:, 0][i]+dust_2_rho_xz[:, 0][i])/M1

        res = fsolve(ff, x0=p0, args=(mDivL[0][i], mDivL[1][i], m_p1[0,0,i])) 
        pp_intra[1,i] = res[0]
        M1 = ff_broken (pp_intra[1,i], 1, mDivL[0][i], mDivL[1][i])
        cc_intra[1,i] = (dust_3_rho[0, 0][i]+dust_4_rho[0, 0][i])/M1

        res = fsolve(ff, x0=p0, args=(mDivL[1][i], mmax[0,0,i], m_p2[0,0,i])) 
        pp_intra[2,i] = res[0]
        M1 = ff_broken (pp_intra[2,i], 1, mDivL[1][i], mmax[0,0,i])
        cc_intra[2,i] = (dust_5_rho[0, 0][i]+dust_6_rho[0, 0][i])/M1
    else:
        pp_intra[:,i] = nan
        cc_intra[:,i] = nan


def ff_integ_many (cpre, pwl, m_bounds, mch):
    """
    bounds: separting the bins (nbin+1)
    mch: characteristic masses (nbin); first & last will not be used
    cpre: only one c, for the first distribution

    for each pwl segment (nbin-1) we have: f(m) = ci *m^-pi
    """
    nbin = len(mch)

    rhobin = zeros(nbin)

    # get the c for other bins: 
    cpreL = zeros(nbin-1)
    cpreL[0] = cpre
    for i in range(1, nbin-1):
        cpreL[i] = cpreL[i-1]*mch[i]**(pwl[i] - pwl[i-1])

    rhobin[0] = ff_broken (pwl[0], cpreL[0], m_bounds[0], m_bounds[1])

    # rhobin[0] = ff_broken (pwl[0], cpre[0], m_bounds[0], m_bounds[1])
    for i in range(1, nbin-1):
        rhobin[i] = ff_broken (pwl[i-1], cpreL[i-1], m_bounds[i], mch[i]) +\
                    ff_broken (pwl[i], cpreL[i], mch[i], m_bounds[i+1])

    rhobin[-1] = ff_broken (pwl[-1], cpreL[-1], m_bounds[-2], m_bounds[-1])

    return rhobin

def ff_inter(vars, m_bounds, mch1, rho):
    c, p1, p2 = vars
    
    m_norm = m_bounds/mch1
    # c = rho[0]*(2-p1)/(m_norm[1]**(2-p1)-m_norm[0]**(2-p1))
    #m_norm = m_norm/mch1

    f1 = ff_broken (p1, c, m_norm[0], m_norm[1]) - rho[0]
    f2 = ff_broken (p1, c, m_norm[1], 1) + ff_broken (p2, c, 1, m_norm[2]) - rho[1]
    f3 = ff_broken (p2, c, m_norm[2], m_norm[3]) - rho[2]

    # f1 = c/(2-p1)*(m_norm[2]**(2-p1)-m_norm[1]**(2-p1)) - rho[0]
    # f2 = c/(2-p1)*(1**(2-p1)-m_norm[1]**(2-p1)) + c/(2-p2)*(m_norm[2]**(2-p2)-1**(2-p2)) - rho[1]
    # f3 = c/(2-p2)*(m_norm[3]**(2-p2)-m_norm[2]**(2-p2)) - rho[2]
    return [f1, f2, f3] 

def ff_inter_new (vars, m_bounds, mch, rho_sim):
    """
    vars = [the_final_c, pwl_0, pwl_1, ...]
    """

    cfinal, pwl = vars[0], vars[1:]
    rho_dis = ff_integ_many (cfinal, pwl, m_bounds, mch)
    fsol = rho_dis - rho_sim

    return fsol


pp_inter = zeros((2, len(rad)))
# cc = zeros(len(rad))

cc_broken = zeros((2,len(rad)))

m_bounds = array([mmin*ones_like(rad), mDivL[0], mDivL[1], mmax[0,0,:]] )
for i in range(len(rad)):
    # p_guess = [pp_intra[0,i], pp_intra[2,i]]
    # p_guess = [11/6, 11/6]
    p_guess = 2 - pp[:,i]
    # guess = [1, p_guess[0], p_guess[1]]
    
    # p_guess = [9/6, 9/6]
    rhos = array([(dust_1_rho[0,0,i]+dust_2_rho[0,0,i]), 
                  (dust_3_rho[0,0,i]+dust_4_rho[0,0,i]), 
                  (dust_5_rho[0,0,i]+dust_6_rho[0,0,i])])

    # res, dout, icode, msg = fsolve(ff_inter, x0=guess, 
    #              args=(m_bounds[:,i], m_p1[0,0,i], rhos*UNIT_Sigma),xtol=1e-10,
    #              full_output=True)

    guess = [10, p_guess[0], p_guess[1]]
    mchs = array([m_p[0,0,i], m_p1[0,0,i], m_p2[0,0,i]])
    res2, dout2, icode2, msg2 = fsolve(ff_inter_new, x0=guess, 
                 args=(m_bounds[:,i], mchs, rhos*UNIT_Sigma),xtol=1e-10,
                 full_output=True)


    residual = ff_inter_new(res2, m_bounds[:,i], mchs, rhos*UNIT_Sigma)

    # if linalg.norm(residual) > 1e-8:
    #     print("solution failed to converge for index {}:".format(i))
    #     import pdb; pdb.set_trace()

    # print("index {}: residual = {}; pwl={}".format(i, linalg.norm(residual), res2[1:]))
    # if icode != 1:
    #     import pdb; pdb.set_trace()

    # pp_inter[:,i] = res[1:]
    # cc[i] = res[0]
    # use res2: 
    pp_inter[:,i] = res2[1:]
    cc_broken[0,i] = res2[0]
    nbin = len(mchs) - 1
    for j in range(1, nbin):
        cc_broken[j,i] = cc_broken[j-1,i]*mchs[j]**(pp_inter[j,i] - pp_inter[j-1,i])

    #test the continuity at m_p1 
    v1 = cc_broken[0,i]*m_p1[0,0,i]**(-pp_inter[0,i]) 
    v2 = cc_broken[1,i]*m_p1[0,0,i]**(-pp_inter[1,i])
    differ = abs(v1 - v2)/((abs(v1)+abs(v2))/2)
    # print(i, differ)
    # if abs(differ) > 1e-8:
    #     import pdb; pdb.set_trace()

    # m_norm = m_bounds[:,i]/m_p1[0,0,i+2]
    # cc[i] = rhos[0]*UNIT_Sigma*(2-pp_inter[0,i])/(m_norm[1]**(2-pp_inter[0,i]) - m_norm[0]**(2-pp_inter[0,i]))

pp_rho = 2 - pp_intra 
# axs[1, 1].plot(rad, pp_rho[0], marker='.', color='k', label='p_0')
# axs[1, 1].plot(rad, pp_rho[1], marker='.', color='gray', label='p_1')
# axs[1, 1].plot(rad, pp_rho[2], marker='.', color='lightgray', label='p_2')
# axs[1, 1].plot(rad, pp[0], marker='.', color='blue', label='p_01')
# axs[1, 1].plot(rad, pp[1], marker='.', color='y', label='p_12')
axs[1, 1].plot(rad_noghost, 2- pp_inter[0,2:-2], marker='.', color='k', label='p_01')
axs[1, 1].plot(rad_noghost, 2- pp_inter[1,2:-2], marker='.', color='gray', label='p_12')

axs[1, 1].axhline(y=1/6, color='r', ls='--', label='1/6')
axs[1, 1].set_xlabel('r (au)')
axs[1, 1].set_ylabel('power law index')
axs[1, 1].set_ylim(-0.1, 1.5)
axs[1, 1].set_xscale('log')

axcomp = axs[1, 1].twinx() 
# axcomp.set_yscale('log')
axcomp.set_ylabel(r'$f_{comp}$')
fice_0 = dust_1_rho_xz.T[0]/(dust_1_rho_xz.T[0]+dust_2_rho[0,0]) 
fice_1 = dust_3_rho_xz.T[0]/(dust_3_rho_xz.T[0]+dust_4_rho[0,0])
fice_2 = dust_5_rho[0,0]/(dust_5_rho[0,0]+dust_6_rho[0,0])
axcomp.plot(rad, fice_0, c = colD['si'], lw=lwD['si'], alpha = alpD['si'])
axcomp.plot(rad, fice_1, c = colD['mi'], lw=lwD['mi'], alpha = alpD['mi'])
axcomp.plot(rad, fice_2, c = colD['li'], lw=lwD['li'], alpha = alpD['li'])
# comp_ticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 0.9, 0.99, 0.999]
# axcomp.set_yticks(comp_ticks)
axcomp.set_ylim(0.,  1.)


SigmaM = zeros((len(rad), 200))
m_bins = logspace(log10(mmin), log10(40), 200)

for i in range(len(rad)):
    idx_max = argwhere(m_bins>mmax[0,0,i])[0][0]

    #so the power law breaks at the m_p1 (the second m_ch)
    idx_mp1 = argwhere(m_bins>m_p1[0,0, i])[0][0]

    #for the first power law: 
    kk1 = (dust_1_rho[0, 0, i]+dust_2_rho[0, 0, i])/(m_p[0,0,i]**(pp[0,i]))
    SigmaM[i,:idx_mp1] = kk1*m_bins[:idx_mp1]**(pp[0,i])*UNIT_Sigma

    #for the second power law: 
    kk2 = (dust_3_rho[0, 0, i]+dust_4_rho[0, 0, i])/(m_p1[0,0,i]**(pp[1,i]))
    SigmaM[i,idx_mp1:idx_max] = kk2*m_bins[idx_mp1:idx_max]**(pp[1,i])*UNIT_Sigma

min_sig = 0.1 
max_sig = 500 
SigmaM[SigmaM<min_sig] = min_sig

axs[0, 1].set_xlim(rad_noghost[0],rad_noghost[-1])
# axs[0, 1].set_facecolor('black')
levels = logspace(log10(min_sig), log10(max_sig), 10)
contour = axs[0, 1].contourf(rad, m_bins, SigmaM.T, levels=levels, norm=LogNorm(), cmap='magma')
cbar = fig.colorbar(contour, ax=axs[0, 1], orientation='horizontal',location='top',
             label=r'$\Sigma$ (g/cm$^2$)')
cbar.set_ticks([min_sig, 10, 100 , max_sig])


axs[0, 1].set_xscale('log')
axs[0, 1].plot(rad, m_p_xz.T[0], '-', c = 'lightgray', lw=lwD['ss'], alpha = alpD['ss'], label='small')
axs[0, 1].plot(rad, m_p1_xz.T[0], '-',c = 'lightgray', lw=lwD['ms'], alpha = alpD['ms'], label='mid')
axs[0, 1].plot(rad, m_p2[0,0], '-',c = 'lightgray', lw=lwD['ls'], alpha = alpD['ls'], label='large')
axs[0, 1].plot(rad_noghost, mmax[0,0,2:-2], 'b--', label='m_max')
axs[0, 1].plot(rad, mDivL[0], c ='gray', ls='--', label='m_div1')
axs[0, 1].plot(rad, mDivL[1], c = 'lightgray', ls='--', label='m_div2')

axs[0, 1].set_yscale('log')
axs[0, 1].set_xlabel('r (au)')
axs[0, 1].set_ylabel('particle mass (g)')
axs[0, 1].set_xlim(rad_noghost[0],rad_noghost[-1])

legend_handles_panel1 = [
    Line2D([0], [0], color='black', lw=2, label='surface density'),
    Line2D([0], [0], color='black', lw=2, linestyle='--', label='number density')
]

for ax in axs.flatten():
    ax.set_xticks(rad_ticks)
    ax.set_xticklabels(rad_ticklabels)
# Add legend to the first subplot
axs[0,0].legend(handles=legend_handles_panel1, loc='upper right', frameon=True, fontsize=15)
# axs[0, 1].legend(loc='upper right')
axs[1, 0].legend(loc='lower right')
axs[1, 1].legend(loc='upper right')
# Create a single figure legend
fig.legend(handles=legend_handles, loc='upper left', ncol=4, frameon=False, fontsize=15, bbox_to_anchor=(0.05,1.00))
plt.tight_layout()
fig.subplots_adjust(top=0.9)
plt.savefig('plots/1ddust_{:05d}.png'.format(int(filenum)), dpi=300)


fig, ax = plt.subplots(figsize=(7, 5))
ax.set_xscale('log') 
ax.set_yscale('log')

# axfm = ax.twinx()
# axfm.set_yscale('log')
# axfm.set_ylabel(r'$m^2 f(m)$')


rad_dd = 0.6 
rad_ice = 2.6
radL = [rad_ice, 2.8, 3.1]

# generate some colors from black to white
colL = [plt.cm.gray(i) for i in linspace(0.1, 0.9, len(radL) + 1)]
# generate some colors from purple to orange 
colLre = [plt.cm.plasma(i) for i in linspace(0.1, 0.9, len(radL) + 1)]

legend_handles = [
    Line2D([0], [0], color='k', marker='o',ls='', lw=0.5, label='Sim'),
    Line2D([0], [0], color='k', marker='x', ls='--', label=r'$m^2 f(m)$'),
    Line2D([0], [0], color='k', marker='s', markerfacecolor='none',ls='',
           label='integ'),
    Line2D([0], [0], color='k', marker='D',markeredgecolor='purple', ls='', label='relax')
]


for i in range(len(radL)):
    idx = argmin(abs(rad - radL[i])) 

    legend_handles.append(
            Line2D([0], [0], color=colL[i], ls='-', label='{:.1f} au'.format(radL[i]))
    )

    m_s = array([m_p[0,0,idx], m_p1[0,0,idx], m_p2[0,0,idx]])
    rhos = array([dust_1_rho[0,0,idx] + dust_2_rho[0,0,idx],
                dust_3_rho[0,0,idx] + dust_4_rho[0,0,idx],
                dust_5_rho[0,0,idx] + dust_6_rho[0,0,idx]])

    ax.plot(m_s, rhos*UNIT_Sigma, marker='o', ls ='', lw=0.5, 
             label=str(radL[i]), c = colL[i])
    
    m_bounds = array([mmin, mDivL[0][idx], mDivL[1][idx], mmax[0,0,idx]])
    mb_norm = m_bounds/m_p1[0,0,idx]

    # rhos_int = []
    # rhos_int.append(ff_broken (pp_inter[0,idx], cc[idx], mb_norm[0], mb_norm[1]) )
    # rhos_int.append(
    #     ff_broken (pp_inter[0,idx], cc[idx], mb_norm[1], 1) + ff_broken (pp_inter[1,idx], cc[idx], 1, mb_norm[2])
    #                )
    # rhos_int.append(ff_broken (pp_inter[1,idx], cc[idx], mb_norm[2], mb_norm[3]) )

    rhos_int = ff_integ_many (cc_broken[0,idx], pp_inter[:,idx], m_bounds,m_s)

    rho_dist = zeros((2, 2))
    # import pdb; pdb.set_trace()

    for j in range(len(cc_broken[:,idx])):
        rho_dist[j] = cc_broken[j,idx]*m_s[j:j+2]**(2-pp_inter[j,idx])
        ax.plot(m_s[j:j+2], rho_dist[j], marker='x', c = colL[i], ls='--' )

    ax.scatter(m_s, array(rhos_int), marker='s', edgecolors= colL[i],facecolors='none'
               ,s = 100)

    #plot the relaxed state 
    M1re = sum(rhos)*UNIT_Sigma 
    c_relax = M1re/(6*(m_bounds[-1]**(1/6)-m_bounds[0]**(1/6)))
    M1re_bins = []
    M2re_bins = []
    for j in range(len(m_bounds)-1):
        M1re_bins.append(ff_broken(11/6, c_relax, m_bounds[j], m_bounds[j+1]))
        M2re_bins.append(ff_broken(5/6, c_relax, m_bounds[j], m_bounds[j+1]))

    Nre_bins = array(M1re_bins)**2/array(M2re_bins)
    mre = array(M2re_bins)/array(M1re_bins)
    ax.plot(mre, M1re_bins, marker='D',markeredgecolor='purple', 
            markerfacecolor = colL[i], ls='' )
    # plot the relaxed state: 
    # mre = [mre[0], mre[2]]
    # M1re_bins = [M1re_bins[0], M1re_bins[2]]
    ax.plot(mre, M1re_bins, marker='D',markeredgecolor='purple', 
            markerfacecolor = colL[i], ls='-', c = 'purple' )
    
    # plt.plot(m_s, rhos_dist2, marker='x', c = colL[i], ls='--' )


ax.set_xlabel('particle mass (g)')
ax.set_ylabel(r'$\Sigma$ (g/cm$^2$)')
ax.legend(handles=legend_handles,ncol=2, frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig('plots/1dinter_{:05d}.png'.format(int(filenum)), dpi=300)


fig, ax = plt.subplots(figsize=(7, 5))
ax.set_xscale('log') 
ax.set_yscale('log')

#reconstruct from intra method 
for i in range(len(radL)):
    idx = argmin(abs(rad - radL[i]))

    m_s = array([m_p[0,0,idx], m_p1[0,0,idx], m_p2[0,0,idx]])
    rhos = array([dust_1_rho[0,0,idx] + dust_2_rho[0,0,idx],
                dust_3_rho[0,0,idx] + dust_4_rho[0,0,idx],
                dust_5_rho[0,0,idx] + dust_6_rho[0,0,idx]])

    ax.plot(m_s, rhos*UNIT_Sigma, marker='o', ls ='--', lw=0.5, 
             label=str(radL[i]), c = colL[i])

    nbin = len(m_s)
    rho_intra = zeros(nbin)
    rho_distbins = zeros((nbin, 2))
    m_bounds = array([mmin, mDivL[0][idx], mDivL[1][idx], mmax[0,0,idx]])

    rho_dist = cc_intra[:,idx]*m_s**(2-pp_intra[:,idx])
    for j in range(nbin):
        rho_intra[j] = ff_broken (pp_intra[j,idx], cc_intra[j,idx], m_bounds[j], m_bounds[j+1])

        rho_distbins[j,0] = cc_intra[j,idx]*m_bounds[j]**(2-pp_intra[j,idx]) 
        rho_distbins[j,1] = cc_intra[j,idx]*m_bounds[j+1]**(2-pp_intra[j,idx])

    ax.plot(m_s, rho_intra*UNIT_Sigma, marker='s',markerfacecolor='none', 
            markeredgecolor = colL[i], markersize=10, ls='' )
    ax.axvline(x=m_bounds[1], color=colL[i], ls='--', lw=0.5)
    ax.axvline(x=m_bounds[2], color=colL[i], ls='--', lw=0.5)
    # ax.plot(m_s, rho_dist, c = array(colL[i])*0.2 , ls=':' )
    for j in range(nbin):
        ax.plot(m_bounds[j:j+2], rho_distbins[j]*UNIT_Sigma,marker='x', c = colL[i], ls='--' )


ax.set_xlabel('particle mass (g)')
ax.set_ylabel(r'$\Sigma$ (g/cm$^2$)')
ax.legend(handles=legend_handles,ncol=2, frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig('plots/1dintra_{:05d}.png'.format(int(filenum)), dpi=300)
    
import pdb; pdb.set_trace()
