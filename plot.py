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
from preplot import get_relaxed_state, read_athinput
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
H_profile = lambda r: 0.033*AU*(r/1.0)**(H_slope)
# default value
T0 = T_profile(a0) # Temperature at planet position
Mdot_gas = 1.e-8*M_sun/YR
alpha = 3.e-3

try: 
    filenum = sys.argv[1]
except:
    print ("please specify a filenumber")
    sys.exit()

dT = 1
# read the data
nstep = filenum
try: 
    filename = sys.argv[2]
    DIR = './../' + filename +'/'
except:
    DIR = '/home/izx/athena_works/snowline_test/'


inputfile = DIR+'athinput.iceline'
athinputs = read_athinput(inputfile)

UNIT_T = athinputs['units']['time_cgs'] 
UNIT_L = athinputs['units']['length_cgs']  # scale height at reference poistion
UNIT_M = athinputs['units']['mass_cgs']  # mass unit in cgs 

Cs0 = sqrt(cons.k_B.cgs.value*T0/(2.34*cons.m_p.cgs.value))
Sigma0 = Mdot_gas/(3.0*pi*alpha*Cs0**2*UNIT_T) # gas surface density at planet position
sigma_profile = lambda r: Sigma0*(r/a0)**(sigma_slope)
# global dimensionless quantity
mu_He = 4
mu_H2 = 2
mu_xy = 2.34

UNIT_V = sqrt(cons.k_B.cgs.value*T0/(mu_xy*cons.m_p.cgs.value))
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

# dimensionless quantity used in intial set-up.
L_norm = (AU/UNIT_L)
r0 = a0*L_norm
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
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.serif'] = 'Times New Roman'  # Replace with your chosen font
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 15})
from copy import deepcopy
half = False

#get the file number by args 


rin = athinputs['mesh']['x1min'] 
rout = athinputs['mesh']['x1max']
Nrad = 300
intpl_numx = 320 
intpl_numz = 160 
xx_exp = linspace(rin/L_norm,rout/L_norm,intpl_numx)
zz_exp = linspace(-1.0,1.0,intpl_numz)[int(intpl_numz/2):]
xx_exp_mesh, zz_exp_mesh = meshgrid(xx_exp,zz_exp)

GM = (r0)**3
tlim = athinputs['time']['tlim']/YR*UNIT_T*2*pi 

# DIR = '/home/yu/Programs/Athena/work/output/snowline_2D/output4/'
# DIR = '/mnt/disk1/dataYu/output/snowline_2D/output37/'
#----------------------------------------
# primitive data read
#----------------------------------------
filename = DIR+'iceline.out1.'+str(nstep).rjust(5,'0')+'.athdf'
print("Reading file: ", filename)
data_prim= athena_read.athdf(filename,face_func_2=face_f_2_power, num_ghost=0)
rad = data_prim['x1v']/ L_norm
theta = data_prim['x2v']
phi = data_prim['x3v']

phi_f = data_prim['x3f']
phi_f[-1] = phi_f[0]
phi[-1] = phi[0] = 0.0
theta_f = data_prim['x2f']
rad_f = data_prim['x1f']/ L_norm

simu_time = data_prim['Time']
hstname = DIR+'iceline.hst'
# data_hst = athena_read.hst(hstname)
# dt = data_hst['dt'][int(filenum)-1]

## (phi*theta*R)
rho = data_prim['rho']
prs = data_prim['press']
vx1 = data_prim['vel1']
vx2 = data_prim['vel2']
vx3 = data_prim['vel3']

dust_id_pat = re.compile(r'^dust_(\d+)_rho$')
dust_ids = sorted(int(m.group(1)) for k in data_prim.keys() for m in [dust_id_pat.match(k)] if m)
N_Z = 2 
N_P = int((dust_ids[-1] - 1)/(N_Z + 1))

dust_rho = {did: data_prim[f'dust_{did}_rho'] for did in dust_ids}
dust_vel = {}
for did in dust_ids:
    dust_vel[did] = {}
    for comp in ['vel1', 'vel2', 'vel3']:
        key = f'dust_{did}_{comp}'
        dust_vel[did][comp] = data_prim[key] if key in data_prim else zeros_like(rho)

def dust_rho_or_zero(did):
    return dust_rho[did] if did in dust_rho else zeros_like(rho)

def dust_vel_or_zero(did, comp):
    if did in dust_vel and comp in dust_vel[did]:
        return dust_vel[did][comp]
    return zeros_like(rho)

# Backward-compatible aliases used by downstream plotting code.
dust_1_rho = dust_rho_or_zero(1)
dust_1_vx1 = dust_vel_or_zero(1, 'vel1')
dust_1_vx2 = dust_vel_or_zero(1, 'vel2')
dust_1_vx3 = dust_vel_or_zero(1, 'vel3')

dust_2_rho = dust_rho_or_zero(2)

dust_3_rho = dust_rho_or_zero(3)
dust_3_vx1 = dust_vel_or_zero(3, 'vel1')
dust_3_vx2 = dust_vel_or_zero(3, 'vel2')
dust_3_vx3 = dust_vel_or_zero(3, 'vel3')

dust_4_rho = dust_rho_or_zero(4)

dust_5_rho = dust_rho_or_zero(5)
dust_5_vx1 = dust_vel_or_zero(5, 'vel1')
dust_5_vx2 = dust_vel_or_zero(5, 'vel2')
dust_5_vx3 = dust_vel_or_zero(5, 'vel3')

dust_6_rho = dust_rho_or_zero(6)
dust_6_vx1 = dust_vel_or_zero(6, 'vel1')
dust_6_vx2 = dust_vel_or_zero(6, 'vel2')
dust_6_vx3 = dust_vel_or_zero(6, 'vel3')
dust_7_rho = dust_rho_or_zero(7)
dust_7_vx1 = dust_vel_or_zero(7, 'vel1')
dust_7_vx2 = dust_vel_or_zero(7, 'vel2')
dust_7_vx3 = dust_vel_or_zero(7, 'vel3')

#-----------------------------------------
# user defined variable read
# #---------------------------------------
data_uov= athena_read.athdf(DIR+'iceline.out2.'+str(nstep).rjust(5,'0')+'.athdf',face_func_2=face_f_2_power, num_ghost=0)
tem = data_uov['Tem']

st_pat = re.compile(r'^st_(\d+)$')
pop_ids_1based = sorted(int(m.group(1)) for k in data_uov.keys() for m in [st_pat.match(k)] if m)
N_pop = len(pop_ids_1based)
if N_pop == 0:
    N_pop = max(1, len(dust_ids)//2)

ice_ids = [2*p + 1 for p in range(N_pop) if (2*p + 1) in dust_ids]
sil_ids = [2*p + 2 for p in range(N_pop) if (2*p + 2) in dust_ids]
vapor_id = 2*N_pop + 1 if (2*N_pop + 1) in dust_ids else None
number_ids = [did for did in dust_ids if did not in set(ice_ids + sil_ids + ([vapor_id] if vapor_id is not None else []))]

st_by_pop = [data_uov[f'st_{pid}'] if f'st_{pid}' in data_uov else zeros_like(tem) for pid in pop_ids_1based]
m_p_by_pop = [data_uov[f'm_p_{pid}'] if f'm_p_{pid}' in data_uov else zeros_like(tem) for pid in pop_ids_1based]
s_p_by_pop = [data_uov[f's_p_{pid}'] if f's_p_{pid}' in data_uov else zeros_like(tem) for pid in pop_ids_1based]

# dif = data_uov['dif']
try:
    t_relax = data_uov['t_relax']
except:
    t_relax = zeros_like(tem) 
mmax = data_uov['mmax']
mmax_xz = mmax[0,:,:].T
mmin = 1e-12
st = st_by_pop[0] if len(st_by_pop) > 0 else zeros_like(tem)
st1 = st_by_pop[1] if len(st_by_pop) > 1 else zeros_like(tem)
# tem_equi = data_uov['st']
m_p = m_p_by_pop[0] if len(m_p_by_pop) > 0 else zeros_like(tem)
m_p1 = m_p_by_pop[1] if len(m_p_by_pop) > 1 else zeros_like(tem)
s_p = s_p_by_pop[0] if len(s_p_by_pop) > 0 else zeros_like(tem)
s_p1 = s_p_by_pop[1] if len(s_p_by_pop) > 1 else zeros_like(tem)
# dfvdt = data_uov['dfvdt']
q_z = data_uov['q_z']
try:
    q_int = data_uov['q_int']
except:
    q_int = ones_like(q_z) 
q_latent = data_uov['q_latent']
q_diff = data_uov['q_diff']
flx_vap_x1 = data_uov['flx_vap_x1']
flx_vap_x2 = data_uov['flx_vap_x2']
flx_x1 = data_uov['flx_x1']
flx_x2 = data_uov['flx_x2']

flx_ice_x1_by_pop = [data_uov[f'flx_ice_x1_{pid}'] if f'flx_ice_x1_{pid}' in data_uov else zeros_like(tem) for pid in pop_ids_1based]
flx_ice_x2_by_pop = [data_uov[f'flx_ice_x2_{pid}'] if f'flx_ice_x2_{pid}' in data_uov else zeros_like(tem) for pid in pop_ids_1based]
flx_sil_x1_by_pop = [data_uov[f'flx_sil_x1_{pid}'] if f'flx_sil_x1_{pid}' in data_uov else zeros_like(tem) for pid in pop_ids_1based]
flx_sil_x2_by_pop = [data_uov[f'flx_sil_x2_{pid}'] if f'flx_sil_x2_{pid}' in data_uov else zeros_like(tem) for pid in pop_ids_1based]

flx_ice_x1 = flx_ice_x1_by_pop[0] if len(flx_ice_x1_by_pop) > 0 else zeros_like(tem)
flx_ice_x2 = flx_ice_x2_by_pop[0] if len(flx_ice_x2_by_pop) > 0 else zeros_like(tem)
flx_ice1_x1 = flx_ice_x1_by_pop[1] if len(flx_ice_x1_by_pop) > 1 else zeros_like(tem)
flx_ice1_x2 = flx_ice_x2_by_pop[1] if len(flx_ice_x2_by_pop) > 1 else zeros_like(tem)
flx_sil_x1 = flx_sil_x1_by_pop[0] if len(flx_sil_x1_by_pop) > 0 else zeros_like(tem)
flx_sil_x2 = flx_sil_x2_by_pop[0] if len(flx_sil_x2_by_pop) > 0 else zeros_like(tem)
flx_sil1_x1 = flx_sil_x1_by_pop[1] if len(flx_sil_x1_by_pop) > 1 else zeros_like(tem)
flx_sil1_x2 = flx_sil_x2_by_pop[1] if len(flx_sil_x2_by_pop) > 1 else zeros_like(tem)

# Keep legacy variable names but bind them to inferred semantic ids.
if vapor_id is not None:
    dust_5_rho = dust_rho_or_zero(vapor_id)
    dust_5_vx1 = dust_vel_or_zero(vapor_id, 'vel1')
    dust_5_vx2 = dust_vel_or_zero(vapor_id, 'vel2')
    dust_5_vx3 = dust_vel_or_zero(vapor_id, 'vel3')
if len(number_ids) > 0:
    dust_6_rho = dust_rho_or_zero(number_ids[0])
    dust_6_vx1 = dust_vel_or_zero(number_ids[0], 'vel1')
    dust_6_vx2 = dust_vel_or_zero(number_ids[0], 'vel2')
    dust_6_vx3 = dust_vel_or_zero(number_ids[0], 'vel3')
if len(number_ids) > 1:
    dust_7_rho = dust_rho_or_zero(number_ids[1])
    dust_7_vx1 = dust_vel_or_zero(number_ids[1], 'vel1')
    dust_7_vx2 = dust_vel_or_zero(number_ids[1], 'vel2')
    dust_7_vx3 = dust_vel_or_zero(number_ids[1], 'vel3')

#get the density change by phase_change process 
drho_exist = True
try:
    drho_i_dt = data_uov['drho_i_dt']
    drho_i1_dt = data_uov['drho_i1_dt']
    drho_v_dt = data_uov['drho_v_dt']
except:
    print("no drho found")
    drho_exist = False
    drho_i_dt = zeros_like(rho)
    drho_i1_dt = zeros_like(rho)
    drho_v_dt = zeros_like(rho)

# gamma = data_uov['gamma']

# face coordinate
index_phi = 0
THETA, PHI, R = meshgrid(theta_f,phi_f,rad_f)
x = R* sin(THETA) * cos(PHI)
y = R* sin(THETA) * sin(PHI)
z = R* cos(THETA)
x_xz = x[index_phi,:,:].T
y_xz = z[index_phi,:,:].T

dust_vel_xz = {
    did: {
        'vel1': dust_vel[did]['vel1'][index_phi,:,:].T,
        'vel2': dust_vel[did]['vel2'][index_phi,:,:].T,
        'vel3': dust_vel[did]['vel3'][index_phi,:,:].T,
    }
    for did in dust_ids
}

dust_1_vx1_xz = dust_vel_xz[1]['vel1'] if 1 in dust_vel_xz else zeros_like(rho[index_phi,:,:].T)
dust_1_vx2_xz = dust_vel_xz[1]['vel2'] if 1 in dust_vel_xz else zeros_like(rho[index_phi,:,:].T)
dust_1_vx3_xz = dust_vel_xz[1]['vel3'] if 1 in dust_vel_xz else zeros_like(rho[index_phi,:,:].T)
dust_3_vx1_xz = dust_vel_xz[3]['vel1'] if 3 in dust_vel_xz else zeros_like(rho[index_phi,:,:].T)
dust_3_vx2_xz = dust_vel_xz[3]['vel2'] if 3 in dust_vel_xz else zeros_like(rho[index_phi,:,:].T)
dust_3_vx3_xz = dust_vel_xz[3]['vel3'] if 3 in dust_vel_xz else zeros_like(rho[index_phi,:,:].T)

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
flx_ice_x1 *= dS_R* UNIT_Fm 
flx_vap_x1 *= dS_R* UNIT_Fm
flx_ice_x2 *= dS_theta* UNIT_Fm
flx_vap_x2 *= dS_theta* UNIT_Fm
flx_x1 *= dS_R* UNIT_Fm
flx_x2 *= dS_theta* UNIT_Fm

flx_ice1_x1 *= dS_R* UNIT_Fm 
flx_ice1_x2 *= dS_theta* UNIT_Fm 
flx_sil_x1 *= dS_R* UNIT_Fm 
flx_sil_x2 *= dS_theta* UNIT_Fm 
flx_sil1_x1 *= dS_R* UNIT_Fm 
flx_sil1_x2 *= dS_theta*UNIT_Fm 

# slices
index_phi = 0
rho_xz = rho[index_phi,:,:].T
dust_rho_xz = {did: dust_rho[did][index_phi,:,:].T for did in dust_ids}
dust_1_rho_xz = dust_rho_xz[1] if 1 in dust_rho_xz else zeros_like(rho_xz)
dust_2_rho_xz = dust_rho_xz[2] if 2 in dust_rho_xz else zeros_like(rho_xz)
dust_3_rho_xz = dust_rho_xz[3] if 3 in dust_rho_xz else zeros_like(rho_xz)
dust_4_rho_xz = dust_rho_xz[4] if 4 in dust_rho_xz else zeros_like(rho_xz)
dust_5_rho_xz = dust_rho_xz[vapor_id] if vapor_id in dust_rho_xz else zeros_like(rho_xz)
dust_6_rho_xz = dust_rho_xz[number_ids[0]] if len(number_ids) > 0 and number_ids[0] in dust_rho_xz else zeros_like(rho_xz)
dust_7_rho_xz = dust_rho_xz[number_ids[1]] if len(number_ids) > 1 and number_ids[1] in dust_rho_xz else zeros_like(rho_xz)
# prs_xz = prs[index_phi,:,:].T
tem_xz = tem[index_phi,:,:].T
# tem_equi_xz = tem_equi[index_phi,:,:].T
st_xz = st[index_phi,:,:].T
st1_xz = st1[index_phi,:,:].T
m_p_xz = m_p[index_phi,:,:].T
m_p1_xz = m_p1[index_phi,:,:].T
s_p_xz = s_p[index_phi,:,:].T
s_p1_xz = s_p1[index_phi,:,:].T
# dif_xz = dif[index_phi,:,:].T
# dfvdt_xz = dfvdt[index_phi,:,:].T
q_z_xz = q_z[index_phi,:,:].T
q_int_xz = q_int[index_phi,:,:].T
q_latent_xz = q_latent[index_phi,:,:].T
q_diff_xz = q_diff[index_phi,:,:].T

# change rate
drho_i_dt_xz = drho_i_dt[index_phi,:,:].T 
drho_i1_dt_xz = drho_i1_dt[index_phi,:,:].T 
drho_v_dt_xz = drho_v_dt[index_phi,:,:].T

#the relaxation timescale 
try:
    t_cool = data_uov['t_cool']
except:
    t_cool = None


#plot the rho
d2g_snow = 1.e-3
dust_rho_mod_xz = {did: deepcopy(arr) for did, arr in dust_rho_xz.items()}
for did in dust_rho_mod_xz:
    dust_rho_mod_xz[did][dust_rho_xz[did]/rho_xz < d2g_snow] = nan

dust_5_rho_mod = dust_rho_mod_xz[5] if 5 in dust_rho_mod_xz else zeros_like(rho_xz)
dust_1_rho_mod = dust_rho_mod_xz[1] if 1 in dust_rho_mod_xz else zeros_like(rho_xz)
dust_3_rho_mod = dust_rho_mod_xz[3] if 3 in dust_rho_mod_xz else zeros_like(rho_xz)
dust_2_rho_mod = dust_rho_mod_xz[2] if 2 in dust_rho_mod_xz else zeros_like(rho_xz)
dust_4_rho_mod = dust_rho_mod_xz[4] if 4 in dust_rho_mod_xz else zeros_like(rho_xz)


#find the scale height location: 
def find_dust_scaleheight(rhos_intpl, y_xz_c):
    rho_p = rhos_intpl[1] 
    Hp_idx = zeros(intpl_numx)
    yy = zeros(intpl_numx)
    for i in range(intpl_numx):
        rho_efold = rho_p[0,i]/exp(1.0)**0.5
        if isnan(rho_efold):
            Hp_idx[i] = nan 
            yy[i] = nan
        else:
            Hp_idx[i] = nanargmin(abs(rho_efold - rho_p[:,i]))
            yy[i] =zz_exp[int(Hp_idx[i])] 

    return Hp_idx, yy 


#import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'  # Switch to sans-serif
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']  # Fallback fonts

R_inner = rin/L_norm 
xs = rout/L_norm 
zs = 0.6
numx = 64
numy = 16
numz = 32
x1_exp_half = linspace(R_inner,xs,numx)
# x2_exp_half = linspace(R_inner,ys,numy)
x3_exp = linspace(0.0,zs,numz)
slice_exp = array([0.0])

# xz
vx,vy,vz = v_Intpl_Sph2car(rad,theta,phi,x1_exp_half,slice_exp,x3_exp,vx1.T,vx2.T,vx3.T) 
vx_xz = vx[:,0,:]
vy_xz = vy[:,0,:]
vz_xz = vz[:,0,:]

# these seems not used
dust_1_vx,dust_1_vy,dust_1_vz = v_Intpl_Sph2car(rad,theta,phi,x1_exp_half,slice_exp,x3_exp,dust_1_vx1.T,dust_1_vx2.T,dust_1_vx3.T) 
dust_1_vx_xz = dust_1_vx[:,0,:]
dust_1_vy_xz = dust_1_vy[:,0,:]
dust_1_vz_xz = dust_1_vz[:,0,:]

dust_3_vx,dust_3_vy,dust_3_vz = v_Intpl_Sph2car(rad,theta,phi,x1_exp_half,slice_exp,x3_exp,dust_3_vx1.T,dust_3_vx2.T,dust_3_vx3.T) 
dust_3_vx_xz = dust_3_vx[:,0,:]
dust_3_vy_xz = dust_3_vy[:,0,:]
dust_3_vz_xz = dust_3_vz[:,0,:]

flx_water_x1 = flx_vap_x1 
flx_water_x2 = flx_vap_x2

flx_x, flx_y, flx_z = v_Intpl_Sph2car(rad,theta,phi,x1_exp_half,slice_exp,x3_exp,flx_x1.T, flx_x2.T, flx_x1.T * 0.0)
flx_x_xz = flx_x[:,0,:]
flx_z_xz = flx_z[:,0,:]

water_flx_x, water_flx_y, water_flx_z = v_Intpl_Sph2car(rad,theta,phi,x1_exp_half,slice_exp,x3_exp, flx_water_x1.T, flx_water_x2.T, flx_water_x1.T * 0.0)
water_flx_x_xz = water_flx_x[:,0,:]
water_flx_z_xz = water_flx_z[:,0,:]

ice_flx_x, ice_flx_y, ice_flx_z = v_Intpl_Sph2car(rad,theta,phi,x1_exp_half,slice_exp,x3_exp,(flx_ice_x1).T,(flx_ice_x2).T, (flx_ice_x1).T * 0.0)
ice_flx_x_xz = ice_flx_x[:,0,:]
ice_flx_z_xz = ice_flx_z[:,0,:]

ice1_flx_x, ice1_flx_y, ice1_flx_z = v_Intpl_Sph2car(rad,theta,phi,x1_exp_half,slice_exp,x3_exp,(flx_ice1_x1).T,(flx_ice1_x2).T, (flx_ice1_x1).T * 0.0)
ice1_flx_x_xz = ice1_flx_x[:,0,:]
ice1_flx_z_xz = ice1_flx_z[:,0,:]

sil_flx_x, sil_flx_y, sil_flx_z = v_Intpl_Sph2car(rad,theta,phi,x1_exp_half,slice_exp,x3_exp,(flx_sil_x1).T,(flx_sil_x2).T, (flx_sil_x1).T * 0.0)
sil_flx_x_xz = sil_flx_x[:,0,:]
sil_flx_z_xz = sil_flx_z[:,0,:]

sil1_flx_x, sil1_flx_y, sil1_flx_z = v_Intpl_Sph2car(rad,theta,phi,x1_exp_half,slice_exp,x3_exp,(flx_sil1_x1).T,(flx_sil1_x2).T, (flx_sil1_x1).T * 0.0)
sil1_flx_x_xz = sil1_flx_x[:,0,:]
sil1_flx_z_xz = sil1_flx_z[:,0,:]

for j in range(numz):
    for i in range(numx):
        ratio_xz = fabs(x1_exp_half[i]/x3_exp[j]) if x3_exp[j] != 0.0 else inf
        if(ratio_xz < tan(0.8) or (x1_exp_half[i]**2 + x3_exp[j]**2 > xs**2) ):
            vx_xz[j,i] = 0.0
            vz_xz[j,i] = 0.0
            dust_1_vx_xz[j,i] = 0.0
            dust_1_vz_xz[j,i] = 0.0
            dust_3_vx_xz[j,i] = 0.0
            dust_3_vz_xz[j,i] = 0.0
            water_flx_x_xz[j,i] = 0.0
            water_flx_z_xz[j,i] = 0.0
            ice_flx_x_xz[j,i] = 0.0
            ice_flx_z_xz[j,i] = 0.0
            ice1_flx_x_xz[j,i] = 0.0 
            ice1_flx_z_xz[j,i] = 0.0


# # 3D
# vx,vy,vz = v_Intpl_Sph2car(rad,theta,phi,x1_exp,x2_exp,x3_exp,vx1.T,vx2.T,vx3.T)
normal = sort(sqrt(vx_xz**2 + vz_xz**2),axis = None)[-10]# velocity normalization
normal2 = sort(sqrt(water_flx_x_xz**2 + water_flx_z_xz**2),axis = None)[-2]
# normal2 = sort(sqrt(ice_flx_x_xz**2 + ice_flx_z_xz**2),axis = None)[-2]


dif = (x/3.0)*0.003
dif_intpl = (xx_exp_mesh/3.0)*0.003

rho_intpl = scaler_Intpl_Sph2car(rad,theta,phi,xx_exp,array([0.0]),zz_exp,rho.T)[:,0,:]
dust_rho_intpl = {
    did: scaler_Intpl_Sph2car(rad,theta,phi,xx_exp,array([0.0]),zz_exp,dust_rho[did].T)[:,0,:]
    for did in dust_ids
}

dust_1_rho_intpl = dust_rho_intpl[1] if 1 in dust_rho_intpl else zeros_like(rho_intpl)
dust_2_rho_intpl = dust_rho_intpl[2] if 2 in dust_rho_intpl else zeros_like(rho_intpl)
dust_3_rho_intpl = dust_rho_intpl[3] if 3 in dust_rho_intpl else zeros_like(rho_intpl)
dust_4_rho_intpl = dust_rho_intpl[4] if 4 in dust_rho_intpl else zeros_like(rho_intpl)
dust_5_rho_intpl = dust_rho_intpl[vapor_id] if vapor_id in dust_rho_intpl else zeros_like(rho_intpl)
dust_6_rho_intpl = dust_rho_intpl[number_ids[0]] if len(number_ids) > 0 and number_ids[0] in dust_rho_intpl else zeros_like(rho_intpl)
dust_7_rho_intpl = dust_rho_intpl[number_ids[1]] if len(number_ids) > 1 and number_ids[1] in dust_rho_intpl else zeros_like(rho_intpl)

dif_intpl = scaler_Intpl_Sph2car(rad,theta,phi,xx_exp,array([0.0]),zz_exp,dif.T)[:,0,:]
st_intpl = scaler_Intpl_Sph2car(rad,theta,phi,xx_exp,array([0.0]),zz_exp,st.T)[:,0,:]
st1_intpl = scaler_Intpl_Sph2car(rad, theta, phi, xx_exp, array([0.0]), zz_exp, st1.T)[:,0,:]
st_intpl *= (xx_exp_mesh/3)**(-1.5) #transfer from t_stop to St
st1_intpl *= (xx_exp_mesh/3)**(-1.5)
dif_peb_intpl = dif_intpl/(1.0 + st_intpl**2)
dif_peb1_intpl = dif_intpl/(1. + st1_intpl**2)
Tem_intpl = scaler_Intpl_Sph2car(rad,theta,phi,xx_exp,array([0.0]),zz_exp,tem.T)[:,0,:]
prs_intpl = scaler_Intpl_Sph2car(rad,theta,phi,xx_exp,array([0.0]),zz_exp,prs.T)[:,0,:]
vx,vy,vz = v_Intpl_Sph2car(rad,theta,phi,xx_exp,array([0.0]),zz_exp,vx1.T,vx2.T,vx3.T)
vx_intpl = vx[:,0,:]
vy_intpl = vy[:,0,:]
vz_intpl = vz[:,0,:]

dust_1_vx,dust_1_vy,dust_1_vz = v_Intpl_Sph2car(rad,theta,phi,xx_exp,array([0.0]),zz_exp,dust_1_vx1.T,dust_1_vx2.T,dust_1_vx3.T)
dust_1_vx_intpl = dust_1_vx[:,0,:]
dust_1_vy_intpl = dust_1_vy[:,0,:]
dust_1_vz_intpl = dust_1_vz[:,0,:]

dust_3_vx,dust_3_vy,dust_3_vz = v_Intpl_Sph2car(rad,theta,phi,xx_exp,array([0.0]),zz_exp,dust_3_vx1.T,dust_3_vx2.T,dust_3_vx3.T)
dust_3_vx_intpl = dust_3_vx[:,0,:]
dust_3_vy_intpl = dust_3_vy[:,0,:]
dust_3_vz_intpl = dust_3_vz[:,0,:]


# flux measured from simulation
flux_ice_x,flux_ice_z,flux_ice_z = v_Intpl_Sph2car(rad,theta,phi,xx_exp,array([0.0]),zz_exp,(flx_ice_x1/dS_R).T, (flx_ice_x2/dS_theta).T, 0.0*flx_ice_x2.T)
flux_ice_x_intpl = flux_ice_x[:,0,:]
# flux_ice_z_intpl = flux_ice_z[:,0,:]

flux_ice1_x,flux_ice1_z,flux_ice1_z = v_Intpl_Sph2car(rad,theta,phi,xx_exp,array([0.0]),zz_exp,(flx_ice1_x1/dS_R).T, (flx_ice1_x2/dS_theta).T, 0.0*flx_ice1_x2.T)
flux_ice1_x_intpl = flux_ice1_x[:,0,:]

flux_sil_x,flux_sil_z,flux_sil_z = v_Intpl_Sph2car(rad,theta,phi,xx_exp,array([0.0]),zz_exp,(flx_sil_x1/dS_R).T, (flx_sil_x2/dS_theta).T, 0.0*flx_sil_x2.T)
flux_sil_x_intpl = flux_sil_x[:,0,:]
# flux_sil_z_intpl = flux_sil_z[:,0,:]

flux_sil1_x,flux_sil1_z,flux_sil1_z = v_Intpl_Sph2car(rad,theta,phi,xx_exp,array([0.0]),zz_exp,(flx_sil1_x1/dS_R).T, (flx_sil1_x2/dS_theta).T, 0.0*flx_sil1_x2.T)
flux_sil1_x_intpl = flux_sil1_x[:,0,:]

flux_vap_x,flux_vap_z,flux_vap_z = v_Intpl_Sph2car(rad,theta,phi,xx_exp,array([0.0]),zz_exp,(flx_vap_x1/dS_R).T, (flx_vap_x2/dS_theta).T, 0.0*flx_vap_x2.T)
flux_vap_x_intpl = flux_vap_x[:,0,:]
flux_vap_z_intpl = flux_vap_z[:,0,:]

flux_gas_x,flux_gas_z,flux_gas_z = v_Intpl_Sph2car(rad,theta,phi,xx_exp,array([0.0]),zz_exp,(flx_x1/dS_R).T, (flx_x2/dS_theta).T, 0.0*flx_x2.T)
flux_gas_x_intpl = flux_gas_x[:,0,:]
flux_gas_z_intpl = flux_gas_z[:,0,:]

for j in range(len(zz_exp)):
    for i in range(len(xx_exp)):
        if((fabs(zz_exp[j]/xx_exp[i]) > tan(pi/2-1.3) or (xx_exp[i]**2 + zz_exp[j]**2 > xs**2) or (xx_exp[i]**2 + zz_exp[j]**2 < R_inner**2))):
            rho_intpl[j,i] = 0.0
            for did in dust_rho_intpl:
                dust_rho_intpl[did][j,i] = 0.0
            flux_vap_x_intpl[j,i] = 0.0
            flux_vap_z_intpl[j,i] = 0.0
            flux_gas_x_intpl[j,i] = 0.0
            flux_gas_z_intpl[j,i] = 0.0 
            flux_ice_x_intpl[j,i] = 0.0 
            flux_ice1_x_intpl[j,i] = 0.0 
            flux_sil_x_intpl[j,i] = 0.0 
            flux_sil1_x_intpl[j,i] = 0.0
            st_intpl[j,i] = 0.0 
            st1_intpl[j,i] = 0.0

# surface density
UNIT_SIGMA = UNIT_DEN*UNIT_L
dz = (zz_exp[1]-zz_exp[0])*AU/UNIT_L
dx = (xx_exp[1]-xx_exp[0])*AU/UNIT_L
sigma_gas =  sum(rho_intpl*dz,axis = 0)*2.0 *UNIT_SIGMA # remember to add up 2 wings
sigma_ice_by_pop = [sum(dust_rho_intpl[iid]*dz,axis=0)*2.0*UNIT_SIGMA for iid in ice_ids]
sigma_sil_by_pop = [sum(dust_rho_intpl[sid]*dz,axis=0)*2.0*UNIT_SIGMA for sid in sil_ids]
sigma_ice0 = sigma_ice_by_pop[0] if len(sigma_ice_by_pop) > 0 else zeros_like(sigma_gas)
sigma_ice1 = sigma_ice_by_pop[1] if len(sigma_ice_by_pop) > 1 else zeros_like(sigma_gas)
sigma_sil0 = sigma_sil_by_pop[0] if len(sigma_sil_by_pop) > 0 else zeros_like(sigma_gas)
sigma_sil1 = sigma_sil_by_pop[1] if len(sigma_sil_by_pop) > 1 else zeros_like(sigma_gas)
sigma_ice = sum(sigma_ice_by_pop, axis=0) if len(sigma_ice_by_pop) > 0 else zeros_like(sigma_gas)
sigma_sil = sum(sigma_sil_by_pop, axis=0) if len(sigma_sil_by_pop) > 0 else zeros_like(sigma_gas)
n_col_by_id = {did: sum(dust_rho_intpl[did]*dz,axis=0)*2.0*UNIT_L**2 for did in number_ids}
n_col0 = n_col_by_id[number_ids[0]] if len(number_ids) > 0 else zeros_like(sigma_gas)
n_col1 = n_col_by_id[number_ids[1]] if len(number_ids) > 1 else zeros_like(sigma_gas)
sigma_vap = sum(dust_rho_intpl[vapor_id]*dz,axis = 0)*2.0 *UNIT_SIGMA if vapor_id is not None else zeros_like(sigma_gas)
dust_2_dif_x = zeros((len(zz_exp), len(xx_exp)))
dust_3_dif_x = zeros((len(zz_exp), len(xx_exp)))
dust_1_dif_x = zeros((len(zz_exp), len(xx_exp)))
dust_4_dif_x = zeros((len(zz_exp), len(xx_exp)))
Hp0_idx, yy0 = find_dust_scaleheight([dust_1_rho_intpl, dust_2_rho_intpl], y_xz_c)
Hp1_idx, yy1 = find_dust_scaleheight([dust_3_rho_intpl, dust_4_rho_intpl], y_xz_c)
Hpg_idx, yy_g = find_dust_scaleheight([[], rho_intpl], y_xz_c)

# calc optical depth
kappa0 = athinputs['problem']['kappa0']
f_vi = athinputs['problem']['f_vi']
tau_opt = zeros(rho_xz.shape)
for j in range(tau_opt.shape[1]):
    dx2 = rad*L_norm*(theta_f[1]-theta_f[0])
    tau_opt[:,j] += tau_opt[:,j-1] + rho_xz[:,j]*Get_kappa(kappa0, 0.0, dust_5_rho_xz/rho_xz)[:,j] * dx2

#problemic
tau_opt_intpl = zeros(rho_intpl.shape)
for j in range(-1, -1-tau_opt_intpl.shape[1], -1):
    dx2 = dz 
    kappatem = Get_kappa(kappa0, 0.0, dust_5_rho_intpl/rho_intpl)[:,j]
    kappatem[isnan(kappatem)] = 0.0
    tau_opt_intpl[:,j] += tau_opt_intpl[:,j+1] + rho_intpl[:,j]*kappatem * dx2

tau_ir = tau_opt/3
tau_ir_intpl = tau_opt_intpl/3

for j in range(len(zz_exp)):
    rho_safe = where(rho_intpl[j,:] > 0.0, rho_intpl[j,:], 1.0)
    d2g_2 = where(rho_intpl[j,:] > 0.0, dust_2_rho_intpl[j,:]/rho_safe, 0.0)
    d2g_3 = where(rho_intpl[j,:] > 0.0, dust_3_rho_intpl[j,:]/rho_safe, 0.0)
    d2g_1 = where(rho_intpl[j,:] > 0.0, dust_1_rho_intpl[j,:]/rho_safe, 0.0)
    d2g_4 = where(rho_intpl[j,:] > 0.0, dust_4_rho_intpl[j,:]/rho_safe, 0.0)
    dust_2_dif_x[j,:] = -rho_intpl[j,:]* dif_peb_intpl[j,:]*dfdx_5pts(xx_exp*L_norm, d2g_2)
    dust_3_dif_x[j,:] = -rho_intpl[j,:]* dif_peb1_intpl[j,:]*dfdx_5pts(xx_exp*L_norm, d2g_3)
    dust_1_dif_x[j,:] = -rho_intpl[j,:]* dif_peb_intpl[j,:]*dfdx_5pts(xx_exp*L_norm, d2g_1)
    dust_4_dif_x[j,:] = -rho_intpl[j,:]* dif_peb1_intpl[j,:]*dfdx_5pts(xx_exp*L_norm, d2g_4)

for j in range(len(zz_exp)):
    for i in range(len(xx_exp)):
        if(isnan(dust_1_dif_x[j,i])):
            dust_1_dif_x[j,i] = 0.0
        if(isnan(dust_2_dif_x[j,i])):
            dust_2_dif_x[j,i] = 0.0
        if(isnan(dust_3_dif_x[j,i])):
            dust_3_dif_x[j,i] = 0.0
        if(isnan(dust_4_dif_x[j,i])):
            dust_4_dif_x[j,i] = 0.0


rr = 2.55
zz = 0.17 
rr_idx = (abs(rad - rr)).argmin()
zz_idx = (abs(theta - arccos(zz/rr))).argmin()

singlepop = False
singlepop = sys.argv[2] == 'single_pop'



#plot a time evolution of the fragmenation velocity and maximum peb mass 
# v_ice = 1000 
# v_sil = 100
# v_frag = zeros(120)
# time = zeros(120)
# for i in range(120):
#     filenum = i 
#     fileprim = DIR+'iceline.out1.'+str(filenum).rjust(5,'0')+'.athdf'
#     data_prim = athena_read.athdf(fileprim,face_func_2=face_f_2_power, num_ghost=0)
#
#     # fileuov = DIR+'iceline.out2.'+str(filenum).rjust(5,'0')+'.athdf'
#     # data_uov = athena_read.athdf(fileuov,face_func_2=face_f_2_power, num_ghost=0)
#
#     dust_1_rho = data_prim['dust_1_rho'][0, zz_idx, rr_idx]
#     dust_3_rho = data_prim['dust_3_rho'][0, zz_idx, rr_idx]
#     dust_2_rho = data_prim['dust_2_rho'][0, zz_idx, rr_idx]
#     dust_4_rho = data_prim['dust_4_rho'][0, zz_idx, rr_idx]
#
#     rho_sil = dust_2_rho + dust_4_rho
#     rho_ice = dust_1_rho + dust_3_rho 
#
#     v_frag[i] = (rho_sil*v_sil + rho_ice*v_ice)/(rho_sil + rho_ice)
#     time[i] = data_prim['Time']*UNIT_T/YR
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(time, v_frag, color = 'k', lw = 2)
# ax.set_xlabel('time [yr]', fontsize = 12)
# ax.set_ylabel('v_frag [cm/s]', fontsize = 12)
# plt.savefig('./plots/vfrag_time.png', dpi = 300, bbox_inches='tight')
# plt.close()

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_ylim(-0.35, 0.35)
ax.set_xlim(rin/L_norm, 3)

crhov =  ax.contourf(x_xz_c,y_xz_c,dust_5_rho_xz*UNIT_DEN,levels = logspace(-19,-9,10), norm = LogNorm(), cmap = 'Greys', alpha = 1.0, extend = 'both',zorder=3, antialiased = True)
ax.contourf(x_xz_c,-y_xz_c,dust_5_rho_xz*UNIT_DEN,levels = logspace(-19,-9,10), norm = LogNorm(), cmap = 'Greys', alpha = 1.0, extend = 'both',zorder=3, antialiased = True)
colors = ['white', 'skyblue', 'deepskyblue', 'dodgerblue', 'blue', 'darkblue']
crho1= ax.contourf(x_xz_c,y_xz_c,(dust_1_rho_mod)/rho_xz,levels = logspace(log10(0.05), log10(1.25),5), norm = LogNorm(), antialiased = True, 
                   colors = colors, alpha = 0.7, extend = 'both',zorder=4)
crho1= ax.contourf(x_xz_c,-y_xz_c,(dust_3_rho_mod)/rho_xz,levels = logspace(log10(0.05), log10(1.25),5), norm = LogNorm(), antialiased = True, 
                   colors = colors, alpha = 0.7, extend = 'both',zorder=4)
# ax0 =  ax.contourf(x_xz_c,-y_xz_c,dust_5_rho_mod,levels = logspace(log10(d2g_snow),log10(1.0),25), norm = LogNorm(), cmap = 'RdPu', alpha = 0.7, extend = 'both',zorder=3, antialiased = True)
# ax00 = ax.contourf(x_xz_c,-y_xz_c,dust_1_rho_mod,levels = logspace(log10(d2g_snow),log10(0.3),20), norm = LogNorm(), cmap = 'Blues', alpha = 1, extend = 'both', antialiased=True, zorder=4)
cbarv = fig.colorbar(crhov, ax = ax, orientation = 'vertical',pad = -0.15, shrink = 0.3, aspect = 12, anchor=(0, 1))
# cbarv.ax.set_title(r'$\rho_{vap}$ [g cm$^{-3}$]', fontsize = 12)
cbarv.ax.set_ylabel(r'$\rho_{vap}$ [g cm$^{-3}$]', fontsize = 12)
cbarv.set_ticks(logspace(-20,-10,6))
cbarv.set_ticklabels([r'$10^{-20}$',r'$10^{-18}$',r'$10^{-16}$',r'$10^{-14}$',r'$10^{-12}$',r'$10^{-10}$'], fontsize = 10)
cbar1 = fig.colorbar(crho1, ax = ax, orientation = 'vertical',pad = -0.15, shrink = 0.3, aspect = 12, anchor=(0, 0.5))
cbar1.ax.set_ylabel(r'$\rho_{ice}/\rho_{gas}$', fontsize = 12)
cbar1.set_ticks([0.1, 0.3, 1])
cbar1.set_ticklabels(['0.1', '0.3', '1'], fontsize = 10)

# overplot vapor colored by temperature region (contourf)
vap_rho = dust_5_rho_xz * UNIT_DEN

# mask vapor by temperature ranges
vap_cold = ma.masked_where(~((tem_xz< 150) & (vap_rho > 0)), vap_rho)
vap_warm = ma.masked_where(~((tem_xz>= 150) & (tem_xz < 400) & (vap_rho > 0)), vap_rho)
vap_hot  = ma.masked_where(~((tem_xz>= 400) & (vap_rho > 0)), vap_rho)

vap_cold_obs = ma.masked_where(~((tau_ir < 1.0) & (vap_cold > 0)), vap_cold)
vap_warm_obs = ma.masked_where(~((tau_ir < 1.0) & (vap_warm > 0)), vap_warm)
vap_hot_obs = ma.masked_where(~(( tau_ir < 1.0) & (vap_hot > 0)), vap_hot)

# integrate to get the mass of hot/warm/cold vapor in the optically thin region
m_cold = 0 
m_warm = 0 
m_hot  = 0
m_cold_M = zeros_like(dust_5_rho_xz)
m_warm_M = zeros_like(dust_5_rho_xz)
m_hot_M = zeros_like( dust_5_rho_xz)

for i in range(len(rad)):
    for j in range(len(theta)):
        if(tem_xz[i,j] < 150 and tau_ir[i,j] < 1.0):
            m_cold_M[i,j] = dust_5_rho_xz[i,j]*rad[i]**2*sin(theta[j])*diff(rad_f)[i]*diff(theta_f)[j]*2*pi*UNIT_M
            m_cold += m_cold_M[i,j] 
        elif(tem_xz[i,j] >= 150 and tem_xz[i,j] < 400 and tau_ir[i,j] < 1.0):
            m_warm_M[i,j] = dust_5_rho_xz[i,j]*rad[i]**2*sin(theta[j])*diff(rad_f)[i]*diff(theta_f)[j]*2*pi*UNIT_M 
            m_warm += m_warm_M[i,j]
        elif(tem_xz[i,j] >= 400 and tau_ir[i,j] < 1.0):
            m_hot_M[i,j]= dust_5_rho_xz[i,j]*rad[i]**2*sin(theta[j])*diff(rad_f)[i]*diff(theta_f)[j]*2*pi*UNIT_M  
            m_hot += m_hot_M[i,j]

# m_cold = 0 
# m_warm = 0 
# m_hot  = 0
# m_cold_M = zeros_like(dust_5_rho_intpl)
# m_warm_M = zeros_like(dust_5_rho_intpl)
# m_hot_M = zeros_like( dust_5_rho_intpl)
# dx = (xx_exp[1]-xx_exp[0])
# for j in range(len(xx_exp)):
#     for i in range(len(zz_exp)):
#         if(Tem_intpl[i,j] < 150 and tau_ir_intpl[i,j] < 1.0):
#             m_cold_M[i,j] = dust_5_rho_intpl[i,j]*dz*dx*AU**2*UNIT_DEN
#             m_cold += m_cold_M[i,j] 
#         elif(Tem_intpl[i,j] >= 150 and Tem_intpl[i,j] < 400 and tau_ir_intpl[i,j] < 1.0):
#             m_warm_M[i,j] = dust_5_rho_intpl[i,j]*dx*dz*AU**2*UNIT_DEN
#             m_warm += m_warm_M[i,j]
#         elif(Tem_intpl[i,j] >= 400 and tau_ir_intpl[i,j] < 1.0):
#             m_hot_M[i,j]= dust_5_rho_intpl[i,j]*dx*dz*AU**2*UNIT_DEN  
#             m_hot += m_hot_M[i,j]
# find the vapor density threshold that encloses 99% of the mass in each temperature region
def _mass_threshold(mass_map, rho_map, thres=0.99):
    cells = []
    total_mass = 0.0
    for i in range(len(rad)):
        for j in range(len(theta)):
            if mass_map[i,j] > 0 and rho_map[i,j] > 0:
                cells.append((rho_map[i,j], mass_map[i,j]))
                total_mass += mass_map[i,j]

    if total_mass == 0:
        return 0.0
    cells.sort(key=lambda x: x[0], reverse=True)
    cum = 0.0
    target = thres * total_mass
    for density, mass in cells:
        cum += mass
        if cum >= target:
            return density
    return cells[-1][0]

threshold_cold = _mass_threshold(m_cold_M, vap_rho)
threshold_warm = _mass_threshold(m_warm_M, vap_rho)
threshold_hot  = _mass_threshold(m_hot_M,  vap_rho, thres = 1.0)

# mask vapor to only the densest cells accounting for 90% of each region's mass
vap_cold_90 = ma.masked_where(~((tau_ir < 1.0) & (vap_cold > 0) & (vap_rho >= threshold_cold)), vap_rho)
vap_warm_90 = ma.masked_where(~((tau_ir < 1.0) & (vap_warm > 0) & (vap_rho >= threshold_warm)), vap_rho)
vap_hot_90  = ma.masked_where(~((tau_ir < 1.0) & (vap_hot > 0) & (vap_rho >= threshold_hot)), vap_rho)

levels_vap = logspace(-20, -8, 10)
ax.contourf(x_xz_c, y_xz_c, vap_cold_90, levels=levels_vap, norm=LogNorm(), colors=['blue'],  alpha=0.5, zorder=6,antialiased = True)
ax.contourf(x_xz_c, y_xz_c, vap_warm_90, levels=levels_vap, norm=LogNorm(), colors=['orange'],alpha=0.5, zorder=6,antialiased = True)
ax.contourf(x_xz_c, y_xz_c, vap_hot_90,  levels=levels_vap, norm=LogNorm(), colors=['red'],   alpha=0.5, zorder=6,antialiased = True)

ax.contourf(x_xz_c, -y_xz_c, vap_cold_90, levels=levels_vap, norm=LogNorm(), colors=['blue'],  alpha=0.5, zorder=6,antialiased = True)
ax.contourf(x_xz_c, -y_xz_c, vap_warm_90, levels=levels_vap, norm=LogNorm(), colors=['orange'],alpha=0.5, zorder=6,antialiased = True)
ax.contourf(x_xz_c, -y_xz_c, vap_hot_90,  levels=levels_vap, norm=LogNorm(), colors=['red'],   alpha=0.5, zorder=6,antialiased = True)

# legend for vapor temperature regions
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue',  alpha=0.5, label=r'$T<150$ K'),
    Patch(facecolor='orange',alpha=0.5, label=r'$150<T<400$ K'),
    Patch(facecolor='red',   alpha=0.5, label=r'$T>400$ K'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.8)
# overplot vapor only above tau_ir = 1 (optically thin region)
# vap_above_tau = ma.masked_where(~((tau_ir < 1.0) & (vap_rho > 0)), vap_rho)
# ax.contourf(x_xz_c, y_xz_c, vap_above_tau, levels=levels_vap, norm=LogNorm(),
#             hatches=['//'], alpha=0.0, zorder=7)
#over plot temperature profile
ax.set_xlabel(r'$R$ [AU]', fontsize = 12)
ax.set_ylabel(r'$z$ [AU]', fontsize = 12)
# ax1 = ax.pcolormesh(x_xz,y_xz,tem_xz,norm = Normalize(vmin = 100,vmax = 600,clip = True) ,cmap = 'coolwarm', alpha = 1)
C_Tem = ax.contour(x_xz_c,y_xz_c,tem_xz,levels = linspace(100,400,5,endpoint=True), cmap = 'coolwarm', alpha = 0.8, linewidths = 1.5, linestyles = 'dashed',zorder = 11)
ax.contour(x_xz_c,-y_xz_c,tem_xz,levels = linspace(100,400,5,endpoint=True),        cmap = 'coolwarm', alpha = 0.8, linewidths = 1.5, linestyles = 'dashed',zorder = 11)

ax.contour(x_xz_c,y_xz_c,tem_xz,levels = linspace(100,400,5,endpoint=True), colors='white', alpha = 0.8, linewidths = 2.8,zorder = 10)
ax.contour(x_xz_c,-y_xz_c,tem_xz,levels = linspace(100,400,5,endpoint=True), colors='white', alpha = 0.8, linewidths = 2.8,zorder = 10)
cbarT = fig.colorbar(C_Tem, ax = ax, orientation = 'vertical',pad = 0.02, shrink = 0.3, aspect = 12, anchor=(0, 0))
cbarT.ax.set_ylabel(r'$T$ [K]', fontsize = 12)
# cbarT.set_ticklabels([r'$100$',r'$200$',r'$300$',r'$400$',r'$500$',r'$600$'], fontsize = 10)
C = ax.contour(x_xz_c,y_xz_c,tau_ir,levels = array([1.0]), colors = 'purple', linestyles = 'dashed', linewidths = 3.0, zorder = 5)
# C = ax.contour(xx_exp_mesh, zz_exp_mesh,tau_ir_intpl,levels = array([0.1, 1.0, 2, 100]), colors = 'purple', linestyles = 'dashed', linewidths = 3.0, zorder = 5)
ax.annotate(r'$\tau_{ir}=1$', xy=(2.5, 0.25), xytext=(2.5, 0.25), fontsize = 20, color = 'purple', zorder = 10, fontweight = 'bold',rotation = 20)

fig.savefig('./plots/vap_obs_{:05d}.png'.format(int(filenum)), bbox_inches='tight', dpi = 500)

fig, axs = plt.subplots(2, 1, figsize=(6, 6))
m_p1_safe = where(m_p1[0].T > 0.0, m_p1[0].T, nan)
rrr = mmax[0].T/m_p1_safe
axs[0].set_title('time: {:.2f} yr'.format(simu_time*UNIT_T/YR),loc='left')
axs[0].scatter([rr], [zz], color = 'red', s = 50, marker = 'o', label = r'$(R,z)=(2.75,0.18)$ AU', zorder = 10)
cbar = axs[0].contourf(x_xz_c, y_xz_c,rrr, levels = logspace(-4,5, 11),
                    norm = LogNorm(vmin=1e-4 ,vmax=100000.0),extend = 'both', cmap = cm.viridis)
axs[0].contour(x_xz_c, y_xz_c, rrr, levels = [7.0], colors = 'white', linewidths = 1.5)
cbar = fig.colorbar(cbar, format=ticker.FuncFormatter(formatnum), ax = axs[0], orientation = 'vertical',)
cbar.set_ticks([1, 10, 100, 1000, 1e4, 1e5])
cbar.ax.set_title('$m_{max}/m_1$')

bbar = axs[1].contourf(x_xz_c, y_xz_c, mmax[0].T, levels = logspace(-12,-1,16), 
                       norm = LogNorm(), extend = 'both',cmap = cm.viridis)
cbarmmax = fig.colorbar(bbar, format=ticker.FuncFormatter(formatnum), ax = axs[1], orientation = 'vertical',)
cbarmmax.set_ticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
cbarmmax.ax.set_title('$m_{max}$')

plt.savefig('./plots/mmax_{:05d}.png'.format(int(filenum)) ,dpi=300)
plt.close()

colD = {'ga':'black', 'ss':'tab:orange', 'ms':'tab:orange', 'ls':'tab:orange', 'si':'tab:blue', 'mi':"tab:blue" ,'li':'tab:blue', 'va':'tab:purple'}
lwD  = {'ga':2, 'ss':1, 'ms':3, 'ls':4, 'si':1, 'mi':3, 'li':4, 'va':2}  
alpD = {'ga':1.0, 'ss':1., 'ms': 0.8, 'ls':0.5, 'si':1., 'mi':0.8, 'li':0.5, 'va':1.}

legend_handles = [
    # Line2D([0], [0], color=colD['ga'], lw=lwD['ga'], alpha=alpD['ga'], label='gas') ,
    Line2D([0], [0], color=colD['va'], lw=lwD['va'], alpha=alpD['va'], label='vapor'),
    Line2D([0], [0], color=colD['ss'], lw=lwD['ss'], alpha=alpD['ss'], label='sil0') ,
    # Line2D([0], [0], color=colD['ms'], lw=lwD['ms'], alpha=alpD['ms'], label='sil1') ,
    Line2D([0], [0], color=colD['ls'], lw=lwD['ls'], alpha=alpD['ls'], label='sil2') ,
    Line2D([0], [0], color=colD['si'], lw=lwD['si'], alpha=alpD['si'], label='ice0') ,
    # Line2D([0], [0], color=colD['mi'], lw=lwD['mi'], alpha=alpD['mi'], label='ice1') ,
    Line2D([0], [0], color=colD['li'], lw=lwD['li'], alpha=alpD['li'], label='ice2') ,
]

# flx_ice_x = sum(dust_1_rho_intpl*dz*dust_1_vx_intpl + dust_1_dif_x*dz,axis = 0)*2.0 *(2*pi*xx_exp*L_norm)
flux_sil = sum(dust_2_rho_intpl*dz*dust_1_vx_intpl + dust_2_dif_x*dz,axis = 0)*2.0 *(2*pi*xx_exp*L_norm)
# flx_ice1_x = sum(dust_3_rho_intpl*dz*dust_3_vx_intpl + dust_3_dif_x*dz,axis = 0)*2.0 *(2*pi*xx_exp*L_norm)
# flux_sil1 = sum(dust_4_rho_intpl*dz*dust_3_vx_intpl + dust_3_dif_x*dz,axis = 0)*2.0 *(2*pi*xx_exp*L_norm)

UNIT_Fm = (UNIT_L**3*UNIT_DEN/UNIT_T)/(M_sun/YR)
flux_sil *= UNIT_Fm 
# flux_sil1 *= UNIT_Fm 
# flx_ice_x *= UNIT_Fm
# flx_ice1_x *= UNIT_Fm

# simulation face flux
dthetaM = empty_like(flx_x1[0])
for i in range(len(dthetaM.T)): 
    dthetaM[:, i] = dtheta
# flux_gas_face = sum((flx_x1[0] - flx_vap_x1[0])*dthetaM,axis = 0) *2.0 *(2*pi*rad*L_norm)  # remember to add up 2 wings
flux_gas_face = sum((flux_gas_x_intpl - flux_vap_x_intpl)*dz,axis = 0) *2.0 *(2*pi*xx_exp*L_norm)  # remember to add up 2 wings
flux_ice_face = sum(flux_ice_x_intpl*dz,axis = 0) *2.0 *(2*pi*xx_exp*L_norm)
flux_vap_face = sum(flux_vap_x_intpl*dz,axis = 0) *2.0 *(2*pi*xx_exp*L_norm)
flux_ice1_face = sum(flux_ice1_x_intpl*dz,axis = 0) *2.0*(2*pi*xx_exp*L_norm)
flux_sil_face = sum(flux_sil_x_intpl*dz,axis = 0)  *2.0*(2*pi*xx_exp*L_norm)
flux_sil1_face = sum(flux_sil1_x_intpl*dz,axis = 0) *2.0*(2*pi*xx_exp*L_norm)
flux_water_face = sum((flux_vap_x_intpl + flux_ice_x_intpl +flux_ice1_x_intpl)*dz,axis = 0) *2.0 *(2*pi*xx_exp*L_norm)

flux_ice1_faced = sum(flx_ice_x1[0, :, :], axis=0) * 2.0
flux_ice_faced = sum(flx_ice1_x1[0, :, :], axis=0) * 2.0 
flux_sil_faced = sum(flx_sil_x1[0, :, :], axis=0) * 2.0 
flux_sil1_faced = sum(flx_sil1_x1[0, :, :], axis=0) * 2.0 
flux_vap_faced = sum(flx_vap_x1[0, :, :], axis=0) * 2.0
flux_water_faced = sum((flx_vap_x1[0, :, :] + flx_ice_x1[0, :, :] + flx_ice1_x1[0, :, :]), axis=0) * 2.0

## advective flux
flux_vap_adv = sum(dust_5_rho_intpl*dz*vx_intpl,axis = 0)*2.0 *(2*pi*xx_exp*L_norm)
flux_vap_adv *= UNIT_Fm

fig, axs = plt.subplots(2, 3, figsize=(22.5, 12), constrained_layout=True,sharex=True)
# fig.subplots_adjust(wspace=0.7)

axs[0,2].set_title("time: {:.2f} yr".format(simu_time*UNIT_T/YR),loc= 'right', y=1.1)
# axs[0,0].set_xscale('log')
# axs[0,0].set_xlabel(r'$R$ [AU]', fontsize = 12)
axs[0,0].set_ylabel(r'$\Sigma$ [g/cm$^2$]', fontsize = 12)

# axs[0,0].set_yscale('log')
axs[0,0].set_ylim(1e-2, 30)
# sax[0].plot(xx_exp,(sigma_gas-sigma_vap)*0.4, color = 'k', alpha = 1.0, label = '$ f_{\mathrm{i/g}} \Sigma_{\mathrm{xy}}$')
# shere the 0.4 is from the 0.8/2, in which 0.8 is the dust-to-gas flux ratio, so the vapor should be the half of it
axs[0,0].plot(xx_exp,(sigma_gas)*0.4, color = 'k', linestyle='-', alpha = 1.0, label = 'gas')
axs[0,0].plot(xx_exp,sqrt(2*pi)*0.4*(xx_exp/3)**(-1)*UNIT_SIGMA,color = 'grey',linewidth = 5, alpha = 0.5)
ice_style = [('si', 'ice {}'), ('li', 'ice {}')]
sil_style = [('ss', 'silicate {}'), ('ls', 'silicate {}')]
for p in range(len(sigma_ice_by_pop)):
    i_style = p if p < len(ice_style) else len(ice_style)-1
    key = ice_style[i_style][0]
    axs[0,0].plot(xx_exp, sigma_ice_by_pop[p], c = colD[key], lw = lwD[key], label = ice_style[i_style][1].format(p))
for p in range(len(sigma_sil_by_pop)):
    i_style = p if p < len(sil_style) else len(sil_style)-1
    key = sil_style[i_style][0]
    axs[0,0].plot(xx_exp, sigma_sil_by_pop[p], c = colD[key], lw = lwD[key], label = sil_style[i_style][1].format(p))
axs[0,0].plot(xx_exp, sigma_vap , c = colD['va'], lw = lwD['va'], label = 'vapor') 

#plot the column density 
# axn = axs[0,0].twinx() 
# axn.set_yscale('log')
# axn.plot(xx_exp, n_col0, color = 'gray', lw = lwD['si'], alpha=alpD['si'], ls = 'dashed', label = 'n0')
# axn.plot(xx_exp, n_col1, color = 'gray', lw = lwD['li'], alpha=alpD['li'], ls = 'dashed', label = 'n1')

# #plot the vapor density contour 
# cvap = axs[1,0].contourf(xx_exp, zz_exp, log10(dust_5_rho_intpl), levels = linspace(-12, -2, 11), cmap = 'RdPu', alpha = 0.8)
# cbar_vap = fig.colorbar(cvap, ax=axs[1,0])
# cbar_vap.ax.set_title(r'$lg \rho_{\mathrm{vap}} [g/cm^3]$', fontsize = 12)

ticks = logspace(-12, -1, 5)
# axs[0,1].plot(rad, H_profile(rad)/AU, '--', c='gray', lw=1)
# axs[0,1].plot(rad, -H_profile(rad)/AU, '--', c='gray', lw=1,zorder=10)
# axs[0,1].set_xlabel(r'$R$ [AU]', fontsize = 12)
axs[0,1].set_ylabel(r'$z$ [AU]', fontsize = 12)
axs[0,1].set_ylim(-0.25, 0.25)
axs[0,1].set_xlim(rin/L_norm, rout/L_norm)
# the vapor
ax0 = axs[0,1].contourf(x_xz_c,y_xz_c,dust_5_rho_mod,levels = logspace(log10(d2g_snow),log10(1.0),25), norm = LogNorm(), cmap = 'RdPu', alpha = 0.7, extend = 'both',zorder=3, antialiased = True)
crho1= axs[0,1].contourf(x_xz_c,y_xz_c,dust_3_rho_mod,levels = logspace(log10(d2g_snow),log10(0.3),20), norm = LogNorm(), cmap = 'Blues', alpha = 1.0, extend = 'both', antialiased = True,zorder=4)
ax0 = axs[0,1].contourf(x_xz_c,-y_xz_c,dust_5_rho_mod,levels = logspace(log10(d2g_snow),log10(1.0),25), norm = LogNorm(), cmap = 'RdPu', alpha = 0.7, extend = 'both',zorder=3, antialiased = True)
ax00 = axs[0,1].contourf(x_xz_c,-y_xz_c,dust_1_rho_mod,levels = logspace(log10(d2g_snow),log10(0.3),20), norm = LogNorm(), cmap = 'Blues', alpha = 1, extend = 'both', antialiased=True, zorder=4)

axs[0,1].plot(xx_exp, -yy0, '--', c='k', lw=1, zorder=10)
axs[0,1].plot(xx_exp, yy1, '--', c='k', lw=1, zorder=10)
#zxl: this we change to the sum of the ice in different populations.
ice_rho_xz_tot = dust_1_rho_xz + dust_3_rho_xz
axs[0,1].contour(x_xz_c,y_xz_c, dust_3_rho_xz/rho_xz,levels = [d2g_snow], cmap = 'Blues_r', norm = LogNorm(), alpha = 0.7, linewidths = 3.0, zorder = 5)
axs[0,1].contour(x_xz_c,-y_xz_c, dust_1_rho_xz/rho_xz,levels = [d2g_snow], cmap = 'Blues_r', alpha = 0.7, linewidths = 3.0, zorder = 4)
axs[0,1].contour(x_xz_c,y_xz_c,tau_ir,levels = array([0.5,1.0]), colors = 'black', linestyles = 'dotted', zorder = 20)
axs[0,1].contour(x_xz_c,-y_xz_c,tau_ir,levels = array([0.5,1.0]), colors = 'black', linestyles = 'dotted', zorder = 20)
#label the panels in the lower left corner
axs[0,1].text(0.05, 0.95, '$pop_1$', transform=axs[0,1].transAxes, fontsize=18, fontweight='bold', va='top', ha='left')
axs[0,1].text(0.05, 0.05, '$pop_0$', transform=axs[0,1].transAxes, fontsize=18, fontweight='bold', va='bottom', ha='left')

# the streamlines
# normalized lw of flux
lw_flx_gas = sqrt(flx_x_xz**2 + flx_z_xz**2)/normal2
lw_flx_ice =sqrt(ice_flx_x_xz**2 + ice_flx_z_xz**2)/normal2
lw_flx_ice1 = sqrt(ice1_flx_x_xz**2 + ice1_flx_z_xz**2)/normal2
lw_flx_water = sqrt(water_flx_x_xz**2 + water_flx_z_xz**2)/normal2
lw_flx_gas = 2.0*sqrt(lw_flx_gas)
lw_flx_ice = 2.0*sqrt(lw_flx_ice)
lw_flx_ice1 = 2.0*sqrt(lw_flx_ice1)
lw_flx_water = 2.0*sqrt(lw_flx_water)
axs[0,0].streamplot(x1_exp_half,x3_exp, ice1_flx_x_xz/normal2, ice1_flx_z_xz/normal2,linewidth = lw_flx_ice1, arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='blue',zorder=10)
axs[0,0].streamplot(x1_exp_half,x3_exp, water_flx_x_xz/normal2, water_flx_z_xz/normal2,linewidth = lw_flx_water, arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='pink',zorder=4)
# Create negative z coordinate (increasing order for streamplot)
z_neg = -x3_exp[::-1]
# axs[0,1].streamplot(x1_exp_half, z_neg,
#                     water_flx_x_xz[::-1, :] / normal2,
#                     -water_flx_z_xz[::-1, :] / normal2,
#                     linewidth=lw_flx_water[::-1, :], arrowstyle='->',
#                     density=1.0, broken_streamlines=True,
#                     color='pink', zorder=4)
axs[0,1].streamplot(x1_exp_half,z_neg, 
                    ice_flx_x_xz[::-1,:]/normal2, 
                    - ice_flx_z_xz[::-1,:]/normal2,linewidth = lw_flx_ice[::-1,:], arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='blue',zorder=4)

# legend
legends = [Line2D([0], [0], color='k', lw=2, label=r'$10^{-3}~\rho_{0}c_{\mathrm{s,0}}$'),
           Line2D([0], [0], color='k', ls = '--', lw=1, label=r'$H_{peb}$')]
axs[0,1].legend(handles=legends, loc='upper right',fontsize = 15, framealpha = 0.6)
# the ices
# cbar1 = fig.colorbar(c1, ax=axs[0,1])
# cbar1.ax.set_title(r'$lg m_{\mathrm{1}} [g]$', fontsize = 12)
# cbar1.set_ticks(ticks)

#move the colorbar to be aligned with the bottom of top figure 
cbarrho = fig.colorbar(crho1, ax=axs[0,1],location = 'right', shrink = 0.45, pad =-0.11,anchor=(0,-0.))
cbarrho.set_ticks([1e-2, 1e-1], labels = ['$10^{-2}$', '$10^{-1}$'])
cbarrho.ax.set_title(r'$\rho_{\mathrm{ice}} [g/cm^3]$', fontsize = 12)
cbarvap = fig.colorbar(ax0, ax=axs[0,1], location = 'right', shrink = 0.45, pad =0.04, anchor=(0,1))
cbarvap.set_ticks([1e-2, 1e-1, 1.0], labels = ['$10^{-2}$', '$10^{-1}$', '$10^{0}$'])
cbarvap.ax.set_title(r'$\rho_{\mathrm{vap}} [g/cm^3]$', fontsize = 12)

den0 = dust_1_rho_xz + dust_2_rho_xz
den1 = dust_3_rho_xz + dust_4_rho_xz
watercomp0 = where(den0 > 0.0, dust_1_rho_xz/den0, 0.0)
watercomp1 = where(den1 > 0.0, dust_3_rho_xz/den1, 0.0)
#plot the dust mass contour 
axs[0,2].set_xlabel(r'$R$ [AU]', fontsize = 12)
axs[0,2].set_ylabel(r'$z$ [AU]', fontsize = 12)
axs[0,2].set_ylim(-0.25, 0.25)
# axs[0,2].plot(rad, H_profile(rad)/AU, '--', c='gray', lw=1)
# axs[0,2].plot(rad, -H_profile(rad)/AU, '--', c='gray', lw=1)
axs[0,2].plot(xx_exp, -yy1, '--', c='k', lw=1, zorder=10)
axs[0,2].plot(xx_exp, yy1, '--', c='k', lw=1, zorder=10)

c1 = axs[0,2].contourf(x_xz_c, y_xz_c, m_p1_xz, levels = logspace(-12, -1, 21), norm = LogNorm(),cmap = 'Purples', alpha = 1.0, extend = 'both')
axs[0,2].contour(x_xz_c, -y_xz_c, watercomp1, levels = [0.5], colors = 'k', linewidths = 2.0)
axs[0,2].contourf(x_xz_c, -y_xz_c, watercomp1, levels = linspace(0.1,0.7,21), cmap = 'Blues', alpha = 0.8, extend = 'both')
cbar0 = fig.colorbar(c1, ax=axs[0,2], location = 'right', shrink = 0.8, pad = 0.04, anchor=(0,0))
cbar0.ax.set_title(r'$m [g]$', fontsize = 12)

cbar0.set_ticks(ticks)

# ax[0].contour(x_xz_c,y_xz_c,r_snow_2d(tem_xz,rho_xz,0.4) ,levels = [1.e-3,1.0,1.e3], cmap = 'Greens_r', alpha = 0.7, linewidths = 5.0)

#plot the water mass fraction 
axs[1,2].plot(xx_exp, -yy0, '--', c='k', lw=1, zorder=10)
axs[1,2].plot(xx_exp, yy0, '--', c='k', lw=1, zorder=10)
axs[1,2].plot(xx_exp, yy_g, '-', c='r', lw=1, zorder=10)
# axs[1,2].plot(rad, H_profile(rad)/AU, '--', c='gray', lw=1)
axs[1,2].set_xlabel(r'$R$ [AU]', fontsize = 12)
axs[1,2].set_ylim(-0.25, 0.25)
axs[1,2].set_ylabel(r'$z$ [AU]', fontsize = 12)
c0 = axs[1,2].contourf(x_xz_c, y_xz_c,m_p_xz, levels = logspace(-12, -1, 21), norm = LogNorm(), cmap = 'Purples', alpha = 1.0, extend='both')
ccomp0 = axs[1,2].contourf(x_xz_c, -y_xz_c, watercomp0, levels = linspace(0.1,0.7,21), cmap = 'Blues', alpha = 0.8, extend = 'both')
#also plot the 1/2 line 
axs[1,2].contour(x_xz_c, -y_xz_c, watercomp0, levels = [0.5], colors = 'k', linewidths = 2.0)

axs[0,2].text(0.05, 0.1, '$pop_1$', transform=axs[0,2].transAxes, fontsize=18, fontweight='bold', va='top', ha='left')
axs[1,2].text(0.05, 0.05, '$pop_0$', transform=axs[1,2].transAxes, fontsize=18, fontweight='bold', va='bottom', ha='left')

cbarcomp0 = fig.colorbar(ccomp0, ax=axs[1,2], location='right', shrink=0.8, pad=0.04, anchor=(0,1))
cbarcomp0.ax.set_title(r'$f_{\mathrm{H_2 O}}$', fontsize = 12)
cbarcomp0.set_ticks([0.3, 0.5, 0.7], labels = ['0.3', '0.5', '0.7'])
cbarcomp0.ax.hlines(0.5, 0,1, color='k', linewidth=2)  # Mark the 0.5 line on the colorbar


#plot the tempearture profile
axs[1,0].set_xlabel(r'$R$ [AU]', fontsize = 12)
axs[1,0].set_ylabel(r'$z$ [AU]', fontsize = 12)
ax1 = axs[1,0].pcolormesh(x_xz,y_xz,tem_xz,norm = Normalize(vmin = 100,vmax = 600,clip = True) ,cmap = 'coolwarm', alpha = 1)
C_Tem = axs[1,0].contour(x_xz_c,y_xz_c,tem_xz,levels = linspace(150,200,11,endpoint=True), cmap = 'Greys_r', alpha = 1.0, linewidths = 1.0)
C = axs[1,0].contour(x_xz_c,y_xz_c,tau_opt,levels = array([0.1,0.5,1.0,5.0]), colors = 'black', linestyles = 'dotted')
# axs[1,0].contour(x_xz_c,y_xz_c,tau_ir,levels = array([0.1,0.5,1.0,5.0]), colors = 'orange', linestyles = 'dotted')
# ax[1].annotate(r'$\tau_{R} = 0.1, 0.5, 1.0, 5.0$',xy = (2.0,0.14),xytext = (2.0,0.14),fontsize = 15)
divider = make_axes_locatable(axs[1,0])
cl1 = fig.colorbar(ax1, ax=axs[1,0])
cl1.set_label(r'$T(\mathrm{K})$')
cb_ymin, cb_ymax = cl1.ax.get_ylim()
# Get colors from the contourf object
# colors = C_Tem.get_array()
plt.draw() # Force the figure to update and draw to get the colors
colors = C_Tem.get_edgecolors()  # Get the edge colors of the contour lines
# colors = C_Tem.colors
# Define levels for the second contour (make sure these are within the range of Z1)
second_contour_levels = C_Tem.levels
# Add lines to the colorbar
color_id = 0
for level in second_contour_levels:
    # Normalize level value to colorbar scale
    fmax = ax1.get_clim()[1]
    fmin = ax1.get_clim()[0]
    norm_level = (level - fmin) / (fmax - fmin)
    # Calculate y position on the colorbar
    y = cb_ymin + norm_level * (cb_ymax - cb_ymin)
    # Choose the color
    color = colors[color_id]
    color_id += 1
    # Draw a horizontal line on the colorbar
    cl1.ax.hlines(y, 0, 1, color=color, linewidth=2)


# sublimation / condensation rate
P_eq = P_eq0*exp(-T_a/tem_xz)
P_vap = dust_3_rho_xz * kB_mp * tem_xz / mu_z
rate_ratio = P_eq/P_vap * (dust_3_rho_xz/dust_1_rho_xz)

# axs[0,1].streamplot(x1_exp_half,x3_exp, water_flx_x_xz/normal2, water_flx_z_xz/normal2,linewidth = lw_flx_water, arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='pink',zorder=4)

legend_handles_panel1 = [
    Line2D([0], [0], color='black', lw=2, label='surface density'),
    Line2D([0], [0], color='black', lw=2, linestyle='--', label='number density')
]


# axs[1,1].plot(rad, H_profile(rad)/AU, '--', c='gray', lw=1)
axs[1,1].set_ylim(0, 0.25)

axs[1,1].plot(xx_exp, yy0, '-.', c='k', lw=1, zorder=10,label = r'$H_{\mathrm{0}}$')
axs[1,1].plot(xx_exp, yy1, '--', c='k', lw=1, zorder=10, label = r'$H_{\mathrm{1}}$')
# axs[0,2].set_xlabel(r'$R$ [AU]', fontsize = 12)
axs[1,1].set_ylabel(r'$z$ [AU]', fontsize = 12)
# reconstruct the dust size distribution. 
# The crude one: 
num_pp = dust_1_rho_xz + dust_2_rho_xz
den_pp = dust_3_rho_xz + dust_4_rho_xz
mass_ratio = where(m_p1_xz > 0.0, m_p_xz/m_p1_xz, nan)
density_ratio = where(den_pp > 0.0, num_pp/den_pp, nan)
pp= log10(density_ratio)/log10(mass_ratio)
ax0 = axs[1,1].contourf(x_xz_c,y_xz_c,pp,levels = linspace(-8, 8, 31), cmap = 'coolwarm', alpha = 0.8)
#label the 1/6 line 
axs[1,1].contour(x_xz_c,y_xz_c, pp, levels = [1.0/6.0], colors = 'k', linewidths = 2.0)
cbar0 = fig.colorbar(ax0, ax=axs[1,1])
cbar0.ax.set_title(r'$p$', fontsize = 12)
#mark the 1/6 line in color bar 
cbar0.ax.plot([1,1],[1.0/6.0,1.0/6.0], color = 'k', linewidth = 2.0)

axs[1,1].legend(frameon=False, loc='upper left', fontsize=12)

axs[0,0].legend(handles=legend_handles_panel1, loc='upper right', frameon=True, fontsize=12)
fig.legend(handles=legend_handles, loc='upper left', ncol=3, frameon=False, fontsize=15, bbox_to_anchor=(0.05,1.00))
# plt.tight_layout()
# if not singlepop:
plt.savefig('./plots/2dprop_{:05d}.png'.format(int(filenum)), dpi = 300, bbox_inches='tight')
plt.close()


if singlepop: 
    fig, [ax, axm] = plt.subplots(1, 2, figsize=(18, 6))
    ticks = logspace(-12, -1, 5)
    ax.set_ylabel(r'$z$ [AU]', fontsize = 12)
    ax.set_ylim(0., 0.25)
    ax.set_xlim(rin/L_norm, rout/L_norm)
    legends = [Line2D([0], [0], color='k', lw=2, marker = '>', label=r'$10^{-3}~\rho_{0}c_{\mathrm{s,0}}$'),
               Line2D([0], [0], color='k', ls = '--', lw=1, label=r'$H_{peb}$'), 
               Line2D([0], [0], color='gray', ls = '-.', lw=1, label=r'$H_{\mathrm{gas}}$')]
    ax.legend(handles=legends, loc='upper right',fontsize = 12, framealpha = 0.6)
# the vapor
    ax0 =  ax.contourf(x_xz_c,y_xz_c,dust_3_rho_mod*UNIT_DEN,levels = logspace(-13, -11,25), norm = LogNorm(), cmap = 'RdPu', alpha = 0.7, extend = 'both',zorder=3, antialiased = True)

    crho1= ax.contourf(x_xz_c,y_xz_c,dust_1_rho_mod*UNIT_DEN,levels = logspace(-13, log10(3e-11),20), norm = LogNorm(), cmap = 'Blues', alpha = 1.0, extend = 'both', antialiased = True,zorder=4)

    ax.contour(x_xz_c,y_xz_c,tau_ir,levels = array([0.5,1.0]), colors = 'black', linestyles = 'dotted', zorder = 20)
    ax.plot(xx_exp, yy0, '--', c='k', lw=1, zorder=10)
    ax.plot(xx_exp, yy_g, '-.', c='gray', lw=1, zorder=10)
    ax.contour(x_xz_c,y_xz_c, dust_1_rho_xz/rho_xz,levels = [d2g_snow], cmap = 'Blues_r', alpha = 0.7, linewidths = 3.0, zorder = 4)
    # ax.text(0.05, 0.95, 'pop$_1$', transform=axs[1,0].transAxes, fontsize=18, va='top', ha='left')
    # ax.text(0.05, 0.05, 'pop$_0$', transform=axs[1,0].transAxes, fontsize=18, va='bottom', ha='left')

    ax.streamplot(x1_exp_half,x3_exp, ice_flx_x_xz/normal2, ice_flx_z_xz/normal2,linewidth = lw_flx_ice, arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='blue',zorder=4)
    ax.streamplot(x1_exp_half,x3_exp, water_flx_x_xz/normal2, water_flx_z_xz/normal2,linewidth = lw_flx_water, arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='pink',zorder=4)
#get the advection flux of the ice and water vapor 

#move the colorbar to be aligned with the bottom of top figure 
    cbarrho = fig.colorbar(crho1, ax=ax,location = 'right', shrink = 0.45, pad =-0.15,anchor=(0,-0.))
    cbarrho.set_ticks([1e-13, 1e-12,1e-11], labels = ['$10^{-13}$', '$10^{-12}$', '$10^{-11}$'])
    cbarrho.ax.set_title(r'$\rho_{\mathrm{ice}} [g/cm^3]$', fontsize = 12)
    cbarvap = fig.colorbar(ax0, ax=ax, location = 'right', shrink = 0.45, pad =0.07, anchor=(0,1))
    cbarvap.set_ticks([1e-13, 1e-12, 1e-11], labels = ['$10^{-13}$', '$10^{-12}$', '$10^{-11}$'])
    cbarvap.ax.set_title(r'$\rho_{\mathrm{vap}} [g/cm^3]$', fontsize = 12)

    axm.set_xlabel(r'$R$ [AU]', fontsize = 13)
    axm.set_ylabel(r'$z$ [AU]', fontsize = 12)
    axm.set_xlim(rin/L_norm, rout/L_norm)
    axm.set_ylim(-0.25, 0.25)
# axs[mlot(rad, H_profile(rad)/AU, '--', c='gray', lw=1)
# axs[mlot(rad, -H_profile(rad)/AU, '--', c='gray', lw=1)
    axm.plot(xx_exp, -yy1, '--', c='k', lw=1, zorder=10)
    axm.plot(xx_exp, yy1, '--', c='k', lw=1, zorder=10)


    cmp = axm.contourf(x_xz_c, y_xz_c, m_p_xz, levels = logspace(0, 2, 31), norm = LogNorm(),cmap = 'Purples', alpha = 1.0,extend = 'both')
    axm.contour(x_xz_c, -y_xz_c, watercomp1, levels = [0.5], colors = 'k', linewidths = 2.0)
    cwc = axm.contourf(x_xz_c, -y_xz_c, watercomp0, levels = linspace(0.4,0.7,21), cmap = 'Blues', alpha = 0.8,extend = 'both')

    cbar0 = fig.colorbar(cmp, ax=axm, location = 'right', shrink = 0.45, pad = - 0.15, anchor=(0,0))
    cbar0.ax.set_title(r'$m [g]$', fontsize = 12)
    cbar0.set_ticks([1e0, 1e1, 1e2], labels = ['$1$', '$10^{1}$', '$10^{2}$'])

    cbarcomp0 = fig.colorbar(cwc, ax=axm, location='right', shrink=0.45, pad=0.04, anchor=(0,1))
    cbarcomp0.ax.set_title(r'$f_{\mathrm{H_2 O}}$', fontsize = 12)
    cbarcomp0.set_ticks([0.4, 0.7], labels = ['0.4', '0.7'])
    

    fig.savefig('./plots/2ddust_{:05d}.png'.format(int(filenum)), dpi = 300, bbox_inches='tight')


    fig = plt.figure(figsize = (7,15),facecolor='white')
    axes = fig.subplots(3,1)
    ax = axes.flatten()
    fig.subplots_adjust(hspace = 0.06)

    ax[0].set_ylim(0, 30)
# ax[0].plot(xx_exp,(sigma_gas-sigma_vap)*0.4, color = 'k', alpha = 1.0, label = '$ f_{\mathrm{i/g}} \Sigma_{\mathrm{xy}}$')
# here the 0.4 is from the 0.8/2, in which 0.8 is the dust-to-gas flux ratio, so the vapor should be the half of it
    ax[0].plot(xx_exp, sigma_ice0, c = colD['si'], lw = lwD['si'], label = 'ice 0')
    ax[0].plot(xx_exp, sigma_sil0, c = colD['ss'], lw = lwD['ss'], label = 'silicate 0')
    ax[0].plot(xx_exp, sigma_vap , c = colD['va'], lw = lwD['va'], label = 'vapor') 

#plot the column density 
# ice mass
    r_in_index = sum(xx_exp < 1.0)
    r_out_index = len(xx_exp) - ((sigma_ice / sigma_sil)[::-1] > 1.5).argmax()
    dust_mass = sum(((sigma_ice)*2*pi* xx_exp* (xx_exp[1]-xx_exp[0]) * AU**2)[r_in_index:r_out_index])
# ax[0].fill_between(xx_exp[r_in_index:r_out_index],0,sigma_ice[r_in_index:r_out_index],color = 'tab:blue',alpha = 0.3)
    ax[0].annotate(r'$M_{\mathrm{ice}} = %.2f~M_{\oplus}$'%(dust_mass/M_e),xy = (0.3,0.5), xycoords = 'axes fraction',fontsize = 15)
    ax[0].annotate(r'$f_{\mathrm{i/g}} \dot{M}_{\mathrm{acc}} /(3 \pi \nu)$',xy = (2.5,72), rotation = -10, fontsize = 16)
# ax[0].set_ylim(0,800)
# ax[0].set_xlabel('$r$ [au]')
    ax[0].set_ylabel(r'$\Sigma$ [g~cm$^{-2}$]')
# ax[0].legend(handles=legend_handles,
#     loc = 'upper right',fontsize = 15,frameon = False)
# d2g
# ax00 = ax[0].twinx()
    ax00 = ax[1]
    # ax00.set_yscale('log')
    ax00.plot(rad, ((dust_1_rho_xz + dust_2_rho_xz)/rho_xz)[:,-1], 'k', lw = 3.0,label = '$d/g$')
# ax00.plot(xx_exp, (sigma_ice+sigma_sil)/sigma_gas, 'k', lw = 3.0,label = '$d/g$')
    ax00.plot(rad, (dust_5_rho_xz/rho_xz)[:,-1],'tab:red', linestyle = '-', lw = 3.0)
    ax00.set_ylim(0.,1.0)


# # ax00.vlines(r_snow, -0.01, 1.2, linestyle = '--', color = 'grey')
# ax00.vlines(r_pk, -0.01, 1.2, linestyle = '--', color = 'grey')
# ax00.plot([2.177,2.617],[d2g_pk/2]*2,linestyle = '--', color = 'grey')
# # ax00.vlines(r_vap_outer, -0.01, 1.2, linestyle = '--', color = 'grey')
# ax00.annotate('$r_{\mathrm{snow,mid}}$',xy = (r_snow,0.8),fontsize = 20)
# ax00.annotate('$r_{\mathrm{pk}}$',xy = (r_pk,0.9),fontsize = 20)
# ax00.annotate('FWHM',xy = (r_pk - 0.1,0.27),fontsize = 15)
# # ax00.annotate('$r_{\mathrm{vap,outer}}$',xy = (r_vap_outer,0.9),fontsize = 20)
# ax00.set_ylabel('solid/vapor-to-gas ratio')

    custom_lines2 = [Line2D([0], [10], color='tab:red', lw=3, linestyle='-',label='vapor'),
                    Line2D([0,0.1], [0,0.1], color='k', lw=3, linestyle='-',label='ice+sil')]
    ax[1].legend(handles=custom_lines2,handlelength = 2, loc = (0.74,0.6))

# flux
    p2g_flux_inp = []
    for i in range(1, N_P*N_Z + 1): 
        kk = 'p2g_flux_'+str(i)
        p2g_flux_inp.append(athinputs['dust'][kk])

    ax[2].set_yscale('symlog', linthresh = 1e-2)
    ax[2].plot(xx_exp,flux_ice_face*1e8, lw =lwD['si'],color='darkblue', alpha = alpD['si'], label = r'$\mathcal{F}_{\mathrm{ice}}$')
# ax[2].plot(xx_exp,(flux_ice_face + flux_ice1_face)*1e8,lw =lwD['li'],color='blue', alpha = 1, label = r'$\mathcal{F}_{\mathrm{ice}}$')
# ax[2].plot(xx_exp,flx_ice1_x*1e8,lw =5,color='skyblue', alpha = 0.8, label = r'$\mathcal{F}_{\mathrm{ice,small}}$')
# ax[2].plot(xx_exp,flx_ice_x*1e8,lw =5,color='black', alpha = 0.4, label = r'$\mathcal{F}_{\mathrm{ice,big}}$')
    ax[2].plot(xx_exp,flux_sil_face*1e8, lw =lwD['ss'],color=colD['ss'], alpha = alpD['ss'], label = r'$\mathcal{F}_{\mathrm{sil}}$')
# ax[2].plot(xx_exp,(flux_sil1_face + flux_sil_face)*1e8,lw =lwD['ls'],color='orange', alpha = 1, label = r'$\mathcal{F}_{\mathrm{sil}}$')
# ax[2].plot(xx_exp,flux_sil*1e8,
# ax[2].plot(xx_exp,flux_sil1*1e8,'tab:blue', lw =2.0 , alpha = 1.0, linestyle = '-', label = r'$\mathcal{F}_{\mathrm{sil,small}}$')

    ax[2].axhline(-0.05, c= 'k', ls='--')
    ax[2].plot(xx_exp,flux_vap_face*1e8,  lw =lwD['va'],color=colD['va'], alpha = alpD['va'], label = r'$\mathcal{F}_{\mathrm{vap}}$')
    ax[2].plot(xx_exp,flux_water_face*1e8,lw =3,color='cyan', alpha = 0.6, label = r'$\mathcal{F}_{\mathrm{water}}$')
    ax[2].plot(xx_exp,flux_gas_face*1e8,lw =3,color='grey', alpha = 0.6, label = r'$\mathcal{F}_{\mathrm{xy}}$')

# ax[2].plot(xx_exp, -xx_exp/xx_exp,'k--')
# ax[2].plot(xx_exp, -xx_exp/xx_exp*0.4,'k--')
    ax[2].axhline(0.0, c= 'k', ls='--')
# ax[2].axhline(-0.4, c= 'k', ls='--')

    ax[2].set_xlim(rin/L_norm,rout/L_norm)
    ax[2].set_ylim(-3,5.0)
    ax[2].annotate(r'$\dot{M}_{\mathrm{acc}}$',xy=(1.0,-0.9),fontsize = 15)
    ax[2].annotate(r'$f_{\mathrm{i/g}} \dot{M}_{\mathrm{acc}}$',xy=(1.0,-0.25),fontsize = 15)
    ax[2].set_ylabel(r'Radial Mass Flux [$10^{-8}M_{\odot}$/yr]',fontsize = 15)

    ax[2].legend(loc='upper right', fontsize = 10)
    for i in range(len(axes)):
        ax[i].set_xlim(rin/L_norm,rout/L_norm)
    for i in range(2):  
        ax[i].set_xticklabels([])   

    ax[2].set_xlabel(r'$r$ [au]')

    ax[0].annotate('(a)',xy = (0.02,0.92),xycoords = 'axes fraction',fontsize = 20)

    ax[1].annotate('(b)',xy = (0.02,0.92),xycoords = 'axes fraction',fontsize = 20)
    ax[2].annotate('(c)',xy = (0.02,0.92),xycoords = 'axes fraction',fontsize = 20)

    ax[1].axvline(xx_exp[51], ls='dotted', c= 'black', lw=1)

    plt.savefig('./plots/fig_snow_2d_{:05d}.png'.format(int(filenum)), bbox_inches='tight', dpi = 500) 
    plt.close()

import pdb; pdb.set_trace()
#==============================================================================
#==============================================================================
fig, axs = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True, sharex=True)
axs[0,1].set_title("time: {:.2f} yr".format(simu_time*UNIT_T/YR),loc= 'right', y=1.05)

axs[0,0].set_ylabel(r'$\Sigma$ [g/cm$^2$]', fontsize = 12)

# axs[0,0].set_yscale('log')
axs[0,0].set_ylim(1e-2, 50)
# sax[0].plot(xx_exp,(sigma_gas-sigma_vap)*0.4, color = 'k', alpha = 1.0, label = '$ f_{\mathrm{i/g}} \Sigma_{\mathrm{xy}}$')
# shere the 0.4 is from the 0.8/2, in which 0.8 is the dust-to-gas flux ratio, so the vapor should be the half of it
axs[0,0].plot(xx_exp,(sigma_gas)*0.4, color = 'k', linestyle='-', alpha = 1.0, label = 'gas')
axs[0,0].plot(xx_exp,sqrt(2*pi)*0.4*(xx_exp/3)**(-1)*UNIT_SIGMA,color = 'grey',linewidth = 5, alpha = 0.5)
ice_style = [('si', 'ice {}'), ('li', 'ice {}')]
sil_style = [('ss', 'silicate {}'), ('ls', 'silicate {}')]
for p in range(len(sigma_ice_by_pop)):
    i_style = p if p < len(ice_style) else len(ice_style)-1
    key = ice_style[i_style][0]
    axs[0,0].plot(xx_exp, sigma_ice_by_pop[p], c = colD[key], lw = lwD[key], label = ice_style[i_style][1].format(p))
for p in range(len(sigma_sil_by_pop)):
    i_style = p if p < len(sil_style) else len(sil_style)-1
    key = sil_style[i_style][0]
    axs[0,0].plot(xx_exp, sigma_sil_by_pop[p], c = colD[key], lw = lwD[key], label = sil_style[i_style][1].format(p))
axs[0,0].plot(xx_exp, sigma_vap , c = colD['va'], lw = lwD['va'], label = 'vapor') 
axs[0,0].legend(handles=legend_handles, loc='upper right', ncol=3, frameon=False, fontsize=15)
# density 
ticks = logspace(-12, -1, 5)
axs[1,0].set_ylabel(r'$z$ [AU]', fontsize = 12)
axs[1,0].set_ylim(-0.25, 0.25)
axs[1,0].set_xlim(rin/L_norm, rout/L_norm)
legends = [Line2D([0], [0], color='k', lw=2, marker = '>', label=r'$10^{-3}~\rho_{0}c_{\mathrm{s,0}}$'),
           Line2D([0], [0], color='k', ls = '--', lw=1, label=r'$H_{peb}$')]
axs[1,0].legend(handles=legends, loc='upper right',fontsize = 15, framealpha = 0.6)
# the vapor
ax0 =  axs[1,0].contourf(x_xz_c,y_xz_c,dust_5_rho_mod*UNIT_DEN,levels = logspace(-13, -11,25), norm = LogNorm(), cmap = 'RdPu', alpha = 0.7, extend = 'both',zorder=3, antialiased = True)
axs[1,0].contourf(x_xz_c,-y_xz_c,dust_5_rho_mod*UNIT_DEN,levels = logspace(-13, -11,25), norm = LogNorm(), cmap = 'RdPu', alpha = 0.7, extend = 'both',zorder=3, antialiased = True)

crho1= axs[1,0].contourf(x_xz_c,y_xz_c,dust_3_rho_mod*UNIT_DEN,levels = logspace(-13, log10(3e-11),20), norm = LogNorm(), cmap = 'Blues', alpha = 1.0, extend = 'both', antialiased = True,zorder=4)
axs[1,0].contourf(x_xz_c,-y_xz_c,dust_1_rho_mod*UNIT_DEN,levels = logspace(-13,log10(3e-11),20), norm = LogNorm(), cmap = 'Blues', alpha = 1, extend = 'both', antialiased=True, zorder=4)

axs[1,0].contour(x_xz_c,y_xz_c,tau_ir,levels = array([0.5,1.0]), colors = 'black', linestyles = 'dotted', zorder = 20)
axs[1,0].contour(x_xz_c,-y_xz_c,tau_ir,levels = array([0.5,1.0]), colors = 'black', linestyles = 'dotted', zorder = 20)
axs[1,0].plot(xx_exp, -yy0, '--', c='k', lw=1, zorder=10)
axs[1,0].plot(xx_exp, yy_g, '-', c='r', lw=1, zorder=10)
# axs[1,0].plot(rad, H_profile(rad)/AU, '--', c='gray', lw=1, zorder=10)
axs[1,0].plot(xx_exp, yy1, '--', c='k', lw=1, zorder=10)
#zxl: 0his we change to the sum of the ice in different populations.
ice_rho_xz_tot = dust_1_rho_xz + dust_3_rho_xz
axs[1,0].contour(x_xz_c,y_xz_c, dust_3_rho_xz/rho_xz,levels = [d2g_snow], cmap = 'Blues_r', norm = LogNorm(), alpha = 0.7, linewidths = 3.0, zorder = 5)
axs[1,0].contour(x_xz_c,-y_xz_c, dust_1_rho_xz/rho_xz,levels = [d2g_snow], cmap = 'Blues_r', alpha = 0.7, linewidths = 3.0, zorder = 4)
#label the panels in the lower left corner
axs[1,0].text(0.05, 0.95, 'pop$_1$', transform=axs[1,0].transAxes, fontsize=18, va='top', ha='left')
axs[1,0].text(0.05, 0.05, 'pop$_0$', transform=axs[1,0].transAxes, fontsize=18, va='bottom', ha='left')

# axs[1,0].streamplot(x1_exp_half,x3_exp, flx_x_xz/normal2, flx_z_xz/normal2,linewidth = lw_flx_gas, arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='black',zorder=4)
axs[1,0].streamplot(x1_exp_half,x3_exp, ice1_flx_x_xz/normal2, ice1_flx_z_xz/normal2,linewidth = lw_flx_ice1, arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='blue',zorder=4)
axs[1,0].streamplot(x1_exp_half,x3_exp, water_flx_x_xz/normal2, water_flx_z_xz/normal2,linewidth = lw_flx_water, arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='pink',zorder=4)
axs[1,0].streamplot(x1_exp_half,z_neg, 
                    water_flx_x_xz[::-1,:]/normal2, 
                    - water_flx_z_xz[::-1,:]/normal2,
                    linewidth = lw_flx_water[::-1,:], arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='pink',zorder=4)
axs[1,0].streamplot(x1_exp_half,z_neg, 
                    ice_flx_x_xz[::-1,:]/normal2, 
                    - ice_flx_z_xz[::-1,:]/normal2,linewidth = lw_flx_ice[::-1,:], arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='blue',zorder=4)
#get the advection flux of the ice and water vapor 
flx_ice_adv_x = dust_1_rho*dust_1_vx1*dS_R*UNIT_Fm
flx_ice_adv_z = dust_1_rho*dust_1_vx2*dS_theta*UNIT_Fm

flx_ice1_adv_x = dust_3_rho*dust_3_vx1*dS_R*UNIT_Fm 
flx_ice1_adv_z = dust_3_rho*dust_3_vx2*dS_theta*UNIT_Fm 

ice_flx_adv_x, ice_flx_adv_y, ice_flx_adv_z = v_Intpl_Sph2car(rad,theta,phi,x1_exp_half,slice_exp,x3_exp,(flx_ice_adv_x).T,(flx_ice_adv_z).T, (flx_ice_adv_z).T * 0.0)
ice_flx_adv_x_xz = ice_flx_adv_x[:,0,:]
ice_flx_adv_z_xz = ice_flx_adv_z[:,0,:]

ice1_flx_adv_x, ice1_flx_adv_y, ice1_flx_adv_z = v_Intpl_Sph2car(rad,theta,phi,x1_exp_half,slice_exp,x3_exp,(flx_ice1_adv_x).T,(flx_ice1_adv_z).T, (flx_ice1_adv_z).T * 0.0)
ice1_flx_adv_x_xz = ice1_flx_adv_x[:,0,:]
ice1_flx_adv_z_xz = ice1_flx_adv_z[:,0,:]


lw_ice_adv = sqrt(ice_flx_adv_x_xz**2 + ice_flx_adv_z_xz**2)/normal2
lw_ice1_adv = sqrt(ice1_flx_adv_x_xz**2 +ice1_flx_adv_z_xz**2)/normal2

# axs[1,0].streamplot(x1_exp_half,x3_exp, ice1_flx_adv_x_xz/normal2, ice1_flx_adv_z_xz/normal2,linewidth = lw_ice1_adv, arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='green',zorder=4)
# axs[1,0].streamplot(x1_exp_half,z_neg, 
#                     ice_flx_adv_x_xz[::-1,:]/normal2, 
#                     ice_flx_adv_z_xz[::-1,:]/normal2,
#                     linewidth = lw_ice_adv[::-1,:], arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='green',zorder=4)
 
# axs[1,0].axhline(y=0.8363558854159211*UNIT_L/AU , color='k', linestyle='-', linewidth=1, zorder=10)
# axs[1,0].axvline(x=9.2467337671015173*UNIT_L/AU , color='k', linestyle='-', linewidth=1, zorder=10)
#
# axs[1,0].axhline(y=0.72631927983798383*UNIT_L/AU , color='k', linestyle='-', linewidth=1, zorder=10)
# axs[1,0].axvline(x=9.2560270543284879*UNIT_L/AU , color='k', linestyle='-', linewidth=1, zorder=10)

#move the colorbar to be aligned with the bottom of top figure 
cbarrho = fig.colorbar(crho1, ax=axs[1,0],location = 'right', shrink = 0.45, pad =-0.085,anchor=(0,-0.))
cbarrho.set_ticks([1e-13, 1e-12,1e-11], labels = ['$10^{-13}$', '$10^{-12}$', '$10^{-11}$'])
cbarrho.ax.set_title(r'$\rho_{\mathrm{ice}} [g/cm^3]$', fontsize = 12)
cbarvap = fig.colorbar(ax0, ax=axs[1,0], location = 'right', shrink = 0.45, pad =0.04, anchor=(0,1))
cbarvap.set_ticks([1e-13, 1e-12, 1e-11], labels = ['$10^{-13}$', '$10^{-12}$', '$10^{-11}$'])
cbarvap.ax.set_title(r'$\rho_{\mathrm{vap}} [g/cm^3]$', fontsize = 12)

# mass and water comp
axs[0,1].set_xlabel(r'$R$ [AU]', fontsize = 12)
axs[0,1].set_ylabel(r'$z$ [AU]', fontsize = 12)
axs[0,1].set_xlim(rin/L_norm, rout/L_norm)
axs[0,1].set_ylim(-0.25, 0.25)
# axs[1,2].plot(rad, H_profile(rad)/AU, '--', c='gray', lw=1)
# axs[1,2].plot(rad, -H_profile(rad)/AU, '--', c='gray', lw=1)
axs[0,1].plot(xx_exp, -yy1, '--', c='k', lw=1, zorder=10)
axs[0,1].plot(xx_exp, yy1, '--', c='k', lw=1, zorder=10)


c1 = axs[0,1].contourf(x_xz_c, y_xz_c, m_p1_xz, levels = logspace(-17, 2, 31), norm = LogNorm(),cmap = 'Purples', alpha = 1.0,extend = 'both')
axs[0,1].contour(x_xz_c, -y_xz_c, watercomp1, levels = [0.5], colors = 'k', linewidths = 2.0)
axs[0,1].contourf(x_xz_c, -y_xz_c, watercomp1, levels = linspace(0.4,0.7,21), cmap = 'Blues', alpha = 0.8,extend = 'both')
cbar0 = fig.colorbar(c1, ax=axs[0,1], location = 'right', shrink = 0.8, pad = 0.04, anchor=(0,0))
cbar0.ax.set_title(r'$m [g]$', fontsize = 12)

cbar0.set_ticks(ticks)

# ax[0].contour(x_xz_c,y_xz_c,r_snow_2d(tem_xz,rho_xz,0.4) ,levels = [1.e-3,1.0,1.e3], cmap = 'Greens_r', alpha = 0.7, linewidths = 5.0)

#plot the water mass fraction 
axs[1,1].plot(xx_exp, -yy0, '--', c='k', lw=1, zorder=10)
axs[1,1].plot(xx_exp, yy0, '--', c='k', lw=1, zorder=10)
# axs[1,2].plot(rad, H_profile(rad)/AU, '--', c='gray', lw=1)
axs[1,1].set_xlabel(r'$R$ [AU]', fontsize = 12)
axs[1,1].set_ylim(-0.25, 0.25)
axs[1,1].set_ylabel(r'$z$ [AU]', fontsize = 12)
c0 = axs[1,1].contourf(x_xz_c, y_xz_c,m_p_xz, levels = logspace(-12, -1, 21), norm = LogNorm(), cmap = 'Purples', alpha = 1.0,extend = 'both')
ccomp0 = axs[1,1].contourf(x_xz_c, -y_xz_c, watercomp0, levels = linspace(0.4,0.7,21), cmap = 'Blues', alpha = 0.8,extend = 'both')
#also plot the 1/2 line 
axs[1,1].contour(x_xz_c, -y_xz_c, watercomp0, levels = [0.5], colors = 'k', linewidths = 2.0)

axs[0,1].text(0.05, 0.1, 'pop$_1$', transform=axs[0,1].transAxes, fontsize=18, va='top', ha='left')
axs[1,1].text(0.05, 0.05, 'pop$_0$', transform=axs[1,1].transAxes, fontsize=18, va='bottom', ha='left')

cbarcomp0 = fig.colorbar(ccomp0, ax=axs[1,1], location='right', shrink=0.8, pad=0.04, anchor=(0,1))
cbarcomp0.ax.set_title(r'$f_{\mathrm{H_2 O}}$', fontsize = 12)
cbarcomp0.set_ticks([0.4, 0.5, 0.7], labels = ['0.4', '0.5', '0.7'])
cbarcomp0.ax.hlines(0.5, 0,1, color='k', linewidth=2)  # Mark the 0.5 line on the colorbar

# axs[1,0].set_ylim(0, 0.25)
#
# axs[1,0].plot(xx_exp, yy0, '-.', c='k', lw=1, zorder=10,label = r'$H_{\mathrm{0}}$')
# axs[1,0].plot(xx_exp, yy1, '--', c='k', lw=1, zorder=10, label = r'$H_{\mathrm{1}}$')
# # axs[0,2].set_xlabel(r'$R$ [AU]', fontsize = 12)
# axs[1,0].set_ylabel(r'$z$ [AU]', fontsize = 12)
# # reconstruct the dust size distribution. 
# # The crude one: 
# pp= log10((dust_1_rho_xz+dust_2_rho_xz)/(dust_3_rho_xz + dust_4_rho_xz))/log10(m_p_xz/m_p1_xz)
# ax0 = axs[1,0].contourf(x_xz_c,y_xz_c,pp,levels = linspace(-1, 1, 21), cmap = 'coolwarm', alpha = 0.8, extend = 'both')
# #label the 1/6 line 
# axs[1,0].contour(x_xz_c,y_xz_c, pp, levels = [1.0/6.0], colors = 'k', linewidths = 2.0, extend = 'both')
# cbar0 = fig.colorbar(ax0, ax=axs[1,0])
# cbar0.ax.set_title(r'$p$', fontsize = 12)
# #mark the 1/6 line in color bar 
# cbar0.ax.plot([1,0],[1.0/6.0,1.0/6.0], color = 'k', linewidth = 2.0)
#
# axs[1,0].legend(frameon=False, loc='upper left', fontsize=12)
#
# axs[0,0].legend(handles=legend_handles_panel1, loc='upper right', frameon=True, fontsize=12)

plt.savefig('./plots/2ddust_{:05d}.png'.format(int(filenum)), dpi = 300, bbox_inches='tight')
plt.close()

    


#==============================================================================
#get the relaxation time and plot it 
t_relax_xz = t_relax[0].T
sign = ones_like(t_relax_xz)
# for i in range(len(t_relax_xz)):
#     for j in range(len(t_relax_xz[0])):
#         rhoi = array([dust_1_rho_xz[i][j], dust_3_rho_xz[i][j] ])
#         rhos = array([dust_2_rho_xz[i][j], dust_4_rho_xz[i][j] ])
#         mbounds = array([mmin, sqrt(mmin*mmax[0].T[i][j]), mmax[0].T[i][j]])
#         rhorelax, M2 = get_relaxed_state(rhoi, rhos, mbounds)
#
#         if rhorelax[-1] < (rhoi[-1] + rhos[-1]):
#             sign[i][j] = -1.0
sign[drho_i1_dt_xz < 0.] = -1.0

fig, axs = plt.subplots(2, 1, figsize=(9, 9))

t_relax_signed = t_relax_xz*sign/YR 
t_relax_pos = t_relax_signed.copy() 
t_relax_pos[t_relax_signed <0.] = nan

t_relax_neg = t_relax_signed.copy() 
t_relax_neg[t_relax_signed >0.] = nan

Omega_2d = sqrt(GM_sun/((x_xz_c**2 + y_xz_c**2)*AU**2)**1.5)
t_relax_norm_pos = t_relax_pos*Omega_2d
t_relax_norm_neg = t_relax_neg*Omega_2d

# bbarneg = axs[0].contourf(x_xz_c, y_xz_c, -t_relax_norm_neg*YR, levels = logspace(0,5, 21), norm = LogNorm(),
#                        extend = 'both',cmap = "Blues")
# cbarneg = fig.colorbar(bbarneg, ax = axs[0], orientation = 'vertical',pad = -0.15, shrink = 0.45, 
#                        anchor=(0,0))
# cbarneg.set_ticks([ 1, 100, 1e4, 1e5], labels = ['-1', '-100', '-10$^{4}$', '-10$^{5}$'])
#
bbarpos = axs[0].contourf(x_xz_c, y_xz_c, 1/3e-3/Omega_2d/YR, levels = logspace(0,3, 21), norm = LogNorm(),
                       cmap = "Reds", extend = 'both')
cbarpos = fig.colorbar(bbarpos, ax = axs[0], orientation = 'vertical',pad = 0.02, shrink = 0.45, 
                       anchor=(0,1))
cbarpos.set_ticks([ 1, 100, 1e3], labels = ['1', '100', '10$^{3}$'])
# cbarpos.ax.set_title(r'$\Omega t_{relax}$')
cbarpos.ax.set_title(r'$1/\alpha \Omega$')

axs[0].plot(xx_exp, yy0, '-.', c='k', lw=1, zorder=10,label = r'$H_{\mathrm{0}}$')
axs[0].plot(xx_exp, yy1, '--', c='k', lw=1, zorder=10, label = r'$H_{\mathrm{1}}$')
axs[0].legend(frameon=False, loc='upper left', fontsize=12)
axs[0].set_ylim(0, 0.25)
axs[0].set_xlim(rin/L_norm, rout/L_norm)

bbarneg = axs[1].contourf(x_xz_c, y_xz_c, -t_relax_neg, levels = logspace(0,5, 21), norm = LogNorm(),
                       extend = 'both',cmap = "Blues")
cbarneg = fig.colorbar(bbarneg, ax = axs[1], orientation = 'vertical',pad = -0.15, shrink = 0.45, 
                       anchor=(0,0))
cbarneg.set_ticks([ 1, 100, 1e4, 1e5], labels = ['-1', '-100', '-10$^{4}$', '-10$^{5}$'])

bbarpos = axs[1].contourf(x_xz_c, y_xz_c, t_relax_pos, levels = logspace(0,3, 21), norm = LogNorm(),
                       cmap = "Reds", extend = 'both')
cbarpos = fig.colorbar(bbarpos, ax = axs[1], orientation = 'vertical',pad = 0.02, shrink = 0.45, 
                       anchor=(0,1))
cbarpos.set_ticks([ 1, 100, 1e3], labels = ['1', '100', '10$^{3}$'])
cbarpos.ax.set_title('$t_{relax}$ [yr]')
axs[1].plot(xx_exp, yy0, '-.', c='k', lw=1, zorder=10,label = r'$H_{\mathrm{0}}$')
axs[1].plot(xx_exp, yy1, '--', c='k', lw=1, zorder=10, label = r'$H_{\mathrm{1}}$')
axs[1].legend(frameon=False, loc='upper left', fontsize=12)
axs[1].set_ylim(0, 0.25)
axs[1].set_xlim(rin/L_norm, rout/L_norm)

plt.tight_layout()
plt.savefig('./plots/trelax_{:05d}.png'.format(int(filenum)) ,dpi=300)
plt.close()

fig, axs = plt.subplots(2, 1, figsize=(9, 9))
cbar = axs[0].contourf(x_xz_c, y_xz_c, st1_xz, levels = logspace(-6,-2, 21),
                    norm = LogNorm(),extend = 'both', cmap = cm.viridis)
# axs[0].contour(x_xz_c, y_xz_c, rrr, levels = [7.0], colors = 'white', linewidths = 1.5)
cbar = fig.colorbar(cbar, format=ticker.FuncFormatter(formatnum), ax = axs[0], orientation = 'vertical',)
# cbar.set_ticks([1, 10, 100, 1000, 1e4, 1e5])
cbar.ax.set_title(r'$St$')
axs[0].plot(xx_exp, yy0, '-.', c='k', lw=1, zorder=10,label = r'$H_{\mathrm{0}}$')
axs[0].plot(xx_exp, yy1, '--', c='k', lw=1, zorder=10, label = r'$H_{\mathrm{1}}$')
axs[0].legend(frameon=False, loc='upper left', fontsize=12)
axs[0].set_ylim(0, 0.25)
axs[0].set_xlim(rin/L_norm, rout/L_norm)

bbar = axs[1].contourf(x_xz_c, y_xz_c, dust_7_rho_xz/UNIT_L**3, levels = logspace(-10,1,16), 
                       norm = LogNorm(), extend = 'both',cmap = cm.viridis)
cbartre = fig.colorbar(bbar, format=ticker.FuncFormatter(formatnum), ax = axs[1], orientation = 'vertical',)
cbartre.ax.set_title('$n_1$')
axs[1].plot(xx_exp, yy0, '-.', c='k', lw=1, zorder=10,label = r'$H_{\mathrm{0}}$')
axs[1].plot(xx_exp, yy1, '--', c='k', lw=1, zorder=10, label = r'$H_{\mathrm{1}}$')
axs[1].legend(frameon=False, loc='upper left', fontsize=12)
axs[1].set_ylim(0, 0.25)
axs[1].set_xlim(rin/L_norm, rout/L_norm)

plt.tight_layout()
plt.savefig('./plots/St_{:05d}.png'.format(int(filenum)) ,dpi=300)
plt.close()

def ff_broken (pwl, prec, ml, mu):
    """integrate f(m)=prec*m**-pwl from ml to mu"""
    return prec/(2-pwl)*(mu**(2-pwl)-ml**(2-pwl))

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_xscale('log') 
ax.set_yscale('log')
radL = [2.5]
zL = [0.01]
legend_handles = []

legend_handles = [
    Line2D([0], [0], color='k', marker='o',ls='', lw=0.5, label='Sim'),
    # Line2D([0], [0], color='k', marker='s', markerfacecolor='none',ls='',
    #        label='integ'),
    Line2D([0], [0], marker='D',markeredgecolor='purple', 
           ls='', markerfacecolor = 'none',
           label=r'$m^*_{ch}$'),
    Line2D([0], [0], marker='D',markeredgecolor='purple', 
           ls='', markerfacecolor = 'k',
           label=r'$m^*_{bounds}$'),
    Line2D([0], [0], color='k', ls='--', label=r'$m^2 f^*(m)$'),
    Line2D([0], [0], color='k', ls='-', lw=0.5, label='$m_{min}$, $m_{max}$'),
    Line2D([0], [0], color='k', ls='--', lw=0.5, label='$m_{Div}$'),
]
for i in range(len(radL)):
    idxr = argmin(abs(rad - radL[i])) 

    legend_handles.append(
        Line2D([0], [0], color='k', ls='-', label='[{:.1f},{:.1f}] au'.format(radL[i], zL[i]))
    )

    m_s = array([m_p[0,-1, idxr], m_p1[0, -1, idxr]])
    rhos = array([dust_1_rho[0, -1, idxr]+ dust_2_rho[0, -1, idxr]
                  , dust_3_rho[0, -1, idxr]+ dust_4_rho[0, -1, idxr]])*UNIT_DEN

    ax.plot(m_s, rhos, marker='o', ls ='', lw=0.5, 
             label=str(radL[i]), c = 'k')

    m_bounds = array([mmin, sqrt(mmin*mmax[0,-1,idxr]), mmax[0,-1,idxr]])
    ax.axvline(x=mmin, color='k', ls='-', lw=0.5)
    ax.axvline(x=m_bounds[-1], color='k', ls='-', lw=0.5)
    ax.axvline(x=m_bounds[1], color='k', ls='--', lw=0.5)

    M1re = sum(rhos)
    c_relax = M1re/(6*(mmax[0,-1,idxr]**(1/6)- mmin**(1/6)))
    M1re_bins = []
    M2re_bins = []
    for j in range(len(m_bounds)-1):
        M1re_bins.append(ff_broken(11/6, c_relax, m_bounds[j], m_bounds[j+1]))
        M2re_bins.append(ff_broken(5/6, c_relax, m_bounds[j], m_bounds[j+1]))

    mre = array(M2re_bins)/array(M1re_bins)
    Nre_bins = array(M1re_bins)**2/array(M2re_bins)
    M1_mmax = rhos[1]*10**(1/6*log10(mmax[0,-1,idxr]/mre[1]))
    M1_mmin = rhos[0]*10**(1/6*log10(mmin/mre[0]))
    rhos_re = concatenate(([M1_mmin], [M1_mmax]))
    mreb = concatenate(([mmin], [mmax[0,-1,idxr]]))
    ax.plot(mre, M1re_bins, marker='D',markeredgecolor='purple', 
            markerfacecolor = 'none', ls='')
    ax.plot(mreb, rhos_re, marker='D',markeredgecolor='purple', 
            markerfacecolor = 'k', ls='')

    mreall = concatenate(([mmin], mre, [mmax[0,-1,idxr]]))
    ax.plot(mreall, c_relax*mreall**(1/6), markerfacecolor='none', ls='--', lw=1, c = 'k')

ax.set_xlabel('particle mass (g)')
ax.set_ylabel(r'$\rho$ (g/cm$^3$)')
ax.legend(handles=legend_handles, frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig('plots/2dinter_{:05d}.png'.format(int(filenum)), dpi=300)


#==============================================================================
#draw a vertical density distribution 
fig,ax = plt.subplots(figsize=(7,5))
r_want = 2.5 
x_idx = argmin(abs(xx_exp - r_want))
ax.set_ylim(1e-18, 3e-12)
ax.set_xlim(0, 0.4)
# ax.set_yscale('log')
# ax.set_xscale('log')

den = (dust_1_rho_intpl[:,x_idx] + dust_2_rho_intpl[:,x_idx])*UNIT_DEN
ax.plot(zz_exp,den,  label='pop0', c = colD['ss'], lw = lwD['ss'])

#fit with a gaussian 
from scipy.optimize import curve_fit
def gaussian_func(x, A, mu, sigma):
    return A * exp(-(x - mu)**2 / (2 * sigma**2))

initial_guess = [5e-12, 0, 0.2]
# popt, pcov = curve_fit(gaussian_func, zz_exp, den, p0=initial_guess)
# A_fit, mu_fit, sigma_fit = popt
# ax.plot(zz_exp, gaussian_func(zz_exp, *popt), 'r--')
ax.plot(zz_exp, gaussian_func(zz_exp, 3e-12, 0, 0.09), 'r--', label='A = {:.2e}, mu = {:.2f}, sigma = {:.2f}'.format(2.9e-12, 0, 0.09))
ax.plot(zz_exp, gaussian_func(zz_exp, 3e-12, 0, 0.08), 'b--', label='A = {:.2e}, mu = {:.2f}, sigma = {:.2f}'.format(2.9e-12, 0, 0.08))
ax.plot(zz_exp, gaussian_func(zz_exp, 3e-12, 0, 0.07), 'k--', label='A = {:.2e}, mu = {:.2f}, sigma = {:.2f}'.format(2.9e-12, 0, 0.07))



fig.savefig('plots/2dvert_{:05d}.png'.format(int(filenum)), dpi=300)
# fig, axs = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
# axs[0].set_ylabel(r'$\Sigma$ [g/cm$^2$]', fontsize = 12)
#
# # axs,0].set_yscale('log')
# axs[0].set_ylim(1e-2, 50)
# axs[0].legend(handles=legend_handles, ncol = 3, loc='upper right', frameon=True, fontsize=12)
# # sax].plot(xx_exp,(sigma_gas-sigma_vap)*0.4, color = 'k', alpha = 1.0, label = '$ f_{\mathrm{i/g}} \Sigma_{\mathrm{xy}}$')
# # she the 0.4 is from the 0.8/2, in which 0.8 is the dust-to-gas flux ratio, so the vapor should be the half of it
# # axs[0].plot(xx_exp,(sigma_gas)*0.4, color = 'k', linestyle='-', alpha = 1.0, label = 'gas')
# # axs[0].plot(xx_exp,sqrt(2*pi)*0.4*(xx_exp/3)**(-1)*UNIT_SIGMA,color = 'grey',linewidth = 5, alpha = 0.5)
# axs[0].plot(xx_exp, sigma_ice0, c = colD['si'], lw = lwD['si'], label = 'ice 0')
# axs[0].plot(xx_exp, sigma_sil0, c = colD['ss'], lw = lwD['ss'], label = 'silicate 0')
# axs[0].plot(xx_exp, sigma_ice1, c = colD['li'], lw = lwD['li'], label = 'ice 1')
# axs[0].plot(xx_exp, sigma_sil1, c = colD['ls'], lw = lwD['ls'], label = 'silicate 1')
# axs[0].plot(xx_exp, sigma_vap , c = colD['va'], lw = lwD['va'], label = 'vapor') 
#
# #plot the tempearture profile
# axs[1].set_xlabel(r'$R$ [AU]', fontsize = 12)
# axs[1].set_ylabel(r'$z$ [AU]', fontsize = 12)
# ax1 =   axs[1].pcolormesh(x_xz,y_xz,tem_xz,norm = Normalize(vmin = 100,vmax = 300,clip = True) ,cmap = 'coolwarm', alpha = 1)
# C_Tem = axs[1].contour(x_xz_c,y_xz_c,tem_xz,levels = linspace(150,200,11,endpoint=True), cmap = 'Greys_r', alpha = 1.0, linewidths = 1.0)
# C =     axs[1].contour(x_xz_c,y_xz_c,tau_opt,levels = array([0.1,0.5,1.0,5.0]), colors = 'black', linestyles = 'dotted')
# # ax[1].annotate(r'$\tau_{R} = 0.1, 0.5, 1.0, 5.0$',xy = (2.0,0.14),xytext = (2.0,0.14),fontsize = 15)
# divider = make_axes_locatable(axs[1])
# cl1 = fig.colorbar(ax1,    ax=axs[1])
# cl1.set_label(r'$T(\mathrm{K})$')
# cb_ymin, cb_ymax = cl1.ax.get_ylim()
# # Get colors from the contourf object
# # colors = C_Tem.get_array()
# plt.draw() # Force the figure to update and draw to get the colors
# colors = C_Tem.get_edgecolors()  # Get the edge colors of the contour lines
# # colors = C_Tem.colors
# # Define levels for the second contour (make sure these are within the range of Z1)
# second_contour_levels = C_Tem.levels
# # Add lines to the colorbar
# color_id = 0
# for level in second_contour_levels:
#     # Normalize level value to colorbar scale
#     fmax = ax1.get_clim()[1]
#     fmin = ax1.get_clim()[0]
#     norm_level = (level - fmin) / (fmax - fmin)
#     # Calculate y position on the colorbar
#     y = cb_ymin + norm_level * (cb_ymax - cb_ymin)
#     # Choose the color
#     color = colors[color_id]
#     color_id += 1
#     # Draw a horizontal line on the colorbar
#     cl1.ax.hlines(y, 0, 1, color=color, linewidth=2)
#
# plt.savefig('./plots/2ddisk_{:05d}.png'.format(int(filenum)), dpi = 300, bbox_inches='tight')
# plt.close()

fig = plt.figure(figsize = (7,15),facecolor='white')
axes = fig.subplots(3,1)
ax = axes.flatten()
fig.subplots_adjust(hspace = 0.06)

ax[0].set_yscale('log')
ax[0].set_ylim(1e-10, 600)
# ax[0].plot(xx_exp,(sigma_gas-sigma_vap)*0.4, color = 'k', alpha = 1.0, label = '$ f_{\mathrm{i/g}} \Sigma_{\mathrm{xy}}$')
# here the 0.4 is from the 0.8/2, in which 0.8 is the dust-to-gas flux ratio, so the vapor should be the half of it
ax[0].plot(xx_exp,(sigma_gas)*0.4, color = 'k', linestyle='-', alpha = 1.0, label = 'gas')
ax[0].plot(xx_exp,sqrt(2*pi)*0.4*(xx_exp/3)**(-1)*UNIT_SIGMA,color = 'grey',linewidth = 5, alpha = 0.5)
ax[0].plot(xx_exp, sigma_ice0, c = colD['si'], lw = lwD['si'], label = 'ice 0')
ax[0].plot(xx_exp, sigma_sil0, c = colD['ss'], lw = lwD['ss'], label = 'silicate 0')
ax[0].plot(xx_exp, sigma_ice1, c = colD['li'], lw = lwD['li'], label = 'ice 1')
ax[0].plot(xx_exp, sigma_sil1, c = colD['ls'], lw = lwD['ls'], label = 'silicate 1')
ax[0].plot(xx_exp, sigma_vap , c = colD['va'], lw = lwD['va'], label = 'vapor') 

#plot the column density 
axn = ax[0].twinx() 
axn.set_yscale('log')
axn.plot(xx_exp, n_col0, color = 'gray', lw = lwD['si'], alpha=alpD['si'], ls = 'dashed', label = 'n0')
axn.plot(xx_exp, n_col1, color = 'gray', lw = lwD['li'], alpha=alpD['li'], ls = 'dashed', label = 'n1')
# ax[0].plot(xx_exp,sigma_ice,'tab:blue', label = 'ice')
# ax[0].plot(xx_exp,sigma_sil,'tab:orange', label = 'slicate')
# ice mass
r_in_index = sum(xx_exp < 1.0)
r_out_index = len(xx_exp) - ((sigma_ice / sigma_sil)[::-1] > 1.5).argmax()
dust_mass = sum(((sigma_ice)*2*pi* xx_exp* (xx_exp[1]-xx_exp[0]) * AU**2)[r_in_index:r_out_index])
# ax[0].fill_between(xx_exp[r_in_index:r_out_index],0,sigma_ice[r_in_index:r_out_index],color = 'tab:blue',alpha = 0.3)
ax[0].annotate(r'$M_{\mathrm{ice}} = %.2f~M_{\oplus}$'%(dust_mass/M_e),xy = (0.3,0.5), xycoords = 'axes fraction',fontsize = 15)
ax[0].annotate(r'$f_{\mathrm{i/g}} \dot{M}_{\mathrm{acc}} /(3 \pi \nu)$',xy = (2.5,72), rotation = -10, fontsize = 16)
# ax[0].set_ylim(0,800)
# ax[0].set_xlabel('$r$ [au]')
ax[0].set_ylabel(r'$\Sigma$ [g~cm$^{-2}$]')
# ax[0].legend(handles=legend_handles,
#     loc = 'upper right',fontsize = 15,frameon = False)
# d2g
# ax00 = ax[0].twinx()
ax00 = ax[1]
ax00.set_yscale('log')
ax00.plot(rad, ((dust_1_rho_xz + dust_2_rho_xz + dust_3_rho_xz + dust_4_rho_xz)/rho_xz)[:,-1], 'k', lw = 3.0,label = '$d/g$')
# ax00.plot(xx_exp, (sigma_ice+sigma_sil)/sigma_gas, 'k', lw = 3.0,label = '$d/g$')
ax00.plot(rad, (dust_5_rho_xz/rho_xz)[:,-1],'tab:red', linestyle = '-', lw = 3.0)
ax00.set_ylim(1e-3,1.0)


# # ax00.vlines(r_snow, -0.01, 1.2, linestyle = '--', color = 'grey')
# ax00.vlines(r_pk, -0.01, 1.2, linestyle = '--', color = 'grey')
# ax00.plot([2.177,2.617],[d2g_pk/2]*2,linestyle = '--', color = 'grey')
# # ax00.vlines(r_vap_outer, -0.01, 1.2, linestyle = '--', color = 'grey')
# ax00.annotate('$r_{\mathrm{snow,mid}}$',xy = (r_snow,0.8),fontsize = 20)
# ax00.annotate('$r_{\mathrm{pk}}$',xy = (r_pk,0.9),fontsize = 20)
# ax00.annotate('FWHM',xy = (r_pk - 0.1,0.27),fontsize = 15)
# # ax00.annotate('$r_{\mathrm{vap,outer}}$',xy = (r_vap_outer,0.9),fontsize = 20)
# ax00.set_ylabel('solid/vapor-to-gas ratio')

custom_lines2 = [Line2D([0], [10], color='tab:red', lw=3, linestyle='-',label='vapor'),
                Line2D([0,0.1], [0,0.1], color='k', lw=3, linestyle='-',label='ice+sil')]
ax[1].legend(handles=custom_lines2,handlelength = 2, loc = (0.74,0.6))

# flux
p2g_flux_inp = []
for i in range(1, N_P*N_Z + 1): 
    kk = 'p2g_flux_'+str(i)
    p2g_flux_inp.append(athinputs['dust'][kk])

ax[2].set_yscale('symlog', linthresh = 1e-2)
ax[2].axhline(-0.2, c= 'k', ls='--')
ax[2].plot(xx_exp,flux_ice_face*1e8, lw =lwD['si'],color='darkblue', alpha = alpD['si'], label = r'$\mathcal{F}_{\mathrm{ice, small}}$')
ax[2].plot(xx_exp,flux_ice1_face*1e8,lw =lwD['li'],color='darkblue', alpha = alpD['li'], label = r'$\mathcal{F}_{\mathrm{ice, big}}$')
ax[2].plot(xx_exp,flux_sil_face*1e8, lw =lwD['ss'],color=colD['ss'], alpha = alpD['ss'], label = r'$\mathcal{F}_{\mathrm{sil, small}}$')
ax[2].plot(xx_exp,flux_sil1_face*1e8,lw =lwD['ls'],color=colD['ls'], alpha = alpD['ls'], label = r'$\mathcal{F}_{\mathrm{sil, big}}$')

ax[2].plot(xx_exp,flux_vap_face*1e8,  lw =lwD['va'],color=colD['va'], alpha = alpD['va'], label = r'$\mathcal{F}_{\mathrm{vap}}$')
ax[2].plot(xx_exp,flux_water_face/xx_exp*1e8,lw =3,color='cyan', alpha = 0.6, label = r'$\mathcal{F}_{\mathrm{water}}$')
ax[2].plot(xx_exp,flux_gas_face*1e8,lw =3,color='grey', alpha = 0.6, label = r'$\mathcal{F}_{\mathrm{xy}}$')

# ax[2].plot(rad, flux_ice_faced*1e8, lw =lwD['si'],color='darkblue', alpha = alpD['si'], ls = '--')
# ax[2].plot(rad, flux_ice1_faced*1e8,lw =lwD['li'],color='darkblue', alpha = alpD['li'], ls = '--')
# ax[2].plot(rad, flux_sil_faced*1e8, lw =lwD['ss'],color=colD['ss'], alpha = alpD['ss'], ls = '--')
# ax[2].plot(rad, flux_sil1_faced*1e8,lw =lwD['ls'],color=colD['ls'], alpha = alpD['ls'], ls = '--')
# ax[2].plot(rad, flux_vap_faced*1e8,  lw =lwD['va'],color=colD['va'], alpha = alpD['va'], ls = '--')
# ax[2].plot(rad, flux_water_faced*1e8,lw =1,color='black', alpha = 0.6, ls = '--')

# ax[2].plot(xx_exp, -xx_exp/xx_exp,'k--')
# ax[2].plot(xx_exp, -xx_exp/xx_exp*0.4,'k--')
ax[2].axhline(0.0, c= 'k', ls='--')
# ax[2].axhline(-0.4, c= 'k', ls='--')



ax[2].set_xlim(rin/L_norm,rout/L_norm)
ax[2].set_ylim(-3,5.0)
ax[2].annotate(r'$\dot{M}_{\mathrm{acc}}$',xy=(1.0,-0.9),fontsize = 15)
ax[2].annotate(r'$f_{\mathrm{i/g}} \dot{M}_{\mathrm{acc}}$',xy=(1.0,-0.25),fontsize = 15)
ax[2].set_ylabel(r'Radial Mass Flux [$10^{-8}M_{\odot}$/yr]',fontsize = 15)

ax[2].legend(loc='upper right', fontsize = 10)
for i in range(len(axes)):
    ax[i].set_xlim(rin/L_norm,rout/L_norm)
for i in range(2):  
    ax[i].set_xticklabels([])   

ax[2].set_xlabel(r'$r$ [au]')

ax[0].annotate('(a)',xy = (0.02,0.92),xycoords = 'axes fraction',fontsize = 20)

ax[1].annotate('(b)',xy = (0.02,0.92),xycoords = 'axes fraction',fontsize = 20)
ax[2].annotate('(c)',xy = (0.02,0.92),xycoords = 'axes fraction',fontsize = 20)

ax[1].axvline(xx_exp[51], ls='dotted', c= 'black', lw=1)

plt.savefig('./plots/fig_snow_2d_{:05d}.png'.format(int(filenum)), bbox_inches='tight', dpi = 500) 
plt.close()



# fig,axes = plt.subplots(nrows = 2, ncols = 1,figsize = (11,9))
# fig.set_facecolor('white')
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace= 0.1)
# ax = axes.flatten()

# # the vapor
# ax0 = ax[0].contourf(x_xz_c,y_xz_c,dust_5_rho_mod,levels = logspace(log10(d2g_snow),log10(2.0),15), norm = LogNorm(), cmap = 'RdPu', alpha = 0.8, extend = 'both',zorder=1)
# # the ices
# ax00 = ax[0].contourf(x_xz_c,y_xz_c,dust_1_rho_mod,levels = logspace(log10(d2g_snow),log10(2.0),15), norm = LogNorm(), cmap = 'Blues', alpha = 0.8, extend = 'both', edgecolor='none', antialiased=True, zorder=3)
# ax000 = ax[0].contourf(x_xz_c,y_xz_c,dust_3_rho_mod,levels = logspace(log10(d2g_snow),log10(2.0),15), norm = LogNorm(), cmap = 'Greens', alpha = 1, extend = 'both', edgecolor='none', antialiased = True,zorder=2)
# # ax0 = ax[0].pcolormesh(x_xz,y_xz,q_diff_xz,norm = Normalize(vmin = -0.01,vmax = 0.01,clip = True) ,cmap = 'coolwarm', alpha = 1)
# # ax0 = ax[0].pcolormesh(x_xz,y_xz,st_xz * (x_xz_c/3.0)**(-1.5),norm = Normalize(vmin = 0.01,vmax = 0.3,clip = True) ,cmap = 'coolwarm', alpha = 1)

# #zxl: this we change to the sum of the ice in different populations.
# ice_rho_xz_tot = dust_1_rho_xz + dust_3_rho_xz
# ax[0].contour(x_xz_c,y_xz_c, dust_1_rho_xz/rho_xz,levels = [d2g_snow], cmap = 'Blues_r', alpha = 0.7, linewidths = 3.0)
# ax[0].contour(x_xz_c,y_xz_c, dust_3_rho_xz/rho_xz,levels = [d2g_snow], cmap = 'Greens_r', alpha = 0.7, linewidths = 3.0)
# # ax[0].contour(x_xz_c,y_xz_c,r_snow_2d(tem_xz,rho_xz,0.4) ,levels = [1.e-3,1.0,1.e3], cmap = 'Greens_r', alpha = 0.7, linewidths = 5.0)

# # normalized lw of flux
# lw_flx_ice =sqrt(ice_flx_x_xz**2 + ice_flx_z_xz**2)/normal2
# lw_flx_ice1 = sqrt(ice1_flx_x_xz**2 + ice1_flx_z_xz**2)/normal2
# lw_flx_water = sqrt(water_flx_x_xz**2 + water_flx_z_xz**2)/normal2
# lw_flx_ice = 2.0*sqrt(lw_flx_ice)
# lw_flx_ice1 = 2.0*sqrt(lw_flx_ice1)
# lw_flx_water = 2.0*sqrt(lw_flx_water)

# # sublimation / condensation rate
# P_eq = P_eq0*exp(-T_a/tem_xz)
# P_vap = dust_3_rho_xz * kB_mp * tem_xz / mu_z
# rate_ratio = P_eq/P_vap * (dust_3_rho_xz/dust_1_rho_xz)

# # legend
# legends = [Line2D([0], [0], color='k', lw=2, label=r'$10^{-3}~\rho_{0}c_{\mathrm{s,0}}$')]
# ax[0].legend(handles=legends, loc='upper left',fontsize = 15,frameon = False)

# ax[0].streamplot(x1_exp_half,x3_exp, ice_flx_x_xz/normal2, ice_flx_z_xz/normal2,linewidth = lw_flx_ice, arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='blue')
# ax[0].streamplot(x1_exp_half,x3_exp, ice1_flx_x_xz/normal2, ice1_flx_z_xz/normal2,linewidth = lw_flx_ice1, arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='cyan')
# ax[0].streamplot(x1_exp_half,x3_exp, water_flx_x_xz/normal2, water_flx_z_xz/normal2,linewidth = lw_flx_water, arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='pink')
# # ax[0].streamplot(x1_exp_half,x3_exp, water_flx_x_xz/normal2, water_flx_z_xz/normal2,linewidth = lw_flx, arrowstyle = '->', density = 1.5, broken_streamlines = True, color ='w')
# # ax[0].streamplot(x1_exp_half,x3_exp, (gas_flx_x_xz- vap_flx_x_xz)/normal2, (gas_flx_z_xz-vap_flx_z_xz)/normal2,linewidth = lw_flx4, arrowstyle = '->', density = 1.6, broken_streamlines = True, color ='white')
# # ax[0].streamplot(x1_exp_half,x3_exp, vx_xz/normal, vz_xz/normal,linewidth = 0.75
# #                 ,arrowstyle = '->', density = 2.0, broken_streamlines = True, color ='grey', norm = LogNorm(1.e-5,1.e-2,clip = True))

# # temperature
# ax1 = ax[1].pcolormesh(x_xz,y_xz,tem_xz,norm = Normalize(vmin = 100,vmax = 300,clip = True) ,cmap = 'coolwarm', alpha = 1)
# C_Tem = ax[1].contour(x_xz_c,y_xz_c,tem_xz,levels = linspace(150,200,11,endpoint=True), cmap = 'Greys_r', alpha = 1.0, linewidths = 1.0)
# C = ax[1].contour(x_xz_c,y_xz_c,tau_opt,levels = array([0.1,0.5,1.0,5.0]), colors = 'black', linestyles = 'dotted')
# # ax[1].annotate(r'$\tau_{R} = 0.1, 0.5, 1.0, 5.0$',xy = (2.0,0.14),xytext = (2.0,0.14),fontsize = 15)

# #===
# divider = make_axes_locatable(ax[0])
# cax = fig.add_axes([ax[0].get_position().x1+0.01,ax[0].get_position().y0,0.02,ax[0].get_position().height])
# cl0 = fig.colorbar(ax0,cax = cax)
# # cl0.set_ticks([0.01,0.03,0.05])
# # cl0.ax.set_title('St',fontsize=12)
# cl0.set_ticks([])
# cl0.ax.set_title(r'$\rho_{\mathrm{vap}}$',fontsize = 15)
# #===
# cax1 = fig.add_axes([ax[0].get_position().x1+0.055,ax[0].get_position().y0,0.02,ax[0].get_position().height])
# cl00 = fig.colorbar(ax00, cax = cax1,format = ticker.FuncFormatter(formatnum))
# cl00.set_ticks([])
# cl00.ax.set_title(r'$\rho_{\mathrm{ice}}$',fontsize = 15)


# cax11 = fig.add_axes([ax[0].get_position().x1+0.115,ax[0].get_position().y0,0.02,ax[0].get_position().height])
# cl000 = fig.colorbar(ax000, cax = cax11,format = ticker.FuncFormatter(formatnum))
# cl000.set_ticks(ticks = logspace(-2,0,3))
# cl000.ax.set_title(r'$\rho_{\mathrm{ice,small}}$',fontsize = 15)
# cl000.set_label(r'$\rho_{0}$', fontsize = 15)

# divider = make_axes_locatable(ax[1])
# cax = fig.add_axes([ax[1].get_position().x1+0.01,ax[1].get_position().y0,0.02,ax[1].get_position().height])
# cl1 = fig.colorbar(ax1,cax = cax)
# cl1.set_label(r'$T(\mathrm{K})$')
# cb_ymin, cb_ymax = cl1.ax.get_ylim()
# # Get colors from the contourf object
# # colors = C_Tem.get_array()
# plt.draw() # Force the figure to update and draw to get the colors
# colors = C_Tem.get_edgecolors()  # Get the edge colors of the contour lines
# # colors = C_Tem.colors
# # Define levels for the second contour (make sure these are within the range of Z1)
# second_contour_levels = C_Tem.levels
# # Add lines to the colorbar
# color_id = 0
# for level in second_contour_levels:
#     # Normalize level value to colorbar scale
#     fmax = ax1.get_clim()[1]
#     fmin = ax1.get_clim()[0]
#     norm_level = (level - fmin) / (fmax - fmin)
#     # Calculate y position on the colorbar
#     y = cb_ymin + norm_level * (cb_ymax - cb_ymin)
#     # Choose the color
#     color = colors[color_id]
#     color_id += 1
#     # Draw a horizontal line on the colorbar
#     cl1.ax.hlines(y, 0, 1, color=color, linewidth=2)

# for i in range(len(list(ax))):
#     ax[i].set_xlim(R_inner,xs)
#     ax[i].set_ylim(0,zs)
#     # ax[i].set_aspect(1)

# ax[0].set_ylabel('$z$ [au]',fontsize =15)
# ax[1].set_ylabel('$z$ [au]',fontsize =15)
# ax[1].set_xlabel('$r$ [au]',fontsize =15)

# plt.savefig('./plots/rho_xz_{:05d}.png'.format(int(filenum)), dpi = 300, bbox_inches='tight')
# plt.close()
# # #plot the relaxation timescale 
# # plt.figure(figsize=(9,4))
# # plt.contourf(x_xz_c,y_xz_c,t_relax[0].T,levels=100, alpha = 1., extend = 'both')
# # plt.colorbar()
# # import cgs
# # plt.axhline(3/tan(1.35416), ls='dotted', c= 'black', lw=1)
# # x = linspace(1,4,100)
# # upper_theta = x/tan(1.32708)
# # plt.plot(x, upper_theta, '--', c='black', lw=1)
# #
# # plt.contour(x_xz_c,y_xz_c,tau_opt,levels = array([0.1,0.5,1.0,5.0]), colors = 'black', linestyles = 'dotted')
# # plt.annotate(r'$\tau_{R} = 0.1, 0.5, 1.0, 5.0$',xy = (2.0,0.14),xytext = (2.0,0.14),fontsize = 15)
# # plt.savefig('./plots/relaxationtime.png',dpi=100)
# #
# # plt.close()

# def normalize(arr):
#     norm = empty_like(arr)
#     for i in range(len(arr)):
#         norm[i] = arr[i]/arr[i].max()

#     return norm
# #plot the rho**2/tem**3 
# # aaa = rho_xz**2/tem_xz**3
# aaa = 1 - normalize(q_z_xz) 
# theta1 = x/tan(theta[8])
# theta2 = x/tan(theta[7])
# theta3 = x/tan(theta[9])

# # plt.figure(figsize=(9,4))
# # plt.plot(x, theta1, '--', c='black', lw=1)
# # plt.plot(x, theta2, '--', c='black', lw=1)
# # plt.plot(x, theta3, '--', c='black', lw=1)
# # plt.contourf(x_xz_c,y_xz_c,aaa,levels=100, alpha = 1., extend = 'both')
# # plt.colorbar()
# #
# # plt.contour(x_xz_c,y_xz_c,tau_opt,levels = array([0.1,0.5,1.0,5.0]), colors = 'black', linestyles = 'dotted')
# # plt.annotate(r'$\tau_{R} = 0.1, 0.5, 1.0, 5.0$',xy = (2.0,0.14),xytext = (2.0,0.14),fontsize = 15)
# # plt.savefig('./plots/dependence.png',dpi=100)
# #
# # plt.close()

# #plot the stokes number here 

# idx_H1L = []
# for i in range(len(rad)):
#     idx_H1L.append(abs(z[0].T[i] - H_profile(rad[i])/AU).argmin())
# idx_H1L = array(idx_H1L)

# # st_xz = st[index_phi, :, :].T 
# st1_xz = st1[index_phi, :, :].T 
# st_xz[st_xz==1e-4] = nan
# st_xz[st_xz==0.5] = nan 
# st1_xz[st1_xz==1e-4] = nan
# st1_xz[st1_xz==0.5] = nan
# # # m_p_xz = m_p[index_phi, :, :].T 
# # # m_p1_xz = m_p1[index_phi, :, :].T 
# # fig,axs = plt.subplots(nrows = 2, ncols = 1,figsize = (9,6), sharex=True)
# #
# # ax1 = axs[0].contourf (x_xz_c, y_xz_c, st_xz, cmap='Blues',alpha=1, norm=LogNorm(vmin=1e-4,vmax=0.5))
# # c1 = fig.colorbar(ax1) 
# # ax2 = axs[1].contourf (x_xz_c, y_xz_c, st1_xz, cmap ='Greens', alpha=1, norm=LogNorm(vmin=1e-4,vmax=0.5))
# # c2 = fig.colorbar(ax2)
# #
# # # c2.set_ticks([])
# # c1.ax.set_title(r'$St_{large}$', fontsize=15)
# # c2.ax.set_title(r'$St_{small}$', fontsize=15)
# #
# # plt.savefig('./plots/stokesnumber.png', bbox_inches ='tight')
# # plt.close()

# s_p1_xz = s_p1[index_phi, :, :].T
# s_p_xz_1H = s_p_xz[range(len(rad)), idx_H1L]
# s_p1_xz_1H = s_p1_xz[range(len(rad)), idx_H1L]

# #plot the profile or pebble size at midplane 
# plt.figure(figsize=(9,6))
# plt.yscale('log')
# plt.plot(rad, s_p_xz.T[-1], color = 'blue', label = 'large midplane') 
# plt.plot(rad, s_p_xz_1H , color = 'cyan', ls='--', label = 'large 1H')
# plt.plot(rad, s_p_xz.T[13], color = 'lightblue', label = 'large upper layer')
# plt.plot(rad, s_p1_xz.T[-1], color = 'red', label = 'small midplane') 
# plt.plot(rad, s_p1_xz_1H , color = 'orange', ls='--', label = 'small 1H') 
# plt.plot(rad, s_p1_xz.T[13], color = 'yellow', label = 'large upper layer')
# plt.xlabel('r [au]') 
# plt.ylabel('particle size [cm]') 
# # plt.plot(rad, st_xz.T[-20], label = 'mid-height large particle Stokes number')
# # plt.plot(xx_exp, st_intpl[0], label = '0')
# # plt.plot(xx_exp, st_intpl[30], label = '30')
# # plt.plot(xx_exp, st_intpl[25], label = '25')
# # plt.plot(xx_exp, st_intpl[20], label = '20')
# # plt.plot(xx_exp, st_intpl[10], label = '10')
# # plt.plot(xx_exp, st_intpl[5], label = '5')

# plt.legend(loc = 'upper right')
# plt.savefig('./plots/sp_profile_{:05d}.png'.format(int(filenum)), dpi = 300)
# plt.close()

# fig,axs = plt.subplots(nrows = 2, ncols = 1,figsize = (9,6), sharex=True)
# # st_xz = st[index_phi, :, :].T 
s_p_xz [isnan(st_xz)] = nan 
s_p1_xz [isnan(st1_xz)] = nan 
#
# #theta[8] 
# x = linspace(1,4,100)
# upper_theta = x/tan(theta[13])
# plt.plot(x, upper_theta, '--', c='black', lw=1)
#
# ax1 = axs[0].contourf (x_xz_c, y_xz_c, s_p_xz, cmap='Blues',alpha=1)
# c1 = fig.colorbar(ax1) 
# ax2 = axs[1].contourf (x_xz_c, y_xz_c, s_p1_xz, cmap ='Greens', alpha=1)
# c2 = fig.colorbar(ax2)
#
# # c2.set_ticks([])
# c1.ax.set_title(r'$s_{large}$', fontsize=15)
# c2.ax.set_title(r'$s_{small}$', fontsize=15)
#
# plt.savefig('./plots/size_map.png', bbox_inches ='tight')
# plt.close()


# fig,ax = plt.subplots(nrows = 1, ncols = 1,figsize = (9,3))
m_p_xz = m_p[index_phi, :, :].T 
m_p1_xz = m_p1[index_phi, :, :].T 
m_p_xz[isnan(st_xz)] = nan 
m_p1_xz[isnan(st1_xz)] = nan
#
# ax1 = ax.contourf (x_xz_c, y_xz_c, m_p_xz,cmap='Blues',alpha=1., norm=LogNorm(vmin=5,vmax=1e3))
# c1 = fig.colorbar(ax1) 
# ax2 = ax.contourf (x_xz_c, y_xz_c, m_p1_xz, cmap ='Greens', alpha=0.6, norm=LogNorm(vmin=5,vmax=1e3))
# c2 = fig.colorbar(ax2)
#
# # c2.set_ticks([])
# c1.ax.set_title(r'$m_{large}$', fontsize=15)
# c2.ax.set_title(r'$m_{small}$', fontsize=15)
#
# plt.savefig('./plots/particlemass.png', bbox_inches ='tight')
# plt.close()


# plt.figure(figsize=(7,5))
# plt.plot(rad, st_xz.T[-1], color = 'blue', label = 'large(-1)') 
# plt.plot(rad, st_xz.T[-10], color = 'skyblue', label = 'large(-10)')
# plt.plot(rad, st_xz.T[13], color = 'lightblue', label = 'large(13)')
# plt.plot(rad, st1_xz.T[-1], color = 'red', label = 'small(-1)') 
# plt.plot(rad, st1_xz.T[-10], color = 'orange', label = 'small(-10)')
# plt.plot(rad, st1_xz.T[13], color = 'yellow', label = 'small(13)')
# plt.xlabel('r [au]') 
# plt.ylabel('Stokes number') 
# # plt.plot(rad, st_xz.T[-20], label = 'mid-height large particle Stokes number')
# # plt.plot(xx_exp, st_intpl[0], label = '0')
# # plt.plot(xx_exp, st_intpl[30], label = '30')
# # plt.plot(xx_exp, st_intpl[25], label = '25')
# # plt.plot(xx_exp, st_intpl[20], label = '20')
# # plt.plot(xx_exp, st_intpl[10], label = '10')
# # plt.plot(xx_exp, st_intpl[5], label = '5')
#
# plt.legend(loc = 'upper right')
# plt.savefig('./plots/St_profile.png', dpi = 300)
# plt.close()

# plt.figure(figsize=(7,5))
# plt.yscale('log')
# plt.plot(rad, m_p_xz.T[-1], color = 'blue', label = 'large(-1)') 
# plt.plot(rad, m_p_xz.T[-10], color = 'skyblue', label = 'large(-10)')
# plt.plot(rad, m_p_xz.T[13], color = 'lightblue', label = 'large(13)')
# plt.plot(rad, m_p1_xz.T[-1], color = 'red', label = 'small(-1)') 
# plt.plot(rad, m_p1_xz.T[-10], color = 'orange', label = 'small(-10)')
# plt.plot(rad, m_p1_xz.T[13], color = 'yellow', label = 'small(13)')
# plt.xlabel('r [au]') 
# plt.ylabel('mass [g]') 
# # plt.plot(rad, st_xz.T[-20], label = 'mid-height large particle Stokes number')
# # plt.plot(xx_exp, st_intpl[0], label = '0')
# # plt.plot(xx_exp, st_intpl[30], label = '30')
# # plt.plot(xx_exp, st_intpl[25], label = '25')
# # plt.plot(xx_exp, st_intpl[20], label = '20')
# # plt.plot(xx_exp, st_intpl[10], label = '10')
# # plt.plot(xx_exp, st_intpl[5], label = '5')
#
# plt.legend(loc = 'upper right')
# plt.savefig('./plots/m_profile.png', dpi = 300)
# plt.close()

#plot the number density map
n_p_xz = (dust_1_rho_xz + dust_2_rho_xz)/m_p_xz 
n_p1_xz = (dust_3_rho_xz + dust_4_rho_xz)/m_p1_xz 

# conden_rate_ratio = n_p_xz*s_p_xz**2 / (n_p1_xz*s_p1_xz**2)
# #find the place where the conden rate ratio are 0.01
# loc01 = []
# loc03 = []
# loc1 = []
# loc01np = []
# loc03np = []
#
# for i in range (len(rad)):
#     loc01.append(z[0].T[i][nanargmin(abs(conden_rate_ratio[i, :] - 0.01))])
#     loc03.append(z[0].T[i][nanargmin(abs(conden_rate_ratio[i, :] - 0.03))])
#     loc1.append(z[0].T[i][nanargmin(abs(conden_rate_ratio[i, :] - 1.0))])
#     loc01np.append(z[0].T[i][nanargmin(abs((n_p_xz/n_p1_xz)[i, :] - 0.0003))])
#     loc03np.append(z[0].T[i][nanargmin(abs((n_p_xz/n_p1_xz)[i, :] - 0.001))])
#     # loc01.append(nanargmin(conden_rate_ratio[i, conden_rate_ratio[i] - 0.03 >=0]))
#
# plt.figure(figsize=(9,4)) 
# plt.ylim(0,0.3)
#
# plt.plot(rad, loc01, '--', c='black', lw=1, label = '0.01') 
# plt.plot(rad, loc03, '--', c='grey', lw=1, label = '0.03')
# plt.plot(rad, loc1, '--', c='lightgrey', lw=1, label = '1.0')
# levels = linspace(0.01,5,100)
# crr_clip = clip(conden_rate_ratio, 0.01, 5)
# cc = plt.contourf(x_xz_c,y_xz_c,crr_clip, levels=levels, cmap='Blues')
# #make the cc show from 0.01 to 5 
# cc.set_clim(0.01,5)
#
# plt.colorbar(cc)
# plt.xlabel('r [au]')
# plt.ylabel('z [au]')
# plt.legend()
#
# plt.savefig('./plots/conden_rate_ratio_{:05d}.png'.format(int(filenum)), dpi = 300)
# plt.close()


# plt.figure(figsize=(9,4)) 
# plt.ylim(0,0.3)
#
# conrate_perpeb = 1/conden_rate_ratio/(n_p1_xz/n_p_xz)
# conrate_perpeb_clip = clip(conrate_perpeb, 0.01, 1.5)
#
# plt.plot(rad, loc01np, '--', c='black', lw=1, label = '0.0003') 
# plt.plot(rad, loc03np, '--', c='grey', lw=1, label = '0.001')
# cc = plt.contourf(x_xz_c,y_xz_c,conrate_perpeb_clip, levels=10, cmap='Blues', vmin = 0.01, vmax = 1.5)
#
# # plt.contour(x_xz_c,y_xz_c,n_p_xz/n_p1_xz, levels = array([0.0003,0.001]), colors = 'black', linestyles = 'dotted') 
# plt.colorbar(cc)
# plt.xlabel('r [au]')
# plt.ylabel('z [au]')
# plt.legend()
#
# plt.savefig('./plots/n_p_ratio.png', dpi = 300)
# plt.close()

#plot the drho_i for large and small pebbles 
if False:
    dlogmdt_large = drho_i_dt_xz/n_p_xz/m_p_xz 
    dlogmdt_small = drho_i1_dt_xz/n_p1_xz/m_p1_xz

    fig, [ax, ax1,ax2] = plt.subplots(3, 1, figsize=(7,9),sharex=True)
    plt.subplots_adjust(hspace = 0.)
    ax.set_title("Time = {}".format(int(filenum)*50),loc='right')
    ax.set_ylim(-1,1)
    ax1.set_ylim(-1,1)
    ax2.set_ylim(-10,10)
    ax.set_ylabel('dlogmi/dt (large)', fontsize = 10)
    ax1.set_ylabel('dlogmi/dt (small)', fontsize = 10)
    ax2.set_ylabel('Ratio of dlogmi/dt (small/large)', fontsize = 10)

    dlogmdt_large_1H = dlogmdt_large[arange(len(rad)), idx_H1L]
    dlogmdt_small_1H = dlogmdt_small[arange(len(rad)), idx_H1L]

    #we should compare the drhodt at different heights and different populations
    ax.plot(rad, dlogmdt_large.T[-1], color = 'blue',     lw=1,   label = 'midplane') 
    ax.plot(rad, dlogmdt_large_1H, 
            color = '#6495ED', lw=1,   label = '1 Hg') 
    # ax.plot(rad, drho_i_dt_xz.T[13]/dust_1_rho_xz.T[-10], color = 'lightblue', lw=1,  label = '(13)') 
    ax1.plot(rad, dlogmdt_small.T[-1], color = 'red',   lw=1,    label = 'midplane')
    ax1.plot(rad, dlogmdt_small_1H,
            color = 'orange', lw=1,    label = '1 Hg')
    # ax1.plot(rad, drho_i1_dt_xz.T[13]/dust_3_rho_xz.T[-10], color = 'yellow',lw=1,    label = '(13)')

    ratio = dlogmdt_small/dlogmdt_large
    ratio_1H = dlogmdt_small_1H/dlogmdt_large_1H
    ax2.plot(rad, ratio.T[-1], color = 'purple',lw=1,    label = 'midplane')
    ax2.plot(rad, ratio_1H, color = 'magenta',lw=1,   label = '1 Hg')
    # ax2.plot(rad, drho_i1_dt_xz.T[13]/drho_i_dt_xz.T[13], color = 'pink',lw=1,    label = '(13)')

    ax.legend(loc = 'upper right')
    ax1.legend(loc = 'upper right')
    ax2.legend(loc = 'upper right')
    plt.savefig('./plots/drhodt_profile_{:05d}.png'.format(int(filenum)), dpi = 300)
    plt.tight_layout()
    plt.close()

    #plot the map of drhodt/np
    fig, [ax, ax1,ax2] = plt.subplots(3, 1, figsize=(7,9),sharex=True)
    plt.subplots_adjust(hspace = 0.)

    ax.set_title("Time = {}".format(int(filenum)*50),loc='right')
    ax.set_ylim(0,0.15)
    ax1.set_ylim(0,0.15) 
    ax2.set_ylim(0,0.15)
    ax2.set_xlim(1,2.6)
    # ax2.set_xlim(1.8, 2.2)
    ax.set_ylabel('dlogmi/dt (large)', fontsize = 10)
    ax1.set_ylabel('dlogmi/dt (small)', fontsize = 10)
    ax2.set_ylabel('Ratio of dlogmi/dt (small/large)', fontsize = 10)

    # c1 = ax.contourf (x_xz_c, y_xz_c, dlogmdt_large[dlogmdt_large>0.0], levels = 10, cmap='Blues', alpha=1.) 
    # c2 = ax1.contourf (x_xz_c, y_xz_c, drho_i1_dt_xz/n_p1_xz/m_p1_xz,levels = 10,  cmap ='Greens', alpha=1.)
    # c3 = ax2.contourf (x_xz_c, y_xz_c, (drho_i1_dt_xz/n_p1_xz/m_p1_xz)/(drho_i_dt_xz/n_p_xz/m_p_xz), levels = 10, cmap ='Purples', alpha=1.)
    vmin_large = -10 
    vmax_large = nanmax(dlogmdt_large) 
    vmin_small = -10 
    vmax_small = nanmax(dlogmdt_small)
    vmin_ratio = 0.0 
    vmax_ratio = 6.0

    dlogmdt_large_clipped = clip(dlogmdt_large, vmin_large, vmax_large)
    dlogmdt_small_clipped = clip(dlogmdt_small, vmin_small, vmax_small)
    dlogmdt_ratio_clipped = clip(dlogmdt_small / dlogmdt_large, vmin_ratio, vmax_ratio)
    from matplotlib.colors import TwoSlopeNorm
    norm_large = TwoSlopeNorm(vmin=vmin_large, vcenter=0, vmax=vmax_large)
    norm_small = TwoSlopeNorm(vmin=vmin_small, vcenter=0, vmax=vmax_small)

# Use a diverging colormap
    c1 = ax.contourf(x_xz_c, y_xz_c, dlogmdt_large_clipped, levels=100, cmap='seismic', norm=norm_large, vmin=vmin_large, vmax=vmax_large)
    c2 = ax1.contourf(x_xz_c, y_xz_c, dlogmdt_small_clipped, levels=100, cmap='seismic', norm=norm_small, vmin=vmin_small, vmax=vmax_small)
    c3 = ax2.contourf(x_xz_c, y_xz_c, dlogmdt_ratio_clipped, levels=100, cmap='Purples', vmin=vmin_ratio, vmax=vmax_ratio)

    # cbar = fig.colorbar(c1, ax=[ax, ax1, ax2], orientation='vertical')
    # cbar.set_label('Value')
    cbar1 = fig.colorbar(c1, ax=ax, orientation='vertical') 
    cbar2 = fig.colorbar(c2, ax=ax1, orientation='vertical')
    cbar3 = fig.colorbar(c3, ax=ax2, orientation='vertical')
    cbar1.set_ticks([-10, -5, 0, 1])
    cbar2.set_ticks([-10, -5, 0, 1])
    cbar3.set_ticks([0, 3, 5])

    ax.plot(rad, H_profile(rad)/AU, '--', c='gray', lw=1)
    ax1.plot(rad, H_profile(rad)/AU, '--', c='gray', lw=1)
    ax2.plot(rad, H_profile(rad)/AU, '--', c='gray', lw=1)

    # ax.legend(loc = 'upper right')
    # ax1.legend(loc = 'upper right')
    # ax2.legend(loc = 'upper right')
    plt.savefig('./plots/dlogmdt_{:05d}.png'.format(int(filenum)), dpi = 300)
    plt.tight_layout()
    plt.close()


    #plot the map of dmdt
    fig, [ax, ax1,ax2] = plt.subplots(3, 1, figsize=(7,9),sharex=True)
    plt.subplots_adjust(hspace = 0.)

    ax.set_title("Time = {}".format(int(filenum)*50),loc='right')
    ax.set_ylim(0,0.15)
    ax1.set_ylim(0,0.15) 
    ax2.set_ylim(0,0.15)
    # ax2.set_xlim(1.8, 2.2)
    ax.set_ylabel('dmi/dt (large)', fontsize = 10)
    ax1.set_ylabel('dmi/dt (small)', fontsize = 10)
    ax2.set_ylabel('Ratio of dmi/dt (small/large)', fontsize = 10)
    ax2.set_xlim(1,2.6)

    dmdt_large = drho_i_dt_xz/n_p_xz
    dmdt_small = drho_i1_dt_xz/n_p1_xz
    # c1 = ax.contourf (x_xz_c, y_xz_c, dlogmdt_large[dlogmdt_large>0.0], levels = 10, cmap='Blues', alpha=1.) 
    # c2 = ax1.contourf (x_xz_c, y_xz_c, drho_i1_dt_xz/n_p1_xz/m_p1_xz,levels = 10,  cmap ='Greens', alpha=1.)
    # c3 = ax2.contourf (x_xz_c, y_xz_c, (drho_i1_dt_xz/n_p1_xz/m_p1_xz)/(drho_i_dt_xz/n_p_xz/m_p_xz), levels = 10, cmap ='Purples', alpha=1.)
    vmin_large = -30 
    vmax_large = 350 
    vmin_small = -30 
    vmax_small = 350 
    vmin_ratio = 0.0 
    vmax_ratio = 4.0

    dmdt_large_clipped = clip(dmdt_large, vmin_large, vmax_large)
    dmdt_small_clipped = clip(dmdt_small, vmin_small, vmax_small)
    dmdt_ratio_clipped = clip(dmdt_small / dmdt_large, vmin_ratio, vmax_ratio)
    from matplotlib.colors import TwoSlopeNorm
    norm_large = TwoSlopeNorm(vmin=vmin_large, vcenter=0, vmax=vmax_large)
    norm_small = TwoSlopeNorm(vmin=vmin_small, vcenter=0, vmax=vmax_small)

# Use a diverging colormap
    c1 = ax.contourf(x_xz_c, y_xz_c, dmdt_large_clipped, levels=100, cmap='seismic', norm=norm_large, vmin=vmin_large, vmax=vmax_large)
    c2 = ax1.contourf(x_xz_c, y_xz_c, dmdt_small_clipped, levels=100, cmap='seismic', norm=norm_small, vmin=vmin_small, vmax=vmax_small)
    c3 = ax2.contourf(x_xz_c, y_xz_c, dmdt_ratio_clipped, levels=100, cmap='Purples', vmin=vmin_ratio, vmax=vmax_ratio)

    # cbar = fig.colorbar(c1, ax=[ax, ax1, ax2], orientation='vertical')
    # cbar.set_label('Value')
    cbar1 = fig.colorbar(c1, ax=ax, orientation='vertical') 
    cbar2 = fig.colorbar(c2, ax=ax1, orientation='vertical')
    cbar3 = fig.colorbar(c3, ax=ax2, orientation='vertical')
    cbar1.set_ticks([vmin_large, 0, 100, 300])
    cbar2.set_ticks([vmin_small, 0, 100, 300])
    cbar3.set_ticks([0, 1,3])
    #plot the ratio of fluxes
    # lw_flxratio = ((vmax_ratio - vmin_ratio)/5)*0.5
    # ax2.streamplot(x1_exp_half,x3_exp, ice1_flx_x_xz/ice_flx_x_xz/normal2, ice1_flx_z_xz/ice_flx_z_xz/normal2, 
    #                linewidth = lw_flxratio, arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='blue')
    ax.streamplot(x1_exp_half,x3_exp, ice_flx_x_xz/normal2, ice_flx_z_xz/normal2,
                   linewidth = lw_flx_ice, arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='cyan')
    ax1.streamplot(x1_exp_half,x3_exp, ice1_flx_x_xz/normal2, ice1_flx_z_xz/normal2,
                   linewidth = lw_flx_ice1, arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='cyan')

    ax.plot(rad, H_profile(rad)/AU, '--', c='gray', lw=1)
    ax1.plot(rad, H_profile(rad)/AU, '--', c='gray', lw=1)
    ax2.plot(rad, H_profile(rad)/AU, '--', c='gray', lw=1)

    # ax.legend(loc = 'upper right')
    # ax1.legend(loc = 'upper right')
    # ax2.legend(loc = 'upper right')
    plt.savefig('./plots/dmdt_{:05d}.png'.format(int(filenum)), dpi = 300)
    plt.tight_layout()
    plt.close()

    #plot the map of drhodt
    fig, [ax, ax1,ax2] = plt.subplots(3, 1, figsize=(7,9),sharex=True)
    plt.subplots_adjust(hspace = 0.)

    ax.set_title("Time = {}".format(int(filenum)*50),loc='right')
    ax.set_ylim(0,0.15)
    ax1.set_ylim(0,0.15) 
    ax2.set_ylim(0,0.15)
    # ax2.set_xlim(1.8, 2.2)
    ax.set_ylabel(r'd$\rho_i$/dt (large)', fontsize = 10)
    ax1.set_ylabel(r'd$\rho_i$/dt (small)', fontsize = 10)
    ax2.set_ylabel(r'Ratio of d$\rho_i$/dt (small/large)', fontsize = 10)
    ax2.set_xlim(1,2.6)

    # c1 = ax.contourf (x_xz_c, y_xz_c, dlogmdt_large[dlogmdt_large>0.0], levels = 10, cmap='Blues', alpha=1.) 
    # c2 = ax1.contourf (x_xz_c, y_xz_c, drho_i1_dt_xz/n_p1_xz/m_p1_xz,levels = 10,  cmap ='Greens', alpha=1.)
    # c3 = ax2.contourf (x_xz_c, y_xz_c, (drho_i1_dt_xz/n_p1_xz/m_p1_xz)/(drho_i_dt_xz/n_p_xz/m_p_xz), levels = 10, cmap ='Purples', alpha=1.)
    vmin_large = nanmin(drho_i_dt_xz)
    vmax_large = nanmax(drho_i_dt_xz) 
    vmin_small = nanmin(drho_i1_dt_xz) 
    vmax_small = nanmax(drho_i1_dt_xz) 
    vmin_ratio = 0.0
    vmax_ratio = 20.0

    drhodt_large_clipped = clip(drho_i_dt_xz, vmin_large, vmax_large)
    dmdt_small_clipped = clip(drho_i1_dt_xz, vmin_small, vmax_small)
    dmdt_ratio_clipped = clip(drho_i1_dt_xz/drho_i_dt_xz, vmin_ratio, vmax_ratio)
    from matplotlib.colors import TwoSlopeNorm
    norm_large = TwoSlopeNorm(vmin=vmin_large, vcenter=0, vmax=vmax_large)
    norm_small = TwoSlopeNorm(vmin=vmin_small, vcenter=0, vmax=vmax_small)

# Use a diverging colormap
    c1 = ax.contourf(x_xz_c, y_xz_c, drhodt_large_clipped, levels=100, cmap='seismic', norm=norm_large, vmin=vmin_large, vmax=vmax_large)
    c2 = ax1.contourf(x_xz_c, y_xz_c, dmdt_small_clipped, levels=100, cmap='seismic', norm=norm_small, vmin=vmin_small, vmax=vmax_small)
    c3 = ax2.contourf(x_xz_c, y_xz_c, dmdt_ratio_clipped, levels=20, cmap='Purples', vmin=vmin_ratio, vmax=vmax_ratio)

    # cbar = fig.colorbar(c1, ax=[ax, ax1, ax2], orientation='vertical')
    # cbar.set_label('Value')
    cbar1 = fig.colorbar(c1, ax=ax, orientation='vertical') 
    cbar2 = fig.colorbar(c2, ax=ax1, orientation='vertical')
    cbar3 = fig.colorbar(c3, ax=ax2, orientation='vertical')
    #reserve 3 decimal points for the colorbar ticks
    cbar1.set_ticks([round(vmin_large,3), 0, 0.004]) 
    cbar2.set_ticks([round(vmin_small,3), 0, 0.04]) 
    cbar3.set_ticks([0, 5, 10, 15])
    # cbar1.set_ticks([vmin_large, 0, 0.002, 0.004])
    # cbar2.set_ticks([vmin_small, 0, 0.03, 0.06])
    # cbar3.set_ticks([0, 5, 10, 15])
    #plot the ratio of fluxes
    # lw_flxratio = ((vmax_ratio - vmin_ratio)/5)*0.5
    # ax2.streamplot(x1_exp_half,x3_exp, ice1_flx_x_xz/ice_flx_x_xz/normal2, ice1_flx_z_xz/ice_flx_z_xz/normal2, 
    #                linewidth = lw_flxratio, arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='blue')
    ax.streamplot(x1_exp_half,x3_exp, ice_flx_x_xz/normal2, ice_flx_z_xz/normal2,
                   linewidth = lw_flx_ice, arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='cyan')
    ax1.streamplot(x1_exp_half,x3_exp, ice1_flx_x_xz/normal2, ice1_flx_z_xz/normal2,
                   linewidth = lw_flx_ice1, arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='cyan')

    ax.plot(rad, H_profile(rad)/AU, '--', c='gray', lw=1)
    ax1.plot(rad, H_profile(rad)/AU, '--', c='gray', lw=1)
    ax2.plot(rad, H_profile(rad)/AU, '--', c='gray', lw=1)

    # ax.legend(loc = 'upper right')
    # ax1.legend(loc = 'upper right')
    # ax2.legend(loc = 'upper right')
    plt.savefig('./plots/drhodt_{:05d}.png'.format(int(filenum)), dpi = 300)
    plt.tight_layout()
    plt.close()
