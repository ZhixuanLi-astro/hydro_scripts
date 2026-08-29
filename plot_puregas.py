import os
import sys
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib import ticker
import astropy.constants as cons
from preplot import pol2car, car2pol, dfdx_2pts, dfdx_5pts, dfdx_7pts, curl_in_polar_rlog,v_Intpl_Sph2car,scaler_Intpl_Sph2car

# path to the athena_read module
sys.path.insert(0, '/home/izx/athena-multifluid-dust/vis/python')
import athena_read

plt.rcParams.update({'font.size': 15})
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

AU = cons.au.cgs.value
YR = (365.2425 * 24 * 3600)
M_sun = cons.M_sun.cgs.value

# ----------------------------------------------------------------------
# disk model parameters (same reference disk as in the iceline setup)
# ----------------------------------------------------------------------
T_slope = -0.5
Cs_slope = T_slope / 2
H_slope = Cs_slope + 1.5
sigma_slope = -(Cs_slope + H_slope)
rho_slope = sigma_slope - H_slope
p_slope = T_slope + rho_slope

a0 = 3.0                       # reference radius [AU]
T_profile = lambda r: 150.0 * (r / 3.0) ** T_slope
T0 = T_profile(a0)             # temperature at the reference position [K]
Mdot_gas = 1.e-8 * M_sun / YR  # reference accretion rate [g/s]
alpha = 3.e-3

mu_xy = 2.34                   # mean molecular weight


def find_dust_scaleheight(rhos_intpl, y_xz_c):
    rho_p = rhos_intpl[1] 
    nz = intpl_numz  
    yy = zeros(intpl_numx)
    for i in range(intpl_numx):
        rho_efold = rho_p[0,i] / exp(1.0)**0.5
        if isnan(rho_efold):
            yy[i] = nan
        else:
            # linear interpolation to find exact z where rho = rho_efold
            diff_rho = rho_p[:,i] - rho_efold
            # find sign change (descending profile)
            idx = argmax(diff_rho[1:] * diff_rho[:-1] <= 0.0)
            if diff_rho[idx] * diff_rho[idx+1] <= 0.0 and idx < nz - 1:
                # linear interp between zz_exp[idx] and zz_exp[idx+1]
                frac = -diff_rho[idx] / (diff_rho[idx+1] - diff_rho[idx])
                yy[i] = zz_exp[idx] + frac * (zz_exp[idx+1] - zz_exp[idx])
            else:
                # fallback: nearest grid point
                idx_fallback = nanargmin(abs(diff_rho))
                yy[i] = zz_exp[idx_fallback]

    return None, yy 


# ----------------------------------------------------------------------
# command line arguments: filenumber [dirname]
# ----------------------------------------------------------------------
try:
    filenum = sys.argv[1]
except IndexError:
    print("please specify a filenumber")
    sys.exit()

try:
    dirname = sys.argv[2]
    DIR = os.path.join('..', dirname) + os.sep
except IndexError:
    DIR = '/mnt/sdc/Zhixuan/athena_works/puregas/'

nstep = filenum

# ----------------------------------------------------------------------
# read the units from the input file
# ----------------------------------------------------------------------
athinputs = athena_read.athinput(DIR + 'athinput.iceline')
UNIT_T = athinputs['units']['time_cgs']
UNIT_L = athinputs['units']['length_cgs']
UNIT_M = athinputs['units']['mass_cgs']

UNIT_V = sqrt(cons.k_B.cgs.value * T0 / (mu_xy * cons.m_p.cgs.value))
# fundamental density unit of the code
UNIT_DEN = UNIT_M / UNIT_L ** 3
# conversion factor: code mass-flux * code area  ->  M_sun / yr
UNIT_Fm = (UNIT_M / UNIT_T) / (M_sun / YR)

L_norm = AU / UNIT_L          # convert code length to AU
r0 = a0 * L_norm


def face_f_2_power(x2min, x2max, cell_width_ratio, num_face):
    # reconstruct the theta face positions for a ratio-spaced (x2rat = -1) grid
    x = linspace(0, 1, num_face)
    w = (x) ** (1 / 2)
    return w * (x2max - x2min) + x2min


# ----------------------------------------------------------------------
# read gas primitive variables (out1)
# ----------------------------------------------------------------------
filename = DIR + 'iceline.out1.' + str(nstep).rjust(5, '0') + '.athdf'
print("Reading file: ", filename)
data_prim = athena_read.athdf(filename, face_func_2=face_f_2_power, num_ghost=0)

rad = data_prim['x1v'] / L_norm    # cell-centered radius [AU]
theta = data_prim['x2v']           # cell-centered polar angle [rad]
phi = data_prim['x3v']
rad_f = data_prim['x1f'] / L_norm
theta_f = data_prim['x2f']
phi_f = data_prim['x3f']

rin = athinputs['mesh']['x1min'] 
rout = athinputs['mesh']['x1max']
x2min = athinputs['mesh']['x2min']
x2max = athinputs['mesh']['x2max']
Nrad = 300
intpl_numx = 320 
intpl_numz = 320 
xx_exp = linspace(rin/L_norm,rout/L_norm,intpl_numx)
zz_exp = linspace(-1.0,1.0,intpl_numz)[int(intpl_numz/2):]
xx_exp_mesh, zz_exp_mesh = meshgrid(xx_exp,zz_exp)

dR = data_prim['x1f'][1:]-data_prim['x1f'][0:-1]
dtheta = data_prim['x2f'][1:]-data_prim['x2f'][0:-1]
dphi = array([2.0*pi])
dtheta_3D, dphi_3D, dR_3D = meshgrid(dtheta,dphi, dR)
theta_3D, phi_3D, R_3D = meshgrid(data_prim['x2v'],array([pi]),data_prim['x1v'])

dS_R = R_3D**2 *sin(theta_3D) * dtheta_3D* dphi_3D
dS_theta = R_3D*sin(theta_3D) * dR_3D* dphi_3D
dS_phi = R_3D*dR_3D*dtheta_3D

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

rho = data_prim['rho']
vel1 = data_prim['vel1']
vel2 = data_prim['vel2']
vel3 = data_prim['vel3']
press = data_prim['press']

# midplane (theta = pi/2, last x2 index) radial pressure-gradient diagnostic:
# steep dlnP/dr near the inner edge is what drives (and is amplified by) the
# radial sloshing mode there.
press_mid = press[index_phi, -1, :]              # (nx1,) midplane pressure
dlnP_dr = gradient(log(press_mid), rad)          # d lnP / dr [1/AU]
dlnP_dr_exp = interp(xx_exp, rad, dlnP_dr)       # onto the interpolation grid

# ----------------------------------------------------------------------
# read gas mass flux from the user-defined output (out2); if it is not
# available, fall back to rho * v (the advective mass flux)
# ----------------------------------------------------------------------
fname_uov = DIR + 'iceline.out2.' + str(nstep).rjust(5, '0') + '.athdf'
print("Reading file: ", fname_uov)
data_uov = athena_read.athdf(fname_uov, face_func_2=face_f_2_power, num_ghost=0)

flx_x1 = data_uov['flx_x1']   # radial mass-flux density (r direction)
flx_x2 = data_uov['flx_x2']   # polar mass-flux density (theta direction)
tem = data_uov['Tem']

simu_time = data_prim['Time']


tem_xz = tem[index_phi,:,:].T

rho_xz = rho[index_phi,:,:].T
kappa0 = athinputs['problem']['kappa0']
f_vi = athinputs['problem']['f_vi']

def Get_kappa(kappa0, d2g, fv):
    return kappa0*(1.0-fv)*UNIT_DEN*UNIT_L

tau_opt = zeros(rho_xz.shape)
for j in range(tau_opt.shape[1]):
    dx2 = rad*L_norm*(theta_f[1]-theta_f[0])
    fv = zeros_like(rho_xz) 
    tau_opt[:,j] += tau_opt[:,j-1] + rho_xz[:,j]*Get_kappa(kappa0, 0.0, fv)[:,j] * dx2
tau_ir = tau_opt/3


flx_x1 *= dS_R* UNIT_Fm
flx_x2 *= dS_theta* UNIT_Fm

flux_gas_x,flux_gas_z,flux_gas_z = v_Intpl_Sph2car(rad,theta,phi,xx_exp,array([0.0]),zz_exp,(flx_x1/dS_R).T, (flx_x2/dS_theta).T, 0.0*flx_x2.T)
flux_gas_x_intpl = flux_gas_x[:,0,:]
flux_gas_z_intpl = flux_gas_z[:,0,:]

dz = (zz_exp[1]-zz_exp[0])*AU/UNIT_L
dx = (xx_exp[1]-xx_exp[0])*AU/UNIT_L
flux_gas_face = sum(flux_gas_x_intpl*dz,axis = 0) *2.0 *(2*pi*xx_exp*L_norm)  # remember to add up 2 wings

rho_intpl = scaler_Intpl_Sph2car(rad,theta,phi,xx_exp,array([0.0]),zz_exp,rho.T)[:,0,:]
Hpg_idx, yy_g = find_dust_scaleheight([[], rho_intpl], y_xz_c)

UNIT_SIGMA = UNIT_DEN*UNIT_L
sigma_gas =  sum(rho_intpl*dz,axis = 0)*2.0 *UNIT_SIGMA # remember to add up 2 wings

fig = plt.figure(figsize = (7,11.5),facecolor='white')
axes = fig.subplots(4,1)
ax = axes.flatten()
fig.subplots_adjust(hspace = 0.06)

aT = ax[0].contourf(x_xz_c, y_xz_c, tem_xz 
               , levels = linspace(100,400,100), cmap = 'coolwarm', extend = 'both')
cbarT = fig.colorbar(aT, ax=ax[0], orientation='vertical')
cbarT.set_label(r'Temperature [K]', fontsize = 15)
cbarT.ax.set_yticks(linspace(100,400,7))

T_levels = linspace(150,200,10)
# some colors from white to black: 
from matplotlib.cm import Greys
colors = Greys(linspace(0.0, 1, len(T_levels)))

ax[0].contour(x_xz_c, y_xz_c, tem_xz,'--', levels = T_levels, colors = colors, 
              linewidths = 1)

ax[0].plot(xx_exp, yy_g, '-.', c='gray', lw=1, zorder=10)
# ax[0].plot(xx_exp, 4*yy_g, '-.', c='k', lw=1, zorder=10)

upper_damping = xx_exp / tan(x2min + (x2max - x2min)*0.4)
upper_bond = xx_exp / tan(theta[0])
ax[0].plot(xx_exp, upper_damping, '--', c='k', lw=1, zorder=10)

ax[0].set_ylim(0,0.4)

ax[0].contour(x_xz_c,y_xz_c,tau_ir,levels = array([1.0]), colors = 'purple', linestyles = 'dashed', linewidths = 3.0, zorder = 5)
# hatch the place where the vertical velocity = 0 
ax[0].contourf(x_xz_c,y_xz_c,vel2[index_phi,:,:].T,levels = array([-1e-9,0.0,1e-9]), zorder = 10)

for i, tt in enumerate(T_levels):
    cbarT.ax.axhline(tt, color=colors[i], linestyle='-', linewidth=1)


ax[1].set_yscale('symlog', linthresh = 1e-5)
ax[1].axhline(-0.015, c= 'k', ls='--', zorder =10)
ax[1].plot(xx_exp,flux_gas_face*1e8,lw =3,color='grey', alpha = 0.6, label = r'$\mathcal{F}_{\mathrm{xy}}$')

ax[1].set_xlim(rin/L_norm,rout/L_norm)
ax[1].set_ylim(-3,5.0)
ax[1].annotate(r'$\dot{M}_{\mathrm{acc}}$',xy=(1.0,-0.9),fontsize = 15)
ax[1].annotate(r'$f_{\mathrm{i/g}} \dot{M}_{\mathrm{acc}}$',xy=(1.0,-0.25),fontsize = 15)
ax[1].set_ylabel(r'Radial Mass Flux [$10^{-8}M_{\odot}$/yr]',fontsize = 15)

ax[1].legend(loc='upper right', fontsize = 10)
for i in range(len(axes)):
    ax[i].set_xlim(rin/L_norm,rout/L_norm)
for i in range(3):  
    ax[i].set_xticklabels([])   

ax[3].set_xlabel(r'$r$ [au]')

ax[2].plot(xx_exp,sigma_gas, color = 'k')

# midplane radial pressure gradient (diagnostic: steep near the inner edge)
ax[3].plot(xx_exp, dlnP_dr_exp, color='k', lw=2)
ax[3].axhline(0.0, color='gray', lw=0.8)
ax[3].set_ylabel(r'$d\ln P/dr$  [AU$^{-1}$]', fontsize = 15)

ax[0].set_title('Time = {:.2f} yr'.format(simu_time*UNIT_T/YR),fontsize = 15, loc = 'left')
ax[0].annotate('(a)',xy = (0.02,0.92),xycoords = 'axes fraction',fontsize = 20)
ax[1].annotate('(b)',xy = (0.02,0.92),xycoords = 'axes fraction',fontsize = 20)
ax[2].annotate('(c)',xy = (0.02,0.92),xycoords = 'axes fraction',fontsize = 20)
ax[3].annotate('(d)',xy = (0.02,0.92),xycoords = 'axes fraction',fontsize = 20)

plt.savefig('./plots/fig_snow_2d_{:05d}.png'.format(int(filenum)), bbox_inches='tight', dpi = 500) 
plt.close()
import pdb; pdb.set_trace()

filelist = range(1823, 1893)
rho_ghost = zeros(len(filelist))
rho_active = zeros(len(filelist))
tL = zeros(len(filelist))
flx_ghost = zeros(len(filelist))
flx_active = zeros(len(filelist))

for i, fn in enumerate(filelist):
    filename = DIR + 'iceline.out1.' + str(fn).rjust(5, '0') + '.athdf'
    print("Reading file: ", filename)
    data_prim = athena_read.athdf(filename, face_func_2=face_f_2_power, num_ghost=2)
    rho = data_prim['rho']
    # vel1 = data_prim['vel1']
    # vel2 = data_prim['vel2']
    # vel3 = data_prim['vel3']

# ----------------------------------------------------------------------
# read gas mass flux from the user-defined output (out2); if it is not
# available, fall back to rho * v (the advective mass flux)
# ----------------------------------------------------------------------
    fname_uov = DIR + 'iceline.out2.' + str(fn).rjust(5, '0') + '.athdf'
    print("Reading file: ", fname_uov)
    data_uov = athena_read.athdf(fname_uov, face_func_2=face_f_2_power, num_ghost=2)

    flx_x1 = data_uov['flx_x1']   # radial mass-flux density (r direction)
    flx_x2 = data_uov['flx_x2']   # polar mass-flux density (theta direction)
    tem = data_uov['Tem']

    simu_time = data_prim['Time']

    tL[i] = simu_time*UNIT_T/YR
    rho_ghost[i] = rho[0,-1,1]*UNIT_DEN
    rho_active[i] = rho[0,-1,2]*UNIT_DEN

    flx_ghost[i] = flx_x1[0,-1,1]*UNIT_Fm * dS_R[0, -1, 1]
    flx_active[i] = flx_x1[0,-1,2]*UNIT_Fm * dS_R[0, -1, 2]

fig, ax = plt.subplots(figsize = (7,5),facecolor='white')
ax.plot(tL, rho_ghost/rho_active, label = r'$\rho_{\mathrm{ghost}}/\rho_{\mathrm{active}}$',
        color = 'k', lw = 2)
ax.set_ylim(-1e-3, 1e-3)
ax.plot(tL, flx_active, label = r'$\mathcal{F}_{\mathrm{active}}$', color = 'b', lw = 2)
ax.set_xlabel('Time [yr]')
ax.set_ylabel(r'Mass Flux [$M_{\odot}$/yr]')

fig.savefig('./plots/ghostflux_{:05d}.png'.format(int(filenum)), bbox_inches='tight', dpi = 500)
import pdb; pdb.set_trace()
