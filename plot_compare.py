#!/usr/bin/env python3
"""
plot_compare.py
Compare 2D dust density maps: single_pop (002241) vs passive_test (002756).
"""

import sys
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import astropy.constants as cons
import re
from copy import deepcopy

sys.path.insert(0, '/home/izx/athena_sublimation/vis/python')
import athena_read
from preplot import scaler_Intpl_Sph2car, v_Intpl_Sph2car, read_athinput

# ── physical constants ───────────────────────────────────────────────────────
AU   = cons.au.cgs.value
YR   = 365.2425 * 24 * 3600
M_sun = cons.M_sun.cgs.value
T_slope = -0.5
Cs_slope = T_slope / 2
H_slope  = Cs_slope + 1.5
sigma_slope = -(Cs_slope + H_slope)
a0  = 3.0
T0  = 150.0
Mdot_gas = 1.e-8 * M_sun / YR
alpha    = 3.e-3
mu_xy = 2.34
rin = 0.3 
rout = 3.0 
intpl_numz = 3000
intpl_numx = 320
xx_exp = linspace(rin,rout,intpl_numx)
zz_exp = linspace(-1.0,1.0,intpl_numz)[int(intpl_numz/2):]
xx_exp_mesh, zz_exp_mesh = meshgrid(xx_exp,zz_exp)

# ── helpers ──────────────────────────────────────────────────────────────────
def face_f_2_power(x2min, x2max, cell_width_ratio, num_face):
    x = linspace(0, 1, num_face)
    w = x**(1/3)
    return w * (x2max - x2min) + x2min
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

# ── data loader ──────────────────────────────────────────────────────────────
def load_run(dir_path, nstep):
    """Return a dict with everything needed for the 2D dust density plot."""
    d = {}
    inputfile = dir_path + 'athinput.iceline'
    ath = read_athinput(inputfile)

    UNIT_T   = ath['units']['time_cgs']
    UNIT_L   = ath['units']['length_cgs']
    UNIT_M   = ath['units']['mass_cgs']

    Cs0    = sqrt(cons.k_B.cgs.value * T0 / (mu_xy * cons.m_p.cgs.value))
    Sigma0 = Mdot_gas / (3.0 * pi * alpha * Cs0**2 * UNIT_T)
    UNIT_DEN = Sigma0 / (sqrt(2*pi) * UNIT_L)

    L_norm = AU / UNIT_L
    r0     = a0 * L_norm
    rin    = ath['mesh']['x1min']
    rout   = ath['mesh']['x1max']

    # interpolation grid
    xx_exp = linspace(rin/L_norm, rout/L_norm, intpl_numx)
    zz_exp = linspace(-1.0, 1.0, intpl_numz)[intpl_numz//2:]

    # ── primitive data ──
    fname1 = dir_path + 'iceline.out1.' + str(nstep).rjust(5, '0') + '.athdf'
    print(f'Reading: {fname1}')
    data_prim = athena_read.athdf(fname1, face_func_2=face_f_2_power, num_ghost=0)
    rad   = data_prim['x1v'] / L_norm
    theta = data_prim['x2v']
    phi   = data_prim['x3v']
    theta_f = data_prim['x2f']
    rad_f   = data_prim['x1f'] / L_norm

    d['simu_time'] = data_prim['Time'] * UNIT_T / YR
    d['UNIT_DEN']  = UNIT_DEN
    d['UNIT_M']    = UNIT_M
    d['L_norm']    = L_norm
    d['rin'] = rin; d['rout'] = rout
    d['rad']   = rad
    d['rad_f'] = rad_f
    d['theta']   = theta
    d['theta_f'] = theta_f

    rho = data_prim['rho']
    dust_id_pat = re.compile(r'^dust_(\d+)_rho$')
    dust_ids = sorted(int(m.group(1)) for k in data_prim.keys()
                      for m in [dust_id_pat.match(k)] if m)

    dust_rho = {did: data_prim[f'dust_{did}_rho'] for did in dust_ids}

    # ── user-defined data ──
    fname2 = dir_path + 'iceline.out2.' + str(nstep).rjust(5, '0') + '.athdf'
    print(f'Reading: {fname2}')
    data_uov = athena_read.athdf(fname2, face_func_2=face_f_2_power, num_ghost=0)
    tem = data_uov['Tem']
    kappa0 = ath['problem']['kappa0']

    st_pat = re.compile(r'^st_(\d+)$')
    pop_ids_1based = sorted(int(m.group(1)) for k in data_uov.keys()
                            for m in [st_pat.match(k)] if m)
    N_pop = len(pop_ids_1based)
    if N_pop == 0:
        N_pop = max(1, len(dust_ids)//2)

    ice_ids  = [2*p + 1 for p in range(N_pop) if (2*p + 1) in dust_ids]
    sil_ids  = [2*p + 2 for p in range(N_pop) if (2*p + 2) in dust_ids]
    vapor_id = 2*N_pop + 1 if (2*N_pop + 1) in dust_ids else None
    d['N_pop']    = N_pop
    d['ice_ids']  = ice_ids
    d['vapor_id'] = vapor_id

    # ── face-coordinate xz slice for contourf ──
    index_phi = 0
    THETA_f, PHI_f, R_f = meshgrid(theta_f, array([0.0]), rad_f)
    x_f = R_f * sin(THETA_f) * cos(PHI_f)
    z_f = R_f * cos(THETA_f)
    x_xz   = x_f[index_phi,:,:].T
    y_xz   = z_f[index_phi,:,:].T
    x_xz_c = x_xz[1:,1:]
    y_xz_c = y_xz[1:,1:]
    d['x_xz_c'] = x_xz_c; d['y_xz_c'] = y_xz_c

    # ── xz slices ──
    rho_xz = rho[index_phi,:,:].T
    dust_rho_xz = {did: dust_rho[did][index_phi,:,:].T for did in dust_ids}
    tem_xz = tem[index_phi,:,:].T
    d['rho_xz'] = rho_xz; d['tem_xz'] = tem_xz

    # ── masked densities (filter tiny d2g) ──
    d2g_snow = 1.e-4
    dust_rho_mod_xz = {did: deepcopy(arr) for did, arr in dust_rho_xz.items()}
    for did in dust_rho_mod_xz:
        dust_rho_mod_xz[did][dust_rho_xz[did]/rho_xz < d2g_snow * 0.5] = nan

    # Get correct arrays by semantic role
    if vapor_id is not None:
        d['vap_rho_mod'] = dust_rho_mod_xz.get(vapor_id, zeros_like(rho_xz))
        d['vap_rho_xz']  = dust_rho_xz.get(vapor_id, zeros_like(rho_xz))
    else:
        d['vap_rho_mod'] = zeros_like(rho_xz)
        d['vap_rho_xz']  = zeros_like(rho_xz)

    d['ice_rho_mod'] = {}; d['ice_rho_xz'] = {}
    for iid in ice_ids:
        d['ice_rho_mod'][iid] = dust_rho_mod_xz.get(iid, zeros_like(rho_xz))
        d['ice_rho_xz'][iid]  = dust_rho_xz.get(iid, zeros_like(rho_xz))

    # silicate densities per population (sil id = 2*p+2), for water-comp maps
    d['sil_rho_mod'] = {}; d['sil_rho_xz'] = {}
    for sid in sil_ids:
        d['sil_rho_mod'][sid] = dust_rho_mod_xz.get(sid, zeros_like(rho_xz))
        d['sil_rho_xz'][sid]  = dust_rho_xz.get(sid, zeros_like(rho_xz))

    # water (ice) mass fraction per population: f_H2O = rho_ice/(rho_ice+rho_sil)
    d['watercomp'] = {}
    for p, iid in enumerate(ice_ids):
        sid = 2*p + 2
        den = d['ice_rho_xz'][iid] + d['sil_rho_xz'].get(sid, zeros_like(rho_xz))
        d['watercomp'][p] = where(den > 0.0, d['ice_rho_xz'][iid]/den, 0.0)

    d['d2g_snow'] = d2g_snow

    # ── optical depth τ_ir ──
    tau_opt = zeros(rho_xz.shape)
    for j in range(tau_opt.shape[1]):
        dx2 = rad * L_norm * (theta_f[1] - theta_f[0])
        fv = d['vap_rho_xz'] / rho_xz
        k = kappa0 * (1.0 - fv) * UNIT_DEN * UNIT_L
        tau_opt[:, j] = tau_opt[:, j-1] + rho_xz[:, j] * k[:, j] * dx2
    d['tau_ir'] = tau_opt / 3

    # ── scale heights via interpolation ──
    rho_intpl = scaler_Intpl_Sph2car(rad, theta, phi, xx_exp, array([0.0]),
                                     zz_exp, rho.T)[:,0,:]
    dust_rho_intpl = {
        did: scaler_Intpl_Sph2car(rad, theta, phi, xx_exp, array([0.0]),
                                  zz_exp, dust_rho[did].T)[:,0,:]
        for did in dust_ids
    }

    # ── interpolate τ_ir and temperature for Col_vap ──
    tau_ir_intpl = scaler_Intpl_Sph2car(rad, theta, phi, xx_exp, array([0.0]),
                                        zz_exp, array([(tau_opt/3).T]).T)[:,0,:]
    tem_intpl = scaler_Intpl_Sph2car(rad, theta, phi, xx_exp, array([0.0]),
                                     zz_exp, tem.T)[:,0,:]

    for j in range(len(zz_exp)):
        for i in range(len(xx_exp)):
            if (fabs(zz_exp[j]/xx_exp[i]) > tan(pi/2-1.3) or
                xx_exp[i]**2 + zz_exp[j]**2 > (rout/L_norm)**2 or
                xx_exp[i]**2 + zz_exp[j]**2 < (rin/L_norm)**2):
                rho_intpl[j,i] = 0.0
                tem_intpl[j,i] = 0.0
                tau_ir_intpl[j,i] = 0.0
                for did in dust_rho_intpl:
                    dust_rho_intpl[did][j,i] = 0.0

    if len(ice_ids) >= 1:
        iid0 = ice_ids[0]
        sid0 = sil_ids[0] if len(sil_ids) > 0 else None
        d1_arr = dust_rho_intpl.get(iid0, zeros_like(rho_intpl))
        d2_arr = dust_rho_intpl.get(sid0, zeros_like(rho_intpl)) if sid0 else zeros_like(rho_intpl)
        _, yy0 = find_dust_scaleheight([d1_arr, d2_arr], y_xz_c)
        d['yy0'] = yy0
    else:
        d['yy0'] = zeros(intpl_numx)

    if len(ice_ids) >= 2:
        iid1 = ice_ids[1]
        sid1 = sil_ids[1] if len(sil_ids) > 1 else None
        d3_arr = dust_rho_intpl.get(iid1, zeros_like(rho_intpl))
        d4_arr = dust_rho_intpl.get(sid1, zeros_like(rho_intpl)) if sid1 else zeros_like(rho_intpl)
        _, yy1 = find_dust_scaleheight([d3_arr, d4_arr], y_xz_c)
        d['yy1'] = yy1
    else:
        d['yy1'] = zeros(intpl_numx)

    # gas scale height
    _, yy_g = find_dust_scaleheight([[], rho_intpl], y_xz_c)

    # smooth scale heights to remove stair-step from discrete z-grid
    from scipy import interpolate as _interp
    mask0 = ~isnan(d['yy0'])
    if mask0.sum() > 3:
        yy0[mask0] = _interp.interp1d(xx_exp[mask0], yy0[mask0], kind='cubic')(xx_exp[mask0])

    mask1 = ~isnan(d['yy1'])
    if mask1.sum() > 3:
        d['yy1'][mask1] = _interp.interp1d(xx_exp[mask1], d['yy1'][mask1], kind='cubic')(xx_exp[mask1])

    maskg = ~isnan(yy_g)
    if maskg.sum() > 3:
        yy_g[maskg] = _interp.interp1d(xx_exp[maskg], yy_g[maskg], kind='cubic')(xx_exp[maskg])

    d['yy_g'] = yy_g
    d['xx_exp'] = xx_exp

    # ── streamplot flux fields ───────────────────────────────────────────────
    UNIT_Fm = (UNIT_L**3 * UNIT_DEN / UNIT_T) / (M_sun / YR)

    # cell areas
    dR = rad_f[1:] - rad_f[:-1]  # in L_norm units
    dtheta_arr = theta_f[1:] - theta_f[:-1]
    dphi_arr = array([2.0*pi])
    dtheta_3D, dphi_3D, dR_3D = meshgrid(dtheta_arr, dphi_arr, dR)
    theta_3D, phi_3D, R_3D = meshgrid(theta, array([pi]), rad * L_norm)
    dS_R = R_3D**2 * sin(theta_3D) * dtheta_3D * dphi_3D
    dS_theta = R_3D * sin(theta_3D) * dR_3D * dphi_3D

    # read flux data from uov (keys: flx_ice_x1_1, flx_ice_x2_1 for pop0)
    flx_ice_x1_raw = data_uov.get('flx_ice_x1_1', zeros_like(tem))
    flx_ice_x2_raw = data_uov.get('flx_ice_x2_1', zeros_like(tem))
    flx_vap_x1_raw = data_uov.get('flx_vap_x1', zeros_like(tem))
    flx_vap_x2_raw = data_uov.get('flx_vap_x2', zeros_like(tem))

    # multiply by area factors
    flx_ice_x1 = flx_ice_x1_raw * dS_R * UNIT_Fm
    flx_ice_x2 = flx_ice_x2_raw * dS_theta * UNIT_Fm
    flx_vap_x1 = flx_vap_x1_raw * dS_R * UNIT_Fm
    flx_vap_x2 = flx_vap_x2_raw * dS_theta * UNIT_Fm

    flx_water_x1 = flx_vap_x1
    flx_water_x2 = flx_vap_x2

    # streamplot grid
    R_inner = rin / L_norm
    xs = rout / L_norm
    numx = 64; numz = 32
    x1_exp_half = linspace(R_inner, xs, numx)
    x3_exp = linspace(0.0, 0.6, numz)
    slice_exp = array([0.0])

    # interpolate fluxes to Cartesian
    water_flx_x, water_flx_y, water_flx_z = v_Intpl_Sph2car(
        rad, theta, phi, x1_exp_half, slice_exp, x3_exp,
        flx_water_x1.T, flx_water_x2.T, (flx_water_x1 * 0.0).T)
    water_flx_x_xz = water_flx_x[:, 0, :]
    water_flx_z_xz = water_flx_z[:, 0, :]

    ice_flx_x, ice_flx_y, ice_flx_z = v_Intpl_Sph2car(
        rad, theta, phi, x1_exp_half, slice_exp, x3_exp,
        flx_ice_x1.T, flx_ice_x2.T, (flx_ice_x1 * 0.0).T)
    ice_flx_x_xz = ice_flx_x[:, 0, :]
    ice_flx_z_xz = ice_flx_z[:, 0, :]

    # also try ice1 fluxes (pop1) if available
    flx_ice1_x1_raw = data_uov.get('flx_ice_x1_2', zeros_like(tem))
    flx_ice1_x2_raw = data_uov.get('flx_ice_x2_2', zeros_like(tem))
    flx_ice1_x1 = flx_ice1_x1_raw * dS_R * UNIT_Fm
    flx_ice1_x2 = flx_ice1_x2_raw * dS_theta * UNIT_Fm
    ice1_flx_x, ice1_flx_y, ice1_flx_z = v_Intpl_Sph2car(
        rad, theta, phi, x1_exp_half, slice_exp, x3_exp,
        flx_ice1_x1.T, flx_ice1_x2.T, (flx_ice1_x1 * 0.0).T)
    ice1_flx_x_xz = ice1_flx_x[:, 0, :]
    ice1_flx_z_xz = ice1_flx_z[:, 0, :]

    # mask outside domain
    for j in range(numz):
        for i in range(numx):
            ratio_xz = fabs(x1_exp_half[i] / x3_exp[j]) if x3_exp[j] != 0.0 else inf
            if ratio_xz < tan(0.8) or (x1_exp_half[i]**2 + x3_exp[j]**2 > xs**2):
                water_flx_x_xz[j, i] = 0.0
                water_flx_z_xz[j, i] = 0.0
                ice_flx_x_xz[j, i] = 0.0
                ice_flx_z_xz[j, i] = 0.0
                ice1_flx_x_xz[j, i] = 0.0
                ice1_flx_z_xz[j, i] = 0.0

    # normalization & linewidths
    normal2 = sort(sqrt(water_flx_x_xz**2 + water_flx_z_xz**2), axis=None)[-2]
    lw_flx_ice = 2.0 * sqrt(sqrt(ice_flx_x_xz**2 + ice_flx_z_xz**2) / normal2)
    lw_flx_water = 2.0 * sqrt(sqrt(water_flx_x_xz**2 + water_flx_z_xz**2) / normal2)
    lw_flx_ice1 = 2.0 * sqrt(sqrt(ice1_flx_x_xz**2 + ice1_flx_z_xz**2) / normal2)

    d['x1_exp_half'] = x1_exp_half
    d['x3_exp'] = x3_exp
    d['normal2'] = normal2
    d['ice_flx_x_xz'] = ice_flx_x_xz
    d['ice_flx_z_xz'] = ice_flx_z_xz
    d['water_flx_x_xz'] = water_flx_x_xz
    d['water_flx_z_xz'] = water_flx_z_xz
    d['ice1_flx_x_xz'] = ice1_flx_x_xz
    d['ice1_flx_z_xz'] = ice1_flx_z_xz
    d['lw_flx_ice'] = lw_flx_ice
    d['lw_flx_water'] = lw_flx_water
    d['lw_flx_ice1'] = lw_flx_ice1

    # ── Col_vap and Tem_col ──
    dz = (zz_exp[1] - zz_exp[0]) * AU / UNIT_L
    UNIT_SIGMA = UNIT_DEN * UNIT_L
    tau_ir_1 = where(tau_ir_intpl < 1.0, 1.0, 0.0)
    tau_ir_1 = where(tau_ir_intpl <= 0.0, 0.0, tau_ir_1)
    d['dust_5_rho_intpl'] = dust_rho_intpl.get(vapor_id, zeros_like(rho_intpl)) if vapor_id else zeros_like(rho_intpl)
    Col_vap = sum(d['dust_5_rho_intpl'] * tau_ir_1 * dz, axis=0) * 2.0 \
              * (2 * pi * xx_exp * L_norm) * UNIT_SIGMA / (18.0 * cons.m_p.cgs.value)
    denom = sum(d['dust_5_rho_intpl'] * tau_ir_1 * dz, axis=0)
    Tem_col = zeros(intpl_numx)
    Tem_col = sum(d['dust_5_rho_intpl'] * tem_intpl * tau_ir_1 * dz, axis=0) / where(denom > 0, denom, 1.0)
    d['Col_vap'] = Col_vap
    d['Tem_col'] = Tem_col
    d['xx_exp'] = xx_exp

    return d


# ══════════════════════════════════════════════════════════════════════════════
#  Load all runs once — every figure below reuses these in-memory dicts
# ══════════════════════════════════════════════════════════════════════════════
BASE = '../../athena_works/'
NSTEP = 530

data_530 = {}
for run in ('DAS', 'DPS', 'DAR', 'DPR'):
    print(f'Loading {run} @ {NSTEP} ...')
    data_530[run] = load_run(BASE + run + '/', NSTEP)

# short aliases used by the figures below
d1 = data_530['DAS']   # single-pop active
d2 = data_530['DPS']   # single-pop passive
d3 = data_530['DAR']   # two-pop active
d4 = data_530['DPR']   # two-pop passive

# ── Col_vap bar comparison figure — all 3 bars in ONE panel ─────────────────
fig_bar, ax_bar = plt.subplots(figsize=(14, 7))

norm_bar = LogNorm(vmin=1e17, vmax=1e26)
ax_bar.set_xlim(0.3, 3.5)
bar_height = 0.4

cases = [
    (d1, '1 pop active',    2.0),
    (d2, '1 pop passive',      3.5),
    (d3, '2 pop active',  5.0),
    (d4, '2 pop passive',  6.5),
]

def _find_crossings(x, y, thresh):
    above = y > thresh
    crossings = []
    for i in where(diff(above.astype(int)))[0]:
        if i + 1 < len(x):
            frac = (thresh - y[i]) / (y[i+1] - y[i])
            crossings.append(x[i] + frac * (x[i+1] - x[i]))
    return crossings

colors_markers = ['gray', 'black']
colTem = ['orange', 'blue']
TemL = [400, 150]

for d, label, y_lo in cases:
    y_hi = y_lo + bar_height
    y_mid = (y_lo + y_hi) / 2

    Col_use = d['Col_vap']
    xx_use = d['xx_exp']

    # restrict to Col_vap >= 1e18
    idx_valid = where(Col_use >= 1e18)[0]
    if len(idx_valid) > 0:
        i0, i1 = idx_valid[0], idx_valid[-1] + 1
        xx_use = xx_use[i0:i1]
        Col_use = Col_use[i0:i1]

    # cell edges
    xx_edges = empty(len(xx_use) + 1)
    xx_edges[1:-1] = 0.5 * (xx_use[1:] + xx_use[:-1])
    xx_edges[0]  = xx_use[0]  - (xx_use[1]  - xx_use[0])  / 2.0
    xx_edges[-1] = xx_use[-1] + (xx_use[-1] - xx_use[-2]) / 2.0

    C2d = Col_use.reshape(1, -1)
    mesh = ax_bar.pcolormesh(xx_edges, [y_lo, y_hi], C2d, norm=norm_bar, cmap='Purples')

    # threshold markers
    for i_th, thresh in enumerate([1e18, 1e19]):
        for xc in _find_crossings(xx_use, Col_use, thresh):
            ax_bar.plot(xc, y_mid, marker='|', color=colors_markers[i_th],
                        markersize=12, markeredgewidth=2, zorder=13)

    # peak
    peak_idx = argmax(Col_use)
    ax_bar.plot(xx_use[peak_idx], y_mid, marker='*', color='gold', markersize=18,
                markeredgecolor='black', markeredgewidth=0.5, zorder=14)

    # temperature markers
    for i_th, thresh in enumerate(TemL):
        for xc in _find_crossings(d['xx_exp'], d['Tem_col'], thresh):
            ax_bar.plot(xc, y_mid, marker='x', color=colTem[i_th],
                        markersize=12, markeredgewidth=2, zorder=13)

ax_bar.set_yticks([d[2] + bar_height/2 for d in cases])
ax_bar.set_yticklabels([d[1] for d in cases])
ax_bar.set_ylim(1, 7)
ax_bar.set_xlabel(r'$R$ [AU]', fontsize=13)

cb = fig_bar.colorbar(mesh, ax=ax_bar, orientation='vertical', shrink=0.8)
cb.set_label(r'Col$_{\rm vap}$ [molecules]', fontsize=12)
for i_th, thresh in enumerate([1e18, 1e19]):
    cb.ax.hlines(y=thresh, xmin=0, xmax=1, color=colors_markers[i_th], linewidth=2)

fig_bar.tight_layout()
fig_bar.savefig('./plots/compare_col_vap.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: ./plots/compare_col_vap.png')

# ── build comparison figure (vertical stack) ────────────────────────────────
plt.rcParams.update({'font.size': 13})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 12))
# fig.subplots_adjust(hspace=0.35)
#
# # ── TOP: single_pop ──────────────────────────────────────────────────────────
# ax1.set_ylabel(r'$z$ [AU]', fontsize=12)
# ax1.set_ylim(0., 0.15)
# ax1.set_xlim(d1['rin']/d1['L_norm'], d1['rout']/d1['L_norm'])
#
# # legend
# legends = [
#     Line2D([0],[0], color='k', lw=2, marker='>', label=r'$10^{-3}\rho_0 c_{\mathrm{s,0}}$'),
#     Line2D([0],[0], color='k', ls='--', lw=1, label=r'$H_{\mathrm{peb}}$'),
#     Line2D([0],[0], color='gray', ls='-.', lw=1, label=r'$H_{\mathrm{gas}}$'),
# ]
# ax1.legend(handles=legends, loc='upper right', fontsize=12, framealpha=0.)
#
# # vapor (RdPu)
# c_vap1 = ax1.contourf(d1['x_xz_c'], d1['y_xz_c'],
#                        d1['vap_rho_mod'] * d1['UNIT_DEN'],
#                        levels=logspace(-13, -11, 25), norm=LogNorm(),
#                        cmap='RdPu', alpha=0.7, extend='both',
#                        zorder=3, antialiased=True)
#
# # ice (Blues) — single pop: only one ice species
# iid0 = d1['ice_ids'][0]
# c_ice1 = ax1.contourf(d1['x_xz_c'], d1['y_xz_c'],
#                        d1['ice_rho_mod'][iid0] * d1['UNIT_DEN'],
#                        levels=logspace(-13, log10(3e-11), 20), norm=LogNorm(),
#                        cmap='Blues', alpha=1.0, extend='both',
#                        antialiased=True, zorder=4)
#
# ax1.contour(d1['x_xz_c'], d1['y_xz_c'], d1['tau_ir'],
#             levels=array([0.5,1.0]), colors='black',
#             linestyles='dotted', zorder=20)
# ax1.plot(d1['xx_exp'], d1['yy0'], '--', c='k', lw=1, zorder=10)
# ax1.plot(d1['xx_exp'], d1['yy_g'], '-.', c='gray', lw=1, zorder=10)
# ax1.contour(d1['x_xz_c'], d1['y_xz_c'],
#             d1['ice_rho_xz'][iid0] / d1['rho_xz'],
#             levels=[d1['d2g_snow']], cmap='Blues_r',
#             alpha=0.7, linewidths=3.0, zorder=4)
#
# # streamlines — single_pop: ice (blue) + water vapor (pink)
# ax1.streamplot(d1['x1_exp_half'], d1['x3_exp'],
#                d1['ice_flx_x_xz'] / d1['normal2'],
#                d1['ice_flx_z_xz'] / d1['normal2'],
#                linewidth=d1['lw_flx_ice'], arrowstyle='->',
#                density=1.0, broken_streamlines=True,
#                color='blue', zorder=4)
# ax1.streamplot(d1['x1_exp_half'], d1['x3_exp'],
#                d1['water_flx_x_xz'] / d1['normal2'],
#                d1['water_flx_z_xz'] / d1['normal2'],
#                linewidth=d1['lw_flx_water'], arrowstyle='->',
#                density=1.0, broken_streamlines=True,
#                color='pink', zorder=4)
#
# # ── BOTTOM: passive_test ────────────────────────────────────────────────────
# ax2.set_xlabel(r'$R$ [AU]', fontsize=12)
# ax2.set_ylabel(r'$z$ [AU]', fontsize=12)
# ax2.set_ylim(-0.15, 0.15)
# ax2.set_xlim(d2['rin']/d2['L_norm'], d2['rout']/d2['L_norm'])
#
# # vapor — both hemispheres
# c_vap2u = ax2.contourf(d2['x_xz_c'],  d2['y_xz_c'],
#                         d2['vap_rho_mod'] * d2['UNIT_DEN'],
#                         levels=logspace(-13, -11, 25), norm=LogNorm(),
#                         cmap='RdPu', alpha=0.7, extend='both',
#                         zorder=3, antialiased=True)
# ax2.contourf(d2['x_xz_c'], -d2['y_xz_c'],
#              d2['vap_rho_mod'] * d2['UNIT_DEN'],
#              levels=logspace(-13, -11, 25), norm=LogNorm(),
#              cmap='RdPu', alpha=0.7, extend='both',
#              zorder=3, antialiased=True)
#
# # ice pop1 — upper hemisphere
# iid1 = d2['ice_ids'][1] if len(d2['ice_ids']) > 1 else d2['ice_ids'][0]
# c_ice2u = ax2.contourf(d2['x_xz_c'], d2['y_xz_c'],
#                         d2['ice_rho_mod'][iid1] * d2['UNIT_DEN'],
#                         levels=logspace(-15, log10(3e-11), 20), norm=LogNorm(),
#                         cmap='Blues', alpha=1.0, extend='both',
#                         antialiased=True, zorder=4)
#
# # ice pop0 — lower hemisphere
# iid0_d2 = d2['ice_ids'][0]
# c_ice2l = ax2.contourf(d2['x_xz_c'], -d2['y_xz_c'],
#                         d2['ice_rho_mod'][iid0_d2] * d2['UNIT_DEN'],
#                         levels=logspace(-15, log10(3e-11), 20), norm=LogNorm(),
#                         cmap='Blues', alpha=1.0, extend='both',
#                         antialiased=True, zorder=4)
#
# # streamlines — passive_test: 4 calls matching non-singlepop 2ddust
# z_neg = -d2['x3_exp'][::-1]
# # upper: ice1 (pop1) + water vapor
# ax2.streamplot(d2['x1_exp_half'], d2['x3_exp'],
#                d2['ice1_flx_x_xz'] / d2['normal2'],
#                d2['ice1_flx_z_xz'] / d2['normal2'],
#                linewidth=d2['lw_flx_ice1'], arrowstyle='->',
#                density=1.0, broken_streamlines=True,
#                color='blue', zorder=4)
# ax2.streamplot(d2['x1_exp_half'], d2['x3_exp'],
#                d2['water_flx_x_xz'] / d2['normal2'],
#                d2['water_flx_z_xz'] / d2['normal2'],
#                linewidth=d2['lw_flx_water'], arrowstyle='->',
#                density=1.0, broken_streamlines=True,
#                color='pink', zorder=4)
# # lower: water vapor + ice (pop0)
# ax2.streamplot(d2['x1_exp_half'], z_neg,
#                d2['water_flx_x_xz'][::-1, :] / d2['normal2'],
#                -d2['water_flx_z_xz'][::-1, :] / d2['normal2'],
#                linewidth=d2['lw_flx_water'][::-1, :], arrowstyle='->',
#                density=1.0, broken_streamlines=True,
#                color='pink', zorder=4)
# ax2.streamplot(d2['x1_exp_half'], z_neg,
#                d2['ice_flx_x_xz'][::-1, :] / d2['normal2'],
#                -d2['ice_flx_z_xz'][::-1, :] / d2['normal2'],
#                linewidth=d2['lw_flx_ice'][::-1, :], arrowstyle='->',
#                density=1.0, broken_streamlines=True,
#                color='blue', zorder=4)
#
# ax1.set_yticks([0.0, 0.05, 0.10, 0.15])
# ax2.set_yticks([-0.15,-0.10,-0.05, 0.0, 0.05, 0.10, 0.15])
#
# # contours and lines
# ax2.contour(d2['x_xz_c'],  d2['y_xz_c'], d2['tau_ir'],
#             levels=array([0.5,1.0]), colors='black',
#             linestyles='dotted', zorder=20)
# ax2.contour(d2['x_xz_c'], -d2['y_xz_c'], d2['tau_ir'],
#             levels=array([0.5,1.0]), colors='black',
#             linestyles='dotted', zorder=20)
# ax2.plot(d2['xx_exp'], -d2['yy0'], '--', c='k', lw=1, zorder=10)
# ax2.plot(d2['xx_exp'],  d2['yy_g'], '-',  c='r', lw=1, zorder=10)
# ax2.plot(d2['xx_exp'],  d2['yy1'], '--', c='k', lw=1, zorder=10)
# ax2.contour(d2['x_xz_c'],  d2['y_xz_c'],
#             d2['ice_rho_xz'][iid1] / d2['rho_xz'],
#             levels=[d2['d2g_snow']], cmap='Blues_r',
#             alpha=0.7, linewidths=3.0, zorder=4)
# ax2.contour(d2['x_xz_c'], -d2['y_xz_c'],
#             d2['ice_rho_xz'][iid0_d2] / d2['rho_xz'],
#             levels=[d2['d2g_snow']], cmap='Blues_r',
#             alpha=0.7, linewidths=3.0, zorder=4)
#
# # labels
# ax2.text(0.05, 0.90, 'pop$_1$', transform=ax2.transAxes, fontsize=16, va='top', ha='left')
# ax2.text(0.05, 0.05, 'pop$_0$', transform=ax2.transAxes, fontsize=16, va='bottom', ha='left')
#
#
# # ── TWO COLORBARS at top, horizontally aligned ───────────────────────────────
# # Shared across both panels
# cb_ice = fig.colorbar(c_ice2u, ax=ax1, location='top',
#                        shrink=0.45, pad=-0.15, anchor=(0., 0.))
# cb_ice.set_ticks([1e-13, 1e-12, 1e-11])
# cb_ice.set_ticklabels(['$10^{-13}$', '$10^{-12}$', '$10^{-11}$'])
#
# cb_vap = fig.colorbar(c_vap1, ax=ax1, location='top',
#                        shrink=0.45, pad=0.06, anchor=(1, 0.))
# cb_vap.set_ticks([1e-13, 1e-12, 1e-11])
# cb_vap.set_ticklabels(['$10^{-13}$', '$10^{-12}$', '$10^{-11}$'])
#
#
# fig.tight_layout()
# fig.savefig('./plots/compare_2ddust.png', dpi=300, bbox_inches='tight')
# print('Saved: ./plots/compare_2ddust.png')
#

# ══════════════════════════════════════════════════════════════════════════════
#  Figure 2: vap_obs comparison
# ══════════════════════════════════════════════════════════════════════════════

def _mass_threshold(mass_map, rho_map, rad, theta, thres=0.99):
    cells = []
    total_mass = 0.0
    for i in range(len(rad)):
        for j in range(len(theta)):
            if mass_map[i, j] > 0 and rho_map[i, j] > 0:
                cells.append((rho_map[i, j], mass_map[i, j]))
                total_mass += mass_map[i, j]
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


def plot_vap_obs(ax, d, show_legend=True):
    """Plot vap_obs on a single axis. Returns (crhov, crho1, C_Tem) for colorbars."""
    from matplotlib.patches import Patch

    ax.set_ylim(0, 0.25)
    ax.set_xlim(d['rin']/d['L_norm'], 3)
    ax.set_ylabel(r'$z$ [AU]', fontsize=12)

    # background vapor density (Greys)
    crhov = ax.contourf(d['x_xz_c'], d['y_xz_c'],
                d['vap_rho_xz'] * d['UNIT_DEN'],
                levels=logspace(-19, -9, 10), norm=LogNorm(),
                cmap='Greys', alpha=1.0, extend='both',
                zorder=3, antialiased=True)

    # ice-to-gas ratio
    if d['N_pop'] == 1:
        ice_ratio = d['ice_rho_mod'][d['ice_ids'][0]] / d['rho_xz']
    else:
        ice_sum = sum(d['ice_rho_mod'][iid] for iid in d['ice_ids'])
        ice_ratio = ice_sum / d['rho_xz']

    ice_colors = ['white', 'skyblue', 'deepskyblue', 'dodgerblue', 'blue', 'darkblue']
    # crho1 = ax.contourf(d['x_xz_c'], d['y_xz_c'], ice_ratio,
    #             levels=logspace(log10(0.001), log10(0.05), 7),
    #             norm=LogNorm(), antialiased=True,
    #             colors=ice_colors, alpha=0.7, extend='both', zorder=4)

    # τ_ir = 1 contour
    ax.contour(d['x_xz_c'], d['y_xz_c'], d['tau_ir'],
               levels=array([1.0]), colors='purple',
               linestyles='dashed', linewidths=3.0, zorder=5)

    # ── vapor colored by temperature ──
    vap_rho = d['vap_rho_xz'] * d['UNIT_DEN']
    tem_xz = d['tem_xz']
    tau_ir = d['tau_ir']
    rad   = d['rad']
    rad_f = d['rad_f']
    theta = d['theta']
    theta_f = d['theta_f']
    UNIT_M = d['UNIT_M']

    vap_cold = ma.masked_where(~((tem_xz < 150) & (vap_rho > 0)), vap_rho)
    vap_warm = ma.masked_where(~((tem_xz >= 150) & (tem_xz < 400) & (vap_rho > 0)), vap_rho)
    vap_hot  = ma.masked_where(~((tem_xz >= 400) & (vap_rho > 0)), vap_rho)

    # mass integration
    m_cold_M = zeros_like(d['vap_rho_xz'])
    m_warm_M = zeros_like(d['vap_rho_xz'])
    m_hot_M  = zeros_like(d['vap_rho_xz'])
    for i in range(len(rad)):
        for j in range(len(theta)):
            if tem_xz[i, j] < 150 and tau_ir[i, j] < 1.0:
                m_cold_M[i, j] = d['vap_rho_xz'][i, j] * rad[i]**2 * sin(theta[j]) \
                    * diff(rad_f)[i] * diff(theta_f)[j] * 2*pi * UNIT_M
            elif tem_xz[i, j] >= 150 and tem_xz[i, j] < 400 and tau_ir[i, j] < 1.0:
                m_warm_M[i, j] = d['vap_rho_xz'][i, j] * rad[i]**2 * sin(theta[j]) \
                    * diff(rad_f)[i] * diff(theta_f)[j] * 2*pi * UNIT_M
            elif tem_xz[i, j] >= 400 and tau_ir[i, j] < 1.0:
                m_hot_M[i, j] = d['vap_rho_xz'][i, j] * rad[i]**2 * sin(theta[j]) \
                    * diff(rad_f)[i] * diff(theta_f)[j] * 2*pi * UNIT_M

    threshold_cold = _mass_threshold(m_cold_M, vap_rho, rad, theta, thres=0.99)
    threshold_warm = _mass_threshold(m_warm_M, vap_rho, rad, theta, thres=0.9)
    threshold_hot  = _mass_threshold(m_hot_M,  vap_rho, rad, theta, thres=0.9)

    m_cold_tot = sum(m_cold_M)
    m_warm_tot = sum(m_warm_M)
    m_hot_tot  = sum(m_hot_M)

    vap_cold_90 = ma.masked_where(~((tau_ir < 1.0) & (vap_cold > 0) & (vap_rho >= threshold_cold)), vap_rho)
    vap_warm_90 = ma.masked_where(~((tau_ir < 1.0) & (vap_warm > 0) & (vap_rho >= threshold_warm)), vap_rho)
    vap_hot_90  = ma.masked_where(~((tau_ir < 1.0) & (vap_hot > 0) & (vap_rho >= threshold_hot)), vap_rho)

    levels_vap = logspace(-20, -8, 10)
    ax.contourf(d['x_xz_c'],  d['y_xz_c'], vap_cold_90, levels=levels_vap,
                norm=LogNorm(), colors=['blue'],  alpha=0.5, zorder=6, antialiased=True)
    ax.contourf(d['x_xz_c'],  d['y_xz_c'], vap_warm_90, levels=levels_vap,
                norm=LogNorm(), colors=['orange'], alpha=0.5, zorder=6, antialiased=True)
    ax.contourf(d['x_xz_c'],  d['y_xz_c'], vap_hot_90,  levels=levels_vap,
                norm=LogNorm(), colors=['red'],   alpha=0.5, zorder=6, antialiased=True)
    ax.contourf(d['x_xz_c'], -d['y_xz_c'], vap_cold_90, levels=levels_vap,
                norm=LogNorm(), colors=['blue'],  alpha=0.5, zorder=6, antialiased=True)
    ax.contourf(d['x_xz_c'], -d['y_xz_c'], vap_warm_90, levels=levels_vap,
                norm=LogNorm(), colors=['orange'], alpha=0.5, zorder=6, antialiased=True)
    ax.contourf(d['x_xz_c'], -d['y_xz_c'], vap_hot_90,  levels=levels_vap,
                norm=LogNorm(), colors=['red'],   alpha=0.5, zorder=6, antialiased=True)

    # temperature contours
    C_Tem = ax.contour(d['x_xz_c'],  d['y_xz_c'], tem_xz,
               levels=linspace(100, 400, 5, endpoint=True),
               cmap='coolwarm', alpha=0.8, linewidths=1.5,
               linestyles='dashed', zorder=11)
    ax.contour(d['x_xz_c'], -d['y_xz_c'], tem_xz,
               levels=linspace(100, 400, 5, endpoint=True),
               cmap='coolwarm', alpha=0.8, linewidths=1.5,
               linestyles='dashed', zorder=11)
    ax.contour(d['x_xz_c'],  d['y_xz_c'], tem_xz,
               levels=linspace(100, 400, 5, endpoint=True),
               colors='white', alpha=0.8, linewidths=2.8, zorder=10)
    ax.contour(d['x_xz_c'], -d['y_xz_c'], tem_xz,
               levels=linspace(100, 400, 5, endpoint=True),
               colors='white', alpha=0.8, linewidths=2.8, zorder=10)

    # legend — only on the left panel
    if show_legend:
        legend_elements = [
            Patch(facecolor='blue',  alpha=0.5, label=r'$T<150$ K'),
            Patch(facecolor='orange', alpha=0.5, label=r'$150<T<400$ K'),
            Patch(facecolor='red',   alpha=0.5, label=r'$T>400$ K'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.8)

    # τ_ir annotation
    ax.annotate(r'$\tau_{ir}=1$', xy=(2.5, 0.25), xytext=(2.5, 0.13),
                fontsize=20, color='purple', zorder=10,
                fontweight='bold', rotation=20)

    return crhov, C_Tem, m_cold_tot, m_warm_tot, m_hot_tot


# ── build vap_obs comparison figure ──────────────────────────────────────────
import matplotlib.gridspec as gridspec
fig2 = plt.figure(figsize=(18, 6))
gs = gridspec.GridSpec(1, 2, figure=fig2, width_ratios=[1, 1.15], wspace=0.08)
ax3 = fig2.add_subplot(gs[0])
ax4 = fig2.add_subplot(gs[1], sharey=ax3)
plt.setp(ax4.get_yticklabels(), visible=False)

crhov1,  C_Tem1, mc1, mw1, mh1 = plot_vap_obs(ax3, d1, show_legend=True)

ax4.set_xlabel(r'$R$ [AU]', fontsize=12)
crhov2,  C_Tem2, mc2, mw2, mh2 = plot_vap_obs(ax4, d2, show_legend=False)
ax4.set_ylabel('')

# ── three colorbars on the right (pad values from original plot.py) ──────────
cbarv = fig2.colorbar(crhov2, ax=ax4, orientation='vertical',
                       pad=-0.15, shrink=0.45, aspect=12, anchor=(0, 1))
cbarv.ax.set_ylabel(r'$\rho_{vap}$ [g cm$^{-3}$]', fontsize=12)
cbarv.set_ticks(logspace(-20, -10, 6))
cbarv.set_ticklabels([r'$10^{-20}$', r'$10^{-18}$', r'$10^{-16}$',
                       r'$10^{-14}$', r'$10^{-12}$', r'$10^{-10}$'], fontsize=10)

# cbar1 = fig2.colorbar(crho1_2, ax=ax4, orientation='vertical',
#                        pad=-0.15, shrink=0.3, aspect=12, anchor=(0, 0.5))
# cbar1.ax.set_ylabel(r'$\rho_{ice}/\rho_{gas}$', fontsize=12)
# cbar1.set_ticks([0.001, 0.05])
# cbar1.set_ticklabels(['0.001', '0.05'], fontsize=10)

cbarT = fig2.colorbar(C_Tem2, ax=ax4, orientation='vertical',
                       pad=0.02, shrink=0.45, aspect=12, anchor=(0, 0))
cbarT.ax.set_ylabel(r'$T$ [K]', fontsize=12)

fig2.savefig('./plots/compare_vap_obs.png', dpi=300, bbox_inches='tight')
print('Saved: ./plots/compare_vap_obs.png')

# ── Print vapor masses in North Sea units ────────────────────────────────────
# North Sea water mass: ~54,000 km³ × 1 g/cm³ ≈ 5.4 × 10^19 g
M_NS = 5.4e19  # g
M_Me = 3.86e21 # g mass of Mediterranean Sea
print(f"\n=== Vapor masses (North Sea water mass = {M_NS:.1e} g) ===")
print(f"single_lowa  (t={d1['simu_time']:.0f} yr):")
print(f"  cold (T<150K):  {mc1/M_NS:.3f} NS  ")
print(f"  warm (150-400K): {mw1/M_NS:.3f} NS ")
print(f"  hot  (T>400K):  {mh1/M_NS:.3f} NS  ")
print(f"low_alpha    (t={d2['simu_time']:.0f} yr):")
print(f"  cold (T<150K):  {mc2/M_NS:.3f} NS  ")
print(f"  warm (150-400K): {mw2/M_NS:.3f} NS ")
print(f"  hot  (T>400K):  {mh2/M_NS:.3f} NS  ")

# ══════════════════════════════════════════════════════════════════════════════
#  compare_2ddust:  DPS / DPR / DAS / DAR   (4 rows x 2 columns)
#  Column A: rho map (vapor RdPu + ice Blues + streamlines + snowline contour)
#  Column B: water (ice) mass fraction f_H2O
#  single-pop runs (DPS, DAS): upper-half style, z in [0, 0.15]
#  two-pop runs  (DPR, DAR):  mirrored two-hemisphere style, z in [-0.15, 0.15]
#  shared colour bars at the TOP of each column
# ══════════════════════════════════════════════════════════════════════════════

def _rho_levels_vap():
    return logspace(-14, -10, 15)

def _rho_levels_ice():
    return logspace(-14, log10(3e-11), 20)

def _wcomp_levels():
    return linspace(0.4, 0.99, 16)


def plot_2ddust_rho_panel(ax, d):
    """Left column: vapor (RdPu) + ice (Blues) density map with streamlines.

    two-pop: mirrored hemispheres (upper = pop1 'Pebbles', lower = pop0 'Dust')
    single-pop: upper-half only (no mirroring), z in [0, 0.15]
    Returns (c_ice, c_vap) mappables for the column colour bars.
    """
    is2 = (d['N_pop'] >= 2)
    xmin = maximum(0.5, d['rin']/d['L_norm'])
    xmax = minimum(3.0, d['rout']/d['L_norm'])

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.15, 0.15) if is2 else ax.set_ylim(0.0, 0.15)

    vap_mod = d['vap_rho_mod'] * d['UNIT_DEN']
    iid_lo = d['ice_ids'][0]                       # pop0
    iid_hi = d['ice_ids'][1] if is2 else None      # pop1 (two-pop only)
    ice_lo = d['ice_rho_mod'][iid_lo] * d['UNIT_DEN']
    ice_hi = d['ice_rho_mod'][iid_hi] * d['UNIT_DEN'] if is2 else None

    lv_vap = _rho_levels_vap()
    lv_ice = _rho_levels_ice()

    # ---- vapor: full disk (both halves for two-pop; single upper only) ----
    cvap = ax.contourf(d['x_xz_c'], d['y_xz_c'], vap_mod, levels=lv_vap,
                       norm=LogNorm(), cmap='RdPu', alpha=0.7, extend='both',
                       zorder=3, antialiased=True)
    if is2:
        ax.contourf(d['x_xz_c'], -d['y_xz_c'], vap_mod, levels=lv_vap,
                    norm=LogNorm(), cmap='RdPu', alpha=0.7, extend='both',
                    zorder=3, antialiased=True)

    # ---- ice ----
    # single-pop: one population over the panel
    # two-pop   : upper = large-pebble population, lower = small-dust population
    if not is2:
        cice = ax.contourf(d['x_xz_c'], d['y_xz_c'], ice_lo, levels=lv_ice,
                           norm=LogNorm(), cmap='Blues', alpha=1.0,
                           extend='both', antialiased=True, zorder=4)
    else:
        cice = ax.contourf(d['x_xz_c'], d['y_xz_c'], ice_hi, levels=lv_ice,
                           norm=LogNorm(), cmap='Blues', alpha=1.0,
                           extend='both', antialiased=True, zorder=4)
        ax.contourf(d['x_xz_c'], -d['y_xz_c'], ice_lo, levels=lv_ice,
                    norm=LogNorm(), cmap='Blues', alpha=1.0, extend='both',
                    antialiased=True, zorder=4)
        ax.axhline(0.0, c='k', lw=4., zorder=15)

    # ---- snowline contour: (rho_ice / rho_gas) = d2g_snow ----
    # single-pop: contour of the single population (Blues_r, as in plot.py)
    # two-pop   : upper = large-pebble pop (darkblue, lw=5), lower = small-dust pop
    if not is2:
        ax.contour(d['x_xz_c'], d['y_xz_c'],
                   d['ice_rho_xz'][iid_lo]/d['rho_xz'],
                   levels=[d['d2g_snow']], cmap='Blues_r', alpha=0.7,
                   linewidths=3.0, zorder=5)
    else:
        ax.contour(d['x_xz_c'], d['y_xz_c'],
                   d['ice_rho_xz'][iid_hi]/d['rho_xz'],
                   levels=[d['d2g_snow']], colors='darkblue', alpha=0.7,
                   linewidths=5.0, zorder=5)
        ax.contour(d['x_xz_c'], -d['y_xz_c'],
                   d['ice_rho_xz'][iid_lo]/d['rho_xz'],
                   levels=[d['d2g_snow']], colors='darkblue', alpha=0.7,
                   linewidths=5.0, zorder=4)

    # ---- scale heights ----
    if is2:
        ax.plot(d['xx_exp'], -d['yy0'], '--', c='k', lw=1, zorder=10)
        ax.plot(d['xx_exp'],  d['yy1'], '--', c='k', lw=1, zorder=10)
    else:
        ax.plot(d['xx_exp'], d['yy0'], '--', c='k', lw=1, zorder=10)
        ax.plot(d['xx_exp'], d['yy_g'], '-.', c='gray', lw=1, zorder=10)

    # ---- streamlines ----
    Xg = d['x1_exp_half']; Zg = d['x3_exp']
    n2 = d['normal2']
    # pop0 ice flux (blue); pop1 ice flux (cyan-blue); water vapor flux (pink)
    if is2:
        ax.streamplot(Xg, Zg, d['ice1_flx_x_xz']/n2, d['ice1_flx_z_xz']/n2,
                      linewidth=d['lw_flx_ice1'], arrowstyle='->',
                      density=1.0, broken_streamlines=True,
                      color='blue', zorder=4)
        ax.streamplot(Xg, Zg, d['water_flx_x_xz']/n2, d['water_flx_z_xz']/n2,
                      linewidth=d['lw_flx_water'], arrowstyle='->',
                      density=2.0, broken_streamlines=True,
                      color='#d6336c', zorder=4)
        z_neg = -d['x3_exp'][::-1]
        ax.streamplot(Xg, z_neg, d['ice_flx_x_xz'][::-1, :]/n2,
                      -d['ice_flx_z_xz'][::-1, :]/n2,
                      linewidth=d['lw_flx_ice'][::-1, :], arrowstyle='->',
                      density=1.0, broken_streamlines=True,
                      color='blue', zorder=4)
        ax.streamplot(Xg, z_neg, d['water_flx_x_xz'][::-1, :]/n2,
                      -d['water_flx_z_xz'][::-1, :]/n2,
                      linewidth=d['lw_flx_water'][::-1, :], arrowstyle='->',
                      density=2.0, broken_streamlines=True,
                      color='#d6336c', zorder=4)
        ax.text(0.05, 0.95, 'Pebbles', transform=ax.transAxes, fontsize=14,
                va='top', ha='left')
        ax.text(0.05, 0.05, 'Dust', transform=ax.transAxes, fontsize=14,
                va='bottom', ha='left')
    else:
        ax.streamplot(Xg, Zg, d['ice_flx_x_xz']/n2, d['ice_flx_z_xz']/n2,
                      linewidth=d['lw_flx_ice'], arrowstyle='->',
                      density=1.0, broken_streamlines=True,
                      color='blue', zorder=4)
        ax.streamplot(Xg, Zg, d['water_flx_x_xz']/n2, d['water_flx_z_xz']/n2,
                      linewidth=d['lw_flx_water'], arrowstyle='->',
                      density=1.0, broken_streamlines=True,
                      color='#d6336c', zorder=4)

    ax.set_xlim(xmin, xmax)
    return cice, cvap


def plot_2ddust_comp_panel(ax, d):
    """Right column: water (ice) mass fraction f_H2O = rho_ice/(rho_ice+rho_sil).

    two-pop: upper = pop1, lower = pop0 (0.5 contour in each half)
    single-pop: the single population (0.5 contour)
    Returns the f_H2O mappable for the column colour bar.
    """
    is2 = (d['N_pop'] >= 2)
    xmin = maximum(0.5, d['rin']/d['L_norm'])
    xmax = minimum(3.0, d['rout']/d['L_norm'])

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.15, 0.15) if is2 else ax.set_ylim(0.0, 0.15)

    lv = _wcomp_levels()
    wc0 = d['watercomp'][0]
    wc1 = d['watercomp'][1] if is2 else None

    if not is2:
        ccomp = ax.contourf(d['x_xz_c'], d['y_xz_c'], wc0, levels=lv,
                            cmap='Blues', alpha=0.8, extend='both')
        ax.contour(d['x_xz_c'], d['y_xz_c'], wc0, levels=[0.5],
                   colors='k', linewidths=2.0)
    else:
        ccomp = ax.contourf(d['x_xz_c'], d['y_xz_c'], wc1, levels=lv,
                            cmap='Blues', alpha=0.8, extend='both')
        ax.contourf(d['x_xz_c'], -d['y_xz_c'], wc0, levels=lv,
                    cmap='Blues', alpha=0.8, extend='both')
        ax.contour(d['x_xz_c'],  d['y_xz_c'], wc1, levels=[0.5],
                   colors='k', linewidths=2.0)
        ax.contour(d['x_xz_c'], -d['y_xz_c'], wc0, levels=[0.5],
                   colors='k', linewidths=2.0)
        ax.axhline(0.0, c='k', lw=4., zorder=15)

    ax.set_xlim(xmin, xmax)
    return ccomp


# ── build the 4x2 comparison figure ─────────────────────────────────────────
runs_2ddust = [('DPS', 'passive, single-pop'),
               ('DPR', 'passive, two-pop'),
               ('DAS', 'active,  single-pop'),
               ('DAR', 'active,  two-pop')]

figC = plt.figure(figsize=(15, 17))
# single-pop rows (DPS, DAS) show only z in [0, 0.15] (half of the two-pop
# z in [-0.15, 0.15]) — give them half the row height so the z scale matches
gsC = gridspec.GridSpec(4, 2, figure=figC, hspace=0.30, wspace=0.10,
                        height_ratios=[1, 2, 1, 2])
axC = [[figC.add_subplot(gsC[i, j]) for j in range(2)] for i in range(4)]

# data_530 was loaded once above — reuse it (no second read of the run files)
run_data = [(name, tag, data_530[name]) for (name, tag) in runs_2ddust]

c_ice_reps, c_vap_reps, c_comp_reps = [], [], []
for row, (name, tag, dd) in enumerate(run_data):
    is2 = dd['N_pop'] >= 2

    # left: rho map
    cice, cvap = plot_2ddust_rho_panel(axC[row][0], dd)
    axC[row][0].set_ylabel(r'$z$ [AU]', fontsize=12)
    axC[row][0].text(0.02, 1.02, name, transform=axC[row][0].transAxes,
                     fontsize=15, fontweight='bold', va='bottom', ha='left')

    # right: water-comp map
    ccomp = plot_2ddust_comp_panel(axC[row][1], dd)
    if is2:
        axC[row][1].text(0.05, 0.95, 'Pebbles', transform=axC[row][1].transAxes,
                         fontsize=14, va='top', ha='left')
        axC[row][1].text(0.05, 0.05, 'Dust', transform=axC[row][1].transAxes,
                         fontsize=14, va='bottom', ha='left')

    # same 0.05 step for every row, so the half-height single-pop rows keep
    # identical spacing to the two-pop rows
    tickz = linspace(-0.15, 0.15, 7) if is2 else linspace(0.0, 0.15, 4)
    axC[row][0].set_yticks(tickz)
    axC[row][1].set_yticks(tickz)

    c_ice_reps.append(cice); c_vap_reps.append(cvap); c_comp_reps.append(ccomp)

for i in range(4):
    if i < 3:
        axC[i][0].set_xticklabels([])
        axC[i][1].set_xticklabels([])
    else:
        axC[i][0].set_xlabel(r'$R$ [AU]', fontsize=12)
        axC[i][1].set_xlabel(r'$R$ [AU]', fontsize=12)

# column titles
axC[0][0].set_title(r'$\rho_{\rm ice}$ / $\rho_{\rm vap}$', fontsize=14)
axC[0][1].set_title(r'$f_{\rm H_2O}$', fontsize=14)

# ── shared colour bars at the top of each column ─────────────────────────────
caxI = figC.add_axes([0.16, 0.945, 0.18, 0.015])
cbar_ice = figC.colorbar(c_ice_reps[0], cax=caxI, orientation='horizontal')
cbar_ice.set_ticks([1e-13, 1e-12, 1e-11])
cbar_ice.set_ticklabels([r'$10^{-13}$', r'$10^{-12}$', r'$10^{-11}$'], fontsize=9)
cbar_ice.ax.set_title(r'$\rho_{\rm ice}$ [g cm$^{-3}$]', fontsize=11)

caxV = figC.add_axes([0.42, 0.945, 0.18, 0.015])
cbar_vap = figC.colorbar(c_vap_reps[0], cax=caxV, orientation='horizontal')
cbar_vap.set_ticks([1e-13, 1e-12, 1e-11])
cbar_vap.set_ticklabels([r'$10^{-13}$', r'$10^{-12}$', r'$10^{-11}$'], fontsize=9)
cbar_vap.ax.set_title(r'$\rho_{\rm vap}$ [g cm$^{-3}$]', fontsize=11)

caxC = figC.add_axes([0.66, 0.945, 0.16, 0.015])
cbar_comp = figC.colorbar(c_comp_reps[0], cax=caxC, orientation='horizontal')
cbar_comp.set_ticks([0.4, 0.5, 0.7, 0.9])
cbar_comp.set_ticklabels([r'$0.4$', r'$0.5$', r'$0.7$', r'$0.9$'], fontsize=9)
cbar_comp.ax.set_title(r'$f_{\rm H_2O}$', fontsize=11)
cbar_comp.ax.axvline(0.5, color='k', linewidth=2)

figC.savefig('./plots/compare_2ddust.png', dpi=300, bbox_inches='tight')
print('Saved: ./plots/compare_2ddust.png')
plt.close(figC)

# ══════════════════════════════════════════════════════════════════════════════
#  fig_snow_compare: 2 rows x 4 columns (columns = DPS / DPR / DAS / DAR)
#  upper row : column densities  Sigma_gas, Sigma_ice, Sigma_sil, Sigma_vap
#  lower row : midplane solid-to-gas (black) and vapor-to-gas (red) ratios
#  (quantities & colours follow plot.py fig_snow_2d)
# ══════════════════════════════════════════════════════════════════════════════

def _sigma_arrays(d):
    """Column densities [g cm^-2] for gas, ice, silicate, vapor.

    xz arrays are indexed [rad, theta]; the slice covers the upper
    hemisphere, and at fixed radius the cell z-extent is
    dz = R * |cos(thf[j+1]) - cos(thf[j])|; the full column is
    2 x the integral (mirror symmetry).
    """
    rho_xz = d['rho_xz']                                             # (n_r, n_t)
    dcos = fabs(cos(d['theta_f'][1:]) - cos(d['theta_f'][:-1]))      # (n_t,)
    dz_cm = d['rad'][:, None] * dcos[None, :] * AU                   # (n_r, n_t)

    def col(rhov):
        return 2.0 * sum(rhov * dz_cm, axis=1) * d['UNIT_DEN']

    sig_gas = col(rho_xz)
    sig_ice = sum([col(d['ice_rho_xz'][iid]) for iid in d['ice_ids']], axis=0)
    sig_sil = sum([col(d['sil_rho_xz'][sid]) for sid in d['sil_rho_xz']], axis=0)
    sig_vap = col(d['vap_rho_xz'])
    return sig_gas, sig_ice, sig_sil, sig_vap


def _midplane_ratios(d):
    """Midplane (theta_max row) solid/gas and vapor/gas density ratios."""
    jm = int(argmax(d['theta']))
    rho_m = d['rho_xz'][:, jm]
    solid_m = sum([d['ice_rho_xz'][iid][:, jm] for iid in d['ice_ids']], axis=0)
    solid_m = solid_m + sum([d['sil_rho_xz'][sid][:, jm]
                             for sid in d['sil_rho_xz']], axis=0)
    denom = where(rho_m > 0.0, rho_m, 1.0)
    d2g = where(rho_m > 0.0, solid_m / denom, 0.0)
    v2g = where(rho_m > 0.0, d['vap_rho_xz'][:, jm] / denom, 0.0)
    return d2g, v2g


figS, axS = plt.subplots(2, 4, figsize=(18, 9), sharex=True,
                         gridspec_kw={'hspace': 0.10, 'wspace': 0.20})

for k, (name, tag, dd) in enumerate(run_data):
    rr = dd['rad']                                   # AU, native grid

    # ---- upper: surface densities (reference colours from plot.py colD) ----
    sig_gas, sig_ice, sig_sil, sig_vap = _sigma_arrays(dd)
    axS[0][k].plot(rr, sig_gas, c='black',      lw=2.5, label=r'$\Sigma_{\rm gas}$')
    axS[0][k].plot(rr, sig_ice, c='tab:blue',   lw=2.5, label=r'$\Sigma_{\rm ice}$')
    axS[0][k].plot(rr, sig_sil, c='tab:orange', lw=2.5, label=r'$\Sigma_{\rm sil}$')
    axS[0][k].plot(rr, sig_vap, c='tab:purple', lw=2.5, label=r'$\Sigma_{\rm vap}$')
    axS[0][k].set_yscale('log')
    axS[0][k].set_ylim(1e-5, 1e4)      # shared across the 4 columns
    axS[0][k].set_xlim(0.5, 3.0)
    axS[0][k].set_title(f'{name}   ($t={dd["simu_time"]:.0f}$ yr)',
                        fontsize=12)

    # ---- lower: midplane ratios ----
    d2g, v2g = _midplane_ratios(dd)
    axS[1][k].plot(rr, d2g, c='k',   lw=3.0, label=r'$(s/g)_{\rm mid}$')
    axS[1][k].plot(rr, v2g, c='red', lw=3.0, label=r'$(v/g)_{\rm mid}$')
    axS[1][k].set_yscale('log')
    axS[1][k].set_ylim(1e-6, 0.2)      # shared across the 4 columns
    axS[1][k].set_xlim(0.5, 3.0)
    axS[1][k].set_xlabel(r'$R$ [AU]', fontsize=12)

    if k == 0:
        axS[0][k].set_ylabel(r'$\Sigma$ [g cm$^{-2}$]', fontsize=12)
        axS[1][k].set_ylabel('midplane ratio', fontsize=12)
        axS[0][k].legend(fontsize=9, loc='upper right', framealpha=0.9)
        axS[1][k].legend(fontsize=9, loc='lower left', framealpha=0.9,
                         ncol=2)
        axS[0][k].annotate('(a)', xy=(0.02, 0.95), xycoords='axes fraction',
                           fontsize=14, va='top')
        axS[1][k].annotate('(b)', xy=(0.02, 0.95), xycoords='axes fraction',
                           fontsize=14, va='top')

figS.savefig('./plots/fig_snow_compare.png', dpi=300, bbox_inches='tight')
print('Saved: ./plots/fig_snow_compare.png')
plt.close(figS)
