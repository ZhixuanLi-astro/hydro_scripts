import sys
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
# import dynamo as dyn
from scipy.integrate import odeint,ode
from scipy import optimize
import astropy.constants as cons

# positive direction: VR(outwards), Vphi(counterclockwise)
def pol2car(r,theta): # transform polar coordinate to cartesian
    lenx = len(r)
    leny = len(theta)
    x1 = zeros((lenx,leny))
    x2 = zeros((lenx,leny))
    for i in range(lenx):
        for j in range(leny):
            x1[i,j] = r[i]*cos(theta[j])
            x2[i,j] = r[i]*sin(theta[j])
    return x1,x2

def car2pol(x1,x2):
    lenx = len(x1)
    leny = len(x2)
    r = zeros((leny,lenx))
    theta = zeros((leny,lenx))
    
    for j in range(leny):
        for i in range(lenx):
            r[j,i] = sqrt(x1[i]**2+x2[j]**2)
            theta[j,i] = angle(x1[i] + 1j*x2[j])
            if(theta[j,i]<0):
                theta[j,i]+=2*pi
#             print(x1[i],x2[j])
    return r,theta

def v_Intplp2c_xy(x1_origin,x2_origin,x1_expect,x2_expect,vx,vy):
    #needed gird in Cartesian, transformed to polar first:
    lenx = len(x1_expect)
    leny = len(x2_expect)
    r = zeros((leny,lenx))
    theta = zeros((leny,lenx))
    array_expect = zeros((lenx*leny,2))
    
    for j in range(leny):
        for i in range(lenx):
            # transfromed to polar coordinate
            r[j,i] = sqrt(x1_expect[i]**2+x2_expect[j]**2)
            theta[j,i] = angle(x1_expect[i] + 1j*x2_expect[j])
            if(theta[j,i]<0):
                theta[j,i]+=2*pi
            # transfromed to 1-D array to interpolate 
            array_expect[lenx*j+i,0] = r[j,i]
            array_expect[lenx*j+i,1] = theta[j,i]
            
    #do interpolation in Polar coordinate
    fvx = interpolate.RegularGridInterpolator((x1_origin,x2_origin),vx,fill_value=None,bounds_error = False)
    fvy = interpolate.RegularGridInterpolator((x1_origin,x2_origin),vy,fill_value=None,bounds_error = False)
    vx_polar = fvx(array_expect).reshape((leny,lenx))
    vy_polar = fvy(array_expect).reshape((leny,lenx))
    
    #cast to cartesian
    vx_cartesian = vx_polar*cos(theta)-vy_polar*sin(theta)
    vy_cartesian = vx_polar*sin(theta)+vy_polar*cos(theta)
    
    return vx_cartesian,vy_cartesian

def v_Intplp2c_xz(x1_origin,x2_origin,x1_expect,x2_expect,vx,vy):
    #needed gird in Cartesian, transformed to polar first:
    lenx = len(x1_expect)
    leny = len(x2_expect)
    r = zeros((leny,lenx))
    theta = zeros((leny,lenx))
    array_expect = zeros((lenx*leny,2))
    
    for j in range(leny):
        for i in range(lenx):
            # transfromed to polar coordinate
            r[j,i] = sqrt(x1_expect[i]**2+x2_expect[j]**2)
            theta[j,i] = pi/2 - angle(x1_expect[i] + 1j*x2_expect[j])
            # transfromed to 1-D array to interpolate 
            array_expect[lenx*j+i,0] = r[j,i]
            array_expect[lenx*j+i,1] = theta[j,i]
            
    #do interpolation in Polar coordinate
    fvx = interpolate.RegularGridInterpolator((x1_origin,x2_origin),vx,fill_value=None,bounds_error = False)
    fvy = interpolate.RegularGridInterpolator((x1_origin,x2_origin),vy,fill_value=None,bounds_error = False)
    vx_polar = fvx(array_expect).reshape((leny,lenx))
    vy_polar = fvy(array_expect).reshape((leny,lenx))
    
    #cast to cartesian
    vx_cartesian = vx_polar*sin(theta)+vy_polar*cos(theta)
    vy_cartesian = vx_polar*cos(theta)-vy_polar*sin(theta)
    
    return vx_cartesian,vy_cartesian

def v_Intpl_Sph2car(x1_origin,x2_origin,x3_origin,x1_expect,x2_expect,x3_expect,v1,v2,v3):
    #needed gird in Cartesian, transformed to spherical first:
    lenx = len(x1_expect)
    leny = len(x2_expect)
    lenz = len(x3_expect)
    r = zeros((lenz,leny,lenx))
    theta = zeros((lenz,leny,lenx))
    phi = zeros((lenz,leny,lenx))
    array_expect = zeros((lenx*leny*lenz,3))
    
    for k in range(lenz):
        for j in range(leny):
            for i in range(lenx):
                # transfromed to polar coordinate
                r[k,j,i] = sqrt(x1_expect[i]**2+x2_expect[j]**2+x3_expect[k]**2)
                theta[k,j,i] = arccos(x3_expect[k]/r[k,j,i])
                phi[k,j,i] = angle(x1_expect[i] + 1j*x2_expect[j])
                if(phi[k,j,i]<0):
                    phi[k,j,i]+=2*pi
                # transfromed to 1-D array to interpolate 
                array_expect[(leny*lenx)*k + lenx*j+i,0] = r[k,j,i]
                array_expect[(leny*lenx)*k + lenx*j+i,1] = theta[k,j,i]
                array_expect[(leny*lenx)*k + lenx*j+i,2] = phi[k,j,i]
            
    #do interpolation in Polar coordinate
    fvx = interpolate.RegularGridInterpolator((x1_origin,x2_origin,x3_origin),v1,fill_value=None,bounds_error = False)
    fvy = interpolate.RegularGridInterpolator((x1_origin,x2_origin,x3_origin),v2,fill_value=None,bounds_error = False)
    fvz = interpolate.RegularGridInterpolator((x1_origin,x2_origin,x3_origin),v3,fill_value=None,bounds_error = False)
    vx_sph = fvx(array_expect).reshape((lenz,leny,lenx))
    vy_sph = fvy(array_expect).reshape((lenz,leny,lenx))
    vz_sph = fvz(array_expect).reshape((lenz,leny,lenx))
    
    #cast to cartesian
    vx_cartesian = sin(theta)*cos(phi)*vx_sph + cos(theta)*cos(phi)*vy_sph - sin(phi)*vz_sph
    vy_cartesian = sin(theta)*sin(phi)*vx_sph + cos(theta)*sin(phi)*vy_sph + cos(phi)*vz_sph
    vz_cartesian = cos(theta)*vx_sph - sin(theta)*vy_sph
    
    return vx_cartesian,vy_cartesian,vz_cartesian

def scaler_Intpl_Sph2car(x1_origin,x2_origin,x3_origin,x1_expect,x2_expect,x3_expect,rho):
    #needed gird in Cartesian, transformed to spherical first:
    lenx = len(x1_expect)
    leny = len(x2_expect)
    lenz = len(x3_expect)
    r = zeros((lenz,leny,lenx))
    theta = zeros((lenz,leny,lenx))
    phi = zeros((lenz,leny,lenx))
    array_expect = zeros((lenx*leny*lenz,3))
    
    for k in range(lenz):
        for j in range(leny):
            for i in range(lenx):
                # transfromed to polar coordinate
                r[k,j,i] = sqrt(x1_expect[i]**2+x2_expect[j]**2+x3_expect[k]**2)
                theta[k,j,i] = arccos(x3_expect[k]/r[k,j,i])
                phi[k,j,i] = angle(x1_expect[i] + 1j*x2_expect[j])
                if(phi[k,j,i]<0):
                    phi[k,j,i]+=2*pi
                # transfromed to 1-D array to interpolate 
                array_expect[(leny*lenx)*k + lenx*j+i,0] = r[k,j,i]
                array_expect[(leny*lenx)*k + lenx*j+i,1] = theta[k,j,i]
                array_expect[(leny*lenx)*k + lenx*j+i,2] = phi[k,j,i]
            
    #do interpolation in Polar coordinate
    f0 = interpolate.RegularGridInterpolator((x1_origin,x2_origin,x3_origin),rho,fill_value=None,bounds_error = False)
    rho_car = f0(array_expect).reshape((lenz,leny,lenx))
    
    return rho_car


# Ref:https://en.wikipedia.org/wiki/Finite_difference_coefficient

def dfdx_2pts(x,y):
    num = len(x)
    dx = abs(x[1]-x[0])
    dfdx = zeros(num)
    for i in range(num):
        if(i>=1):
            dfdx[i] = (y[i]-y[i-1])/dx
        else:
            dfdx[i] = (y[i+1]-y[i])/dx
    return dfdx

def dfdx_5pts(x,y):
    num = len(x)
    dx = abs(x[1]-x[0])
    dfdx = zeros(num)
    for i in range(num):
        if((i>=2) and ((num-i)>=3)):
            dfdx[i] = (-y[i+2]+8*y[i+1]-8*y[i-1]+y[i-2])/(12*dx)
        elif(i<2):
            dfdx[i] = (-25/12*y[i]+4*y[i+1]-3*y[i+2]+4/3*y[i+3]-1/4*y[i+4])/dx
        else:
            dfdx[i] = -(-25/12*y[i]+4*y[i-1]-3*y[i-2]+4/3*y[i-3]-1/4*y[i-4])/dx
    return dfdx

def dfdx_7pts(x,y):
    num = len(x)
    dx = abs(x[1]-x[0])
    dfdx = zeros(num)
    for i in range(num):
        if((i>=3) and ((num-i)>=4)):
            dfdx[i] = (-1/60*y[i-3]+3/20*y[i-2]-3/4*y[i-1]+3/4*y[i+1]-3/20*y[i+2]+1/60*y[i+3])/dx
        elif(i<3):
            dfdx[i] = (-49/20*y[i]+6*y[i+1]-15/2*y[i+2]+20/3*y[i+3]-15/4*y[i+4]+6/5*y[i+5]-1/6*y[i+6])/dx
        elif(num-i<4):
            dfdx[i] = -(-49/20*y[i]+6*y[i-1]-15/2*y[i-2]+20/3*y[i-3]-15/4*y[i-4]+6/5*y[i-5]-1/6*y[i-6])/dx
    return dfdx


def curl_in_polar_rlog(r,theta,vr,vtheta):
    numr= len(r)
    numtheta = len(theta)
    logr = log10(r)
    rvtheta = zeros((numr,numtheta))
    dfdlogr = zeros((numr,numtheta))
    dthetadvr = zeros((numr,numtheta))
    
    for i in range(numr):
        rvtheta[i,:] = r[i]*vtheta[i,:]
        dthetadvr[i,:] = dfdx_7pts(theta,vr[i,:]) /(r[i])
        
    for j in range(numtheta):
        dfdlogr[:,j] = dfdx_7pts(logr,rvtheta[:,j])
    for i in range(numr):
        dfdlogr[i,:] /=r[i]**2
    return 1/log(10)*dfdlogr -dthetadvr


def search_separatrix2(diff,saddle,xs,r_dev,theta):
    Rd = 0
    index = 0
    
    dt = 0.1
    t_total =40
    f_intgl = ode(diff).set_integrator('dopri5',first_step = 1e-4)
    while(Rd<xs and index < 100):
        xy_dev = [r_dev*cos(theta[index]),r_dev*sin(theta[index])]
        f_intgl.set_initial_value(array(saddle)+array(xy_dev))
        f_intgl.t = 0
        while (f_intgl.successful() and (f_intgl.t < t_total) and abs(f_intgl.y[0])<xs and abs(f_intgl.y[1])<ys):
            f_intgl.integrate(dt+f_intgl.t)
        pos = f_intgl.y
        Rd = sqrt(pos[0]**2+pos[1]**2)
        index +=1
        
    pos0 = pos
    d = 0
    if(index == 1):
        while(Rd>xs and d<2*ys  and index <100):
            xy_dev = [r_dev*cos(theta[index]),r_dev*sin(theta[index])]
            f_intgl.set_initial_value(array(saddle)+array(xy_dev))
            f_intgl.t = 0
            while (f_intgl.successful() and (f_intgl.t < t_total) and abs(f_intgl.y[0])<xs and abs(f_intgl.y[1])<ys):
                f_intgl.integrate(dt+f_intgl.t)
            pos = f_intgl.y
            Rd = sqrt(pos[0]**2+pos[1]**2)
            index +=1
            d = sqrt(sum((pos-pos0)**2))
            print(index,pos)
            pos0 = pos
            
#             print(d)
    return index-1


def search_separatrix(diff,saddle,r_dev,theta):
    d_max = 2e-2
    dt = 0.1
    d = 0
    dir_change = False
    index = 0
    f_intgl = ode(diff).set_integrator('dopri5',first_step = 1e-4)
    # first loop to define dx0
    xy_dev = [r_dev*cos(theta[index]),r_dev*sin(theta[index])]
    f_intgl.set_initial_value(array(saddle)+array(xy_dev))
    f_intgl.t = 0
    
    while (f_intgl.successful() and (d < d_max)):
        f_intgl.integrate(dt+f_intgl.t)
        d = sqrt(sum((f_intgl.y-saddle)**2))
    dx0 = f_intgl.y-saddle
    index +=1
    print(dx0)
    
    # the following loops
    while(dir_change == False):
        d = 0
        xy_dev = [r_dev*cos(theta[index]),r_dev*sin(theta[index])]
        f_intgl.set_initial_value(array(saddle)+array(xy_dev))
        f_intgl.t = 0
        while (f_intgl.successful() and (d < d_max)):
            f_intgl.integrate(dt+f_intgl.t)
            d = sqrt(sum((f_intgl.y-saddle)**2))
        dx = f_intgl.y-saddle
        print(index,dx)
        if(dx[0]*dx0[0]<0 or dx[1]*dx0[1]<0):
            dir_change = True
        index += 1
    return index-1


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

def face_f_1_log(x1min,x1max,cell_width_ratio,num_face):
    x = linspace(0,1,num_face)
    w = log10(x1min) + x*(log10(x1max)-log10(x1min))
    tmp = 10**w
    return tmp

#some functions for powerlaw reconstruction 
def mDiv(mmin, mmax, binnum, mode='small'):
    mDivs = zeros(binnum-1) 
    if mode == 'small':
        mu = mmax
        for i in range(binnum-2, -1,-1):
            mDivs[i] = sqrt(mu*mmin) 
            mu = mDivs[i]
    elif mode == 'log':
        mDivs = logspace(log10(mmin), log10(mmax), binnum+1)[1:-1]

    return mDivs

def ff_inter_new (vars, m_bounds, mch, rho_sim):
    """
    vars = [the_final_c, pwl_0, pwl_1, ...]
    """

    cfinal, pwl = vars[0], vars[1:]
    rho_dis = ff_integ_many (cfinal, pwl, m_bounds, mch)
    fsol = rho_dis - rho_sim

    return fsol

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

def ff_broken (pwl, prec, ml, mu):
    """integrate f(m)=prec*m**-pwl from ml to mu"""
    return prec/(2-pwl)*(mu**(2-pwl)-ml**(2-pwl))

def get_relaxed_state (rhoi, rhos, m_bounds):
    """
    To Get the relaxed state densities
    rhoi: a list of ice densities in each bin 
    rhos: a list of silicate densities in each bin 
    m_bounds: [mmin, mDiv1, mDiv2, ..., mmax]
    """
    M1re = sum(rhos + rhoi)
    mmax = m_bounds[-1]
    mmin = m_bounds[0]
    c_relax = M1re/(6*(mmax**(1/6)- mmin**(1/6)))
    M1re_bins = []
    M2re_bins = []
    for j in range(len(m_bounds)-1):
        M1re_bins.append(ff_broken(11/6, c_relax, m_bounds[j], m_bounds[j+1]))
        M2re_bins.append(ff_broken(5/6, c_relax, m_bounds[j], m_bounds[j+1]))

    return M1re_bins, M2re_bins
    
def read_athinput(filename):
    config = {}
    current_section = None

    with open(filename, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # 去除行内注释
            comment_pos = line.find('#')
            if comment_pos != -1:
                line = line[:comment_pos].strip()
                if not line:
                    continue

            # 段落开始
            if line.startswith('<') and line.endswith('>'):
                section_name = line[1:-1].strip()
                current_section = section_name
                config[current_section] = {}
                continue

            # 键值对
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # 尝试将 value 转换为 float
                try:
                    value = float(value)
                except ValueError:
                    # 保持原字符串（包括 true/false）
                    pass

                if current_section is not None:
                    config[current_section][key] = value
                else:
                    # 若段落外出现键值对，可忽略或警告
                    pass

    return config


# import matplotlib.gridspec as gs
# fig = plt.figure(figsize=(18, 12), constrained_layout=True)
# grid = gs.GridSpec(2, 2, figure=fig, hspace=0.0)
# axs = empty((2, 2), dtype=object) 
# axs[0, 0] = fig.add_subplot(grid[0, 0])
# axs[0, 1] = fig.add_subplot(grid[0, 1])
# axs[1, 0] = fig.add_subplot(grid[1, 0])
# axs[1, 1] = fig.add_subplot(grid[1, 1])
#
# xin = 0.6 
# xout = 2.5 
# yin = -0.1 
# yout = 0.1 
# for ax in axs.flatten():
#     ax.set_xlim(xin, xout)
#     ax.set_ylim(yin, yout)
#
# axs[0,1].set_title("time: {:.2f} yr".format(simu_time*UNIT_T/YR),loc= 'right', y=1.05)
#
# axs[0,0].set_ylabel(r'$\Sigma$ [g/cm$^2$]', fontsize = 12)
#
# # axs[0,0].set_yscale('log')
# axs[0,0].set_ylim(1e-2, 200)
# # sax[0].plot(xx_exp,(sigma_gas-sigma_vap)*0.4, color = 'k', alpha = 1.0, label = '$ f_{\mathrm{i/g}} \Sigma_{\mathrm{xy}}$')
# # shere the 0.4 is from the 0.8/2, in which 0.8 is the dust-to-gas flux ratio, so the vapor should be the half of it
# axs[0,0].plot(xx_exp,(sigma_gas)*0.4, color = 'k', linestyle='-', alpha = 1.0, label = 'gas')
# axs[0,0].plot(xx_exp,sqrt(2*pi)*0.4*(xx_exp/3)**(-1)*UNIT_SIGMA,color = 'grey',linewidth = 5, alpha = 0.5)
# ice_style = [('si', 'ice {}'), ('li', 'ice {}')]
# sil_style = [('ss', 'silicate {}'), ('ls', 'silicate {}')]
# for p in range(len(sigma_ice_by_pop)):
#     i_style = p if p < len(ice_style) else len(ice_style)-1
#     key = ice_style[i_style][0]
#     axs[0,0].plot(xx_exp, sigma_ice_by_pop[p], c = colD[key], lw = lwD[key], label = ice_style[i_style][1].format(p))
# for p in range(len(sigma_sil_by_pop)):
#     i_style = p if p < len(sil_style) else len(sil_style)-1
#     key = sil_style[i_style][0]
#     axs[0,0].plot(xx_exp, sigma_sil_by_pop[p], c = colD[key], lw = lwD[key], label = sil_style[i_style][1].format(p))
# axs[0,0].plot(xx_exp, sigma_vap , c = colD['va'], lw = lwD['va'], label = 'vapor') 
# axs[0,0].legend(handles=legend_handles, loc='upper right', ncol=3, frameon=False, fontsize=15)
# # density 
# ticks = logspace(-8, 3, 5)
# axs[1,0].set_ylabel(r'$z$ [AU]', fontsize = 12)
# legends = [Line2D([0], [0], color='darkblue', lw=6, alpha=0.7,label=r'$\rho_{ice}/\rho_{g} = 10^{-3}$'),
#            Line2D([0], [0], color='k', ls = '--', lw=1, label=r'$H_{peb}$')]
# axs[1,0].legend(handles=legends, loc='upper right',fontsize = 15, framealpha = 0.6)
# # the vapor
# ax0 =  axs[1,0].contourf(x_xz_c,y_xz_c,dust_5_rho_mod*UNIT_DEN,levels = logspace(-14, -10,15), norm = LogNorm(), cmap = 'RdPu', alpha = 0.7, extend = 'both',zorder=3, antialiased = True)
# axs[1,0].contourf(x_xz_c,-y_xz_c,dust_5_rho_mod*UNIT_DEN,levels = logspace(-14, -10,15), norm = LogNorm(), cmap = 'RdPu', alpha = 0.7, extend = 'both',zorder=3, antialiased = True)
#
# crho1= axs[1,0].contourf(x_xz_c,y_xz_c,dust_3_rho_mod*UNIT_DEN,levels = logspace(-14, log10(3e-11),20), norm = LogNorm(), cmap = 'Blues', alpha = 1.0, extend = 'both', antialiased = True,zorder=4)
# axs[1,0].contourf(x_xz_c,-y_xz_c,dust_1_rho_mod*UNIT_DEN,levels = logspace(-14,log10(3e-11),20), norm = LogNorm(), cmap = 'Blues', alpha = 1, extend = 'both', antialiased=True, zorder=4)
#
# axs[1,0].axhline(0.0, c= 'k', ls='-',linewidth = 4., zorder=15)
# # axs[1,0].contour(x_xz_c,y_xz_c,tau_ir,levels = array([0.5,1.0]), colors = 'black', linestyles = 'dotted', zorder = 20)
# # axs[1,0].contour(x_xz_c,-y_xz_c,tau_ir,levels = array([0.5,1.0]), colors = 'black', linestyles = 'dotted', zorder = 20)
# axs[1,0].plot(xx_exp, -yy0, '--', c='k', lw=1, zorder=10)
# # axs[1,0].plot(xx_exp, yy_g, '-', c='r', lw=1, zorder=10)
# axs[1,0].plot(xx_exp, yy1, '--', c='k', lw=1, zorder=10)
# #zxl: 0his we change to the sum of the ice in different populations.
# ice_rho_xz_tot = dust_1_rho_xz + dust_3_rho_xz
# axs[1,0].contour(x_xz_c,y_xz_c, dust_3_rho_xz/rho_xz,levels = [d2g_snow], colors='darkblue', alpha = 0.7, linewidths = 5.0, zorder = 5)
# axs[1,0].contour(x_xz_c,-y_xz_c, dust_1_rho_xz/rho_xz,levels = [d2g_snow],colors='darkblue', alpha = 0.7, linewidths = 5.0, zorder = 4)
# #label the panels in the lower left corner
# axs[1,0].text(0.05, 0.95, 'pebble', transform=axs[1,0].transAxes, fontsize=18, va='top', ha='left')
# axs[1,0].text(0.05, 0.05, 'dust', transform=axs[1,0].transAxes, fontsize=18, va='bottom', ha='left')
# axs[1,0].set_xlabel(r'$R$ [AU]', fontsize = 12)
# axs[1,0].plot([0.8, 0.8, 1.3, 1.3, 0.8], [-0.04, 0.04, 0.04, -0.04, -0.04], color='r', lw=3, ls='-', zorder=20)
#
# # axs[1,0].streamplot(x1_exp_half,x3_exp, 
# #                     flx_x_xz/normal2, flx_z_xz/normal2, 
# #                     linewidth = 2, arrowstyle = '->', 
# #                     density = 1.0, broken_streamlines = True, 
# #                     color ='orange',zorder=4,arrowsize = 1.5)
# axs[1,0].streamplot(x1_exp_half,x3_exp, 
#                     ice1_flx_x_xz/normal2, ice1_flx_z_xz/normal2, 
#                     linewidth = lw_flx_ice1, arrowstyle = '->',
#                     density = 2.5, broken_streamlines = True, 
#                     color ='blue',zorder=4)
# axs[1,0].streamplot(x1_exp_half,x3_exp, 
#                     water_flx_x_xz/normal2, water_flx_z_xz/normal2,
#                     linewidth = lw_flx_water, arrowstyle = '->', 
#                     density = 3.0, broken_streamlines = True, color ='#d6336c',zorder=4)
#
# axs[1,0].streamplot(x1_exp_half,z_neg, 
#                     ice_flx_x_xz[::-1,:]/normal2,
#                     - ice_flx_z_xz[::-1,:]/normal2,
#                     linewidth = lw_flx_ice[::-1,:], arrowstyle = '->', density = 2.0, broken_streamlines = True, color ='blue',zorder=4)
# axs[1,0].streamplot(x1_exp_half,z_neg, 
#                     water_flx_x_xz[::-1,:]/normal2, 
#                     - water_flx_z_xz[::-1,:]/normal2,
#                     linewidth = lw_flx_water[::-1,:], arrowstyle = '->', 
#                     density = 3.0, broken_streamlines = True, color ='#d6336c',zorder=4)
# # axs[1,0].streamplot(x1_exp_half,z_neg, 
# #                     flx_x_xz[::-1,:]/normal2, 
# #                     - flx_z_xz[::-1,:]/normal2,
# #                     linewidth = 1.5, arrowstyle = '->', density = 1.0, broken_streamlines = True, 
# #                     color ='orange',zorder=4)
#
# #get the advection flux of the ice and water vapor 
# # flx_ice_adv_x = dust_1_rho*dust_1_vx1*dS_R*UNIT_Fm
# # flx_ice_adv_z = dust_1_rho*dust_1_vx2*dS_theta*UNIT_Fm
# #
# # flx_ice1_adv_x = dust_3_rho*dust_3_vx1*dS_R*UNIT_Fm 
# # flx_ice1_adv_z = dust_3_rho*dust_3_vx2*dS_theta*UNIT_Fm 
# #
# # ice_flx_adv_x, ice_flx_adv_y, ice_flx_adv_z = v_Intpl_Sph2car(rad,theta,phi,x1_exp_half,slice_exp,x3_exp,(flx_ice_adv_x).T,(flx_ice_adv_z).T, (flx_ice_adv_z).T * 0.0)
# # ice_flx_adv_x_xz = ice_flx_adv_x[:,0,:]
# # ice_flx_adv_z_xz = ice_flx_adv_z[:,0,:]
# #
# # ice1_flx_adv_x, ice1_flx_adv_y, ice1_flx_adv_z = v_Intpl_Sph2car(rad,theta,phi,x1_exp_half,slice_exp,x3_exp,(flx_ice1_adv_x).T,(flx_ice1_adv_z).T, (flx_ice1_adv_z).T * 0.0)
# # ice1_flx_adv_x_xz = ice1_flx_adv_x[:,0,:]
# # ice1_flx_adv_z_xz = ice1_flx_adv_z[:,0,:]
# #
# #
# # lw_ice_adv = sqrt(ice_flx_adv_x_xz**2 + ice_flx_adv_z_xz**2)/normal2
# # lw_ice1_adv = sqrt(ice1_flx_adv_x_xz**2 +ice1_flx_adv_z_xz**2)/normal2
#
# # axs[1,0].streamplot(x1_exp_half,x3_exp, ice1_flx_adv_x_xz/normal2, ice1_flx_adv_z_xz/normal2,linewidth = lw_ice1_adv, arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='green',zorder=4)
# # axs[1,0].streamplot(x1_exp_half,z_neg, 
# #                     ice_flx_adv_x_xz[::-1,:]/normal2, 
# #                     ice_flx_adv_z_xz[::-1,:]/normal2,
# #                     linewidth = lw_ice_adv[::-1,:], arrowstyle = '->', density = 1.0, broken_streamlines = True, color ='green',zorder=4)
#
#
# #move the colorbar to be aligned with the bottom of top figure 
# cbarrho = fig.colorbar(crho1, ax=axs[1,0],location = 'right', shrink = 0.45, pad =-0.085,anchor=(0,-0.))
# cbarrho.set_ticks([1e-13, 1e-12,1e-11], labels = ['$10^{-13}$', '$10^{-12}$', '$10^{-11}$'])
# cbarrho.ax.set_title(r'$\rho_{\mathrm{ice}} [g/cm^3]$', fontsize = 12)
# cbarvap = fig.colorbar(ax0, ax=axs[1,0], location = 'right', shrink = 0.45, pad =0.04, anchor=(0,1))
# cbarvap.set_ticks([1e-13, 1e-12, 1e-11], labels = ['$10^{-13}$', '$10^{-12}$', '$10^{-11}$'])
# cbarvap.ax.set_title(r'$\rho_{\mathrm{vap}} [g/cm^3]$', fontsize = 12)
#
# # mass and water comp
# axs[0,1].set_xlabel(r'$R$ [AU]', fontsize = 12)
# axs[0,1].set_ylabel(r'$z$ [AU]', fontsize = 12)
# # axs[1,2].plot(rad, H_profile(rad)/AU, '--', c='gray', lw=1)
# # axs[1,2].plot(rad, -H_profile(rad)/AU, '--', c='gray', lw=1)
# axs[0,1].plot(xx_exp, -yy0, '--', c='k', lw=1, zorder=10)
# axs[0,1].plot(xx_exp, yy1, '--', c='k', lw=1, zorder=10)
# axs[0,1].plot([0.8, 0.8, 1.3, 1.3, 0.8], [-0.04, 0.04, 0.04, -0.04, -0.04], color='r', lw=3, ls='-', zorder=20)
#
#
# cmap_mass = LinearSegmentedColormap.from_list('mass_cmap', 
#     ['white', '#fee090', '#fc8d59', '#c2518a', 'purple'], N=256)
# # c1 = axs[0,1].contourf(x_xz_c, y_xz_c, m_p1_xz, levels = logspace(-8, 4.5, 31), norm = LogNorm(),cmap = cmap_mass, alpha = 1.0,extend = 'both')
# c1 = axs[0,1].contourf(x_xz_c, y_xz_c, s_p1_xz, levels = logspace(-2.5, 2, 21), norm = LogNorm(),cmap = cmap_mass, alpha = 1.0,extend = 'both')
# axs[0,1].contourf(x_xz_c, -y_xz_c, s_p_xz, levels = logspace(-2.5, 2, 21), norm = LogNorm(),cmap = cmap_mass, alpha = 1.0,extend = 'both')
# axs[0,1].axhline(0.0, c= 'k', ls='-',linewidth = 4., zorder=15)
#
# cbar0 = fig.colorbar(c1, ax=axs[0,1], location = 'right', shrink = 1, pad = 0.04, anchor=(0,0))
# cbar0.ax.set_title(r'$\mathbf{s [cm]}$', fontsize = 20,fontweight = 'bold')
#
# ticks = logspace(-2, 2, 5)
# cbar0.set_ticks(ticks)
#
# # ax[0].contour(x_xz_c,y_xz_c,r_snow_2d(tem_xz,rho_xz,0.4) ,levels = [1.e-3,1.0,1.e3], cmap = 'Greens_r', alpha = 0.7, linewidths = 5.0)
#
# #plot the water mass fraction 
# axs[1,1].plot(xx_exp, -yy0, '--', c='k', lw=1, zorder=10)
# axs[1,1].plot(xx_exp, yy1, '--', c='k', lw=1, zorder=10)
# axs[1,1].axhline(0.0, c= 'k', ls='-',linewidth = 4., zorder=15)
# axs[1,1].plot([0.8, 0.8, 1.3, 1.3, 0.8], [-0.04, 0.04, 0.04, -0.04, -0.04], color='r', lw=3, ls='-', zorder=20)
# # axs[1,2].plot(rad, H_profile(rad)/AU, '--', c='gray', lw=1)
# axs[1,1].set_xlabel(r'$R$ [AU]', fontsize = 12)
# axs[1,1].set_ylabel(r'$z$ [AU]', fontsize = 12)
# # c0 = axs[1,1].contourf(x_xz_c, y_xz_c,m_p_xz, levels = logspace(-8, 3.5, 21), norm = LogNorm(), cmap = cmap_mass, alpha = 1.0,extend = 'both')
# ccomp0 = axs[1,1].contourf(x_xz_c, -y_xz_c, watercomp0, levels = linspace(0.4,0.9,16), cmap = 'Blues', alpha = 0.8,extend = 'both')
# axs[1,1].contour(x_xz_c, -y_xz_c, watercomp0, levels = [0.5], colors = 'k', linewidths = 2.0)
#
# axs[1,1].contour(x_xz_c,  y_xz_c, watercomp1, levels = [0.5], colors = 'k', linewidths = 2.0)
# axs[1,1].contourf(x_xz_c, y_xz_c, watercomp1, levels = linspace(0.4,0.9,16), cmap = 'Blues', alpha = 0.8,extend = 'both')
# #also plot the 1/2 line 
#
# # axs[0,1].text(0.05, 0.1, 'pebble', transform=axs[0,1].transAxes, fontsize=18, va='top', ha='left',zorder =23)
# # axs[1,1].text(0.05, 0.05, 'dust', transform=axs[1,1].transAxes, fontsize=18, va='bottom', ha='left',zorder =22)
#
# cbarcomp0 = fig.colorbar(ccomp0, ax=axs[1,1], location='right', shrink=1, pad=0.04, anchor=(0,1))
# cbarcomp0.ax.set_title(r'$\mathbf{f_{\mathrm{H_2 O}}}$', fontsize = 20, fontweight = 'bold')
# cbarcomp0.set_ticks([0.1, 0.5, 0.9], labels = ['0.1', '0.5', '0.9'])
# cbarcomp0.ax.hlines(0.5, 0,1, color='k', linewidth=2)  # Mark the 0.5 line on the colorbar
#
# # axs[1,0].set_ylim(0, 0.25)
# #
# # axs[1,0].plot(xx_exp, yy0, '-.', c='k', lw=1, zorder=10,label = r'$H_{\mathrm{0}}$')
# # axs[1,0].plot(xx_exp, yy1, '--', c='k', lw=1, zorder=10, label = r'$H_{\mathrm{1}}$')
# # # axs[0,2].set_xlabel(r'$R$ [AU]', fontsize = 12)
# # axs[1,0].set_ylabel(r'$z$ [AU]', fontsize = 12)
# # # reconstruct the dust size distribution. 
# # # The crude one: 
# # pp= log10((dust_1_rho_xz+dust_2_rho_xz)/(dust_3_rho_xz + dust_4_rho_xz))/log10(m_p_xz/m_p1_xz)
# # ax0 = axs[1,0].contourf(x_xz_c,y_xz_c,pp,levels = linspace(-1, 1, 21), cmap = 'coolwarm', alpha = 0.8, extend = 'both')
# # #label the 1/6 line 
# # axs[1,0].contour(x_xz_c,y_xz_c, pp, levels = [1.0/6.0], colors = 'k', linewidths = 2.0, extend = 'both')
# # cbar0 = fig.colorbar(ax0, ax=axs[1,0])
# # cbar0.ax.set_title(r'$p$', fontsize = 12)
# # #mark the 1/6 line in color bar 
# # cbar0.ax.plot([1,0],[1.0/6.0,1.0/6.0], color = 'k', linewidth = 2.0)
# #
# # axs[1,0].legend(frameon=False, loc='upper left', fontsize=12)
# #
# # axs[0,0].legend(handles=legend_handles_panel1, loc='upper right', frameon=True, fontsize=12)
#
# plt.savefig('./plots/2ddust_{:05d}.png'.format(int(filenum)), dpi = 300, bbox_inches='tight')
# plt.close()
#
