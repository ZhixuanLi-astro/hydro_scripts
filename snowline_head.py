import astropy.constants as cons
from numpy import *
from copy import deepcopy

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
k_B = cons.k_B.cgs.value
m_p = cons.m_p.cgs.value
G = cons.G.cgs.value

class chem:
    name = ''

    def __init__(self,name,mu,T_a,P_eq,L_heat):
        self.name = name
        self.mu = mu
        self.T_a = T_a
        self.P_eq = P_eq
        self.R = k_B/m_p/mu
        self.L_heat = L_heat
        
# gas property [cgs]
# water:
mu_water = 18
P_eq_water = 1.14e13
L_heat_water = 2.75e10
R_water = k_B/m_p/mu_water
T_a_water = 6062
chem_H2O = chem('H2O',18,T_a_water,P_eq_water,L_heat_water)

class chem:
    name = ''
    def __init__(self,name,mu,T_a,P_eq,L_heat):
        self.name = name
        self.mu = mu
        self.T_a = T_a
        self.P_eq = P_eq
        self.R = k_B/m_p/mu
        self.L_heat = L_heat
        
# gas property
# water:
mu_water = 18
P_eq_water = 1.14e13 # cgs
L_heat_water = 2.75e10 # cgs
T_a_water = 6062 # K
chem_H2O = chem('H2O',mu_water,T_a_water,P_eq_water,L_heat_water)

def dust_redist(rad, theta, solid, rho_xz, st_max, bin_edges, a_max, alpha, a_size=-3.5):
    '''
    bin_edges: array of dust size bin edges in cm
    a_max: maximum dust size in cm
    a_size: size distribution index, default is -3.5 (MRN distribution)
    '''
    
    # gas scale height
    rho_normal = deepcopy(rho_xz)
    for j in range(len(theta)):
        rho_normal[:,j] /= rho_normal[:,-1]
    H_g = zeros(len(rad))
    for i in range(len(rad)):
        index = sum(rho_normal[i,:] < exp(-0.5))
        H_g[i] = rad[i]*(pi/2-theta[index])
        
    bin_center =  10**((log10(bin_edges[:-1]) + log10( bin_edges[1:]))*0.5)
    
    # mass weights of different bins
    # bin_width = fabs(bin_edges[0:-1] - bin_edges[1:])
    # weights = bin_center ** (a_size + 3.0) * bin_width
    # weights /= sum(weights)
    weights = (bin_edges[0:-1])**(a_size+4.0) - (bin_edges[1:])**(a_size+4.0)
    weights /= sum(weights)
    
    st_bins = zeros((len(bin_center),len(solid['rad'])))
    H_d_bins = zeros((len(bin_center),len(solid['rad'])))
    
    for i in range(len(bin_center)):
        st_bins[i,:] = st_max / a_max * bin_center[i]
        H_d_bins[i,:] = sqrt(alpha/(st_bins[i,:] + alpha)) * H_g
        
    # depends on whether a_max is a single value or an array
    if isinstance(a_max, float):
        for i in range(len(bin_center)):
            solid['H_d' + str(i)] = H_d_bins[i,:]
            solid['bin' + str(i)] = solid['sigma_all']*weights[i]
    elif isinstance(a_max, ndarray):
        for i in range(len(bin_center)):
            solid['H_d' + str(i)] = H_d_bins[i,:]
            solid['bin' + str(i)] = zeros(len(rad))
        for j in range(len(rad)):
            n_bin_in = sum(bin_edges > a_max[j])
            n_bin_in = maximum(1,n_bin_in)
            n_bin_in = minimum(n_bin_in, len(bin_center)-1)
            weight_norm = weights / sum(weights[n_bin_in-1:])
            weight_norm[:n_bin_in-1] = 0.0
            
            for i in range(len(bin_center)):
                solid.loc[j,'bin' + str(i)] = solid['sigma_all'][j]* weight_norm[i]
                
    return solid

def get_rho(sigma, H_d, rad, theta, dtheta):
    rad2, theta2 = meshgrid(rad,theta)
    rad2 = rad2.T
    theta2 = theta2.T
    
    x = rad2
    z = rad2*(pi/2-theta2)
    dz = dtheta*rad2
    
    rho_d = zeros_like(x)
    for i in range(len(theta)):
        rho_d[:,i] = sigma/(sqrt(2*pi)*H_d)*exp(-0.5*(z[:,i]/H_d)**2)
    
    return rho_d * 2.0

def get_rho2(sigma, St, alpha, rad, theta, dtheta):
    rad2, theta2 = meshgrid(rad,theta)
    rad2 = rad2.T
    theta2 = theta2.T
    
    x = rad2*sin(theta2)
    z = rad2*cos(theta2)
    dz = dtheta*rad2
    
    H_g = 0.15 * (x/(31.0*AU))**(0.25) * x
    
    H_peb_z = H_g * (1.0 + St/alpha* (2.0*(H_g/z)**2) * (exp(0.5*(z/H_g)**2)-1.0))**(-0.5)
    
    rho_d = exp(-0.5 * (z/H_peb_z)**2)
    
    for i in range(len(rad)):
        tmp = sum(rho_d[i,:] * dz[i,:])
        rho_d[i,:] *= sigma[i] / tmp
    
    return rho_d