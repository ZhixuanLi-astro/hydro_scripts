import numpy as np 
import cgs
from preplot import read_athinput
import sys

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
# I need to get: m_p1, m_p0, St1, St0, H1, H0, all at midplane maybe  
mmin = 1.e-12

UNIT_M = athinputs['units']['mass_cgs']
UNIT_L = athinputs['units']['length_cgs']
UNIT_DEN = UNIT_M / UNIT_L**3
L_norm = UNIT_L/cgs.au
rout = athinputs['mesh']['x1max']*L_norm
Mstar = 1.0 

p_value = athinputs['problem']['pvalue']
q_value = athinputs['problem']['qvalue']
T0 = athinputs['problem']['T0']

def calc_ini(rhoint, v_frag, Tem, alpha, rad, Mstar):
    """
    rad: in au 
    Mstar: in Msun
    """
    mu_xy = 2.34 
    rhog = rhog_midplane(-p_value, UNIT_DEN, 3.0, rad) # in g/cm^3
    cs2 = cgs.kB*Tem/cgs.mp/mu_xy
    vth = np.sqrt(cs2*8/np.pi)
    OmegaK = np.sqrt(cgs.gC*Mstar*cgs.Msun/(rad*cgs.au)**3)
    Hg = np.sqrt(cs2)/OmegaK 

    St_frag = v_frag**2/cs2 / (3*alpha)
    s_max = St_frag*rhog*vth / (OmegaK*rhoint)
    mmax = 4/3*np.pi*s_max**3*rhoint 

    #get the m1 and m0 under relaxed state:
    #the boundaries are [mmin, sqrt(mmax*mmin), mmax]
    m_div = np.sqrt(mmax*mmin)
    p = 11/6
    m0 = get_m_ch(p, mmin, m_div)
    m1 = get_m_ch(p, m_div, mmax)

    St0 = Stokes_number(m0, rhoint, rhog, OmegaK, vth)
    St1 = Stokes_number(m1, rhoint, rhog, OmegaK, vth)

    Hd0 = dust_scale_height(St0, Hg, alpha)
    Hd1 = dust_scale_height(St1, Hg, alpha)

    print ("m0, m1, St0, St1, Hd0, Hd1: ", m0, m1, St0, St1, Hd0, Hd1)
    import pdb; pdb.set_trace()
    return m0, m1, St0, St1, Hd0, Hd1


def get_m_ch(p, m_l, m_u):
    mch = (p-2)/(p-3)*(m_l**(3-p) - m_u**(3-p)) / (m_l**(2-p) - m_u**(2-p))
    return mch 

def Stokes_number(m, rhoint, rhog, OmegaK, vth):
    s = (3*m/(4*np.pi*rhoint))**(1/3)
    St = rhoint*s/(rhog*vth) * OmegaK 
    return St

def dust_scale_height(St, Hg, alpha):
    Hd = Hg * (1 + St/alpha * (1+2*St)/(1+St))**(-0.5)
    return Hd

def rhog_midplane(q, rho0, r0, rad):
    rhog = rho0 * (rad/r0)**(-q)
    return rhog

def get_rhoint(fice, fsil, rhoint_ice, rhoint_sil):
    rhoint = rhoint_ice*rhoint_sil/(fice*rhoint_sil + fsil*rhoint_ice) 
    return rhoint

def get_tem(rad, T0, q, r0):
    Tem = T0 * (rad/r0)**(-q)
    return Tem


if __name__ == "__main__":
    rhoint = get_rhoint(0.50, 0.50, 1.0, 3.0)
    alpha = athinputs['problem']['alpha_vis']
    Tem = get_tem(rout, T0, -q_value, 3.0)
    v_frag = 1000 
    m0, m1, St0, St1, Hd0, Hd1 = calc_ini(rhoint, v_frag, Tem, alpha, rout, Mstar)
