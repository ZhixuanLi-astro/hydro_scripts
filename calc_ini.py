import numpy as np 
import cgs
from preplot import read_athinput
import sys
import re

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

    print ("m0, m1, St0, St1, Hd0, Hd1: ", m0, m1, St0, St1, Hd0/Hg, Hd1/Hg)
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

def write_athinput_dust(inputfile, m0, m1, St0, St1, Hratio0, Hratio1):
    """
    Write calculated dust initial conditions into the <dust> block of an athinput file.

    Each population (small=0, large=1) shares the same properties across ice & silicate
    species because grains are assumed to be 50% ice + 50% silicate.

    Parameters
    ----------
    inputfile : str
        Path to the athinput file to modify (in-place).
    m0, m1 : float
        Characteristic mass [g] for small and large populations.
    St0, St1 : float
        Stokes numbers for small and large populations.
    Hratio0, Hratio1 : float
        Hd/Hg scale-height ratios for small and large populations.
    """
    with open(inputfile, 'r') as f:
        content = f.read()

    replacements = [
        # Species 1: small ice  ─┐ same small-pop values
        (r'(Stokes_number_1\s*=\s*)[\d.eE+\-]+', rf'\g<1>{St0:.6g}'),
        (r'(Hratio_1\s*=\s*)[\d.eE+\-]+',        rf'\g<1>{Hratio0:.6g}'),
        (r'(m_p0_1\s*=\s*)[\d.eE+\-]+',           rf'\g<1>{m0:.6g}'),
        # Species 2: small silicate ─┘
        (r'(Stokes_number_2\s*=\s*)[\d.eE+\-]+', rf'\g<1>{St0:.6g}'),
        (r'(Hratio_2\s*=\s*)[\d.eE+\-]+',        rf'\g<1>{Hratio0:.6g}'),
        (r'(m_p0_2\s*=\s*)[\d.eE+\-]+',           rf'\g<1>{m0:.6g}'),
        # Species 3: large ice  ─┐ same large-pop values
        (r'(Stokes_number_3\s*=\s*)[\d.eE+\-]+', rf'\g<1>{St1:.6g}'),
        (r'(Hratio_3\s*=\s*)[\d.eE+\-]+',        rf'\g<1>{Hratio1:.6g}'),
        (r'(m_p0_3\s*=\s*)[\d.eE+\-]+',           rf'\g<1>{m1:.6g}'),
        # Species 4: large silicate ─┘
        (r'(Stokes_number_4\s*=\s*)[\d.eE+\-]+', rf'\g<1>{St1:.6g}'),
        (r'(Hratio_4\s*=\s*)[\d.eE+\-]+',        rf'\g<1>{Hratio1:.6g}'),
        (r'(m_p0_4\s*=\s*)[\d.eE+\-]+',           rf'\g<1>{m1:.6g}'),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    with open(inputfile, 'w') as f:
        f.write(content)

    print(f"[write_athinput_dust] Updated species 1-4 in {inputfile}")

if __name__ == "__main__":
    alpha = athinputs['problem']['alpha_vis']
    Tem = get_tem(rout, T0, -q_value, 3.0)
    v_frag = 1200

    # combined internal density: 50% ice + 50% silicate
    rhoint = get_rhoint(0.50, 0.50,
                        athinputs['problem']['rho_ice_inter'],
                        athinputs['problem']['rho_sil_inter'])
    print("\n=== Combined grains (rhoint = {:.2f} g/cm^3) ===".format(rhoint))
    m0, m1, St0, St1, Hd0, Hd1 = calc_ini(rhoint, v_frag, Tem, alpha, rout, Mstar)

    # Hratio = Hd / Hg  (calc_ini returns absolute Hd; recompute Hg for the ratio)
    mu_xy = 2.34
    cs2_ref = cgs.kB * Tem / cgs.mp / mu_xy
    OmegaK_ref = np.sqrt(cgs.gC * Mstar * cgs.Msun / (rout * cgs.au)**3)
    Hg_ref = np.sqrt(cs2_ref) / OmegaK_ref
    Hratio0 = Hd0 / Hg_ref
    Hratio1 = Hd1 / Hg_ref

    # --- write to input file ---
    write_athinput_dust(inputfile, m0, m1, St0, St1, Hratio0, Hratio1)
