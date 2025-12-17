"""

Conventions / units used in these functions:
- Magnetic field B: tesla (T)
- Mechanical stress sigma_max: pascal (Pa)
- lengths a, b, c, R0: meters (m)
- Power P_E: megawatts (MW) unless otherwise noted in function docstrings
- Wall loading P_W: MW / m^2 (this is a typical engineering convention)
- Temperature T_keV: kilo-electron-volts (keV)
- <sigma v>: m^3 / s
- Pressure returned: pascal (Pa) and atmospheres (atm)
- mu0: magnetic permeability of free space (SI)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication

# constants
mu0 = 4 * math.pi * 1e-7         # H/m
PA_PER_ATM = 101325.0            # Pa per atm
eV_to_J = 1.602176634e-19        # J per eV
keV_to_J = 1e3 * eV_to_J

def xi_from_B_sigma(B_max_T: float, sigma_max_Pa: float) -> float:
    """
    xi = B_c^2 / (4 mu0 sigma_max)
    where B_c should be set to B_max (T).
    """
    return (B_max_T**2) / (4.0 * mu0 * sigma_max_Pa)

def a_from_b(b_m: float, xi: float) -> float:
    """a = (1 + xi) / (2 sqrt(xi)) * b"""
    if xi <= 0:
        raise ValueError("xi must be positive")
    return ((1.0 + xi) / (2.0 * math.sqrt(xi))) * b_m

def c_from_b(b_m: float, xi: float) -> float:
    """c = xi^{1/2} (1 + xi^{1/2})/(1 - xi^{1/2}) * b"""
    sqrt_xi = math.sqrt(xi)
    if abs(1.0 - sqrt_xi) < 1e-12:
        raise ValueError("xi^{1/2} near 1 produces division by zero")
    return (sqrt_xi * (1.0 + sqrt_xi) / (1.0 - sqrt_xi)) * b_m

def VI_over_PE(b_m: float, P_W_MWpm2: float, xi: float) -> float:
    """
    Equation (5.30): V_I / P_E = 1.58 * (1 + xi) / (1 - sqrt(xi))^2 * (b / P_W)
    Units:
      - returns V_I / P_E in [m^3 / MW] if b in m, P_W in MW/m^2, P_E in MW.
    """
    sqrt_xi = math.sqrt(xi)
    denom = (1.0 - sqrt_xi)**2
    if denom == 0:
        raise ValueError("Denominator (1 - sqrt(xi))^2 is zero")
    return 1.58 * (1.0 + xi) / denom * (b_m / P_W_MWpm2)

def R0_from_PE_a_PW(P_E_MW: float, a_m: float, P_W_MWpm2: float,
                    eta_t: float = 0.4,
                    E_n_MeV: float = 14.1,
                    E_alpha_MeV: float = 3.5,
                    E_Li_MeV: float = 0.0) -> float:
    """
    General formula for R0:
      R0 = ( 1/(4 pi^2 eta) * E_n/(E_alpha + E_n + E_Li) ) * P_E/(a P_W)
    Inputs:
      - P_E_MW: electric power in MW
      - a_m: minor radius a in m
      - P_W_MWpm2: wall loading in MW/m^2
      - energies in MeV
    Output:
      - R0 in meters
    Note: The short form R0 = 0.04 * P_E/(a P_W) assumes particular numerical
    values for energies and eta; use the general function above for clarity.
    """
    denom_energy = E_alpha_MeV + E_n_MeV + E_Li_MeV
    if denom_energy <= 0:
        raise ValueError("Sum of energies must be positive")
    prefactor = (1.0 / (4.0 * math.pi**2 * eta_t)) * (E_n_MeV / denom_energy)
    return prefactor * (P_E_MW / (a_m * P_W_MWpm2))

def R0_convenience_numeric(P_E_MW: float, a_m: float, P_W_MWpm2: float) -> float:
    """
    Convenience version using the numeric factor given in your text:
      R0 = 0.04 * P_E / (a * P_W)   [meters]
    Use only if you want the same unit assumptions as original derivation
    (P_E in MW, P_W in MW/m^2, a in m).
    """
    return 0.04 * P_E_MW / (a_m * P_W_MWpm2)

def plasma_area(R0_m: float, a_m: float) -> float:
    """A_P = 4 pi^2 R0 a (m^2)"""
    return 4.0 * math.pi**2 * R0_m * a_m

def plasma_volume(R0_m: float, a_m: float) -> float:
    """V_P = 2 pi^2 R0 a^2 (m^3)"""
    return 2.0 * math.pi**2 * R0_m * a_m**2

def pressure_from_prefactor(T_keV: float, sigma_v_m3s: float,
                            prefactor_atm: float = 8.4e-12) -> tuple:
    """
    Implements eqn (5.37) as given in your text:
      p [atm] = prefactor_atm * sqrt( T_k^2 / <sigma v> )
      where: T_k is T in keV (they wrote T_k in the text), and <sigma v> in m^3/s
    Returns (p_atm, p_Pa)
    Note: This follows the exact algebra in your text and returns pressure
    both in atm and Pa. If you prefer a fully SI-derived expression starting
    from first principles, we can implement that too.
    """
    if sigma_v_m3s <= 0:
        raise ValueError("<sigma v> must be positive")
    p_atm = prefactor_atm * math.sqrt((T_keV**2) / sigma_v_m3s)
    p_Pa = p_atm * PA_PER_ATM
    return p_atm, p_Pa


def n_from_p_T(T_kev:float, p_Pa:float):
    """From the assumption that p = 2 nT 
    calculates the density from the pressure and temperature"""
    n = p_Pa/(2*T_kev*keV_to_J)
    return n

def tauE_from_min_ignition(p_atm: float, p_tauE_atm_s: float = 8.3,H=1) -> float:
    """
    From minimum ignition: p * tau_E = p_tauE_atm_s (atm * s)
    Solve for tau_E in seconds given p in atm.
    """
    if p_atm <= 0:
        raise ValueError("pressure must be positive (in atm)")
    return p_tauE_atm_s / p_atm *H

def B0_on_axis(R0_m: float, a_m: float, b_m: float, B_max_T: float) -> float:
    """
    B0 = (R0 - a - b)/R0 * B_max
    Units: dimensions in meters, B_max in tesla -> B0 in tesla
    """
    if R0_m == 0:
        raise ValueError("R0 must be non-zero")
    return ((R0_m - a_m - b_m) / R0_m) * B_max_T

def beta_from_p_B0(p_Pa: float, B0_T: float) -> float:
    """
    beta = p / (B0^2 / (2 mu0))
    Inputs:
      - p_Pa: pressure in pascals
      - B0_T: magnetic field at axis in tesla
    Output:
      - beta (dimensionless)
    """
    denom = (B0_T**2) / (2.0 * mu0)
    if denom <= 0:
        raise ValueError("Magnetic energy density denominator must be positive")
    return p_Pa / denom

def Coil_Current(R0,B_max):
    """Returns the current based on the major radius and magnetic field strength"""
    I = 2*np.pi*R0*B_max/mu0
    IM = I/(1*10**6)
    return I,IM

def Inverse_aspectRatio(a,R0):
    epsilon = a/R0
    return epsilon

def AspectRatio(a,R0):
    asp = R0/a
    return asp

def Fusion_Power(n,V_p):
    P_f = 0.25*17.6*10**6*1.602*10**(-19)*n**2*sigma_v*V_p
    return P_f

def Plasma_current(a,B0,R0):
    """Defines a plasma current based on a desired
      safety factor at the edge of the plasma"""
    qa = 1 #The desired q limit at the edge of the plasma
    I_P = 2*np.pi*a**2*B0/(mu0*R0*qa)
    # I_PM = 12.1*a**(1.63)*B0**(-0.16) # Friedberg publication alternative
    I_PM = I_P/(1*10**6)
    return I_P,I_PM

def CurrentFrom_H_mode_scaling_law(P_E_MW,R0,kappa,B0,A,tau_E):
    """Calcs the plasma current from the H-mode scaling law"""
    #Check whether B field has to be the toroidal field..,.
    IM = tau_E/(0.082*P_E_MW**(-0.5)*R0**(1.6)*kappa**(-0.2)*B0**(0.15)*A**(0.5)) #In MA
    return IM




#################################################################
### Plasma physics constraints


def safety_limit(R0,B0,a,I_PM,kappa):
    """Applies beta limit to the design parameters"""

    qstar = 5*a**2*kappa*B0/(R0*I_PM)
    # qstar = 0.112*a**(1.37)*B0**(1.16) #Friedberg publication
    # print("Qstar:",qstar)
    if qstar>2:
        # print("q limit met")
        return qstar
    else:
        # print("q limit NOT met")
        return qstar
    
def beta_limit(beta_val,B0,I_PM,a):
    """Applies Troyon limit to the design parameters"""
    limit = 0.03*I_PM/(a*B0)
    # print("beta limit:", limit)
    if beta_val<limit:
        # print("Troyon/Beta limit met")
        return limit
    else:
        # print("Troyon/Beta limit NOT met")
        return limit

def Density_limit(n,a,I_PM):
    """Applies Greenwald limit to the design parameters"""
    limit = I_PM/(np.pi*a**2)
    # print("limit greenwald:",limit)
    if n/(1*10**(20))<limit:
        # print("Greenwald lim met")
        return limit
    else:
        # print("Greenwald lim NOT met")
        return limit

def bootstrap_frac(kappa,beta_val,B0,I_P,R0,a,epsilon):
    """Applies Greenwald limit to the design parameters"""
    qstar = 5*a**2*kappa*B0/(R0*I_P)
    fB = 1.3*kappa**(0.25)*beta_val*qstar/(epsilon**(0.5))
    limit=0.8
    if fB>limit:
        # print("Bootstrap fraction met")
        return fB
    else:
        # print("Bootstrap fraction NOT met")
        return fB
    

###############################################################


def CALC_PARAMS(a,H,B_max=13.0):
    # B_max = 13.0               # T ()
    sigma_max = 300*10**6         # Pa ( structural allowable stress)
    b = 1.2                    # m (blanket thickness)
    P_E_MW = 1000.0             # MW, electrical output (example)
    P_W_MWpm2 = 4.0            # MW/m^2, wall loading (example)
    T_keV = 15.0
    sigma_v = 3e-22            # m^3/s
    kappa = 1.7
    A = 2.5 #In atomic mass units, scaling law asks for atomic mass H-mode scaling

    xi = xi_from_B_sigma(B_max, sigma_max)
    # a = a_from_b(b, xi)
    c = c_from_b(b, xi)
    VI_over_PE_val = VI_over_PE(b, P_W_MWpm2, xi)
    R0 = R0_from_PE_a_PW(P_E_MW, a, P_W_MWpm2, eta_t=0.4, E_n_MeV=14.1, E_alpha_MeV=3.5, E_Li_MeV=4.72)
    # (Used E_Li ~ 4.72 MeV to match the numeric 0.04 prefactor; change if you want.)
    A_p = plasma_area(R0, a)
    V_p = plasma_volume(R0, a)
    p_atm, p_Pa = pressure_from_prefactor(T_keV, sigma_v)
    tau_E_s = tauE_from_min_ignition(p_atm, H=H)
    B0 = B0_on_axis(R0, a, b, B_max)
    beta_val = beta_from_p_B0(p_Pa, B0)
    n = n_from_p_T(T_keV,p_Pa)
    I,IM = Coil_Current(R0,B_max)
    I_P,I_M = Plasma_current(a,B0,R0) ## Plasma current from q requirement
    I_PM = CurrentFrom_H_mode_scaling_law(P_E_MW,R0,kappa,B0,A,tau_E_s) #Plasma current from H-mode scaling law
    epsilon = Inverse_aspectRatio(a,R0)
    P_f = Fusion_Power(n,V_p)
    qstar = safety_limit(R0,B0,a,I_PM,kappa)
    beta_T = beta_limit(beta_val,B0,I_PM,a)
    nG = Density_limit(n,a,I_PM)
    fB = bootstrap_frac(kappa,beta_val,B0,I_PM,R0,a,epsilon)


    return c,R0,A_p,V_p,p_atm,B0,beta_val,n,I_PM,epsilon,P_f,fB,qstar,beta_T,nG


# -------------------------
# -------------------------
if __name__ == "__main__":
    B_max = 13.0               # T ()
    sigma_max = 300*10**6         # Pa ( structural allowable stress)
    b = 1.2                    # m (blanket thickness)
    P_E_MW = 1000.0             # MW, electrical output (example)
    P_W_MWpm2 = 4.0            # MW/m^2, wall loading (example)
    T_keV = 15.0
    sigma_v = 3e-22            # m^3/s
    kappa = 1.7
    A=1.008

    xi = xi_from_B_sigma(B_max, sigma_max)
    a = a_from_b(b, xi)
    c = c_from_b(b, xi)
    VI_over_PE_val = VI_over_PE(b, P_W_MWpm2, xi)
    R0 = R0_from_PE_a_PW(P_E_MW, a, P_W_MWpm2, eta_t=0.4, E_n_MeV=14.1, E_alpha_MeV=3.5, E_Li_MeV=4.72)
    # (Used E_Li ~ 4.72 MeV to match the numeric 0.04 prefactor; change if you want.)
    A_p = plasma_area(R0, a)
    V_p = plasma_volume(R0, a)
    p_atm, p_Pa = pressure_from_prefactor(T_keV, sigma_v)
    tau_E_s = tauE_from_min_ignition(p_atm)
    B0 = B0_on_axis(R0, a, b, B_max)
    beta_val = beta_from_p_B0(p_Pa, B0)
    n = n_from_p_T(T_keV,p_Pa)
    I,IM = Coil_Current(R0,B_max)
    I_PM_H = CurrentFrom_H_mode_scaling_law(P_E_MW,R0,kappa,B0,A,tau_E_s)
    I_P,I_PM = Plasma_current(a,B0,R0)
    epsilon = Inverse_aspectRatio(a,R0)
    P_f = Fusion_Power(n,V_p)
    asp = AspectRatio(a,R0)
    fB = bootstrap_frac(kappa,beta_val,B0,I_PM,R0,a,epsilon)

    # print("xi =", xi)
    # print("a (m) =", a)
    # print("c (m) =", c)
    # print("V_I / P_E (m^3 / MW) =", VI_over_PE_val)
    # print("R0 (m) =", R0)
    # print("Aspect Ratio =", asp)
    # print("A_p (m^2) =", A_p)
    # print("V_p (m^3) =", V_p)
    # print("p (atm) =", p_atm, "p (Pa) =", p_Pa)
    # print("tau_E required (s) =", tau_E_s)
    # print("B0 (T) =", B0)
    # print("Plasma Current (A) =", I_P)
    # print("Plasma Current (MA) =", I_PM_H)
    # print("beta =", beta_val)
    # print("Density (m^-3) n=", n)
    # print("Fusion power (W):",P_f)
    # print("Bootstrap fraction fB =", fB)
    # # print("Coil Current (m^-3) I=", I)
    # print("---------------- \n LIMITS:")

    # safety_limit(R0,B0,a,I_PM,kappa)
    # beta_limit(beta_val,B0,I_PM,a)
    # Density_limit(n,a,I_PM)
    # bootstrap_frac(kappa,beta_val,B0,I_PM,R0,a,epsilon)





# a_vals =np.arange(0.5,1.75,0.1)
# H_vals =np.arange(0.5,10,0.1)


def Param_Sweep(vals,param_name, unit = "-"):
    beta_hist = np.zeros(len(vals))
    betaT_hist = np.zeros(len(vals))
    qstar_hist = np.zeros(len(vals))
    nG_hist = np.zeros(len(vals))
    n_hist = np.zeros(len(vals))
    fB_hist = np.zeros(len(vals))

    for idx, val in enumerate(vals):
        kwargs = {
            'a': 1.993,
            "H": 1.0,
            param_name: val 
        }
        c,R0,A_p,V_p,p_atm,B0,beta_val,n,I_PM,epsilon,P_f,fB,qstar,beta_T,nG = CALC_PARAMS(**kwargs)
        beta_hist[idx] = beta_val
        betaT_hist[idx] = beta_T
        qstar_hist[idx]=qstar
        n_hist[idx] = n
        nG_hist[idx] = nG
        fB_hist[idx] = fB

    qk = 2  #Q profile limit at the edge
    fNB = 0.8  #Desired bootstrap fraction

    # Plot
    fig = plt.figure(figsize=(8,6))

    thickness = 3
    plt.plot(vals, beta_hist/betaT_hist, label=r'$\beta / \beta_T$', color='red', linestyle='solid', linewidth=thickness)
    plt.plot(vals, qk/qstar_hist, label=r'$q_k / q^*$', color='green', linestyle='dashed', linewidth=thickness)
    plt.plot(vals, n_hist/(nG_hist*(10**20)), label=r'$n / n_G$', color='black', linestyle='dashdot', linewidth=thickness)
    plt.plot(vals, fNB/fB_hist, label=r'$f_{NC} / f_B$', color='blue', linestyle='dotted', linewidth=thickness)

    plt.xlabel(f'{param_name} [{unit}]', size=15)
    # plt.ylim(0,5)
    plt.ylabel('Normalized constraints', size=15)
    plt.title(f'Tokamak operational limits vs {param_name}', size=17)
    plt.legend(fontsize=15, loc='upper right')
    plt.grid(True)
    # plt.savefig('Tokamak_Constraints_vs_Bmax.png', dpi=300)

    plt.fill_between(np.arange(0,100), 0, 1, color='green', alpha=0.1)
    plt.fill_between(np.arange(0,100), 1, 100, color='red', alpha=0.1)
    plt.xlim(vals[0], vals[-1])
    plt.ylim(0, 6)

    ax = plt.gca()
    ax.tick_params(
    axis='both',
    which='both',
    direction='in',
    top=True,
    right=True
    )

    ax.minorticks_on()

    return fig

    
def Get_Constraint_Value(a=1.993, H=1.0, B_max=13.0):
    qk = 2  #Q profile limit at the edge
    fNB = 0.8  #Desired bootstrap fraction

    _,_,_,_,_,_,beta_val,n,_,_,_,fB,qstar,beta_T,nG = CALC_PARAMS(a=a, H=H, B_max=B_max)

    norm_beta = beta_val/beta_T
    norm_qstar = qk/qstar
    norm_nG = n/(nG*(10**20))
    norm_fB = fNB/fB

    return norm_beta, norm_qstar, norm_nG, norm_fB

default = Get_Constraint_Value()
print("Constraint values at default params (a=1.993, H=1.0, B_max=13.0): \n", "Beta: ", default[0], " Qstar: ", default[1], " nG: ", default[2], " fB: ", default[3])
a1_5 = Get_Constraint_Value(a=1.5)
print("Constraint values at a = 1.5: \n", "Beta: ", a1_5[0], " Qstar: ", a1_5[1], " nG: ", a1_5[2], " fB: ", a1_5[3])
Bmax25 = Get_Constraint_Value(B_max=25)
print("Constraint values at B_max = 25: \n", "Beta: ", Bmax25[0], " Qstar: ", Bmax25[1], " nG: ", Bmax25[2], " fB: ", Bmax25[3])

B_vals = np.arange(10, 31, 1)
figs = []
figs.append(Param_Sweep(B_vals, 'B_max', 'T'))

H_vals = np.arange(0.5, 2.6, 0.1)
figs.append(Param_Sweep(H_vals, 'H', '-'))
a_vals = np.arange(0.5, 2.6, 0.1)
figs.append(Param_Sweep(a_vals, 'a', 'm'))

Layout = "Export" #Options: Tiled, Export
if(Layout=="Tiled"):
    app = QApplication.instance()
    screen = app.primaryScreen().geometry()

    W = screen.width()
    H = screen.height()
    n = 3

    for i, fig in enumerate(figs):
        mgr = fig.canvas.manager
        mgr.window.setGeometry(
            i * W // n,  # x
            0,          # y
            W // n,     # width
            H           # height
        )
elif(Layout=="Export"):
    for i, fig in enumerate(figs):
        fig.set_size_inches(8,6)
        fig.savefig(f'Tokamak_Constraints_vs_param_{i}.png', dpi=300)

plt.show()

