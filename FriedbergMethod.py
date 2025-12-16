import math
import numpy as np
import matplotlib.pyplot as plt

# ------------------- Constants -------------------
mu0 = 4 * math.pi * 1e-7       # H/m
eV_to_J = 1.602176634e-19
keV_to_J = 1e3 * eV_to_J

# ------------------- User Parameters -------------------
R0 = 5.0                # major radius (m)
B_max = 13.0            # maximum field at coil (T)
b = 1.2                 # blanket thickness (m)
P_over_V_MWpm3 = 4.9    # fusion power density (MW/m^3)
T_keV = 15.0            # plasma temperature
sigma_v = 3e-22         # <sigma v> (m^3/s)
kappa = 1.7             # elongation
q_edge_target = 3.0     # desired edge safety factor
fB_required = 0.75      # reference bootstrap fraction
qk = 2.0                # reference q limit

# ------------------- Helper Functions -------------------
def plasma_area(R0, a):
    return 4 * math.pi**2 * R0 * a

def plasma_volume(R0, a):
    return 2 * math.pi**2 * R0 * a**2

def density_from_power_density(P_over_V_MWpm3, sigma_v, E_fusion_MeV=17.6):
    """Compute plasma density from P/V = (1/4) n^2 <σv> E_fusion"""
    P_Wpm3 = P_over_V_MWpm3 * 1e6
    E_J = E_fusion_MeV * 1e6 * eV_to_J
    n = math.sqrt(4 * P_Wpm3 / (sigma_v * E_J))
    return n

def pressure_from_n_T(n, T_keV):
    return 2 * n * keV_to_J * T_keV  # Pa

def beta_from_p_B0(p_Pa, B0_T):
    return p_Pa / (B0_T**2 / (2 * mu0))

def Ip_from_qedge(a, R0, B0, q_edge=3.0):
    return 2 * math.pi * a**2 * B0 / (mu0 * R0 * q_edge)  # A

def qstar_formula(a, kappa, B0, R0, Ip_A):
    Ip_MA = Ip_A / 1e6
    return 5 * a**2 * kappa * B0 / (R0 * Ip_MA)

def greenwald_nG(Ip_MA, a):
    return Ip_MA / (math.pi * a**2) * 1e20

def troyon_beta_T_percent(Ip_MA, a, B0):
    return 0.03 * Ip_MA / (a * B0) * 100

def bootstrap_fraction(beta_N, kappa, qstar, epsilon, nu=1.0):
    return 4 * nu * (kappa**0.25) * beta_N * qstar / math.sqrt(epsilon)

def B0_on_axis(R0, a, b, B_max):
    return (R0 - a - b) / R0 * B_max

def Inverse_aspectRatio(a, R0):
    return a / R0

# ------------------- Main Sweep -------------------
a_vals = np.arange(0.5, 1.75 + 1e-9, 0.05)
n_vals = np.zeros_like(a_vals)
beta_vals = np.zeros_like(a_vals)
betaT_vals = np.zeros_like(a_vals)
qstar_vals = np.zeros_like(a_vals)
nG_vals = np.zeros_like(a_vals)
fNC_vals = np.zeros_like(a_vals)
ratio_beta_betaT = np.zeros_like(a_vals)
ratio_qk_qstar = np.zeros_like(a_vals)
ratio_n_nG = np.zeros_like(a_vals)
ratio_fNC_fB = np.zeros_like(a_vals)

n_fixed = density_from_power_density(P_over_V_MWpm3, sigma_v)

for idx, a in enumerate(a_vals):
    Vp = plasma_volume(R0, a)
    n = n_fixed
    p_Pa = pressure_from_n_T(n, T_keV)
    B0 = B0_on_axis(R0, a, b, B_max)
    beta = beta_from_p_B0(p_Pa, B0)       # dimensionless
    beta_percent = beta * 100              # for printing

    Ip_A = Ip_from_qedge(a, R0, B0, q_edge_target)
    Ip_MA = Ip_A / 1e6

    qstar = qstar_formula(a, kappa, B0, R0, Ip_A)
    nG = greenwald_nG(Ip_MA, a)
    betaT_percent = troyon_beta_T_percent(Ip_MA, a, B0)
    epsilon = Inverse_aspectRatio(a, R0)

    beta_N = (beta * a * B0) / Ip_MA       # <--- corrected, beta as fraction
    fNC = bootstrap_fraction(beta_N, kappa, qstar, epsilon)

    n_vals[idx] = n
    beta_vals[idx] = beta_percent
    betaT_vals[idx] = betaT_percent
    qstar_vals[idx] = qstar
    nG_vals[idx] = nG
    fNC_vals[idx] = fNC

    ratio_beta_betaT[idx] = beta_percent / betaT_percent
    ratio_qk_qstar[idx] = qk / qstar
    ratio_n_nG[idx] = n / nG
    ratio_fNC_fB[idx] = fNC / fB_required / 10



# ------------------- Plot -------------------
plt.figure(figsize=(8,6))
plt.plot(a_vals, ratio_beta_betaT, label=r'$\beta / \beta_T$', color='red')
plt.plot(a_vals, ratio_qk_qstar, label=r'$q_k / q^*$', color='green')
plt.plot(a_vals, ratio_n_nG, label=r'$n / n_G$', color='black')
plt.plot(a_vals, ratio_fNC_fB, label=r'$f_{NC} / f_B$', color='blue')
plt.axhline(1.0, color='gray', linestyle='--', linewidth=0.6)
plt.xlabel('Minor radius a [m]')
plt.ylabel('Constraint ratio')
plt.title('Friedberg-type Constraint Sweep')
plt.legend()
plt.grid(True)
plt.show()

# ------------------- Table -------------------
print("a [m]  beta%   beta_T%   beta/betaT   qstar   qk/qstar   n (1e20 m^-3)   n/nG   fNC/fB")
for i, a in enumerate(a_vals[::4]):
    print(f"{a:4.2f}  {beta_vals[i]:6.3f}  {betaT_vals[i]:6.3f}  {ratio_beta_betaT[i]:6.3f}  "
          f"{qstar_vals[i]:6.3f}  {ratio_qk_qstar[i]:6.3f}  {n_vals[i]/1e20:6.3f}  "
          f"{ratio_n_nG[i]:6.3f}  {ratio_fNC_fB[i]:6.3f}")
