"""
Steady-state data generator — Atkinson and How (1974) trickling-filter model.

Independent variables : Si, Q, T, h        (Latin Hypercube)
Derived               : k0(T), phi, f, lumped product, Se
Fixed                 : k0_20 = 9.66 1/h, calibrated to reproduce the Germain (1966)
                        removal of 79 % at the base case.

Base case : Si = 200 g/m3, D = 6.0 m, w = 90 1/m, Q = 1.0 m/h, h = 300 um,
            T = 20 C, Df = 1e-4 m2/h  ->  Se = 42.0 g/m3, 79.0 % removal.

Output: steady_state_dataset.csv   (10,000 rows)
"""
import numpy as np, pandas as pd
from scipy.stats import qmc

SEED, N = 42, 10_000

D_FILT = 6.0        # m      filter depth              (design)
W_AREA = 90.0       # 1/m    specific surface area     (design)
DF     = 1e-4       # m2/h   effective diffusivity     (assumed)
THETA  = 1.035      # -      Arrhenius coefficient
K0_20  = 9.66       # 1/h    intrinsic rate at 20 C    (calibrated)

RANGES = {
    "Si": (50.0, 300.0),     # g/m3   influent BOD
    "Q":  (0.3, 4.0),        # m/h    hydraulic loading
    "T":  (10.0, 30.0),      # C      temperature
    "h":  (1e-4, 5e-4),      # m      biofilm thickness
}

def k0_of_T(T):
    """Arrhenius temperature correction, Eq. (3.3)."""
    return K0_20 * THETA ** (np.asarray(T) - 20.0)

def effectiveness(h, k0_T):
    """Thiele modulus and effectiveness factor, Eq. (3.2). f is derived, never sampled."""
    phi = h * np.sqrt(np.maximum(k0_T, 1e-12) / DF)
    return np.tanh(phi) / phi, phi

def effluent(Si, Q, lump):
    """Atkinson and How steady-state removal, Eq. (3.1)."""
    return Si * np.exp(-lump * (W_AREA * D_FILT / Q))

def generate(n=N, seed=SEED):
    keys = list(RANGES)
    u = qmc.LatinHypercube(d=len(keys), seed=seed).random(n)
    lo = np.array([RANGES[k][0] for k in keys])
    hi = np.array([RANGES[k][1] for k in keys])
    v = qmc.scale(u, lo, hi)
    Si, Q, T, h = (v[:, i] for i in range(4))

    k0_T = k0_of_T(T)
    f, phi = effectiveness(h, k0_T)
    lump = f * h * k0_T
    Se = effluent(Si, Q, lump)

    df = pd.DataFrame(dict(Si=Si, Se=Se, Q=Q, T=T, h_m=h,
                           k0_20=K0_20, k0_T=k0_T, phi=phi, f=f,
                           lump_fhk0=lump, removal_pct=100 * (1 - Se / Si),
                           D_m=D_FILT, w_1m=W_AREA))
    keep = (df.Se >= 0) & (df.Se <= df.Si) & (df.f > 0) & (df.f <= 1)
    return df[keep].reset_index(drop=True)

def base_case():
    k = k0_of_T(20.0); f, phi = effectiveness(300e-6, k)
    Se = effluent(200.0, 1.0, f * 300e-6 * k)
    return dict(k0_T=k, phi=phi, f=f, Se=Se, removal=100 * (1 - Se / 200.0))

if __name__ == "__main__":
    bc = base_case()
    print("base case: f = %.4f  phi = %.4f  Se = %.2f g/m3  removal = %.1f %%"
          % (bc["f"], bc["phi"], bc["Se"], bc["removal"]))
    df = generate()
    df.to_csv("steady_state_dataset.csv", index=False)
    print("written steady_state_dataset.csv : %d rows, mean removal %.1f %%"
          % (len(df), df.removal_pct.mean()))
