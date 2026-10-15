"""
Stage 2 — filter-scale dynamic model, and generation of disturbance trajectories.

Governing equation (advection with first-order biofilm reaction):

    dS/dt = -Q(t) dS/dz - w f h k0(T) S        0 <= z <= D,   S(0,t) = Si(t)

At steady state this integrates to the algebraic Atkinson form used in Stage 1,
which is the cross-consistency check in verify().

Parameterisation is identical to steady_state_generator.py (k0_20 = 9.66 1/h,
f derived from the Thiele relationship).

NOTE ON THE SOLVER.  The rain forcing contains step discontinuities.  An adaptive
implicit solver takes large steps through the constant intervals either side of a
short event and can integrate straight across it, producing a trajectory in which
the disturbance is absent.  The equations are therefore integrated in SEGMENTS
split at every discontinuity, with the solver restarted at each step change.

Output: pde_trajectories.npz
"""
import numpy as np
from scipy.integrate import solve_ivp

D_FILT, W_AREA, DF, THETA, K0_20 = 6.0, 90.0, 1e-4, 1.035, 9.66
H_M   = 300e-6          # m, locked biofilm thickness for the dynamic stage
ND    = 241             # grid cells through the depth
DT_H  = 0.2             # h, 12-minute sampling
HOURS = 36.0

def k0_of_T(T):  return K0_20 * THETA ** (T - 20.0)
def effectiveness(h, k0_T):
    phi = h * np.sqrt(k0_T / DF)
    return float(np.tanh(phi) / phi)

z  = np.linspace(0.0, D_FILT, ND)
dz = z[1] - z[0]

def _rhs(t, S, Si, Q, C):
    d = np.empty_like(S)
    d[0]  = (Si - S[0]) * 1e3                  # Dirichlet inlet by fast relaxation
    d[1:] = -Q * (S[1:] - S[:-1]) / dz - C * S[1:]
    return d

def _segment(S0, Si, Q, C, t0, t1, teval):
    sol = solve_ivp(_rhs, (t0, t1), S0, args=(Si, Q, C), t_eval=teval,
                    method="BDF", rtol=1e-8, atol=1e-10)
    return sol.t, sol.y[-1, :], sol.y[:, -1]

def simulate(Si0=200.0, Q0=1.0, T=20.0, rain_start=18.0, rain_dur=2.5,
             q_factor=1.25, si_factor=0.75, hours=HOURS):
    """Baseline, a rectangular rain event, then recovery. Integrated segment by segment."""
    C = effectiveness(H_M, k0_of_T(T)) * H_M * k0_of_T(T) * W_AREA
    edges = [0.0, rain_start, rain_start + rain_dur, hours]
    states = [(Si0, Q0), (Si0 * si_factor, Q0 * q_factor), (Si0, Q0)]
    S = Si0 * np.exp(-(C / Q0) * z)                     # start from steady profile
    T_all, Se_all, Si_all, Q_all = [], [], [], []
    for k in range(3):
        Si, Q = states[k]
        n = max(2, int(round((edges[k + 1] - edges[k]) / DT_H)) + 1)
        te = np.linspace(edges[k], edges[k + 1], n)
        tt, se, S = _segment(S, Si, Q, C, edges[k], edges[k + 1], te)
        sl = slice(1, None) if k else slice(None)
        T_all.append(tt[sl]); Se_all.append(se[sl])
        Si_all.append(np.full(tt[sl].shape, Si)); Q_all.append(np.full(tt[sl].shape, Q))
    return (np.concatenate(T_all), np.concatenate(Si_all),
            np.concatenate(Q_all), np.full(sum(len(x) for x in T_all), T),
            np.concatenate(Se_all))

def verify():
    """Cross-consistency: the dynamic model must reduce to the algebraic one."""
    C = effectiveness(H_M, k0_of_T(20.0)) * H_M * k0_of_T(20.0) * W_AREA
    S = np.full(ND, 200.0)
    _, se, _ = _segment(S, 200.0, 1.0, C, 0.0, 120.0, np.linspace(0, 120, 25))
    analytic = 200.0 * np.exp(-C * D_FILT / 1.0)
    print("verification: numerical %.3f  analytical %.3f  error %.3f mg/L (%.2f %%)"
          % (se[-1], analytic, abs(se[-1] - analytic), 100 * abs(se[-1] - analytic) / analytic))

def build(n_traj=60, seed=7, out="pde_trajectories.npz"):
    rng = np.random.default_rng(seed)
    X, M = [], []
    for _ in range(n_traj):
        Si0 = rng.uniform(120, 280); Q0 = rng.uniform(0.6, 1.6); T = rng.uniform(12, 26)
        st  = rng.uniform(14, 22);   du = rng.uniform(1.5, 4.0)
        t, Si, Q, Tv, Se = simulate(Si0, Q0, T, st, du)
        X.append(np.column_stack([Si, Q, Tv, Se]).astype(np.float32))
        M.append([st, du])
    L = min(len(a) for a in X)
    X = np.stack([a[:L] for a in X]); M = np.asarray(M, np.float32)
    np.savez_compressed(out, X=X, M=M, dt_h=DT_H)
    print("written %s : %d trajectories x %d steps at %.0f-min sampling"
          % (out, X.shape[0], X.shape[1], DT_H * 60))

if __name__ == "__main__":
    verify()
    build()
