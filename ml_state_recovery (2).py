"""
Stage 1 — recover the internal biofilm states from boundary observations.

Inputs  : Si, Se, Q, T          (the four quantities a works measures)
Targets : lumped f*h*k0(T), and f, h, k0(T) individually
Models  : SVR (RBF), Random Forest, XGBoost
Split   : 70 / 20 / 10, fixed seed; all metrics on the held-out test set.

Targets are standardised on the TRAINING partition only and predictions are
returned to physical units before scoring, because h (~1e-4 m) and k0 (~10 1/h)
differ by orders of magnitude.

Requires steady_state_dataset.csv from steady_state_generator.py
"""
import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

SEED = 42
FEATURES = ["Si", "Se", "Q", "T"]
TARGETS  = {"lumped f*h*k0(T)": "lump_fhk0", "f": "f", "h": "h_m", "k0(T)": "k0_T"}

def make(name):
    if name == "SVR":
        return Pipeline([("sc", StandardScaler()),
                         ("m", SVR(kernel="rbf", C=10, gamma="scale", epsilon=1e-3))])
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    return XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.9,
                        random_state=SEED, n_jobs=-1)

def split(n, seed=SEED):
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(.7 * n), int(.9 * n)
    return idx[:a], idx[a:b], idx[b:]

def run(csv="steady_state_dataset.csv"):
    df = pd.read_csv(csv)
    tr, va, te = split(len(df))
    Xtr, Xte = df.iloc[tr][FEATURES].values, df.iloc[te][FEATURES].values
    rows = []
    for label, col in TARGETS.items():
        ytr, yte = df.iloc[tr][col].values, df.iloc[te][col].values
        sy = StandardScaler().fit(ytr.reshape(-1, 1))          # fit on train only
        for name in ["SVR", "Random Forest", "XGBoost"]:
            mdl = make(name)
            t0 = time.perf_counter()
            mdl.fit(Xtr, sy.transform(ytr.reshape(-1, 1)).ravel())
            fit_s = time.perf_counter() - t0
            pred = sy.inverse_transform(mdl.predict(Xte).reshape(-1, 1)).ravel()   # descale
            rows.append(dict(target=label, model=name,
                             R2=r2_score(yte, pred),
                             RMSE=np.sqrt(mean_squared_error(yte, pred)),
                             MAE=mean_absolute_error(yte, pred),
                             train_s=fit_s))
    res = pd.DataFrame(rows)
    res.to_csv("ml_recovery_results.csv", index=False)
    print(res.pivot(index="target", columns="model", values="R2").round(3).to_string())
    print("\nwritten ml_recovery_results.csv")
    return res

if __name__ == "__main__":
    run()
