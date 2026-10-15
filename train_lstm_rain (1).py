"""
Stage 2/3 - dynamic rainfall experiment.

Generates PDE trajectories for normal operation and for rainfall disturbances,
trains an LSTM to follow the PDE effluent response, and evaluates it on a
rainfall event that was never presented during training.

Rainfall scenario:   Si x 0.75  (dilution)   and   Q x 1.25  (extra flow)

Architecture and optimiser follow Table 3.3 of the dissertation: 48 hidden units,
output head 48 -> 24 -> 1, AdamW at 8e-4 with weight decay 1e-4, batch 2048,
Smooth L1 loss.  Trajectories are sampled at 12 minutes.

HISTORY WINDOW.  The window must span the hydraulic residence time.  Rain at
17.8-20.3 h produces an effluent minimum at 26.3 h, because the diluted water
needs D/Q hours to cross the bed.  A 3 h window cannot see the cause of that
drop, so the window is 45 samples (9 h).

Splits are by COMPLETE trajectory, never by time point, and the normal/stress
proportion is preserved in each partition.

Outputs: lstm_rain.pt, lstm_rain_predictions.csv, lstm_rain_metrics.csv
"""
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pde_characteristics import simulate

SEED      = 42
HISTORY   = 45          # 45 samples x 12 min = 9 h, one hydraulic residence time
DT        = 0.2         # h, 12-minute sampling
HOURS     = 38.0
EPOCHS    = 40
N_TRAIN, N_VAL, N_TEST = 300, 50, 50
STRESS_FRACTION        = 0.80
Q_FACTOR, SI_FACTOR    = 1.25, 0.75

torch.manual_seed(SEED); np.random.seed(SEED)
FEATURES = ["Influent_Si_mgL", "Hydraulic_Loading_Q_mh", "Temperature_C", "Effluent_Se_mgL"]

class SeLSTM(nn.Module):
    def __init__(self, n_in=4, hidden=48):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 24), nn.ReLU(), nn.Linear(24, 1))
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1, :]).squeeze(-1)

def one_event(rng, stress):
    kw = dict(Si0=rng.uniform(160, 250), Q0=rng.uniform(0.8, 1.3), T=20.0)
    if stress:
        kw.update(rain_start=rng.uniform(6, 26), rain_dur=rng.uniform(0.8, 6.0),
                  q_factor=Q_FACTOR, si_factor=SI_FACTOR)
    else:
        kw.update(rain_start=1e6, rain_dur=0.0, q_factor=1.0, si_factor=1.0)
    t, Si, Q, T, Se = simulate(dt=DT, hours=HOURS, **kw)
    return t, np.column_stack([Si, Q, T, Se]).astype(np.float32)

def build_dataset(seed=SEED):
    rng = np.random.default_rng(seed); rows = []
    plan = [("train", N_TRAIN), ("val", N_VAL), ("test", N_TEST)]
    for tag, n in plan:
        for i in range(n):
            stress = rng.random() < STRESS_FRACTION
            if tag == "test" and i == 0:                 # canonical event for the figure
                t, a = one_event(rng, True)
                t, Si, Q, T, Se = simulate(dt=DT, hours=HOURS, Si0=200.0, Q0=1.0, T=20.0,
                                           rain_start=17.8, rain_dur=2.5,
                                           q_factor=Q_FACTOR, si_factor=SI_FACTOR)
                a = np.column_stack([Si, Q, T, Se]).astype(np.float32)
            else:
                t, a = one_event(rng, stress)
            rows.append(pd.DataFrame({"Event_ID": f"{tag}_{i:02d}", "Time_Hours": t,
                                      FEATURES[0]: a[:, 0], FEATURES[1]: a[:, 1],
                                      FEATURES[2]: a[:, 2], FEATURES[3]: a[:, 3],
                                      "Stress": bool(stress) or (tag == "test" and i == 0)}))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv("pde_rain_dataset.csv", index=False)
    print("dataset: %d events (%d train / %d val / %d test), %.0f %% stress, %d rows"
          % (df.Event_ID.nunique(), N_TRAIN, N_VAL, N_TEST,
             100*df.groupby("Event_ID").Stress.first().mean(), len(df)))
    return df

def make_xy(df, tag):
    ids = sorted(e for e in df.Event_ID.unique() if e.startswith(tag))
    X, y, meta = [], [], []
    for eid in ids:
        g = df[df.Event_ID == eid].sort_values("Time_Hours")
        a = g[FEATURES].to_numpy(np.float32); t = g.Time_Hours.to_numpy()
        for j in range(HISTORY, len(a)):
            X.append(a[j-HISTORY:j]); y.append(a[j, 3]); meta.append((eid, t[j]))
    return np.asarray(X, np.float32), np.asarray(y, np.float32), meta

def main():
    df = build_dataset()
    Xtr, ytr, _        = make_xy(df, "train")
    Xva, yva, _        = make_xy(df, "val")
    Xte, yte, meta_te  = make_xy(df, "test")
    print("windows: train %d, val %d, test %d" % (len(Xtr), len(Xva), len(Xte)))

    mu = Xtr.reshape(-1, 4).mean(0); sd = Xtr.reshape(-1, 4).std(0) + 1e-8
    ym, ys = ytr.mean(), ytr.std() + 1e-8            # scaling fitted on TRAIN only
    nx = lambda a: (a - mu) / sd

    net = SeLSTM()
    opt = torch.optim.AdamW(net.parameters(), lr=8e-4, weight_decay=1e-4)
    lossf = nn.SmoothL1Loss()
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.from_numpy(nx(Xtr)), torch.from_numpy((ytr - ym)/ys)),
        batch_size=2048, shuffle=True)
    Xv = torch.from_numpy(nx(Xva))

    best, best_state, hist = np.inf, None, []
    for ep in range(1, EPOCHS+1):
        net.train()
        for xb, yb in dl:
            opt.zero_grad(); l = lossf(net(xb), yb); l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        with torch.no_grad(): pv = net(Xv).numpy()*ys + ym
        mse = float(np.mean((pv - yva)**2)); hist.append((ep, mse))
        if mse < best:
            best, best_state, best_ep = mse, {k: v.clone() for k, v in net.state_dict().items()}, ep
        if ep % 10 == 0: print("  epoch %2d  validation MSE %.5f" % (ep, mse))
    net.load_state_dict(best_state); net.eval()       # best epoch on validation
    with torch.no_grad(): pred = net(torch.from_numpy(nx(Xte))).numpy()*ys + ym

    out = pd.DataFrame(meta_te, columns=["Event_ID", "Time_Hours"])
    out["PDE_Se_mgL"] = yte; out["LSTM_Se_mgL"] = pred
    out.to_csv("lstm_rain_predictions.csv", index=False)

    ev = out[out.Event_ID == "test_00"]
    rows = [dict(scope="all test events", R2=r2_score(yte, pred),
                 RMSE=np.sqrt(mean_squared_error(yte, pred)), MAE=mean_absolute_error(yte, pred)),
            dict(scope="canonical rain event", R2=r2_score(ev.PDE_Se_mgL, ev.LSTM_Se_mgL),
                 RMSE=np.sqrt(mean_squared_error(ev.PDE_Se_mgL, ev.LSTM_Se_mgL)),
                 MAE=mean_absolute_error(ev.PDE_Se_mgL, ev.LSTM_Se_mgL)),
            dict(scope="persistence baseline (all)", R2=r2_score(yte[1:], yte[:-1]),
                 RMSE=np.sqrt(mean_squared_error(yte[1:], yte[:-1])),
                 MAE=mean_absolute_error(yte[1:], yte[:-1]))]
    met = pd.DataFrame(rows); met["best_epoch"] = best_ep
    met.to_csv("lstm_rain_metrics.csv", index=False)
    print("\nbest epoch %d" % best_ep); print(met.to_string(index=False))
    torch.save(net.state_dict(), "lstm_rain.pt")
    print("\nwritten lstm_rain.pt, lstm_rain_predictions.csv, lstm_rain_metrics.csv")

if __name__ == "__main__":
    main()
