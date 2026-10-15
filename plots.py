"""
Figures for Chapter 4, generated from the saved results.

Run after ml_state_recovery.py and lstm_dynamic.py:
    python plots.py

Produces
    fig_accuracy.png        R2 by target and model            (bar chart)
    fig_clouds.png          predicted vs true, XGBoost        (4 panels)
    fig_learning_curve.png  accuracy vs training-set size
    fig_learning_speed.png  training time vs training-set size
    fig_lstm.png            LSTM vs PDE on held-out data

R2 values are annotated to 3 decimal places, matching the tables in the text.
"""
import numpy as np, pandas as pd, time, warnings, matplotlib
warnings.filterwarnings("ignore"); matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from ml_state_recovery import make, split, FEATURES, TARGETS, SEED

plt.rcParams.update({"savefig.dpi":300,"font.family":"DejaVu Sans","axes.grid":True,
                     "grid.alpha":.3,"axes.axisbelow":True,"font.size":11})
C={"SVR":"#3498db","Random Forest":"#2ecc71","XGBoost":"#e74c3c"}
MODELS=list(C)

# ---------- 1. accuracy bar chart ----------
def fig_accuracy(res):
    lab={"lumped f*h*k0(T)":"f·h·k₀(T)\n(lumped)","f":"f","h":"h","k0(T)":"k₀(T)"}
    order=["lumped f*h*k0(T)","f","h","k0(T)"]
    piv=res.pivot(index="target",columns="model",values="R2")
    fig,ax=plt.subplots(figsize=(10,5.3)); x=np.arange(4); w=.25
    for k,m in enumerate(MODELS):
        v=[piv.loc[t,m] for t in order]
        b=ax.bar(x+(k-1)*w,v,w,label=m,color=C[m],edgecolor="white")
        for bb,val in zip(b,v):
            ax.text(bb.get_x()+bb.get_width()/2,val+.008,"%.3f"%val,ha="center",fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([lab[t] for t in order])
    ax.set_ylim(0,1.07); ax.set_ylabel("Test R² (higher is better)")
    ax.set_title("Recovery of the lumped product and the individual states\n"
                 "(k₀,₂₀ = 9.66 h⁻¹ fixed; f derived from the Thiele relationship)",fontsize=12)
    ax.legend(loc="lower right",fontsize=9.5)
    fig.tight_layout(); fig.savefig("fig_accuracy.png",bbox_inches="tight",facecolor="white"); plt.close()

# ---------- 2. predicted vs true ----------
def fig_clouds(df,tr,te):
    info=[("lumped f*h*k0(T)","lump_fhk0","f·h·k₀(T)  (m h⁻¹)",1.0),
          ("f","f","f  (–)",1.0),("h","h_m","h  (µm)",1e6),("k0(T)","k0_T","k₀(T)  (h⁻¹)",1.0)]
    Xtr,Xte=df.iloc[tr][FEATURES].values,df.iloc[te][FEATURES].values
    fig,axs=plt.subplots(1,4,figsize=(16,4.1))
    for a,(lbl,col,nm,sc) in zip(axs,info):
        ytr,yte=df.iloc[tr][col].values,df.iloc[te][col].values
        sy=StandardScaler().fit(ytr.reshape(-1,1))
        m=make("XGBoost"); m.fit(Xtr,sy.transform(ytr.reshape(-1,1)).ravel())
        p=sy.inverse_transform(m.predict(Xte).reshape(-1,1)).ravel()
        a.scatter(yte*sc,p*sc,s=6,alpha=.3,color=C["XGBoost"])
        lo,hi=(yte*sc).min(),(yte*sc).max(); a.plot([lo,hi],[lo,hi],"k--",lw=1.5)
        a.set_xlabel("True "+nm); a.set_ylabel("Predicted")
        a.set_title("%s\nR² = %.3f"%(nm.split("(")[0].strip(),r2_score(yte,p)),
                    fontsize=11,fontweight="bold")
    fig.suptitle("XGBoost: predicted against true on the held-out test set",
                 fontsize=12.5,fontweight="bold",y=1.03)
    fig.tight_layout(); fig.savefig("fig_clouds.png",bbox_inches="tight",facecolor="white"); plt.close()

# ---------- 3 & 4. learning curve and learning speed ----------
def fig_learning(df,tr,te,sizes=(800,2000,4000,6000,7000)):
    Xte,yte=df.iloc[te][FEATURES].values,df.iloc[te]["lump_fhk0"].values
    acc={m:[] for m in MODELS}; spd={m:[] for m in MODELS}
    for n in sizes:
        sub=tr[:n]; X=df.iloc[sub][FEATURES].values; y=df.iloc[sub]["lump_fhk0"].values
        sy=StandardScaler().fit(y.reshape(-1,1))
        for m in MODELS:
            mdl=make(m); t0=time.perf_counter()
            mdl.fit(X,sy.transform(y.reshape(-1,1)).ravel()); dt=time.perf_counter()-t0
            p=sy.inverse_transform(mdl.predict(Xte).reshape(-1,1)).ravel()
            acc[m].append(r2_score(yte,p)); spd[m].append(dt)
    fig,ax=plt.subplots(figsize=(9,5.2))
    for m in MODELS: ax.plot(sizes,acc[m],"o-",color=C[m],lw=2.2,ms=7,label=m)
    ax.set_xlabel("Number of training samples"); ax.set_ylabel("Test R² (lumped product)")
    ax.set_title("Learning curve — accuracy against training-set size",fontsize=12)
    ax.legend(); fig.tight_layout()
    fig.savefig("fig_learning_curve.png",bbox_inches="tight",facecolor="white"); plt.close()
    fig,ax=plt.subplots(figsize=(9,5.2))
    for m in MODELS: ax.plot(sizes,spd[m],"s-",color=C[m],lw=2.2,ms=7,label=m)
    ax.set_xlabel("Number of training samples"); ax.set_ylabel("Training time (s)")
    ax.set_title("Learning speed — training time against training-set size",fontsize=12)
    ax.legend(); fig.tight_layout()
    fig.savefig("fig_learning_speed.png",bbox_inches="tight",facecolor="white"); plt.close()
    for m in MODELS:
        print("  %-14s R² %.3f -> %.3f   time %.2f s at %d samples"
              %(m,acc[m][0],acc[m][-1],spd[m][-1],sizes[-1]))

# ---------- 5. LSTM ----------
def fig_lstm(npz="lstm_predictions.npz"):
    import os
    if not os.path.exists(npz): print("  (skipped: run lstm_dynamic.py first)"); return
    d=np.load(npz); pred,true,pers=d["pred"],d["true"],d["persistence"]
    fig,ax=plt.subplots(1,2,figsize=(13,4.6))
    n=min(240,len(true)); t=np.arange(n)*0.2
    ax[0].plot(t,true[:n],color="#2c3e50",lw=2.6,label="PDE truth")
    ax[0].plot(t,pred[:n],color="#e74c3c",lw=1.9,ls="--",label="LSTM prediction")
    ax[0].set_xlabel("Time (h)"); ax[0].set_ylabel("$S_e$ (mg/L)")
    ax[0].set_title("Held-out trajectory segment",fontsize=11,fontweight="bold"); ax[0].legend(fontsize=9)
    ax[1].scatter(true,pred,s=5,alpha=.25,color="#e74c3c")
    lo,hi=true.min(),true.max(); ax[1].plot([lo,hi],[lo,hi],"k--",lw=1.4)
    ax[1].set_xlabel("True $S_e$ (mg/L)"); ax[1].set_ylabel("Predicted")
    ax[1].set_title("LSTM R² = %.3f   ·   persistence R² = %.3f"
                    %(r2_score(true,pred),r2_score(true,pers)),fontsize=11,fontweight="bold")
    fig.tight_layout(); fig.savefig("fig_lstm.png",bbox_inches="tight",facecolor="white"); plt.close()

if __name__=="__main__":
    df=pd.read_csv("steady_state_dataset.csv")
    tr,va,te=split(len(df))
    res=pd.read_csv("ml_recovery_results.csv")
    print("accuracy chart");   fig_accuracy(res)
    print("prediction clouds");fig_clouds(df,tr,te)
    print("learning curve and speed"); fig_learning(df,tr,te)
    print("lstm figure");      fig_lstm()
    print("\nwritten: fig_accuracy.png fig_clouds.png fig_learning_curve.png "
          "fig_learning_speed.png fig_lstm.png")
