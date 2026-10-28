import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

class CoefficientLeastSquares:
    """Original centred-difference least-squares analysis (1,000 sampled rows)."""

    def __init__(self, df, number_of_rows=1000):
        if not isinstance(number_of_rows, int) or isinstance(number_of_rows, bool) or number_of_rows < 4:
            raise ValueError("number_of_rows must be an integer of at least four.")
        self.df = df.copy()
        self.number_of_rows = number_of_rows

    def fit(self):
        """Estimate a, b, c, K and evaluate residuals outside the fitting rows."""
        df = self.df
        dt = 1.0
        df["V2_speed"] = (df["V2"].shift(-1) - df["V2"].shift(1)) / (2 * dt)
        df["V2_acceleration"] = (
            df["V2"].shift(-1) - 2 * df["V2"] + df["V2"].shift(1)
        ) / dt**2
        df["V2_3"] = df["V2"] ** 3
        valid = df.dropna(subset=["V2_speed", "V2_acceleration", "V2", "V2_3", "V1"])
        if len(valid) < self.number_of_rows:
            raise ValueError("number_of_rows exceeds the available valid derivative rows.")
        sample = valid.sample(n=self.number_of_rows, random_state=42)
        A = sample[["V2_speed", "V2", "V2_3", "V1"]].to_numpy()
        B = -sample["V2_acceleration"].to_numpy()
        X = np.linalg.lstsq(A, B, rcond=None)[0]
        print("a, b, c, K:", X)
        df["calculated"] = (
            df["V2_acceleration"] + X[0] * df["V2_speed"]
            + X[1] * df["V2"] + X[2] * df["V2_3"] + X[3] * df["V1"]
        )
        valid = df.loc[valid.index.difference(sample.index), ["V2_acceleration", "calculated"]]
        self.residual_rms = np.sqrt((valid["calculated"] ** 2).mean())
        acceleration_rms = np.sqrt((valid["V2_acceleration"] ** 2).mean())
        self.relative_residual = (
            100 * self.residual_rms / acceleration_rms if acceleration_rms > 0 else np.nan
        )
        print(f"Residual RMS: {self.residual_rms:.6g}")
        print(f"Relative residual: {self.relative_residual:.2f}%")
        self.X, self.sample = X, sample
        return X

    def plot(self):
        """Plot the equation residual calculated by fit()."""
        if not hasattr(self, "X"):
            raise RuntimeError("Call fit() before plot().")
        self.df["calculated"].plot(figsize=(12, 4))
        plt.xlabel("Sample index")
        plt.ylabel("Calculated value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class QuantumSilverbox:
    def __init__(self, df, start, count, history=96,
                 dt=1.0, label_fraction=1.0, seed=42):
        self.start, self.h, self.dt = start, history, dt
        self.count = count
        self.df = df
        data = df[['V1', 'V2']].iloc[start:start+count].copy()
        if len(data) != count or not np.isfinite(data.to_numpy()).all():
            raise ValueError('Selected segment unavailable or contains invalid data.')
        self.u = data.V1.to_numpy(dtype=float)
        self.y = data.V2.to_numpy(dtype=float).copy()
        n=len(data); tr=int(.6*n); va=int(.8*n)
        self.train=np.arange(history,tr-1)
        self.valid=np.arange(tr+history,va-1)
        self.test=np.arange(va+history,n-1)
        if not 0 < label_fraction <= 1:
            raise ValueError('label_fraction must be in (0,1].')
        budget=int(round(label_fraction*len(self.train)))
        if budget < 32:
            raise ValueError('At least 32 labelled samples required for calibration.')
        rng=np.random.default_rng(seed)
        # Distributed short contiguous blocks retain adjacent observations
        # for estimating derivatives WITHOUT using hidden measurements.
        if budget == len(self.train):
            self.labelled=self.train.copy()
        else:
            block_count=min(8,max(1,budget//16))
            strata=np.array_split(self.train,block_count)
            allocations=np.full(block_count,budget//block_count,dtype=int)
            allocations[:budget % block_count]+=1
            pieces=[]
            for stratum,length in zip(strata,allocations):
                offset=int(rng.integers(0,len(stratum)-length+1))
                pieces.append(stratum[offset:offset+length])
            self.labelled=np.concatenate(pieces)
        # Hide unavailable training outputs explicitly. Held-out segments
        # remain available for VALIDATION and final TEST evaluation only.
        visible=np.zeros(n,dtype=bool);visible[self.labelled]=True
        self.y[:tr]=np.where(visible[:tr],self.y[:tr],np.nan)
        um,us=self.u[:tr].mean(),max(self.u[:tr].std(),1e-12)
        x=.5*np.tanh((self.u-um)/us)
        self.inputs=np.zeros((n,history))
        for lag in range(history):self.inputs[lag:,lag]=x[:n-lag]
        self.ym=self.y[self.labelled].mean()
        self.ys=max(self.y[self.labelled].std(),1e-12)
        # Every measured y used for physics calibration must be labelled.
        i=self.train[visible[self.train-1] & visible[self.train]
                     & visible[self.train+1]]
        self.calibration_indices=i.copy()
        if len(i)<8:raise ValueError('Too few fully observed triples.')
        v=(self.y[i+1]-self.y[i-1])/(2*dt)
        acc=(self.y[i+1]-2*self.y[i]+self.y[i-1])/dt**2
        A=np.column_stack([v,self.y[i],self.y[i]**3,self.u[i]])
        scale=np.maximum(np.sqrt(np.mean(A*A,axis=0)),1e-12)
        self.coef=np.linalg.lstsq(A/scale,-acc,rcond=None)[0]/scale
        self.rs=max(np.sqrt(np.mean(acc**2)),1e-12)
        raw=self.measure(np.zeros(2))
        self.fm=raw[self.train].mean(0)
        self.fs=np.maximum(raw[self.train].std(0),1e-8)
        self.reg=1e-6

    def state(self, theta, inputs):
        x=inputs.reshape(-1)
        angle=np.arcsin(x)
        p=angle+theta[0]*x
        q=angle+theta[1]*x
        phase=np.zeros_like(x)
        c0,s0=np.cos(p/2),np.sin(p/2)
        c1,s1=np.cos(q/2),np.sin(q/2)
        # RY(p)|0> tensor RY(q)|0>, then RZ(phase) on wire 0.
        state=np.column_stack([c0*c1,c0*s1,s0*c1,s0*s1]).astype(complex)
        state[:,:2]*=np.exp(-.5j*phase)[:,None]
        state[:,2:]*=np.exp(.5j*phase)[:,None]
        # Entangle with CNOT(control=0,target=1).
        return state[:,[0,1,3,2]]

    def measure(self, theta):
        s=self.state(theta,self.inputs)
        # All features below are expectation values of Pauli observables.
        X1=2*np.real(s[:,0].conj()*s[:,1]+s[:,2].conj()*s[:,3])
        Z0X1=2*np.real(s[:,0].conj()*s[:,1]-s[:,2].conj()*s[:,3])
        X0=2*np.real(s[:,0].conj()*s[:,2]+s[:,1].conj()*s[:,3])
        n=len(self.y)
        return np.concatenate([z.reshape(n,self.h) for z in [X1,Z0X1,X0]],axis=1)

    def design(self, theta):
        f=(self.measure(theta)-self.fm)/self.fs
        return np.column_stack([np.ones(len(f)),f])

    def residual(self,w,F,indices,weight,jac=False):
        # Supervision: ONLY the revealed output labels.
        j=self.labelled
        Cdata=F[j]
        pred_data=self.ym+self.ys*(Cdata@w)
        data_res=(pred_data-self.y[j])/self.ys/np.sqrt(len(j))
        # Physics: all eligible TRAINING time points, labelled or not.
        # This branch never reads measured y at these time points.
        i=indices;a,b,c,K=self.coef;dt=self.dt
        C=F[i];D1=(F[i+1]-F[i-1])/(2*dt)
        D2=(F[i+1]-2*C+F[i-1])/dt**2
        pred=self.ym+self.ys*(C@w)
        r=(self.ys*(D2@w)+a*self.ys*(D1@w)+b*pred
           +c*pred**3+K*self.u[i])/self.rs
        residual=np.concatenate([data_res,np.sqrt(weight/len(i))*r,
                                  np.sqrt(self.reg)*w[1:]])
        if not jac:return residual
        Jr=self.ys*(D2+a*D1+(b+3*c*pred**2)[:,None]*C)/self.rs
        R=np.zeros((len(w)-1,len(w)))
        R[:,1:]=np.eye(len(w)-1)*np.sqrt(self.reg)
        J=np.vstack([Cdata/np.sqrt(len(j)),np.sqrt(weight/len(i))*Jr,R])
        return residual,J

    def fit_head(self,F,weight,w=None):
        C=F[self.labelled]; target=(self.y[self.labelled]-self.ym)/self.ys
        R=np.eye(C.shape[1])*len(self.labelled)*self.reg; R[0,0]=0
        if w is None: w=solve(C.T@C+R,C.T@target,assume_a='pos')
        if weight==0: return solve(C.T@C+R,C.T@target,assume_a='pos')
        # Damped Gauss--Newton updates of the classical readout only.
        for _ in range(10):
            r,J=self.residual(w,F,self.train,weight,jac=True)
            H=J.T@J+1e-8*np.eye(len(w))
            delta=solve(H,-J.T@r,assume_a='pos')
            old=r@r
            for alpha in [1.,.5,.25,.125,.0625]:
                candidate=w+alpha*delta
                rr=self.residual(candidate,F,self.train,weight)
                if rr@rr < old:
                    w=candidate
                    break
            else: break
            if np.linalg.norm(alpha*delta)<1e-7: break
        return w

    def metric(self,theta,inputs,eps=1e-5):
        psi=self.state(theta,inputs)
        deriv=np.empty((len(psi),4,2),complex)
        for j in range(2):
            e=np.eye(2)[j]*eps
            deriv[:,:,j]=(self.state(theta+e,inputs)-self.state(theta-e,inputs))/(2*eps)
        first=np.einsum('bsi,bsj->bij',deriv.conj(),deriv)
        overlap=np.einsum('bs,bsi->bi',psi.conj(),deriv)
        second=np.einsum('bi,bj->bij',overlap.conj(),overlap)
        G=np.real(first-second).mean(0);G=(G+G.T)/2
        ev,V=np.linalg.eigh(G)
        return (V*np.maximum(ev,0))@V.T

    def loss(self,theta,w,weight):
        r=self.residual(w,self.design(theta),self.train,weight)
        return r@r

    def fit(self,weight=0.,steps=10):
        theta=np.zeros(2); damping=.01; eta=.1;eps=1e-5
        F=self.design(theta);w=self.fit_head(F,weight)
        best=None; bestval=np.inf;records=[]
        rng=np.random.default_rng(42)
        subset=rng.choice(self.train,min(32,len(self.train)),replace=False)
        for step in range(steps+1):
            F=self.design(theta);w=self.fit_head(F,weight,w)
            trainpred=self.ym+self.ys*F[self.labelled]@w
            valpred=self.ym+self.ys*F[self.valid]@w
            trainrmse=np.sqrt(np.mean((trainpred-self.y[self.labelled])**2))
            valrmse=np.sqrt(np.mean((valpred-self.y[self.valid])**2))
            records.append([step,trainrmse,valrmse,damping,*theta])
            if valrmse<bestval:
                bestval=valrmse;best=(theta.copy(),w.copy())
            if step%5==0:print(f'physics={weight:g} step={step:02d} train={trainrmse:.6f} val={valrmse:.6f} theta={theta}',flush=True)
            if step==steps:break
            grad=np.empty(2)
            for j in range(2):
                e=np.eye(2)[j]*eps
                grad[j]=(self.loss(theta+e,w,weight)-self.loss(theta-e,w,weight))/(2*eps)
            G=self.metric(theta,self.inputs[subset]);old=self.loss(theta,w,weight)
            for attempt in range(8):
                M=G+damping*np.eye(2);delta=np.linalg.solve(M,-eta*grad)
                expected=-(grad@delta+delta@M@delta/(2*eta))
                if expected<=1e-15:break
                actual=old-self.loss(theta+delta,w,weight);rho=actual/expected
                if np.isfinite(rho) and rho>.1:
                    theta+=delta
                    if rho>.75:damping=max(damping/2,1e-7)
                    elif rho<.25:damping=min(damping*2,1e5)
                    break
                damping=min(damping*4,1e5)
        return {'theta':best[0],'weights':best[1],'physics_weight':weight,
                'validation_rmse':bestval,'history':pd.DataFrame(records,columns=['step','train_rmse','validation_rmse','damping','theta0','theta1'])}

    def predict(self,model,indices):
        return self.ym+self.ys*self.design(model['theta'])[indices]@model['weights']

    def check(self):
        t=np.array([.1,-.2]);s=self.state(t,self.inputs[self.train[:3]])
        assert np.allclose(np.sum(abs(s)**2,axis=1),1,atol=1e-12)
        G=self.metric(t,self.inputs[self.train[:3]])
        assert np.linalg.eigvalsh(G).min()>-1e-10
        F=self.design(t);w=self.fit_head(F,0.)
        i=self.train[:8];r,J=self.residual(w,F,i,.01,True)
        e=np.zeros_like(w);e[5]=1e-6
        numerical=(self.residual(w+e,F,i,.01)-self.residual(w-e,F,i,.01))/(2e-6)
        assert np.allclose(numerical,J[:,5],atol=1e-6,rtol=1e-5)


    @threadpool_limits.wrap(limits=2)
    def run_limited_label_experiment(self, fractions=(.05,.10,.25,1.0),
            seeds=(42,), weights=(0.,.001,.01,.1), steps=10, plot_count=150):
        """Fixed full-label validation/test segments; budget applies to TRAINING
        outputs only. Calibration observations count within that same budget.
        For a final study repeat seeds, tune comparable regularisation budgets,
        add classical nonlinear baselines and use an independent final record.
        """
        if 0.0 not in weights:raise ValueError('Include 0.0 as the no-physics baseline.')
        if not isinstance(plot_count, int) or isinstance(plot_count, bool) or plot_count <= 0:
            raise ValueError("plot_count must be a positive integer.")
        rows=[];validation_rows=[]
        self.predictions = {}
        self.fitted_models = {}
        for fraction in fractions:
            for seed in seeds:
                q=QuantumSilverbox(self.df, start=self.start, count=self.count,
                    history=self.h, dt=self.dt, label_fraction=fraction, seed=seed)
                q.check()
                print(f'\nLABEL FRACTION={fraction:g}; seed={seed}; '
                      f'labels={len(q.labelled)}; calibration triples={len(q.calibration_indices)}')
                candidates=[q.fit(weight=w,steps=steps) for w in weights]
                for model in candidates:
                    validation_rows.append({'fraction':fraction,'seed':seed,
                        'physics_weight':model['physics_weight'],
                        'validation_rmse':model['validation_rmse']})
                # Selection uses VALIDATION ONLY. Physics is not forced to win.
                base=next(m for m in candidates if m['physics_weight']==0.)
                selected=min(candidates,key=lambda m:m['validation_rmse'])
                            # Same fixed test section for every label fraction.
                sl = slice(0, min(plot_count, len(q.test)))

                self.fitted_models[(fraction, seed)] = {"experiment": q, "baseline": base, "selected": selected}
                self.predictions[(fraction, seed)] = pd.DataFrame({
                    "sample_index": q.start + q.test, "V2_actual": q.y[q.test],
                    "QML without physics": q.predict(base, q.test),
                    "Validation-selected QML": q.predict(selected, q.test)
                })
                sample_index = (q.start + q.test)[sl]
                actual = q.y[q.test][sl]
                without_physics = q.predict(base, q.test)[sl]
                with_physics = q.predict(selected, q.test)[sl]

                fig, ax = plt.subplots(figsize=(7, 3.5))

                ax.plot(
                    sample_index, actual,
                    color="black", linewidth=1.8,
                    label="Actual"
                )
                ax.plot(
                    sample_index, without_physics,
                    color="#0072B2", linestyle="--", linewidth=1.4,
                    label="QML without physics"
                )
                ax.plot(
                    sample_index, with_physics,
                    color="#D55E00", linestyle=":", linewidth=1.6,
                    label=f"QML (physics weight={selected['physics_weight']:g})"
                )

                ax.set_title(f"{fraction:.0%} training labels")
                ax.set_xlabel("Sample index")
                ax.set_ylabel("Output voltage V2 (V)")
                ax.legend(frameon=False, fontsize=9)
                ax.grid(alpha=0.2)
                fig.tight_layout()

                filename = f"prediction_{int(round(fraction * 100))}pct_seed{seed}"
                fig.savefig(f"{filename}.pdf", bbox_inches="tight")
                fig.savefig(
                    f"{filename}.png", dpi=600, bbox_inches="tight"
                )
                plt.show()
                for name,model in [('QML without physics',base),
                                   ('Validation-selected QML',selected)]:
                    pred=q.predict(model,q.test);actual=q.y[q.test]
                    err=pred-actual;den=np.sum((actual-actual.mean())**2)
                    rows.append({'fraction':fraction,'seed':seed,'model':name,
                        'training_labels':len(q.labelled),
                        'calibration_triples':len(q.calibration_indices),
                        'selected_physics_weight':model['physics_weight'],
                        'RMSE':np.sqrt(np.mean(err**2)),
                        'MAE':np.mean(abs(err)),
                        'R2':1-np.sum(err**2)/den if den>0 else np.nan})
        results=pd.DataFrame(rows);validation=pd.DataFrame(validation_rows)
        summary=results.groupby(['fraction','model']).agg(
            mean_RMSE=('RMSE','mean'),std_RMSE=('RMSE','std'),
            mean_R2=('R2','mean')).reset_index()
        print('\nRESULTS\n',results.to_string(index=False))
        print('\nSUMMARY\n',summary.to_string(index=False))
        fig,ax=plt.subplots(figsize=(9,4))
        for name in summary.model.unique():
            part=summary[summary.model==name]
            ax.plot(100*part.fraction,part.mean_RMSE,'o-',label=name)
        ax.set(xlabel='Revealed training outputs (%)',ylabel='Mean test RMSE')
        ax.legend();ax.grid(alpha=.3);fig.tight_layout();plt.show()
        self.results, self.validation, self.summary = results, validation, summary
        return results,validation,summary

