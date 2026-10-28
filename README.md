# Physics-Informed Quantum Feature Regression

Nonlinear input–output modelling of the Silverbox benchmark using a simulated two-qubit feature map, classical regression, and an approximate physical equation. The experiment compares a model without physics against a model whose physics weight and checkpoint are selected using validation data under limited training-output availability.

This repository contains the implementation and mathematical derivations. It does not demonstrate quantum computational advantage or guarantee improvement on other systems.

## Contents

- `silverbox_models.py`: two classes, `CoefficientLeastSquares` and `QuantumSilverbox`.
- `Silverbox_Analysis.ipynb`: data loading, independent coefficient analysis, experiment execution, and result inspection.
- `requirements.txt`: Python dependencies, including Jupyter for opening the notebook.
- `silverbox.csv`: local input file; download it using the instructions below. It is not supplied by these code files.

Keep the Python module and notebook in the same directory.

## Installation

Python 3.11 is a suitable environment. From Windows Command Prompt:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m notebook
```

Dependencies are not version-pinned. Record `python --version` and `python -m pip freeze` when preparing a reproducible experimental release.

## Dataset

The Silverbox benchmark records an electronic system with Duffing-type dynamics. `V1` is the input voltage and `V2` is the measured output voltage. Preserve observation order.

- [Official benchmark description](https://www.nonlinearbenchmark.org/benchmarks/silverbox)
- [CSV mirror used by this project](https://raw.githubusercontent.com/biaslab/nsi-silverbox/master/data/Silverbox/SNLS80mV.csv)

Download and prepare the local file in Python:

```python
import pandas as pd

url = "https://raw.githubusercontent.com/biaslab/nsi-silverbox/master/data/Silverbox/SNLS80mV.csv"
df = pd.read_csv(url).loc[:, ["V1", "V2"]]
df.to_csv("silverbox.csv", index=False)
```

Input columns must contain finite numeric values. The supplied default uses sample-time units, `dt=1.0`; it does not assume one second between measurements. Coefficients depend on this time convention.

## Quick start

### Independent physical coefficient analysis

```python
import pandas as pd
from silverbox_models import CoefficientLeastSquares, QuantumSilverbox

df = pd.read_csv("silverbox.csv")
coefficients = CoefficientLeastSquares(df, number_of_rows=1000)
X = coefficients.fit()  # [a, b, c, K]
coefficients.plot()
```

The class copies the dataframe, computes centred differences, samples valid rows using seed 42, and fits the physical equation. Residual metrics use valid rows excluded from the sampled fitting rows. This is an exploratory equation-residual analysis, not an independent trajectory forecasting test. Adjacent derivative stencils can share measurements.

**These exploratory coefficients are not passed to QML.** Each limited-label QML experiment recalibrates its own physical coefficients using only revealed training measurements.

### Limited-label QML experiment

```python
qml = QuantumSilverbox(df, start=50000, count=4000)
results, validation, summary = qml.run_limited_label_experiment(
    fractions=(0.02, 0.05, 0.08, 0.10, 0.25, 1.0),
    seeds=(42,),
    weights=(0.0, 0.001, 0.01, 0.1),
    steps=10,
    plot_count=150,
)
```

- `start`: required zero-based starting position in the original dataframe.
- `count`: required number of consecutive observations.
- `history=96`: current input and 95 preceding inputs.
- `fractions`: fractions of eligible training outputs to reveal.
- `seeds`: label-block selection seeds. The original fitting code retains a fixed seed of 42 for sampling histories for metric estimation.
- `weights`: candidate physics penalties; must include zero for the baseline.
- `steps`: outer optimisation budget per candidate, plus evaluation of the initial checkpoint.
- `plot_count=150`: maximum number of test prediction points shown in each trajectory plot. If fewer are available, all available points are shown. It does not change metric calculation.

The constructor's `label_fraction` and `seed` configure that individual object. The experiment method constructs separate models for the `fractions` and `seeds` passed to the method, preserving the object's segment, history, and time increment.

Access outputs:

```python
print(results)
print(validation)
print(summary)
print(qml.predictions[(0.02, 42)].head())

results.to_csv("results_A.csv", index=False)
validation.to_csv("validation_A.csv", index=False)
```

`qml.fitted_models[(fraction, seed)]` contains the corresponding experiment object, no-physics model, and validation-selected model. The latter may itself have zero physics weight.

Trajectory plots are displayed and saved as PNG/PDF files in the current directory. Filenames include the fraction and seed but not segment boundaries; use different working directories when running configurations that could overwrite earlier plots.

### Second experimental configuration

```python
qml_B = QuantumSilverbox(df, start=8000, count=5000)
results_B, validation_B, summary_B = qml_B.run_limited_label_experiment(
    fractions=(0.02, 0.05, 0.08, 0.10, 0.25, 1.0),
    seeds=(50,),
    steps=10,
    plot_count=150,
)
```

## Protocol and information availability

Each segment is split chronologically into 60% training, 20% validation, and 20% test observations. Initial history-dependent and final neighbour-dependent indices are excluded separately within each partition.

| Configuration | Zero-based segment | Observations | Seed | Eligible train / validation / test |
|---|---|---:|---:|---|
| A | 50000:54000 | 4000 | 42 | 2303 / 703 / 703 |
| B | 8000:13000 | 5000 | 50 | 2903 / 903 / 903 |

For history length H, eligible indices begin at H within each partition and stop before its final observation, matching the supplied implementation. This discards H+1 prediction centres per partition; it is an implementation convention, not a universal requirement.

For each fraction, the rounded label budget is distributed over short contiguous blocks positioned randomly within training strata. The implementation requires at least 32 revealed labels and at least eight fully observed neighbouring triples. Different fractions need not reveal nested sets of measurements.

- All input measurements remain available.
- Hidden training outputs are replaced by NaN inside each experiment.
- A calibration centre is usable only if its previous, current, and next outputs are revealed.
- All measurements used for calibration count within the same training-label budget.
- Input and quantum-feature statistics use training inputs only.
- Output statistics use revealed training outputs only.
- Validation outputs select checkpoints and physics weights.
- Test outputs are used for final metrics, not candidate selection.

Thus, limited supervision refers to **training outputs**, not the total measurement acquisition budget including validation and testing.

## Mathematical model and derivations

### 1. Physical equation and centred differences

The approximate voltage equation is

$$
\ddot y+a\dot y+by+cy^3+Ku=0.
$$

Here u and y are input and output voltages; a, b, c, and K are fitted coefficients. They should not automatically be interpreted as literal mechanical mass, damping, or stiffness constants.

For a smooth trajectory, Taylor expansion gives

$$
y(t\pm h)=y(t)\pm h y'(t)+\frac{h^2}{2}y''(t)\pm\frac{h^3}{6}y'''(t)+\frac{h^4}{24}y^{(4)}(t)+O(h^5).
$$

Subtracting the expansions cancels the even-order terms, while adding them and subtracting twice y cancels the constant and odd-order terms. Consequently,

$$
D_1y_k=\frac{y_{k+1}-y_{k-1}}{2h}=y'(t_k)+O(h^2),
$$

$$
D_2y_k=\frac{y_{k+1}-2y_k+y_{k-1}}{h^2}=y''(t_k)+O(h^2).
$$

These are approximations for sufficiently smooth signals. Differentiation can amplify measurement noise.

### 2. Least-squares coefficient estimation

For every eligible calibration centre k, define

$$
A_k=[D_1y_k,\ y_k,\ y_k^3,\ u_k],\qquad B_k=-D_2y_k.
$$

Stack these rows into A and B, and let beta denote the column vector of coefficients:

$$
\beta=[a,b,c,K]^T,\qquad \hat\beta=\arg\min_\beta\|A\beta-B\|_2^2.
$$

Expanding and differentiating gives

$$
\nabla_\beta\|A\beta-B\|_2^2=2A^T(A\beta-B).
$$

At a minimiser the gradient is zero, so

$$
A^TA\hat\beta=A^TB.
$$

If A has full column rank, the unique solution is

$$
\hat\beta=(A^TA)^{-1}A^TB.
$$

The code uses `numpy.linalg.lstsq`, not explicit inversion. In the QML calibration branch, columns are first scaled by their RMS values. Writing S as the diagonal matrix of column scales, solve with A S^{-1} for gamma and recover beta = S^{-1} gamma. This changes numerical conditioning without changing the full-rank least-squares problem. The independent coefficient-analysis class preserves the original unscaled fit.

### 3. Input scaling and histories

Using the training-input mean and population standard deviation, define

$$
x_k=\tfrac12\tanh((u_k-\mu_u)/\sigma_u).
$$

Then x is in (-1/2, 1/2), so arcsin(x) is well defined. Each prediction uses

$$
h_k=[x_k,x_{k-1},\ldots,x_{k-H+1}],\qquad H=96.
$$

Measured output history is excluded from the predictor.

### 4. Two-qubit encoding

Each scalar history entry x is processed separately in the Hilbert space

$$
\mathcal H=\mathbb C^2\otimes\mathbb C^2\cong\mathbb C^4.
$$

The history is not a 192-qubit state. Two circuit parameters are shared across all scalar entries:

$$
p=\arcsin x+\theta_0x,\qquad q=\arcsin x+\theta_1x.
$$

The rotation and encoded state are

$$
R_y(v)=\begin{bmatrix}\cos(v/2)&-\sin(v/2)\\\sin(v/2)&\cos(v/2)\end{bmatrix},
$$


```math
\lvert \psi_\theta(x) \rangle = \mathrm{CNOT}\left(R_y(p)\otimes R_y(q)\right)\lvert 00 \rangle
```



With CNOT control qubit 0 and target qubit 1, in basis 00,01,10,11, this becomes

$$
|\psi\rangle=[AC,AD,BD,BC]^T,
$$

where A=cos(p/2), B=sin(p/2), C=cos(q/2), and D=sin(q/2). The gates are unitary, hence the state remains normalised.

### 5. Proof of the three observable features

For a normalised state and Hermitian observable O, the expectation is

$$
\langle O\rangle=\langle\psi|O|\psi\rangle.
$$

Using the Pauli matrices X and Z, the three features are

$$
\phi_1=\langle I\otimes X\rangle=2(AC\,AD+BD\,BC)=2CD=\sin q,
$$

$$
\phi_2=\langle Z\otimes X\rangle=2(AC\,AD-BD\,BC)=2CD(A^2-B^2)=\cos p\sin q,
$$

$$
\phi_3=\langle X\otimes I\rangle=2(AC\,BD+AD\,BC)=4ABCD=\sin p\sin q.
$$

The equalities use A^2+B^2=1 and the double-angle identities. At theta=0, these reduce to x, x sqrt(1-x^2), and x^2. They admit inexpensive classical evaluation; the quantum representation alone does not imply computational advantage.

### 6. Feature standardisation and prediction

Concatenate the three observables across H history positions into Phi_k of dimension d=3H=288. For feature column j, calculate fixed reference statistics over eligible training indices T at theta=0:

$$
m_j=\frac1{N_T}\sum_{k\in\mathcal T}\Phi_{k,j}(0),\qquad
s_j=\sqrt{\frac1{N_T}\sum_{k\in\mathcal T}(\Phi_{k,j}(0)-m_j)^2}.
$$

Standardise and add an intercept:

$$
z_{k,j}(\theta)=\frac{\Phi_{k,j}(\theta)-m_j}{s_j},\qquad F_k=[1,z_{k,1},\ldots,z_{k,d}].
$$

The feature standard deviation has a positive floor. These statistics are fixed throughout optimisation and reused for validation/test inputs.

Using only revealed training-output indices L, compute their mean mu_y and standard deviation sigma_y. Define

$$
t_k=(y_k-\mu_y)/\sigma_y,\qquad \hat y_k=\mu_y+\sigma_y F_k(\theta)w.
$$

The output is linear in w at fixed theta but nonlinear in the original input history.

### 7. Physics-informed objective

Let P contain eligible training physics centres, including unlabelled centres. With separately estimated physical coefficients,

$$
r_k=D_2\hat y_k+\hat a D_1\hat y_k+\hat b\hat y_k+\hat c\hat y_k^3+\hat K u_k.
$$

Only predicted outputs are needed to evaluate r. The reference acceleration scale is

$$
s_r=\sqrt{\frac1{N_C}\sum_{k\in\mathcal C}(D_2y_k)^2},
$$

where C is the measured calibration-centre set. Positive floors protect all required divisions. The loss is

$$
\mathcal L=\frac1{N_L}\sum_{k\in\mathcal L}(F_kw-t_k)^2+
\frac{\lambda_{\mathrm{phys}}}{N_P}\sum_{k\in\mathcal P}(r_k/s_r)^2+
\lambda_{\mathrm{reg}}\sum_{j=1}^{d}w_j^2.
$$

N_L, N_P, and N_C are the sizes of L, P, and C. The intercept w_0 is not regularised, and lambda_reg=10^{-6}. The physical equation is a soft penalty, not an exactly enforced equality.

### 8. Ridge solution without physics

For fixed theta and zero physics weight, write R=diag(0,1,...,1). The objective gradient is

$$
\frac{2}{N_L}F_L^T(F_Lw-t)+2\lambda_{\mathrm{reg}}Rw.
$$

Setting it to zero yields

$$
(F_L^TF_L+N_L\lambda_{\mathrm{reg}}R)w=F_L^Tt.
$$

The implementation solves this linear system directly.

### 9. Damped Gauss–Newton solution with physics

With positive physics weight the cubic prediction term makes the objective nonlinear in w. Stack the residuals

$$
e_{\mathrm{data},k}=(F_kw-t_k)/\sqrt{N_L},\qquad
e_{\mathrm{physics},k}=\sqrt{\lambda_{\mathrm{phys}}/N_P}\,r_k/s_r,\qquad
e_{\mathrm{reg},j}=\sqrt{\lambda_{\mathrm{reg}}}w_j.
$$

Their squared norm equals the objective. Let J be their Jacobian with respect to w. Linearising e gives e(w+Delta w) approximately e+J Delta w. Minimising the damped local least-squares expression gives

$$
\nabla_{\Delta w}\big(\|e+J\Delta w\|^2+\varepsilon\|\Delta w\|^2\big)
=2J^Te+2(J^TJ+\varepsilon I)\Delta w=0.
$$

Therefore,

$$
(J^TJ+\varepsilon I)\Delta w=-J^Te.
$$

The physical-residual Jacobian follows by the chain rule:

$$
\frac{\partial r_k}{\partial w}=\sigma_y[D_2F_k+\hat aD_1F_k+(\hat b+3\hat c\hat y_k^2)F_k].
$$

Finite differences apply to neighbouring feature rows. The forcing term has zero weight derivative. The code uses damping 10^{-8}, up to ten readout updates per fitting call, and backtracking factors 1, 0.5, 0.25, 0.125, 0.0625. A step is accepted only when the actual training objective decreases.

### 10. Circuit gradient and state-space metric

At fixed w, loss derivatives are approximated with a central parameter perturbation epsilon_theta=10^{-5}:

$$
b_j\approx\frac{\mathcal L(\theta+\epsilon_\theta e_j,w)-\mathcal L(\theta-\epsilon_\theta e_j,w)}{2\epsilon_\theta}.
$$

Here e_j is a coordinate unit vector; b is the circuit-loss gradient, not the physical coefficient b.

The Fubini–Study metric is

$$
g_{ij}=\operatorname{Re}[\langle\partial_i\psi|\partial_j\psi\rangle-\langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle].
$$

For the one-qubit state r(p)=[cos(p/2),sin(p/2)]^T, its angle derivative is [-sin(p/2),cos(p/2)]^T/2. This derivative has squared norm 1/4 and zero overlap with r. As dp/dtheta_0=x and dq/dtheta_1=x, each diagonal metric entry is x^2/4. The product-state cross terms vanish, and the parameter-independent CNOT preserves inner products. Thus

$$
g(x)=\frac{x^2}{4}I_2,\qquad G=\kappa I_2,
$$

where kappa averages x^2/4 over the sampled histories and their lag positions. The code estimates state derivatives numerically and samples up to 32 eligible training histories.

### 11. Geometric update and acceptance

The local model is

$$
Q(\Delta\theta)=b^T\Delta\theta+\frac1{2\eta}\Delta\theta^T(G+\delta I)\Delta\theta.
$$

Differentiation and setting the gradient to zero give

$$
b+\frac1\eta(G+\delta I)\Delta\theta=0,
\qquad \Delta\theta=-\eta(G+\delta I)^{-1}b.
$$

Since G is positive semidefinite and delta>0, the local quadratic has a unique minimiser. For this circuit,

$$
\Delta\theta=-\frac{\eta}{\kappa+\delta}b.
$$

Thus geometry changes the overall scale, not the gradient direction. This quadratic term is a metric penalty, not the exact loss Hessian.

The ratio of actual reduction to predicted reduction is

$$
\rho=\frac{\mathcal L(\theta,w)-\mathcal L(\theta+\Delta\theta,w)}{-Q(\Delta\theta)}.
$$

The code starts with eta=0.1 and damping delta=0.01. It accepts finite ratios above 0.1, adjusts damping based on agreement, and allows up to eight attempts. This is a damped geometric step with a trust-ratio test, not an explicitly radius-constrained trust-region subproblem.

### 12. Model selection and evaluation

Readout and circuit parameters are updated alternately. For each candidate physics weight, retain the checkpoint with lowest validation RMSE. Compare the zero-physics candidate against the candidate with lowest validation RMSE across all weights.

Metrics are calculated on all eligible test points:

$$
\operatorname{RMSE}=\sqrt{\frac1N\sum_k(\hat y_k-y_k)^2},\qquad
\operatorname{MAE}=\frac1N\sum_k|\hat y_k-y_k|,
$$

$$
R^2=1-\frac{\sum_k(\hat y_k-y_k)^2}{\sum_k(y_k-\bar y)^2}.
$$

RMSE and MAE are in volts. Negative R^2 means squared error exceeds the constant test-mean reference in this definition. With one seed, the summary standard deviation is NaN because variability cannot be estimated from one run.

The derivations above justify the formulas implemented; they are not proofs of global convergence, universal generalisation, or quantum advantage.

## Reported experiments

The following values were supplied from the development experiments; they are not newly rerun benchmark measurements in this repository refactor.

| Revealed outputs | A: QML RMSE | A: selected RMSE | B: QML RMSE | B: selected RMSE |
|---|---:|---:|---:|---:|
| 2% | 0.041952 | 0.018336 | 0.036709 | 0.011672 |
| 5% | 0.032828 | 0.013434 | 0.034252 | 0.023529 |
| 8% | 0.043426 | 0.012164 | 0.030175 | 0.023737 |
| 10% | 0.024108 | 0.012837 | 0.058927 | 0.019417 |
| 25% | 0.011650 | 0.011205 | 0.012708 | 0.012462 |
| 100% | 0.008768 | 0.008768 | 0.008339 | 0.008340 |

Configuration A selected physics weight 0.1 throughout. Configuration B selected 0.1, 0.001, 0.1, 0.1, 0.1, and 0.01 in the order above. Both configurations improved in all tested limited-label conditions; full-supervision performance was essentially unchanged. Segment, length, and seed changed together, so their separate effects cannot be identified from this comparison.

## Limitations

- Two configurations of one benchmark are not evidence of performance on every system.
- There is no matched classical nonlinear model with the same physics penalty in this experiment.
- Exact classical state-vector simulation excludes finite-shot and device noise.
- The metric is isotropic for this circuit, so directional geometric benefits are not established.
- The approximate physical equation and numerical derivatives may be inaccurate for other conditions.
- Very short segments or small label budgets cannot support the history and calibration requirements. Reducing plot_count does not reduce the amount of data needed for training.
- Internal `check()` verifies state normalisation, metric positivity, and a residual-Jacobian column; it is not a complete proof of implementation correctness.

## References

- Wigren, T., and Schoukens, J. (2013). *Three free data sets for development and benchmarking in nonlinear system identification*. ECC, 2933–2938. https://doi.org/10.23919/ECC.2013.6669201
- Raissi, M., Perdikaris, P., and Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics, 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045
- Schuld, M., and Killoran, N. (2019). *Quantum Machine Learning in Feature Hilbert Spaces*. Physical Review Letters, 122, 040504. https://doi.org/10.1103/PhysRevLett.122.040504
- Stokes, J., Izaac, J., Killoran, N., and Carleo, G. (2020). *Quantum Natural Gradient*. Quantum, 4, 269. https://doi.org/10.22331/q-2020-05-25-269
- Du, Y., et al. (2025). *Quantum Machine Learning: A Hands-on Tutorial for Machine Learning Practitioners and Researchers*. https://arxiv.org/abs/2502.01146
- Alwan, N. A., Obaiys, S. J., Al-Saidi, N. M. G., and Chadli, R. (2027). *A Trust Optimization Region of the Quantum Natural Gradient for the Large Damping Domain*. ICCSA 2026 Workshops, LNCS 16758, 284–300. https://doi.org/10.1007/978-3-032-30497-1_18

## Attribution and licensing

Consult the accompanying paper for the final author list and contribution statement. No software licence is selected automatically by this project documentation. Dataset availability does not imply permission to redistribute it; consult the benchmark provider's terms.
