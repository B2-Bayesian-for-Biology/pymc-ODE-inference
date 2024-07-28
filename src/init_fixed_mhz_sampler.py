import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import pymc as pm
from numba import njit
import pytensor.tensor as pt
import arviz as az
from pytensor.compile.ops import as_op




## all functions.

def cells_grow(y,t,mu_max,Ks,Qn):
    N, P = y
    dydt = [0,0]
    dydt[0] = -Qn*(mu_max*N)/(N+Ks) *(P*1e6)
    dydt[1] = P*(mu_max*N)/(N+Ks)

    return dydt

def solved_num_cells(y0,t,mu_max,Ks,Qn):
    sol = odeint(cells_grow, y0, t, args=(mu_max,Ks,Qn))
    return sol[:,1]








# 1

data = pd.read_csv("./../data/in_silico_growth_curve.csv")
plt.plot(data['times'],data['cells'],'o',color ='orange')
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])


#2

t = np.linspace(0, 15, 100)
mu_max = 0.6
Ks = 0.09
Qn = 6.7e-10
y0 = [ 6e2, data['cells'][0] ]
num_cells_solved = solved_num_cells(y0,t,mu_max,Ks,Qn)

plt.plot(t,num_cells_solved)
plt.plot(data['times'],data['cells'],'o')
plt.yscale('log')
plt.legend(['Simulation','real data'])
plt.xlabel('Time')
plt.ylabel('cells')
plt.show()



@njit
def cells_grow_pymc(y,t,theta):
    # unpack parameters
    N, P = y
    mu_max,Ks,Qn = theta
    
    dp_dt = -Qn*(mu_max*N)/(N+Ks) *(P*1e6)
    dq_dt = P*(mu_max*N)/(N+Ks)

    return [dp_dt, dq_dt]


# decorator with input and output types a Pytensor double float tensors
@as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
def pytensor_forward_model_matrix(theta):
    return odeint(func = cells_grow_pymc, y0=[ 6e2, data['cells'][0] ], t=data['times'], args=(theta,))

def plot_model_trace(ax, trace_df, row_idx, lw=1, alpha=0.2):
    cols = ["mu_max", "Ks", "Qn"]
    row = trace_df.iloc[row_idx, :][cols].values

    time = data['times']
    theta = row
    x_y = odeint(func= cells_grow_pymc, y0 = [ 6e2, data['cells'][0] ], t=time, args=(theta,))
    plot_model(ax, x_y, time=time, lw=lw, alpha=alpha);


def plot_inference(
    ax,
    trace,
    num_samples=200,
    title="Cell growth Bayesian fit",
    plot_model_kwargs=dict(lw=1, alpha=0.2),
):
    trace_df = az.extract(trace, num_samples=num_samples).to_dataframe()
    plot_data(ax, lw=0)
    for row_idx in range(num_samples):
        plot_model_trace(ax, trace_df, row_idx, **plot_model_kwargs)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(title, fontsize=16)


# plot data function for reuse later
def plot_data(ax, lw=2, title="cells growth"):
    ax.plot(data['times'], data['cells'], color="k", lw=lw, marker="o", markersize=12, label="data)")
    ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Cells", fontsize=14)
    ax.set_title(title, fontsize=16)
    return ax



# plot model function
def plot_model(
    ax,
    x_y,
    time=data['times'],
    alpha=1,
    lw=3,
    title="bayesian",
):
    ax.plot(time, x_y[:, 1], color="b", alpha=alpha, lw=lw, label="Bayesian")
    ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(title, fontsize=16)
    ax.set_yscale('log')
    return ax






#3


with pm.Model() as model_nuts_sampler:
    # Priors
    mu_max = pm.Normal('mu_max', mu=0.6, sigma=0.06)
    Ks = pm.TruncatedNormal('Ks', mu=0.09, sigma=0.01,  lower=0.05, upper=0.13) 
    #I had to use the truncation, else the proposal samples would go to infeasible values. (Ks<0) and there will be no ODE solutions.
    
    Qn = pm.Lognormal('Qn', mu=np.log(6.7e-10), sigma=1.0)
    sigma = pm.HalfNormal("sigma", 10)

    # Ode solution function
    ode_solution = pytensor_forward_model_matrix(
        pm.math.stack([mu_max,Ks,Qn])
    )

    # Likelihood
    pm.Normal("Y_obs", mu=np.log(ode_solution[:,1]), sigma=sigma, observed= np.log(data['cells'].values))


tune = draws = 1500
with model_nuts_sampler:
    trace_pymc_ode = pm.sample(2000, tune=8000,step=pm.DEMetropolisZ(), return_inferencedata=True)



# 4
#df_trace = pm.trace_to_dataframe(trace_pymc_ode)
df_trace = az.convert_to_inference_data(obj=trace_pymc_ode).to_dataframe(include_coords=False,groups='posterior')
df_trace.to_csv('./../results/trace_pymc_ode_init_fixed_mhz_sampler.csv', index=False)


# 5


az.plot_trace(trace_pymc_ode,compact='False',var_names=("Ks"))
plt.show()

az.plot_trace(trace_pymc_ode,compact='False',var_names=("mu_max"))
plt.show()

az.plot_trace(trace_pymc_ode,compact='False',var_names=("Qn"))
plt.show()

az.plot_trace(trace_pymc_ode,compact='False',var_names=("sigma"))
plt.show()


# 6
az.plot_pair(trace_pymc_ode, kind='kde', divergences=True, marginals=True)
plt.show()




# 7 Convergence test for the chains - Gelman-Rubin, Geweke, and Autocorrelation
rhat = az.rhat(trace_pymc_ode)
#geweke = az.geweke(trace_pymc_ode)
autocorr = az.plot_autocorr(trace_pymc_ode)
print(f'Rhat:\n{rhat}\n')
plt.show()

# 8
summary = az.summary(trace_pymc_ode)
print(summary)

# 9
burn_in_length = tune  # Assuming 'tune' variable represents burn-in length


#10
fig, ax = plt.subplots(figsize=(12, 4))
plot_inference(ax, trace_pymc_ode,num_samples=200);
plt.show()





