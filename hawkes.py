import numpy as np
import math

from typing import List, Callable, Tuple

import itertools

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from scipy import linalg
from scipy.optimize import minimize
import multiprocessing

def exp_intensity(j, mu, gamma, alpha, beta, history, t):
	I = mu[j] + gamma[j] * t
	for i, ts in history:
		I += alpha[i, j] * math.exp(-beta[i, j] * (t - ts))
	return I

def MHP(
    d: int,
    intensity: Callable[[int, List[List[float]], float], float],
    T: float, numEvents=1000) -> List[List[float]]:
    
    dim = d
    t = 0.0
    n = 0
    history = list()

    if numEvents == None:
        numEvents = np.iinfo(int).max

    while (n < numEvents) & (t <= T):
        tau_n = np.zeros(dim)
        for j in range(dim):
            r = 0
            tau = t
            intens  = intensity(j, history, tau)
            while True:
                E = np.random.exponential()
                tau = tau + E / intens

                U = np.random.uniform()
                u = U * intens

                intens  = intensity(j, history, tau)
                if u < intens:
                    tau_n[j] = tau
                    break
        dn = tau_n.argmin()
        tn = tau_n[dn]

        history.append((dn, tn))
        n += 1
        t = tn

    if (history[-1][1] > T):
         history = history[:-1]
    return history

def discreize_trajectory(trajectory: List[Tuple[int, float]], discreete_group):
    d = max([_elem[0] for _elem in trajectory]) + 1
    
    hawkes_df = pd.DataFrame(trajectory, columns=["type", "t"])
    hawkes_df['delta'] = pd.cut(hawkes_df["t"], discreete_group)
    
    hs_x = hawkes_df.groupby(["type", "delta"]).count()
    hs_x = hs_x.reset_index(drop=False).sort_values(by=["delta", "type"]).reset_index(drop=True)


    dfX = pd.DataFrame(discreete_group, columns=["group"])
    for typ in hs_x["type"].unique():
        dfX[f"{typ}"] = hs_x.loc[hs_x["type"] == typ, "t"].to_numpy()

    return dfX

def branching_coeff_estimation(trajectory: List[Tuple[int, float]], t_max: float, delta: float, p: int):
    dim = max([_elem[0] for _elem in trajectory]) + 1

    intervals = pd.interval_range(start=0, end=t_max, freq=delta)
    dfX = discreize_trajectory(trajectory, intervals)

    # zdyskretyzowana trajektoria to_numpy
    X = dfX[[f"{i}" for i in range(dim)]].reset_index(drop=True).to_numpy().transpose()
    # print(X)
    n = X.shape[1]
    X_mat = X

    Z = np.vstack([
        np.concatenate([
            X_mat[:,[_idx]].transpose().tolist()[0] for _idx in np.arange(i, p+i)][::-1] + [[1]]) 
        for i in range(n-p)])
    
    Y = np.hstack(
        [X_mat[:,[_idx]] for _idx in np.arange(p,n)]
    ).transpose()

    est = np.linalg.inv(Z.transpose() @ Z) @ (Z.transpose()) @ Y / delta


    A = np.hsplit(est.transpose(), np.cumsum([dim for _ in range(p)]))
    Ai = A[:-1]
    a0 = A[-1]

    return Ai, a0

def exp_intensity_mean_approximation(d, tmax, mu, gamma, alpha, beta):
    u_t = [lambda t, fixed_i = i: mu[fixed_i] + gamma[fixed_i] * t for i in range(d)]
    
    def _B(t):
        b = np.zeros((d+1, d))
        b[0,:] = [_u_t(t) for _u_t in u_t]

        for _a in range(d):
            for _b in range(d):
                b[_a+1, _b] = alpha[_b][_a] * u_t[_b](t)
        return b.flatten(order='C')

    def _A():
        a = np.zeros((d*(d+1), d*(d+1)))
        for m in range(d):
            a[m, (m+1)*d:(m+2)*d] = 1
        for _a in range(1, d+1):
            for _b in range(1, d+1):
                m = _a * d + _b
                if _a != _b:
                    a[m-1,m-1] = -beta[_b-1][_a-1]
                    for n in range(_b*d+1, _b*d+d+1):
                        if n != m:
                            a[m-1,n-1] = alpha[_b-1][_a-1]
                else:
                    a[m-1,m-1] = alpha[_a-1][_a-1]-beta[_a-1][_a-1]
                    for n in range(_b*d+1, _b*d+d+1):
                        if n != m:
                            a[m-1,n-1] = alpha[_a-1][_a-1]
        return a
        
    A_m = _A()
    G0 = _B(0)

    def G(t):
        return G0 @ linalg.expm(A_m * t) + sum([linalg.expm(A_m * (t - s / 2.0)) @ _B(s / 2.0) for s in range(int(2 * t))]) / 2.0    

    return [G(t) for t in np.linspace(0, tmax, 20)]

def gradient_estimation(d, tmax, trajectory):
    def logL_r(x, d, tmax, t):
        mu, alpha, beta = np.split(x, np.cumsum([d, d**2]))
        alpha = np.reshape(alpha, (d, d))
        beta = np.reshape(beta, (d, d))

        logL_res = 0.0
        for m in range(d):
            logL_m = -mu[m] * tmax - sum([
                alpha[m,n] / beta[m,n] * sum([
                    (1 - math.exp(-beta[m,n] * (tmax - tn_k)))
                    for tn_k in t[n]
                ])
                for n in range(d)
            ]) + sum([
                math.log(
                    mu[m] + sum([
                        alpha[m,n] * sum([
                            math.exp(-beta[m,n] * (tm_k - tn_i)) for tn_i in t[n] if tn_i < tm_k
                        ])
                        for n in range(d)
                    ])
                )
                for tm_k in t[m]
            ])

            logL_res += logL_m

        return logL_res 


    def logL_grad(x, d, tmax, t):
        mu, alpha, beta = np.split(x, np.cumsum([d, d**2]))
        alpha = np.reshape(alpha, (d, d))
        beta = np.reshape(beta, (d, d))

        Dmu = np.zeros(d)
        Dalpha = np.zeros((d,d))
        Dbeta = np.zeros((d,d))

        def R(_m,_n, _tm_k):
            return sum([
                math.exp(-beta[_m,_n] * (_tm_k - tn_i)) for tn_i in t[_n] if tn_i < _tm_k
            ])

        def dR(_m,_n, _tm_k):
            return sum([
                (_tm_k - tn_i) * math.exp(-beta[_m,_n] * (_tm_k - tn_i)) for tn_i in t[_n] if tn_i < _tm_k
            ])

        # po mu:
        for m in range(d):
            dL_m_dMu_m = -tmax + sum([1.0 / (mu[m] + sum([
                alpha[m,j] * R(m,j,tm_k)
                for j in range(d)
            ])) for tm_k in t[m]])

            Dmu[m] = dL_m_dMu_m

        # po alpha mn:
        for m in range(d):
            for n in range(d):
                dLm_dAmn = -1.0 / beta[m,n] * sum([(1 - math.exp(-beta[m,n]*(tmax - tn_k))) for tn_k in t[n]]) + sum([
                    R(m,n,tm_k) / (mu[m]  + sum([
                        alpha[m,j] * R(m,j, tm_k) 
                        for j in range(d)
                    ])) for tm_k in t[m]
                ])

                Dalpha[m,n] += dLm_dAmn
        
        # po beta mn:
        for m in range(d):
            for n in range(d):
                dLm_dBmn = (alpha[m,n] / beta[m,n]**2) * sum([
                    (1 - math.exp(-beta[m,n] * (tmax - tn_k)))
                    for tn_k in t[n]
                ]) - (alpha[m,n] / beta[m,n]) * sum([
                    (tmax - tn_k) * ((tmax - tn_k) * math.exp(-beta[m,n] * (tmax - tn_k)))
                    for tn_k in t[n]
                ]) - sum(
                    alpha[m,n] * dR(m,n,tm_k) / (mu[m] + sum([alpha[m,j] * R(m,j, tm_k) for j in range(d)]))
                    for tm_k in t[m]
                )

                Dbeta[m,n] += dLm_dBmn

        return np.hstack([
            Dmu, Dalpha.flatten(), Dbeta.flatten()
        ])
    
    # OPTYMALISATION:
    t = [[] for _ in range(d)]
    for i_n, t_n in trajectory:
        t[i_n].append(t_n)

    N = d + 2 * d**2
    x0 = np.full(N, 1.0)

    bds = [(1.0e-15, 10.0)]*d + [(0.0, 10.0)]*d**2 + [(1.0e-15, 10.0)]*d**2
    x0 = np.full(N, 2.0)
    res = minimize(
        lambda x, args: -logL_r(x, *args) + 0.0 * np.sum(x[d:d+d**2]),
        args=((d, tmax, t),),
        x0=x0, 
        bounds=bds, 
        method='SLSQP',
        options={'maxiter': 100},
        jac=lambda x, args: -logL_grad(x, *args))
    
    return res

def PRM(intensity : Callable[[float], float], intensity_max : float, tmax: float) -> List[float]:
    t = 0
    T = []

    while t < tmax:
        U1 = np.random.uniform(0,1)
        t = t - (math.log(U1) / intensity_max)
        
        if t > tmax:
            break;
        
        U2 = np.random.uniform(0,1)
        if intensity_max * U2 <= intensity(t):
            T.append(t)

    return T

def generate_family(
    h : List[List[Callable[[float], float]]],
    max_h : List[List[float]],
    tmax : float, type_ : int):
    
    d = len(h)
    assert(np.all([len(l) == d for l in h]))
    assert(np.all([len(l) == d for l in max_h]))
    assert(len(max_h) == d)
    assert(0 <= type_ <= d - 1)

    G_i0 = []
    G_i0g = [[] for _ in range(d)]
    g = 0
    # dodaj w t = 0 zdarzenie klasy type_
    G_i0g[type_].append(0.0)
    G_i0.append(G_i0g)

    while np.any([len(obj) > 0 for obj in G_i0[g]]):

        g += 1
        G_i0g = [[] for _ in range(d)]

        for j in range(d):
            for i in range(d):
                if max_h[i][j] == None or max_h[i][j] <= 0.0 or h[i][j] == None:
                    continue

                for idx in range(len(G_i0[g-1][i])):
                    old_event = G_i0[g-1][i][idx]
                    new_events = PRM(h[i][j], max_h[i][j], tmax=tmax-old_event)
                    
                    for new_ev in new_events:
                        G_i0g[j].append(old_event + new_ev)

        G_i0.append(G_i0g)

    return G_i0

def generate_hawkes_by_branching(
    eta : List[float], 
    h : List[List[Callable[[float], float]]],
    max_h : List[List[float]],
    tmax : float
    ):
    
    d = len(eta)

    res = []

    for i0 in range(d):
        if eta[i0] <= 0.0:
            continue

        I_i0 = PRM(lambda t: eta[i0], eta[i0], tmax=tmax)
        for t0 in I_i0:
            family = generate_family(h, max_h, tmax - t0, i0)
            # przesun rodzine o czas t0..
            for g in range(len(family)):
                for j in range(d):
                    for idx in range(len(family[g][j])):
                        family[g][j][idx] += t0

            res.append((i0,family))

    return res

def family_to_jump_times(trajectory, d):
    jump_times = []

    for f in trajectory:
        family = f[1]

        for g in range(len(family)):
                for j in range(d):
                    for idx in range(len(family[g][j])):
                        jump_times.append((j, family[g][j][idx]))

    jump_times.sort(key=lambda _val: _val[1])
    return jump_times

if __name__ == '__main__':
    
    def generate_and_show_1d_traj():
        # generate 1d traj:
        mu = np.array([1.0])
        gamma = np.array([0.0])
        alpha = np.array([[.3]])
        beta = np.array([[.85]])

        traj_1d = MHP(1, lambda i, hist, t: exp_intensity(i,mu, gamma, alpha, beta, hist, t), 45.0)

        print(f"Done generating hawkes. Traj len: {len(traj_1d)}")

        traj_1d_int = list(map(
            lambda t: exp_intensity(0, mu, gamma, alpha, beta, list(filter(lambda el: el[1] < t, traj_1d)), t),
            np.linspace(0, traj_1d[-1][1], 1000)
        ))

        jumps = list(map(lambda x: x[1], traj_1d))

        # show 1d traj:
        fig, ax = plt.subplots()
        fig.set_size_inches(w=14, h=8)

        ax.grid(which='major', visible=True, linestyle='--')
        ax.set_ylabel("$N_t$", color='tab:blue', fontsize=20)

        ax2 = ax.twinx()

        ax.plot(jumps, np.arange(len(jumps)), zorder=3, linewidth=3.5)
        ax2.plot(np.linspace(0, traj_1d[-1][1], 1000), traj_1d_int, color='k', zorder=2, linewidth=1.0, linestyle="--")

        ax.tick_params(axis='y', labelcolor='tab:blue', labelsize=15)
        ax2.tick_params(axis='y', labelcolor='black', labelsize=15)
        ax2.set_ylabel("$\lambda_t$", fontsize=20)
        ax.set_xlabel("$t$", fontsize=20)

        plt.show()

    # generate_and_show_1d_traj()

    def make_intensity_traj(j, intensity, history, tmax, density = 1000):
        traj_intens = list(map(
            lambda t: intensity(j, list(filter(lambda el: el[1] < t, history)), t),
            np.linspace(0, tmax, density)
        ))

        return traj_intens
    
    def generate_and_show_2d_traj():
        reproduction = {
            (0, 0): lambda t: .5 * np.exp(-1.5 * t),
            (1, 0): lambda t: 0.0,
            (0, 1): lambda t: 5**4/(3*2) * np.exp(-5*t)*t**(3),
            (1, 1): lambda t: .1 * np.exp(-1.0 * t)
        }

        mu = np.array([.5, 0.5])
        gamma = np.array([0.0, 0.0])

        def intensity2d(j, history, t):
            I = mu[j] + gamma[j] * t
            for i, ts in history:
                I += reproduction[(i,j)](t - ts)
            return I
        
        dim = 2
        t_max = 400.0
        traj_2d = MHP(dim, lambda i, hist, t: intensity2d(i, hist, t), t_max)
        
        jumps_dim_agg = [[] for _ in range(dim)]
        intensity_dim_agg = [[] for _ in range(dim)]

        hs_df = pd.DataFrame(traj_2d, columns=["type", "t"])
        for _i in range(dim):
            jumps_dim_agg[_i] = hs_df.loc[hs_df["type"] == _i, "t"].to_numpy()
            intensity_dim_agg[_i] = make_intensity_traj(_i, intensity2d, traj_2d, t_max)


        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(w=14, h=8)

        ax.grid(which='major', visible=True, linestyle='--')
        ax.set_ylabel("$N^{(i)}_t$", fontsize=20)

        ax2 = ax.twinx()

        for _i in range(dim):
            hs_i = jumps_dim_agg[_i] 
            ax.plot([0.]+list(hs_i)+[t_max], np.arange(0, len(hs_i)+2), zorder=3, linewidth=2.5, label=f"i = {_i}")
            ax2.plot(np.linspace(0, t_max, 1000), intensity_dim_agg[_i], zorder=2, linewidth=1.0, linestyle="--")

        ax.legend()
        ax.tick_params(axis='y', labelsize=15)
        ax2.tick_params(axis='y', labelsize=15)
        ax2.set_ylabel("$\lambda^{(i)}_t$", fontsize=20)
        ax.set_xlabel("$t$", fontsize=20)

        plt.show()

    # generate_and_show_2d_traj()
        
    def branching_coeff_estym_example():
        reproduction = {
            (0, 0): lambda t: .5 * np.exp(-1.5 * t),
            (1, 0): lambda t: 0.0,
            (0, 1): lambda t: 5**4/(3*2) * np.exp(-5*t)*t**(3),
            (1, 1): lambda t: .1 * np.exp(-1.0 * t)
        }

        mu = np.array([.5, 0.5])
        gamma = np.array([0.0, 0.0])

        def intensity2d(j, history, t):
            I = mu[j] + gamma[j] * t
            for i, ts in history:
                I += reproduction[(i,j)](t - ts)
            return I
        
        dim = 2
        t_max = 1000.0
        traj_2d = MHP(dim, lambda i, hist, t: intensity2d(i, hist, t), t_max, 10000)
        delta = 0.05
        p = 40
        s = p * delta

        Ai, a0 = branching_coeff_estimation(traj_2d, t_max, delta, p)
        
        fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0, wspace=0)
        fig.set_size_inches(w=14, h=8)
        fig.suptitle("Pointwise estimation for reproduction intensities", fontsize=20, y=0.92)

        for i,j in itertools.product(list(range(dim)), list(range(dim))):
            s = p * delta

            x = np.linspace(0, s, 1000)
            y = list(map(reproduction[(i,j)], x))

            ax[i,j].plot(x, y, color='royalblue')

            for k in range(p):
                x = k / p * s
                y = Ai[k][j,i]
                ax[i,j].plot(x, y, marker='x', color='.1')

        plt.show()
        
    # branching_coeff_estym_example()

    def expotential_mean_approx_example():
        dim = 2
        mu = np.array([0.1, 0.6])
        gamma = np.array([0.01, -0.004])
        alpha = np.array([
            [0.4, 0.0],
            [0.0, 0.5]])
        beta = np.array([
            [5.0, 0.0],
            [0.0, 3.0]
        ])
        t_max = 100.0
        NUMBER_TRAJ = 20

        exp_mean = exp_intensity_mean_approximation(dim, t_max, mu, gamma, alpha, beta)

        trajectories = []
        for n in range(NUMBER_TRAJ):
            traj_2d = MHP(dim, lambda i, hist, t: exp_intensity(i,mu, gamma, alpha, beta, hist, t), t_max)
            trajectories.append(traj_2d)

        fig, ax = plt.subplots(2,1, sharex=True)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        fig.set_size_inches(w=14, h=8)

        empirical_mean_x = np.linspace(0, t_max, 1000)
        empirical_mean_y = [[0] * len(empirical_mean_x) for _ in range(dim)]

        for traj_2d in trajectories:
            hs_df = pd.DataFrame(traj_2d, columns=["type", "t"])
            jumps_dim_agg = [[] for _ in range(dim)]
            for _i in range(dim):
                jumps_dim_agg[_i] = hs_df.loc[hs_df["type"] == _i, "t"].to_numpy()

            for _i in range(dim):
                hs_i = jumps_dim_agg[_i]
                ax[_i].plot([0.]+list(hs_i)+[t_max], np.arange(0, len(hs_i)+2), zorder=3, linewidth=.5, linestyle=':', color='.5')

            for _i in range(dim):
                hs_i = jumps_dim_agg[_i]
                for x_idx in range(len(empirical_mean_x)):
                    empirical_mean_y[_i][x_idx] += len(list(filter(lambda t: t <= empirical_mean_x[x_idx], hs_i)))
        
        for _i in range(dim):
            for _j in range(len(empirical_mean_y[_i])):
                empirical_mean_y[_i][_j] /= NUMBER_TRAJ

        for _i in range(dim):
            mu = [exp_mean[n][_i] for n in range(len(exp_mean))]
            t_ = list(np.array(range(len(mu)))).copy()
            t_ = t_ / max(t_) * t_max
            ax[_i].plot(t_, mu, c='b', linewidth=1.2)
        
        for _i in range(dim):
            ax[_i].grid()
            ax[_i].text(.9, .9, f"i = {_i+1}", transform=ax[_i].transAxes)
            ax[_i].plot(empirical_mean_x, empirical_mean_y[_i], c='r', linewidth=1.2)

        plt.legend(handles=[
            Line2D([0], [0], c='b', linewidth=1.2, label="empirical mean"),
            Line2D([0], [0], c='r', linewidth=1.2, label="theoretical mean"),
            Line2D([0], [0], linewidth=.5, linestyle=':', color='.5', label="trajectories"),
        ])
        plt.show()

    expotential_mean_approx_example()

    def parameters_logL_estimate_example_1d():
         # generate 1d traj:
        d = 1
        mu = np.array([1.0])
        gamma = np.array([0.0])
        alpha = np.array([[.3]])
        beta = np.array([[.85]])
        tmax = 300.0

        traj_1d = MHP(d, lambda i, hist, t: exp_intensity(i,mu, gamma, alpha, beta, hist, t), tmax)

        res = gradient_estimation(d, tmax, traj_1d)

        print(res)
        e_mu, e_alpha, e_beta = np.split(res.x, np.cumsum([d, d**2]))
        e_alpha.reshape((d,d))
        e_beta.reshape((d,d))

        print(f"REAL ARGS:\n\t mu={mu}\n\t alpha={alpha} \n\t beta={beta}\t")
        print(f"ESTYM ARGS:\n\t mu={e_mu}\n\t alpha={e_alpha} \n\t beta={e_beta}\t")

    # parameters_logL_estimate_example_1d()
    
    def parameters_logL_estimate_example_2d():
         # generate 1d traj:
        d = 2
        mu = np.array([1.0, 0.5])
        gamma = np.array([0.0, 0.0])
        alpha = np.array([[.3, .1], [0.1, .2]])
        # beta = np.array([[.85]])
        beta = np.ones((2,2))
        tmax = 500.0

        traj_2d = MHP(d, lambda i, hist, t: exp_intensity(i,mu, gamma, alpha, beta, hist, t), tmax, 10000)
        print(f"number events = {len(traj_2d)}")
        res = gradient_estimation(d, tmax, traj_2d)

        print(res)
        e_mu, e_alpha, e_beta = np.split(res.x, np.cumsum([d, d**2]))
        e_alpha = e_alpha.reshape((d,d))
        e_beta = e_beta.reshape((d,d))

        print(f"REAL ARGS:\n\t mu=\n{mu}\n\t alpha=\n{alpha} \n\t beta=\n{beta}\t")
        print(f"ESTYM ARGS:\n\t mu=\n{e_mu}\n\t alpha=\n{e_alpha} \n\t beta=\n{e_beta}\t")

        print(f"REAL $\dfrac{{\\alpha}}{{\\beta}}=\n {alpha/beta}$")
        print(f"ESTYM $\dfrac{{\\alpha}}{{\\beta}}=\n {e_alpha/e_beta}$")
    
    # parameters_logL_estimate_example_2d()

    def branching_simulation_example_2d():
        
        dim = 2
        h = [
            [lambda t: .5 * np.exp(-1.5 * t),   lambda t: 5**4/(3*2) * np.exp(-5*t)*t**(3)],
            [lambda t: 0.0,                     lambda t: .1 * np.exp(-1.0 * t)]
        ]
        max_h = [
            [.5, .5],
            [.01, .1]
        ]
        eta = np.array([.5, .5])
        t_max = 100.0
        
        hawkes_branch = generate_hawkes_by_branching(eta, h, max_h, t_max)
        traj_2d = family_to_jump_times(hawkes_branch, dim)
        

        jumps_dim_agg = [[] for _ in range(dim)]

        hs_df = pd.DataFrame(traj_2d, columns=["type", "t"])
        for _i in range(dim):
            jumps_dim_agg[_i] = hs_df.loc[hs_df["type"] == _i, "t"].to_numpy()

        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(w=14, h=8)

        ax.grid(which='major', visible=True, linestyle='--')
        ax.set_ylabel("$N^{(i)}_t$", fontsize=20)

        for _i in range(dim):
            hs_i = jumps_dim_agg[_i] 
            ax.plot([0.]+list(hs_i)+[t_max], np.arange(0, len(hs_i)+2), zorder=3, linewidth=2.5, label=f"i = {_i}")

        ax.legend()
        ax.tick_params(axis='y', labelsize=15)
        ax.set_xlabel("$t$", fontsize=20)

        plt.show()

    # branching_simulation_example_2d()