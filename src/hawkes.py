from __future__ import annotations
from dataclasses import dataclass, field

import sys
sys.setrecursionlimit(10000)

from abc import ABCMeta, abstractmethod
from math import exp, log, ceil, floor

from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, TypeAlias, overload

import numpy as np
from scipy import linalg, stats
from scipy.optimize import minimize, OptimizeResult

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import pylab as pl

from itertools import groupby

from pprint import pprint


class HawkesHistory(list[tuple[float, int]]):
    pass

class BaseIntensity(metaclass=ABCMeta):
    @abstractmethod
    def get_val(self, to_: int, at_: float) -> float: pass

class ExcitationFunction(metaclass=ABCMeta):
    @abstractmethod
    def get_val(self, from_: int, to_: int, at_: float, history_: HawkesHistory) -> float: pass
    
    @abstractmethod
    def get_val(self, to_: int, at_: float, history_: HawkesHistory) -> float: pass

class HawkesSpec(metaclass=ABCMeta): pass

@HawkesSpec.register
class IntensityBasedHawkesSpec(HawkesSpec):
    _base_intensity_: BaseIntensity
    _excitation_function_: ExcitationFunction
    _dim_: int

    def __init__(self, base_: BaseIntensity, excitation_fun_: ExcitationFunction) -> None:
        pass

# PRAWDZIWY PRZYKLAD:
# Lambda_{j -> i}(t) = mu_{i} + gamma_{i}*t + sum_{T^j_n < t} alpha_{i,j} * exp(-beta_{i,j} * (t - T^j_n))

@BaseIntensity.register
class LinearBaseIntensity(BaseIntensity):
    _mu_: np.array
    _gamma_: np.array
    
    def __init__(self, mu_: np.array, gamma_: np.array) -> None:
        self._mu_ = mu_.copy()
        self._gamma_ = gamma_.copy()

    def get_val(self, to_: int, at_: float) -> float:
        return self._mu_[to_] + self._gamma_[to_] * at_

@ExcitationFunction.register
class ExpotentialKernelExcitationFunction(ExcitationFunction):
    _alpha_: np.ndarray
    _beta_: np.ndarray

    def __init__(self, alpha_: np.array, beta_: np.array) -> None:
        self._alpha_ = alpha_.copy()
        self._beta_ = beta_.copy()

    def get_val(self, from_: int, to_: int, at_: float, history_: HawkesHistory) -> float:
        pass

    def get_val(self, to_: int, at_: float, history_: HawkesHistory) -> float:
        res_ = 0.0
        for ts, i in history_:
            if ts <= at_:
                res_ = res_ + self._alpha_[i, to_] * exp(-self._beta_[i, to_] * (at_ - ts))

        return res_

@IntensityBasedHawkesSpec.register
class LinearBaseExpotentialKernelHawkesScpec(IntensityBasedHawkesSpec):
    _mu_: np.array
    _gamma_ : np.array
    _alpha_ : np.ndarray
    _beta_ : np.ndarray

    def __init__(self, mu_: np.array, gamma_: np.array, alpha_: np.ndarray, beta_: np.ndarray) -> None:
        assert(mu_.ndim == 1 and gamma_.ndim == 1 and alpha_.ndim == 2 and beta_.ndim == 2) 
        self._dim_ = mu_.shape[0]
        assert(self._dim_ >= 1 and gamma_.shape == (self._dim_, ) and alpha_.shape == (self._dim_, self._dim_) and beta_.shape == (self._dim_, self._dim_)) 
        
        self._base_intensity_ = LinearBaseIntensity(mu_=mu_, gamma_=gamma_)
        self._excitation_function_ = ExpotentialKernelExcitationFunction(alpha_=alpha_, beta_=beta_)

        self._mu_ = mu_
        self._gamma_ = gamma_
        self._alpha_ = alpha_
        self._beta_ = beta_

@IntensityBasedHawkesSpec.register
class ConstBaseExpotentialKernelHawkesScpec(IntensityBasedHawkesSpec):
    _mu_: np.array

    _alpha_ : np.ndarray
    _beta_ : np.ndarray

    def __init__(self, mu_: np.array, alpha_: np.ndarray, beta_: np.ndarray) -> None:
        assert(mu_.ndim == 1 and alpha_.ndim == 2 and beta_.ndim == 2) 
        self._dim_ = mu_.shape[0]
        assert(self._dim_ >= 1 and alpha_.shape == (self._dim_, self._dim_) and beta_.shape == (self._dim_, self._dim_)) 
        
        self._base_intensity_ = LinearBaseIntensity(mu_=mu_, gamma_=np.zeros(self._dim_))
        self._excitation_function_ = ExpotentialKernelExcitationFunction(alpha_=alpha_, beta_=beta_)

        self._mu_ = mu_
        self._alpha_ = alpha_
        self._beta_ = beta_

@HawkesSpec.register
class BranchingHawkesSpec(HawkesSpec):
    _eta_: List[Callable[[float], float]]
    _eta_max_: List[float]
    _h_: List[List[Callable[[float], float]]]
    _h_max_: List[List[float]]
    _dim_: int 

    def __init__(self, eta_, h_, eta_max_=None, h_max_=None) -> None:
        assert(all(map(lambda x: x is not None, [eta_max_, h_max_]))) # TODO: add support for None-like max values later
        self._dim_ = len(eta_)

        self._eta_ = eta_
        self._eta_max_ = eta_max_
        self._h_ = h_
        self._h_max_ = h_max_

class BranchingStructure(HawkesHistory):
    class Family:
        @dataclass
        class Event:
            type: int
            time: float
            childrens: List[Any]|None
        type: int
        event: Event

        def for_each_event(self, callable: Callable[[Event, Optional[Any]], None], *args: Optional[Any])->None:
            def __rec__(ev: self.Event):
                for children in ev.childrens:
                    __rec__(ev=children)
                callable(ev, *args)
            __rec__(ev=self.event)

    families: List[Family] = []

# SIMULATIONS:

class HawkesSimulation:
    _t_max_: float | None
    _max_event: int | None 
    _spec_: HawkesSpec

    def __init__(self, spec_: HawkesSpec, t_max: float | None = None, max_event: int | None = None) -> None:
        assert(any(map(lambda x: x is not None, [t_max, max_event])))
        self._t_max_ = t_max
        self._max_event = max_event
        self._spec_ = spec_

    def run(self, **kwargs) -> HawkesHistory:
        if isinstance(self._spec_, IntensityBasedHawkesSpec):
            return self.__impl_1(**kwargs)
        elif isinstance(self._spec_, BranchingHawkesSpec):
            return self.__impl_2(**kwargs)
        else:
            raise NotImplementedError(f"Simulation algorithm for {type(self._spec_)} unknown")
        
    def __impl_1(self, **kwargs):
        def terminate_condition(n: int, t: float) -> bool:
            return any([
                self._t_max_ is not None and t > self._t_max_,
                self._max_event is not None and n > self._max_event
            ])
        
        def intensity(to: int, at: float, history: HawkesHistory):
            bi = self._spec_._base_intensity_.get_val(to, at)
            ei = self._spec_._excitation_function_.get_val(to, at, history)
            # print(f"j={to: }, t={at}, len={len(history)}, bi_val ={bi} ei_val={ei}")
            return bi + ei
        
        hhistory = HawkesHistory()
        n, t = 0, 0.0 
        while(not terminate_condition(n, t)):
            tau_n = np.zeros(self._spec_._dim_)
            for j in range(self._spec_._dim_):
                tau = t
                intens  = intensity(j, tau, hhistory)
                while True:
                    E = np.random.exponential()
                    tau = tau + E / max(intens, 1.0e-19)
                    U = np.random.uniform()
                    u = U * intens
                    intens  = intensity(j, tau, hhistory)
                    if u < intens:
                        tau_n[j] = tau
                        break
            dn = tau_n.argmin()
            tn = tau_n[dn]

            # print("add")
            hhistory.append((tn, dn))
            n += 1
            t = tn
        
        # remove last event if it is post terminate condition occurence
        last_t, _ = hhistory[-1]
        if terminate_condition(len(hhistory), last_t):
            hhistory = hhistory[:-1]

        return hhistory
    
    def __impl_2(self, **kwargs):
        spec: BranchingHawkesSpec = self._spec_
        
        def PRM(intensity : Callable[[float], float], intensity_max : float, tmax: float) -> List[float]: #poisson random measure
            t = 0
            T = []

            while t < tmax:
                U1 = np.random.uniform(0,1)
                t = t - (log(U1) / intensity_max)

                if t > tmax:
                    break

                U2 = np.random.uniform(0,1)
                if intensity_max * U2 <= intensity(t):
                    T.append(t)

            return T
        
        def generate_family(h : List[List[Callable[[float], float]]], max_h : List[List[float]], tmax : float, type_ : int):
            family = BranchingStructure.Family()
            family.type = type_

            event = BranchingStructure.Family.Event(type=type_, time=0.0, childrens=[])
            family.event = event

            def _rec_(__event):
                for j in range(spec._dim_):
                    if max_h[__event.type][j] == None or max_h[__event.type][j] <= 0.0 or h[__event.type][j] == None:
                        continue
                
                    new_events_t = PRM(h[__event.type][j], max_h[__event.type][j], tmax=tmax-__event.time)
                    for new_ev_t in new_events_t:
                        new_ev = BranchingStructure.Family.Event(type=j, time=new_ev_t, childrens=[])
                        _rec_(new_ev)
                        __event.childrens.append(new_ev)
                
            _rec_(family.event)
            return family

        class MoveEventTime():
            t: float
            def __init__(self, t_: float) -> None:
                self.t = t_
            def __call__(self, event: BranchingStructure.Family.Event):
                event.time = event.time + self.t

        res = BranchingStructure()
        for i0 in range(spec._dim_):
            if all(map(lambda t: spec._eta_[i0](t) <= 0.0, np.linspace(0, self._t_max_, 1000))): # czy to nie jest zerowa ekscytacja
                continue

            I_i0 = PRM(spec._eta_[i0], spec._eta_max_[i0], tmax=self._t_max_)
            for t0 in I_i0:
                family = generate_family(spec._h_, spec._h_max_, self._t_max_ - t0, i0)
                moveEvent = MoveEventTime(t0)
                family.for_each_event(moveEvent)
                res.families.append(family)


        class CollectEventWithType():
            def __init__(self) -> None:
                pass
            def __call__(self, event:BranchingStructure.Family.Event, list_: List[Tuple[float, int]]) -> None:
                list_.append((event.time, event.type))
        
        for fam in res.families:
            collect = CollectEventWithType()
            list_ = []
            fam.for_each_event(collect, list_)
            res.extend(list_)
        
        res.sort(key=lambda ev: ev[0])
        return res

# INFERENCE:

class ExpotentialKernelIntensityMean():
    @dataclass
    class HawkesMeanTrajectory():
        time: List[float] = 0
        values: List[Tuple[int, List[float]]] = field(default_factory=list)

    _exp_kernel_: ExpotentialKernelExcitationFunction
    _base_intensity: BaseIntensity
    _dim_: int
    _acaccuracy_steps_: int

    def __init__(self, base_intensity_: BaseIntensity, exp_kernel_: ExpotentialKernelExcitationFunction, accuracy_steps: int = 20) -> None:
        self._dim_ = exp_kernel_._alpha_.shape[0]        

        self._exp_kernel_ = exp_kernel_
        self._base_intensity = base_intensity_
        self._acaccuracy_steps_ = accuracy_steps
    
    def __call__(self, to: float) -> HawkesMeanTrajectory:
        def multivariate_mean_algorithm(d, tmax, u_t, beta, alpha):
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

            return np.array([G(t) for t in np.linspace(0, tmax, 20)])
        
        mean = multivariate_mean_algorithm(
            self._dim_, to, 
            [lambda t, i=i: self._base_intensity.get_val(i, t) for i in range(self._dim_)], 
            self._exp_kernel_._beta_, self._exp_kernel_._alpha_)

        mean_traj = self.HawkesMeanTrajectory()
        mean_traj.time = list(np.array(range(self._acaccuracy_steps_)))
        mean_traj.time = mean_traj.time / max(mean_traj.time) * to
        
        for i in range(self._dim_):
            # a = mean[:,i].tolist()
            # print(mean)
            mean_traj.values.append((i, mean[:,i].tolist()))

        return mean_traj

# ESTIMATIONS:

class LogLikelihoodMaximizationEstimator():
    _trajectory_: HawkesHistory
    _dim_: int
    _tmax_ : float
    _traj_agg_ : List[List[float]] # trajectory representation as by-type aggregate

    def __init__(self, trajectory: HawkesHistory) -> None:
        self._trajectory_ = trajectory
        self._dim_ = max([i for _, i in self._trajectory_]) + 1
        self._tmax_ = max([ts for ts, _ in self._trajectory_]) + 1.0
        
        self._traj_agg_ = [[] for _ in range(self._dim_)]
        for ts, i in self._trajectory_:
            self._traj_agg_[i].append(ts)
    
    def with_gradient(self, *args: Any, **kwds: Any) -> Tuple[ConstBaseExpotentialKernelHawkesScpec, OptimizeResult]:
        def logL_r(x: np.array):
            mu, alpha, beta = np.split(x, np.cumsum([self._dim_, self._dim_**2]))
            alpha = np.reshape(alpha, (self._dim_, self._dim_))
            beta = np.reshape(beta, (self._dim_, self._dim_))

            logL_res = 0.0
            for m in range(self._dim_):
                logL_m = -mu[m] * self._tmax_ - sum([
                    alpha[m,n] / beta[m,n] * sum([
                        (1 - exp(-beta[m,n] * (self._tmax_ - tn_k)))
                        for tn_k in self._traj_agg_[n]
                    ])
                    for n in range(self._dim_)
                ]) + sum([
                    log(
                        mu[m] + sum([
                            alpha[m,n] * sum([
                                exp(-beta[m,n] * (tm_k - tn_i)) for tn_i in self._traj_agg_[n] if tn_i < tm_k
                            ])
                            for n in range(self._dim_)
                        ])
                    )
                    for tm_k in self._traj_agg_[m]
                ])

                logL_res += logL_m

            return logL_res 

        def logL_grad(x: np.array):
            mu, alpha, beta = np.split(x, np.cumsum([self._dim_, self._dim_**2]))
            alpha = np.reshape(alpha, (self._dim_, self._dim_))
            beta = np.reshape(beta, (self._dim_, self._dim_))

            Dmu = np.zeros(self._dim_)
            Dalpha = np.zeros((self._dim_, self._dim_))
            Dbeta = np.zeros((self._dim_, self._dim_))

            def R(_m,_n, _tm_k):
                return sum([
                    exp(-beta[_m,_n] * (_tm_k - tn_i)) for tn_i in self._traj_agg_[_n] if tn_i < _tm_k
                ])

            def dR(_m,_n, _tm_k):
                return sum([
                    (_tm_k - tn_i) * exp(-beta[_m,_n] * (_tm_k - tn_i)) for tn_i in self._traj_agg_[_n] if tn_i < _tm_k
                ])

            # po mu:
            for m in range(self._dim_):
                dL_m_dMu_m = -self._tmax_ + sum([1.0 / (mu[m] + sum([
                    alpha[m,j] * R(m,j,tm_k)
                    for j in range(self._dim_)
                ])) for tm_k in self._traj_agg_[m]])

                Dmu[m] = dL_m_dMu_m

            # po alpha mn:
            for m in range(self._dim_):
                for n in range(self._dim_):
                    dLm_dAmn = -1.0 / beta[m,n] * sum([(1 - exp(-beta[m,n]*(self._tmax_ - tn_k))) for tn_k in self._traj_agg_[n]]) + sum([
                        R(m,n,tm_k) / (mu[m]  + sum([
                            alpha[m,j] * R(m,j, tm_k) 
                            for j in range(self._dim_)
                        ])) for tm_k in self._traj_agg_[m]
                    ])

                    Dalpha[m,n] += dLm_dAmn

            # po beta mn:
            for m in range(self._dim_):
                for n in range(self._dim_):
                    dLm_dBmn = (alpha[m,n] / beta[m,n]**2) * sum([
                        (1 - exp(-beta[m,n] * (self._tmax_ - tn_k)))
                        for tn_k in self._traj_agg_[n]
                    ]) - (alpha[m,n] / beta[m,n]) * sum([
                        (self._tmax_ - tn_k) * ((self._tmax_ - tn_k) * exp(-beta[m,n] * (self._tmax_ - tn_k)))
                        for tn_k in self._traj_agg_[n]
                    ]) - sum(
                        alpha[m,n] * dR(m,n,tm_k) / (mu[m] + sum([alpha[m,j] * R(m,j, tm_k) for j in range(self._dim_)]))
                        for tm_k in self._traj_agg_[m]
                    )

                    Dbeta[m,n] += dLm_dBmn

            return np.hstack([
                Dmu, Dalpha.flatten(), Dbeta.flatten()
            ])

        N = self._dim_ + 2 * self._dim_**2
        x0 = np.full(N, 1.0)

        bds = [(1.0e-15, 10.0)]*self._dim_ + [(0.0, 10.0)]*self._dim_**2 + [(1.0e-15, 10.0)]*self._dim_**2
        x0 = np.full(N, 2.0)
        res = minimize(
            lambda x, args: -logL_r(x) + 0.0 * np.sum(x[self._dim_:self._dim_+self._dim_**2]),
            args=((),),
            x0=x0, 
            bounds=bds, 
            method='SLSQP',
            options={'maxiter': 500},
            jac=lambda x, args: -logL_grad(x))
        
        mu, alpha, beta = np.split(res.x, np.cumsum([self._dim_, self._dim_**2]))
        alpha = np.reshape(alpha, (self._dim_, self._dim_))
        beta = np.reshape(beta, (self._dim_, self._dim_))
        
        estimated_params = ConstBaseExpotentialKernelHawkesScpec(mu_=mu, alpha_=alpha, beta_=beta)
        return estimated_params, res
    
    def no_gradient(self, *args: Any, **kwds: Any) -> Tuple[LinearBaseExpotentialKernelHawkesScpec, OptimizeResult]:
        def logL_for_traj(x_: List[float]):
            logL = 0.0
            
            for i in range(self._dim_):
                def mu_si(s: float):
                    return x_[2 * i] + x_[2 * i + 1] * s

                def part1():
                    I1 = x_[2 * i] * self._tmax_ + x_[2 * i + 1] * self._tmax_**2 / 2.0
                    I2 = 0
                    for j in range(self._dim_):
                        if x_[2 * self._dim_ + self._dim_** 2 + i * self._dim_ + j] > 0: # B_{i,j} > 0:
                            l = [-x_[2 * self._dim_ + i * self._dim_ + j] / x_[2 * self._dim_ + self._dim_ ** 2 + i * self._dim_ + j] * (
                                    exp(
                                        x_[2 * self._dim_ + self._dim_ ** 2 + i * self._dim_ + j] * (tk - self._tmax_)
                                    ) - 1)
                                    for tk in self._traj_agg_[j]
                                ]

                            I2 += sum(
                                l
                            )

                        else:
                            I2 += sum(
                                [
                                    x_[2 * self._dim_ + i * self._dim_ + j] * (self._tmax_ - tk) for tk in self._traj_agg_[j]
                                ]
                            )

                    return I1 + I2

                def part2():
                    try:
                        return sum(
                            [
                                log(mu_si(self._traj_agg_[i][n]) + 
                                sum(
                                    [
                                    sum([x_[2 * self._dim_ + i * self._dim_ + j] * exp(-x_[2 * self._dim_ + self._dim_ ** 2 + i * self._dim_ + j] * (self._traj_agg_[i][n] - tk)) if tk < self._traj_agg_[i][n] else 0 for tk in self._traj_agg_[j]]) 
                                    for j in range(self._dim_)]
                                )) 
                                for n in range(len(self._traj_agg_[i]))
                            ]
                        )
                    except:
                        print(f"Not valid arguments for step: arg=:\n{x_}\n")
                        return 0.0

                logL -= part1()
                logL += part2()
            return logL

        logArgs = [1 for _ in range(2 * (self._dim_ ** 2) + 2 * self._dim_)]
        bds = [(0.0000001, 10.0) for _ in range(2 * (self._dim_ ** 2) + 2 * self._dim_)]
        res = minimize(
            lambda x, args: -logL_for_traj(x), 
            x0 = logArgs, args=((),), bounds=bds, method='Nelder-Mead', options={'disp': True, 'maxiter': 240})
        

        _mu = np.array([res.x[2*i] for i in range(self._dim_)])
        _gamma = np.array([res.x[2*i+1] for i in range(self._dim_)])
        _alpha = _alpha = np.array(res.x[2*self._dim_:(2*self._dim_ + self._dim_**2)])
        _alpha = _alpha.reshape((self._dim_, self._dim_))
        
        _beta = _beta = np.array(res.x[(2*self._dim_ + self._dim_**2):(2*self._dim_ + 2*self._dim_**2)])
        _beta = _beta.reshape((self._dim_, self._dim_))
        
        hs = LinearBaseExpotentialKernelHawkesScpec(mu_=_mu, gamma_=_gamma, alpha_=_alpha, beta_=_beta)
        return hs, res
    
class BranchingModelEstimation():
    @dataclass
    class Edge:
        _from_: int
        _to_: int
    
    class EstimationResult:
        @dataclass
        class FuncEstym:
            ege: Any
            time: np.array
            vals: np.array

        _A_: np.array
        _eta_: np.array
        _edges_: List[Any] #edges
        _points_: List[FuncEstym] 

    _trajectory_: HawkesHistory
    _dim_: int
    _tmax_: float
    _traj_agg_: List[List[float]]
    def __init__(self, trajectory: HawkesHistory) -> None:
        self._trajectory_ = trajectory
        self._dim_ = max([i for _, i in self._trajectory_]) + 1
        self._tmax_ = max([ts for ts, _ in self._trajectory_]) + 1.0
        
        self._traj_agg_ = [[] for _ in range(self._dim_)]
        for ts, i in self._trajectory_:
            self._traj_agg_[i].append(ts)

    def egdes_estim(self, delta_s_: float, s_:float, alpha_: float) -> List[Edge]:
        assert(0 < alpha_ < 1)

        d = len(self._traj_agg_)
        p = ceil(s_ / delta_s_)
        n = floor(self._tmax_ / delta_s_)

        B = np.hstack([
            *[np.eye(d)] * p, np.zeros((d, 1))
        ])

        _X_ = np.array([
            np.histogram(jumps_dim, bins=n, range=(0.0, n * delta_s_))[0] for jumps_dim in self._traj_agg_])

        Z = np.vstack(
            [np.append(np.array(_X_[:,(p-1)::-1]).flatten(order='F'),1)] + 
            [np.append(np.array(_X_[:,(_p):_p-p:-1]).flatten(order='F'),1) for _p in range(p, n-1)])

        Y = Y = _X_[:, p:n].transpose()
        H = np.linalg.inv(Z.transpose() @ Z) @ Z.transpose() @ Y / delta_s_
        A = delta_s_ * B @ H

        E = np.hstack([
            *[np.eye(d**2)] * p, np.zeros((d**2, d))])

        _tmp_ = E @ np.kron(np.linalg.inv(Z.transpose() @ Z) @ Z.transpose(), np.eye(d))
        C = np.vstack(np.split(np.hstack(_tmp_), np.cumsum([d] * (d**2*(n-p) - 1))))

        U = Y - delta_s_ * Z @ H
        Urep = np.vstack([*[U] * d**2])

        _tmp2_ = np.multiply(C, Urep)

        _tmp3_ = np.sum(np.vstack(np.split((np.square(np.sum(_tmp2_, axis=1))), np.cumsum([n-p] * (d**2 - 1)))), axis=1)

        SIG = np.vstack(np.split(_tmp3_, np.cumsum([d] * (d-1))))

        z_inv = stats.norm.ppf(1-alpha_)
        selected = np.where(A > np.sqrt(SIG) * z_inv)

        accepted_edges = [self.Edge(_from_=int(i), _to_=int(j)) for i,j in zip(list(selected[0]), list(selected[1]))]
        return accepted_edges


    def estim(self, s_: float, delta_g_: float, delta_s_: float, alpha_: float = 0.05) -> EstimationResult:
        assert(0 < delta_g_ < delta_s_)
        p = ceil(s_ / delta_g_)
        n = floor(self._tmax_ / delta_g_)
        edges = self.egdes_estim(s_=s_, delta_s_=delta_s_, alpha_=alpha_)

        PA = np.zeros((self._dim_, self._dim_))
        for e in edges:
            PA[e._from_, e._to_] = 1
        PA = PA.transpose()

        d_j = np.sum(PA, axis = 1)

        _Xg = np.array([
            np.histogram(jumps_dim, bins=n, range=(0.0, n * delta_g_))[0] for jumps_dim in self._traj_agg_])
        
        H_all = []
        for j in range(self._dim_):
            pa_j = PA[j]
            _Xg_paj = _Xg[pa_j == 1, :]
            _X_j = _Xg[j, :]
            Zj = np.vstack(
                [np.append(np.array(_Xg_paj[:,(p-1)::-1]).flatten(order='F'),1)] + 
                [
                    np.append(np.array(_Xg_paj[:,(_p):_p-p:-1]).flatten(order='F'),1) for _p in range(p, n-1)
            ])

            Yj = _X_j[p:n].transpose()

            Hg_j = 1 / delta_g_ * np.linalg.inv(Zj.transpose() @ Zj) @ Zj.transpose() @ Yj

            H_all.append(Hg_j)

        EdgesEstym = []
        for j in range(self._dim_):
            pa_j = np.where(PA[j] == 1.0)[0]
            dj = int(sum(PA[j]))
            for l in range(dj):
                b_lj = np.zeros((dj * p + 1, 1))
                b_lj[l:((p-1)*dj + l+1):dj] = 1.0
                a_ilj = b_lj.transpose() @ H_all[j] / p * s_
                if a_ilj > 0:
                    EdgesEstym.append((pa_j[l], j, a_ilj))

        EdgesEstymMat = np.zeros((self._dim_, self._dim_))
        for edge in EdgesEstym:
            i,j, a_ij = edge
            EdgesEstymMat[i,j] = a_ij

        eta = np.array([h[-1] for h in H_all])
        
        
        er = self.EstimationResult()
        er._points_ = []
        for i,j, _ in EdgesEstym:
            Hj = H_all[j]
            dj = int(sum(PA[j]))
            i_paj = int(sum(PA[j][:i]))

            h_ij_est_samples = Hj[i_paj:((p-1)*dj + i_paj+1):dj]
            time = np.linspace(0, s_, len(h_ij_est_samples))
            vals = h_ij_est_samples
            er._points_.append(self.EstimationResult.FuncEstym(self.Edge(_from_=i, _to_=j), time=time, vals=vals))

        er._A_ = EdgesEstymMat
        er._edges_ = edges
        er._eta_ = eta
        return er

# PLOT:

class HawkesPlot():
    @staticmethod
    def get_dim(history: HawkesHistory) -> int:
        return max([i for _, i in history]) + 1
    
    @staticmethod
    def plot_hawkes_trajectories(history: HawkesHistory, ax: plt.Axes, **kwargs) -> None:
        dim = HawkesPlot.get_dim(history)
        
        colormap = pl.cm.twilight_shifted
        colors = colormap(np.linspace(0, 1, dim))

        for key, value in kwargs.items():
            if key == "colormap": 
                assert(isinstance(value, matplotlib.colors.Colormap))
                colormap = value
                colors = colormap(np.linspace(0, 1, dim))
            elif key == "colors":
                assert(len(value) == dim)
                colors=value

        history.sort(key=lambda x: x[1])
        max_t = max([t for t, _ in history])

        if isinstance(ax, Iterable):
            assert(len(ax) == dim)

            for i, ts_g in groupby(history, key=lambda x: x[1]):
                ts = [t for t, _ in ts_g]
                ts.sort()  
                x = [0.0, *ts, max_t]
                y = [*list(range(len(ts) + 1)), len(ts)]
                ax[i].plot(x, y, label=i, color='.2', linewidth=1., linestyle='--')
        
        elif isinstance(ax, plt.Axes):

            for i, ts_g in groupby(history, key=lambda x: x[1]):
                ts = [t for t, _ in ts_g]
                ts.sort()  
                x = [0.0, *ts, max_t]
                y = [*list(range(len(ts) + 1)), len(ts)]
                ax.plot(x, y, label=i, color=colors[i], linewidth=1., linestyle='--')

    @staticmethod
    def plot_hawkes_mean(mean_traj: ExpotentialKernelIntensityMean.HawkesMeanTrajectory, ax: plt.Axes, **kwargs) -> None:
        dim = len(mean_traj.values)
        
        colormap = pl.cm.twilight_shifted
        colors = colormap(np.linspace(0, 1, dim))

        for key, value in kwargs.items():
            if key == "colormap": 
                assert(isinstance(value, matplotlib.colors.Colormap))
                colormap = value
                colors = colormap(np.linspace(0, 1, dim))
            elif key == "colors":
                assert(len(value) == dim)
                colors=value

        if isinstance(ax, Iterable):
            assert(len(ax) == dim)

            for i, vals in mean_traj.values:
                x = mean_traj.time
                y = vals
                ax[i].plot(x, y, label=i, color='b', linewidth=2., linestyle='-')
        
        elif isinstance(ax, plt.Axes):

            for i, vals in mean_traj.values:
                x = mean_traj.time
                y = vals
                ax.plot(x, y, label=i, color=colors[i], linewidth=2., linestyle='-')
        

if __name__ == "__main__":
    np.random.seed(1)

    # mu = np.array([0.1, 0.6])
    # gamma = np.array([0.0, 0.0])
    # alpha = np.array([
    #     [0.4, 0.0],
    #     [0.0, 0.5]])
    # beta = 5 * np.eye(2)
    # mu = np.array([0.1, 0.6])
    # gamma = np.array([0.01, -0.004])
    # alpha = np.array([
    #     [0.4, 0.0],
    #     [0.0, 0.5]])
    # beta = np.array([
    #     [5.0, 0.0],
    #     [0.0, 3.0]])

    # mu = np.array([1.0, 0.05, 0.4])
    # gamma = np.array([0.0, 0.0, 0.0])
    # alpha = np.array([
    #     [0.5, 0.0, 0.0],
    #     [1.5, 0.0, 0.0],
    #     [0.0, 0.0, 2.0]])
    # beta = np.ones((3, 3)) * 3.0
    # tmax = 1000.0

    # spec = LinearBaseExpotentialKernelHawkesScpec(mu.copy(), gamma.copy(), alpha.copy(), beta.copy())
    # mean = ExpotentialKernelIntensityMean(spec._base_intensity_, spec._excitation_function_)
    # mean_traj = mean(1000.0)

    # assert isinstance(spec, HawkesSpec)
    # simulation = HawkesSimulation(spec, t_max=1000.0)
    # trajectories = [simulation.run() for _ in range(4)]
    # # print(res)
    
    # fig1, ax = plt.subplots()
    # for r in trajectories:
    #     HawkesPlot.plot_hawkes_trajectories(history=r, ax=ax, colors=['r', 'g', 'b'])
    # HawkesPlot.plot_hawkes_mean(mean_traj, ax, colors=['r', 'g', 'b'])
    # # ax.legend()

    # plt.show()

    # logL_estimation = LogLikelihoodMaximizationEstimator(trajectories[0])
    # estimadet_x, optim_res = logL_estimation.with_gradient()    
    
    # pprint(optim_res)
    # pprint(f'_mu_ = {estimadet_x._mu_}')
    # pprint(f'_alpha_ = {estimadet_x._alpha_}')
    # pprint(f'_beta_ = {estimadet_x._beta_}')
    
    def gamma64(t):
        if t >= 0:
            return (t**5) * exp(-4 * t) * (4 ** 6) / (5*4*3*2)
        else:
            return 0;

    h_heavy = lambda t: 1.5 * gamma64(t)
    h_light = lambda t: 0.5 if 1.0 <= t <= 2.0 else 0.0
    h_super_light = lambda t: 0.1 if 1.0 <= t <= 2.0 else 0.0
    h_zero = lambda t: 0.0

    _eta_max = [1.0, .0, .0, .0, .0, .0, 1.0, .0, .0, 1.0]
    # _eta_max = [1.0, .0, .0, .0, .0, .0, .0, .0, .0, .0]
    _eta = list(map(lambda v: lambda t: v, _eta_max))

    _h = [
        [h_light, h_heavy, h_zero, h_zero, h_zero, h_zero, h_zero, h_zero, h_zero, h_zero],
        [h_zero, h_zero, h_light, h_heavy, h_zero, h_zero, h_zero, h_zero, h_zero, h_zero],
        [h_zero, h_zero, h_zero, h_zero, h_light, h_zero, h_zero, h_zero, h_zero, h_zero],
        [h_zero, h_zero, h_light, h_zero, h_zero, h_light, h_light, h_zero, h_zero, h_zero],
        [h_zero, h_zero, h_light, h_zero, h_zero, h_zero, h_super_light, h_zero, h_zero, h_zero],
        [h_zero, h_zero, h_zero, h_zero, h_zero, h_zero, h_zero, h_zero, h_zero, h_zero],
        [h_zero, h_zero, h_zero, h_zero, h_zero, h_zero, h_zero, h_light, h_zero, h_zero],
        [h_zero, h_zero, h_zero, h_zero, h_zero, h_zero, h_zero, h_zero, h_heavy, h_zero],
        [h_zero, h_zero, h_zero, h_zero, h_zero, h_zero, h_light, h_zero, h_zero, h_zero],
        [h_zero, h_zero, h_zero, h_zero, h_zero, h_zero, h_zero, h_zero, h_zero, h_zero]]

    _h_max = [
        [.5, .35, .0, .0, .0, .0, .0, .0, .0, .0],
        [.0, .0, .5, .35, .0, .0, .0, .0, .0, .0],
        [.0, .0, .0, .0, .5, .0, .0, .0, .0, .0],
        [.0, .0, .5, .0, .0, .5, .5, .0, .0, .0],
        [.0, .0, .5, .0, .0, .0, .1, .0, .0, .0],
        [.0, .0, .0, .0, .0, .0, .0, .0, .0, .0],
        [.0, .0, .0, .0, .0, .0, .0, .5, .0, .0],
        [.0, .0, .0, .0, .0, .0, .0, .0, .35, .0],
        [.0, .0, .0, .0, .0, .0, .5, .0, .0, .0],
        [.0, .0, .0, .0, .0, .0, .0, .0, .0, .0]]
    
    spec = BranchingHawkesSpec(eta_=_eta, h_=_h, eta_max_=_eta_max, h_max_=_h_max)
    simulation = HawkesSimulation(spec, t_max=5000)
    res = simulation.run()

    # fig1, ax = plt.subplots()
    # HawkesPlot.plot_hawkes_trajectories(history=res, ax=ax)
    # ax.legend()
    
    # fig2, axs = plt.subplots(nrows=10, ncols=1)
    # HawkesPlot.plot_hawkes_trajectories(history=res, ax=axs)
    
    # plt.show()

    bme = BranchingModelEstimation(res)
    print([len(da) for da  in bme._traj_agg_])
    _s = 5.0
    _delta_s = 0.2
    _alpha = 0.05
    _delta_g = 0.05
    estim_res = bme.estim(alpha_=_alpha, s_=_s, delta_s_=_delta_s, delta_g_=_delta_g)

    pprint(estim_res._A_)
    pprint(estim_res._eta_)

    # d = len(_eta)
    # fig, ax = plt.subplots(d,d, sharex=True, sharey=True, gridspec_kw=dict(hspace=0, wspace=0))
    # fig.set_size_inches(20, 20)

    # for i in range(d):
    #     for j in range(d):
    #         ax[i][j].plot(np.linspace(0, _s, 1000), [_h[i][j](x) for x in np.linspace(0, _s, 1000)], c='.5')
    #         ax[i][j].set_xticklabels([])
    #         ax[i][j].set_yticklabels([])
    #         if(i == d-1):
    #             ax[i][j].set_xlabel(j+1)
    #         if(j == 0):
    #             ax[i][j].set_ylabel(i+1)

    # for info in estim_res._points_:
        
    #     ax[info.ege._from_][info.ege._to_].scatter(x=info.time, y=info.vals, c='k', s=0.2)

    # plt.show()

    
