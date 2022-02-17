from pyhedging.Derivative import Derivative
from pyhedging.BSPricer import BSPricer
import numpy as np
import numpy.random as rd
from scipy.stats import norm
from typing import List


class BSDataset:
    def __init__(self, derivative: Derivative,
                 derivative_price: float = 0,
                 n_sim: int = 10 ** 5):
        self._N = n_sim
        self._S0 = derivative.initial_stock_price
        self._T = derivative.maturity
        self._n_points = derivative.n_points
        self._r = derivative.rate
        self._price = derivative_price
        self._sigma = derivative.volatility
        self._strike = derivative.strike
        self._dt = self._T / self._n_points
        self._derivative_name = derivative.payoff

    def generate_dataset(self, hedging_inst: List[Derivative] = None, rnn=False):
        if rnn:
            x = [self._price * np.ones((self._N, 1)), np.zeros((self._N, 1+len(hedging_inst)))]
        else:
            x = [self._price * np.ones((self._N, 1)), np.zeros((self._N, 1))]
        s = self._S0 * np.ones((self._N, 1))
        if hedging_inst is None:
            x.append(s)
            for t in range(2, self._n_points + 2):
                s = s + self._r * self._dt * s + \
                    self._sigma * np.sqrt(self._dt) * s * rd.normal(size=(self._N, 1))
                x.append(s)
        else:
            c = BSPricer.get_vanilla_prices(s, hedging_inst)
            x.append(np.concatenate((s, c), axis=1))
            for t in range(2, self._n_points + 1):
                s = s + self._r * self._dt * s + \
                    self._sigma * np.sqrt(self._dt) * s * rd.normal(size=(self._N, 1))
                c = BSPricer.get_vanilla_prices(s, hedging_inst, time=(t-1)*self._dt)
                x.append(np.concatenate((s, c), axis=1))
            s = s + self._r * self._dt * s + \
                self._sigma * np.sqrt(self._dt) * s * rd.normal(size=(self._N, 1))
            c = BSPricer.get_vanilla_payoffs(s, hedging_inst)
            x.append(np.concatenate((s, c), axis=1))
        y = np.zeros((self._N, 1))
        return x, y

    def generate_delta_BS(self, x_test):
        delta_BS = []
        for t in range(1, self._n_points + 1):
            time_to_maturity = self._T - (t - 1) * self._dt
            d1 = (np.log(x_test[t + 1] / self._strike) + time_to_maturity * (self._sigma ** 2 / 2)) / (
                    self._sigma * np.sqrt(time_to_maturity))
            if self._derivative_name == 'call':
                delta_BS.append(norm.cdf(d1))
            elif self._derivative_name == 'put':
                delta_BS.append(-norm.cdf(-d1))
        return delta_BS
