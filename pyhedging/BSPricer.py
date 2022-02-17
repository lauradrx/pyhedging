import QuantLib as ql
from pyhedging.Derivative import Derivative, BarrierOption, VanillaOption, AsianOption, DigitalOption
import numpy as np
from scipy.stats import norm
from typing import List
from copy import deepcopy


class BSPricer:

    @staticmethod
    def get_price_ql(derivative, running_acc: float = 1, past_fixings: int = 0):
        derivative_payoff = derivative.payoff
        n_points = derivative.n_points
        strike = derivative.strike

        exercise_type = derivative.exercise_type

        today = ql.Date().todaysDate()
        maturity = today + ql.Period(n_points, ql.Days)
        eu_exercise = ql.EuropeanExercise(maturity)

        # Underlying Price
        u = ql.SimpleQuote(derivative.initial_stock_price)
        # Risk-free Rate
        r = ql.SimpleQuote(derivative.rate)
        # Sigma
        sigma = ql.SimpleQuote(derivative.volatility)

        # Build flat curves and volatility
        rf_curve = ql.FlatForward(today, ql.QuoteHandle(r), ql.Actual360())
        volatility = ql.BlackConstantVol(today, ql.TARGET(), ql.QuoteHandle(sigma), ql.Actual360())

        option_type_dic = {'call': ql.Option.Call,
                           'put': ql.Option.Put}

        if type(derivative) == VanillaOption:
            payoff = ql.PlainVanillaPayoff(option_type_dic[derivative_payoff], strike)
            process = ql.BlackScholesProcess(ql.QuoteHandle(u),
                                             ql.YieldTermStructureHandle(rf_curve),
                                             ql.BlackVolTermStructureHandle(volatility))
            option = ql.VanillaOption(payoff, eu_exercise)
            option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

        elif type(derivative) == BarrierOption:
            payoff = ql.PlainVanillaPayoff(option_type_dic[derivative_payoff], strike)
            barrier = derivative.barrier
            barrier_type = derivative.barrier_type

            if barrier_type == 'up_in' and derivative.initial_stock_price >= barrier:
                payoff = ql.PlainVanillaPayoff(option_type_dic[derivative_payoff], strike)
                process = ql.BlackScholesProcess(ql.QuoteHandle(u),
                                                 ql.YieldTermStructureHandle(rf_curve),
                                                 ql.BlackVolTermStructureHandle(volatility))
                option = ql.VanillaOption(payoff, eu_exercise)
                option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

            elif barrier_type == 'down_in' and derivative.initial_stock_price <= barrier:
                payoff = ql.PlainVanillaPayoff(option_type_dic[derivative_payoff], strike)
                process = ql.BlackScholesProcess(ql.QuoteHandle(u),
                                                 ql.YieldTermStructureHandle(rf_curve),
                                                 ql.BlackVolTermStructureHandle(volatility))
                option = ql.VanillaOption(payoff, eu_exercise)
                option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

            elif barrier_type == 'up_out' and derivative.initial_stock_price >= barrier:
                return 0

            elif barrier_type == 'down_out' and derivative.initial_stock_price <= barrier:
                return 0

            else:
                process = ql.BlackScholesProcess(ql.QuoteHandle(u),
                                                 ql.YieldTermStructureHandle(rf_curve),
                                                 ql.BlackVolTermStructureHandle(volatility))

                rebate = 0
                barrier_type_dic = {'down_in': ql.Barrier.DownIn,
                                    'down_out': ql.Barrier.DownOut,
                                    'up_in': ql.Barrier.UpIn,
                                    'up_out': ql.Barrier.UpOut}

                barrier_type = barrier_type_dic[derivative.barrier_type]

                option = ql.BarrierOption(barrier_type, barrier, rebate, payoff, eu_exercise)
                option.setPricingEngine(ql.AnalyticBarrierEngine(process))

        elif type(derivative) == AsianOption:
            payoff = ql.PlainVanillaPayoff(option_type_dic[derivative_payoff], strike)
            dividend = ql.YieldTermStructureHandle(ql.FlatForward(today, 0, ql.Actual360()))
            process = ql.GeneralizedBlackScholesProcess(ql.QuoteHandle(u), dividend,
                                                        ql.YieldTermStructureHandle(rf_curve),
                                                        ql.BlackVolTermStructureHandle(volatility))

            geometric_avg = ql.Average().Geometric
            running_accumulator = running_acc
            future_fixing_dates = [today + ql.Period(i, ql.Days) for i in range(n_points+1)]
            option = ql.DiscreteAveragingAsianOption(geometric_avg,
                                                     running_accumulator,
                                                     past_fixings, future_fixing_dates,
                                                     payoff, eu_exercise)
            option.setPricingEngine(ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(process))

        elif type(derivative) == DigitalOption:
            payoff = ql.CashOrNothingPayoff(option_type_dic[derivative_payoff], strike, 1)
            option = ql.VanillaOption(payoff, eu_exercise)
            process = ql.BlackScholesProcess(ql.QuoteHandle(u),
                                             ql.YieldTermStructureHandle(rf_curve),
                                             ql.BlackVolTermStructureHandle(volatility))
            option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

        else:
            return 'The derivative type has not been specified or is not handled'

        return option.NPV()

    @staticmethod
    def get_vanilla_prices(stock_price, derivatives: List[Derivative], time: float = 0):
        prices = []
        for derivative in derivatives:
            vanilla_type = derivative.payoff
            maturity = derivative.maturity - time
            strike = derivative.strike
            vol = derivative.volatility
            d1 = (np.log(stock_price / strike) + maturity * vol ** 2 / 2) / (vol * np.sqrt(maturity))
            d2 = d1 - vol * np.sqrt(maturity)
            if vanilla_type == 'call':
                prices.append(stock_price * norm.cdf(d1) - strike * norm.cdf(d2))
            elif vanilla_type == 'put':
                prices.append(-stock_price * norm.cdf(-d1) + strike * norm.cdf(-d2))
            else:
                return 'invalid vanilla type'
        return np.concatenate(prices, axis=1)

    @staticmethod
    def get_vanilla_payoffs(stock_price, derivatives: List[Derivative]):
        payoffs = []
        N = len(stock_price)
        for derivative in derivatives:
            strike = derivative.strike
            payoffs.append(np.maximum(np.zeros((N, 1)), stock_price-strike))
        return np.concatenate(payoffs, axis=1)

    @staticmethod
    def get_vanilla_delta_gamma(x, derivative: Derivative):
        delta = []
        gamma = []

        maturity = derivative.maturity
        n_points = derivative.n_points
        strike = derivative.strike
        volatility = derivative.volatility
        dt = maturity/n_points

        for t in range(n_points):
            time_to_maturity = maturity - t*dt
            d1 = (np.log(x[t+2][:, 0] / strike) + time_to_maturity * (volatility ** 2 / 2)) / (
                    volatility * np.sqrt(time_to_maturity))
            delta.append(norm.cdf(d1))
            gamma.append(norm.pdf(d1)/(volatility*x[t+2][:, 0]*np.sqrt(time_to_maturity)))
        return delta, gamma

    @staticmethod
    def get_delta_gamma(x, derivative):
        deltas = []
        gammas = []

        payoff_name = derivative.payoff
        maturity = derivative.maturity
        vol = derivative.volatility
        strike = derivative.strike
        n_points = derivative.n_points
        dt = maturity / n_points

        n_paths = len(x[2])

        derivative_to_price = deepcopy(derivative)

        h = 0.001
        if type(derivative) == VanillaOption:
            if payoff_name == 'put':
                for t in range(n_points):
                    time_to_maturity = maturity - t * dt
                    d1 = (np.log(x[t + 2][:, 0] / strike) + time_to_maturity * (vol ** 2 / 2)) / (
                            vol * np.sqrt(time_to_maturity))
                    deltas.append(-norm.cdf(-d1))
                    gammas.append(norm.pdf(d1) / (vol * x[t + 2][:, 0] * np.sqrt(time_to_maturity)))

            elif payoff_name == 'call':
                for t in range(n_points):
                    time_to_maturity = maturity - t * dt
                    d1 = (np.log(x[t + 2][:, 0] / strike) + time_to_maturity * (vol ** 2 / 2)) / (
                            vol * np.sqrt(time_to_maturity))
                    deltas.append(norm.cdf(d1))
                    gammas.append(norm.pdf(d1) / (vol * x[t + 2][:, 0] * np.sqrt(time_to_maturity)))

            deltas = np.array(deltas).reshape((n_points, n_paths)).T
            gammas = np.array(gammas).reshape((n_points, n_paths)).T

        elif type(derivative) == AsianOption:
            running_acc = np.ones(n_paths)

            p = BSPricer.get_price_ql(derivative_to_price)
            initial_stock_price = derivative_to_price.initial_stock_price
            running_acc *= initial_stock_price

            derivative_to_price.initial_stock_price = initial_stock_price + h
            p_plus = BSPricer.get_price_ql(derivative_to_price)

            derivative_to_price.initial_stock_price = initial_stock_price - h
            p_minus = BSPricer.get_price_ql(derivative_to_price)

            delta = (p_plus - p_minus) / (2 * h)
            gamma = (p_plus - 2 * p + p_minus) / h ** 2

            deltas.append(delta * np.ones(n_paths))
            gammas.append(gamma * np.ones(n_paths))

            for t in range(1, n_points):
                derivative_to_price.n_points = n_points-t
                derivative_to_price.maturity = maturity - t*dt
                stock_price = x[t + 2][:, 0]

                delta = []
                gamma = []

                for price, acc in zip(stock_price, running_acc):

                    derivative_to_price.initial_stock_price = price
                    p = BSPricer.get_price_ql(derivative_to_price, running_acc=acc, past_fixings=t)

                    derivative_to_price.initial_stock_price = price+h
                    p_plus = BSPricer.get_price_ql(derivative_to_price, running_acc=acc, past_fixings=t)

                    derivative_to_price.initial_stock_price = price-h
                    p_minus = BSPricer.get_price_ql(derivative_to_price, running_acc=acc, past_fixings=t)

                    delta.append((p_plus - p_minus) / (2 * h))
                    gamma.append((p_plus - 2 * p + p_minus) / h ** 2)

                running_acc *= stock_price
                deltas.append(delta)
                gammas.append(gamma)
            deltas = np.array(deltas).T
            gammas = np.array(gammas).T

        elif type(derivative) == BarrierOption:
            barrier = derivative.barrier
            barrier_type = derivative.barrier_type
            barrier_touched = np.zeros(n_paths)
            p = BSPricer.get_price_ql(derivative_to_price)
            initial_stock_price = derivative_to_price.initial_stock_price

            derivative_to_price.initial_stock_price = initial_stock_price + h
            p_plus = BSPricer.get_price_ql(derivative_to_price)

            derivative_to_price.initial_stock_price = initial_stock_price - h
            p_minus = BSPricer.get_price_ql(derivative_to_price)

            delta = (p_plus - p_minus) / (2 * h)
            gamma = (p_plus - 2 * p + p_minus) / h ** 2

            deltas.append(delta*np.ones(n_paths))
            gammas.append(gamma*np.ones(n_paths))

            for t in range(1, n_points):
                derivative_to_price = deepcopy(derivative)
                derivative_to_price.n_points = n_points-t
                derivative_to_price.maturity = maturity - t*dt
                stock_price = x[t + 2][:, 0]

                if 'down' in barrier_type:
                    barrier_touched = barrier_touched + (stock_price <= derivative_to_price.barrier)
                else:
                    barrier_touched = barrier_touched + (stock_price >= derivative_to_price.barrier)

                delta = []
                gamma = []

                for price, is_touched in zip(stock_price, barrier_touched):
                    if is_touched and ('in' in barrier_type):
                        exercise_type = derivative.exercise_type
                        derivative_to_price = VanillaOption(payoff_name, exercise_type)
                        derivative_to_price.n_points = n_points - t
                        derivative_to_price.maturity = maturity - t * dt

                        derivative_to_price.initial_stock_price = price
                        p = BSPricer.get_price_ql(derivative_to_price)

                        derivative_to_price.initial_stock_price = price - h
                        p_minus = BSPricer.get_price_ql(derivative_to_price)

                        derivative_to_price.initial_stock_price = price + h
                        p_plus = BSPricer.get_price_ql(derivative_to_price)

                    elif is_touched and ('out' in barrier_type):
                        p = 0
                        p_plus = 0
                        p_minus = 0

                    else:
                        derivative_to_price.initial_stock_price = price
                        p = BSPricer.get_price_ql(derivative_to_price)

                        derivative_to_price.initial_stock_price = price + h
                        p_plus = BSPricer.get_price_ql(derivative_to_price)

                        derivative_to_price.initial_stock_price = price - h
                        p_minus = BSPricer.get_price_ql(derivative_to_price)

                    delta.append((p_plus - p_minus) / (2 * h))
                    gamma.append((p_plus - 2 * p + p_minus) / h ** 2)
                deltas.append(delta)
                gammas.append(gamma)
            deltas = np.array(deltas).T
            gammas = np.array(gammas).T

        elif type(derivative) == DigitalOption:
            p = BSPricer.get_price_ql(derivative_to_price)
            initial_stock_price = derivative_to_price.initial_stock_price

            derivative_to_price.initial_stock_price = initial_stock_price + h
            p_plus = BSPricer.get_price_ql(derivative_to_price)

            derivative_to_price.initial_stock_price = initial_stock_price - h
            p_minus = BSPricer.get_price_ql(derivative_to_price)

            delta = (p_plus - p_minus) / (2 * h)
            gamma = (p_plus - 2 * p + p_minus) / h ** 2

            deltas.append(delta * np.ones(n_paths))
            gammas.append(gamma * np.ones(n_paths))

            for t in range(1, n_points):
                derivative_to_price.n_points = n_points - t
                derivative_to_price.maturity = maturity - t * dt
                stock_price = x[t + 2][:, 0]

                delta = []
                gamma = []

                for price in stock_price:

                    derivative_to_price.initial_stock_price = price
                    p = BSPricer.get_price_ql(derivative_to_price)

                    derivative_to_price.initial_stock_price = price + h
                    p_plus = BSPricer.get_price_ql(derivative_to_price)

                    derivative_to_price.initial_stock_price = price - h
                    p_minus = BSPricer.get_price_ql(derivative_to_price)

                    delta.append((p_plus - p_minus) / (2 * h))
                    gamma.append((p_plus - 2 * p + p_minus) / h ** 2)
                deltas.append(delta)
                gammas.append(gamma)
            deltas = np.array(deltas).T
            gammas = np.array(gammas).T

        return deltas, gammas

    @staticmethod
    def get_delta_gamma_distribution(derivative, stock_prices, time=0):

        if type(derivative) == VanillaOption:
            strike = derivative.strike
            volatility = derivative.volatility
            time_to_maturity = derivative.maturity - time
            d1 = (np.log(stock_prices / strike) + time_to_maturity * (volatility ** 2 / 2)) / (
                    volatility * np.sqrt(time_to_maturity))
            if derivative.payoff == 'call':
                delta_bs = norm.cdf(d1)
            elif derivative.payoff == 'put':
                delta_bs = -norm.cdf(-d1)
            else:
                return 'The payoff has not been specified or is not handled'

            gamma_bs = norm.pdf(d1) / (volatility * stock_prices * np.sqrt(time_to_maturity))

        else:
            derivative_to_price = deepcopy(derivative)
            derivative_to_price.maturity = derivative_to_price.maturity - time
            h = 0.001
            delta_bs = []
            gamma_bs = []
            for stock_price in stock_prices:
                derivative_to_price.initial_stock_price = stock_price
                p = BSPricer.get_price_ql(derivative_to_price)

                derivative_to_price.initial_stock_price = stock_price + h
                p_plus = BSPricer.get_price_ql(derivative_to_price)

                derivative_to_price.initial_stock_price = stock_price - h
                p_minus = BSPricer.get_price_ql(derivative_to_price)

                delta_bs.append((p_plus - p_minus) / (2 * h))
                gamma_bs.append((p_plus - 2 * p + p_minus) / h ** 2)
            delta_bs = np.array(delta_bs)

        return delta_bs, gamma_bs

