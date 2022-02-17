from pyhedging.Derivative import Derivative, BarrierOption, AsianOption, VanillaOption, DigitalOption
from pyhedging.BSDataset import BSDataset
from pyhedging.BSPricer import BSPricer
from pyhedging.ExponentialUtility import ExponentialUtility

import tensorflow as tf
from keras.models import Model
from keras.layers.core import Dense
from keras import Input
from tensorflow.keras.optimizers import Adam
from keras.layers import Subtract, Multiply, Lambda, Add, Concatenate, Minimum, Average, Maximum
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from scipy.stats import norm


class RNNModel:
    def __init__(self, derivative,
                 derivative_price=0,
                 neurons=None,
                 hedging_inst: List[Derivative] = None):
        if neurons is None:
            neurons = [256, 128]
        self._layers = len(neurons) + 1 if neurons else 1
        self._neurons = neurons
        self._n_points = derivative.n_points
        self._strike = derivative.strike
        self._payoff_name = derivative.payoff
        self._exercise_type = derivative.exercise_type
        self._dt = derivative.maturity / self._n_points
        self._model = None
        self._derivative = derivative
        self._price = derivative_price
        self._x_train = None
        self._y_train = None
        self._neurons = neurons
        self._loss_name = None
        self._lam = None
        self._derivative_name = None
        self._train_size = None
        if hedging_inst is None:
            self._n_hedging_inst = 1
        else:
            self._n_hedging_inst = len(hedging_inst) + 1
        self._hedging_inst = hedging_inst

    def build(self):
        hedging_inst = Input(shape=(self._n_hedging_inst,))
        pnl = Input(shape=(1,))
        strategy = Input(shape=(self._n_hedging_inst,))
        inputs = [pnl, strategy, hedging_inst]
        layers = []

        for i in range(self._layers - 1):
            layer = Dense(self._neurons[i], activation='relu',
                          kernel_initializer='random_normal',
                          bias_initializer='zeros',
                          name=str(i + 1))
            layers.append(layer)

        layer = Dense(self._n_hedging_inst, activation='linear',
                      kernel_initializer='random_normal',
                      bias_initializer='zeros',
                      name=str(self._layers))
        layers.append(layer)

        for t in range(self._n_points):
            layer_input = Concatenate(axis=1)([hedging_inst, strategy])
            for i in range(self._layers - 1):
                temp = layers[i](layer_input)
                layer_input = temp
            strategy = layers[self._layers - 1](layer_input)

            next_hedging_inst = Input(shape=(self._n_hedging_inst,))

            change_in_price = Subtract()([next_hedging_inst, hedging_inst])
            pnl_t = Lambda(lambda x: K.sum(x, axis=1))(Multiply()([strategy, change_in_price]))
            pnl = Add()([pnl_t, pnl])

            hedging_inst = next_hedging_inst
            inputs.append(hedging_inst)

        payoff = self._get_payoff(inputs)
        pnl = Subtract()([pnl, payoff])
        self._model = Model(inputs, outputs=pnl)

    def _get_payoff(self, inputs):
        stock_price = inputs[-1]

        if type(self._derivative) == VanillaOption:
            if self._payoff_name == 'call':
                if self._exercise_type == 'european':
                    payoff = Lambda(lambda x: K.relu(x[:, 0] - self._strike))(stock_price)
                    self._derivative_name = 'european_call'
                else:
                    return 'The exercise type has not been specified or is not handled'
            elif self._payoff_name == 'put':
                if self._exercise_type == 'european':
                    payoff = Lambda(lambda x: K.relu(self._strike - x[:, 0]))(stock_price)
                    self._derivative_name = 'european_put'
                else:
                    return 'The exercise type has not been specified or is not handled'
            else:
                return 'The payoff has not been specified or is not handled'

        elif type(self._derivative) == BarrierOption:
            if self._payoff_name == 'put':
                if self._derivative.barrier_type == 'down_in':
                    min_stock_price = Minimum()(inputs[2:])
                    barrier = tf.constant(self._derivative.barrier, dtype=tf.float32)
                    is_down = tf.cast(tf.math.less_equal(min_stock_price[:, 0], barrier), dtype=tf.float32)
                    put_payoff = Lambda(lambda x: K.relu(self._strike - x[:, 0]))(stock_price)
                    payoff = Multiply()([is_down, put_payoff])
                    self._derivative_name = 'put_down_in'
                else:
                    return 'The barrier type has not been specified or is not handled'
            elif self._payoff_name == 'call':
                if self._derivative.barrier_type == 'up_in':
                    max_stock_price = Maximum()(inputs[2:])
                    barrier = tf.constant(self._derivative.barrier, dtype=tf.float32)
                    is_up = tf.cast(tf.math.greater_equal(max_stock_price[:, 0], barrier), dtype=tf.float32)
                    call_payoff = Lambda(lambda x: K.relu(x[:, 0] - self._strike))(stock_price)
                    payoff = Multiply()([is_up, call_payoff])
                    self._derivative_name = 'call_down_in'
                else:
                    return 'The barrier type has not been specified or is not handled'
            else:
                return 'The payoff has not been specified or is not handled'

        elif type(self._derivative) == AsianOption:
            avg_price = Average()(inputs[2:])
            if self._payoff_name == 'call':
                payoff = Lambda(lambda x: K.relu(x[:, 0] - self._strike))(avg_price)
                self._derivative_name = 'asian_call'
            elif self._payoff_name == 'put':
                payoff = Lambda(lambda x: K.relu(self._strike - x[:, 0]))(avg_price)
                self._derivative_name = 'asian_put'
            else:
                return 'The payoff has not been specified or is not handled'

        elif type(self._derivative) == DigitalOption:
            if self._payoff_name == 'call':
                payoff = tf.cast(tf.math.greater_equal(stock_price[:, 0], self._strike), dtype=tf.float32)
                self._derivative_name = 'digital_call'
            elif self._payoff_name == 'put':
                payoff = tf.cast(tf.math.less_equal(stock_price[:, 0], self._strike), dtype=tf.float32)
                self._derivative_name = 'digital_put'
            else:
                return 'The payoff has not been specified or is not handled'

        else:
            return 'The derivative is not handled'

        return payoff

    def train(self, loss_name='mean_squared_error', lam=None, batch_size=32, epochs=5, lr=None, n_cycle=1,
              n_paths=10 ** 5):
        self._loss_name = loss_name

        if loss_name == 'exponential_utility':
            exponential_utility = ExponentialUtility(lam=lam)
            loss = exponential_utility.loss_function
            self._lam = lam
        elif loss_name == 'mean_squared_error':
            loss = loss_name
        else:
            return 'loss name not valid'

        self._train_size = n_paths

        lr = lr if lr else [0.01, 0.001]

        dataset = BSDataset(self._derivative, self._price, n_sim=n_paths)
        self._x_train, self._y_train = dataset.generate_dataset(self._hedging_inst, rnn=True)

        for i in range(n_cycle):
            opt = Adam(learning_rate=lr[0])
            self._model.compile(optimizer=opt, loss=loss)
            self._model.fit(self._x_train, self._y_train, epochs=epochs, batch_size=batch_size)
            opt = Adam(learning_rate=lr[1])
            self._model.compile(optimizer=opt, loss=loss)
            self._model.fit(self._x_train, self._y_train, epochs=epochs, batch_size=batch_size)

    def evaluate_train(self):
        return self._model.evaluate(self._x_train, self._y_train, batch_size=self._train_size)

    def build_hist(self, bins=100, xmin=-0.03, xmax=0.03, save=False):
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(10, 5))
        y_pred = self._model.predict(self._x_train)
        if self._loss_name == 'exponential_utility':
            y_pred += self._model.evaluate(self._x_train, self._y_train, batch_size=self._train_size)
        plt.xlim(xmin, xmax)
        plt.hist(y_pred, bins=bins, density=True)
        plt.show()

        if save:
            fig_name = 'RNN'+self._derivative_name+'_pnl_distribution_'+self._loss_name
            if self._loss_name == 'exponential_utility':
                fig_name += '_' + str(self._lam)
            fig_name += '.png'
            fig.savefig(fig_name)

    def test(self, x_test, y_test):
        return self._model.evaluate(x_test, y_test)

    def get_sample_paths(self, x_test=None, n_paths=10, save=False):

        if not x_test:
            dataset = BSDataset(derivative=self._derivative, derivative_price=self._price, n_sim=n_paths)
            x_test, y_test = dataset.generate_dataset(self._hedging_inst, rnn=True)
        else:
            assert len(x_test[0]) == n_paths

        strategies = []
        stock_price = Input(shape=(self._n_hedging_inst,))
        strategy = Input(shape=(self._n_hedging_inst,))
        input_layer = Concatenate(axis=1)([stock_price, strategy])
        for i in range(self._layers):
            layer = self._model.get_layer(str(i + 1))
            output = layer(input_layer)
            input_layer = output

        sub_model = Model([stock_price, strategy], outputs=output)

        strategy = np.zeros((n_paths, self._n_hedging_inst))
        for t in range(self._n_points):
            sub_model_input = [x_test[t + 2], strategy]
            strategy = sub_model.predict(sub_model_input)
            strategies.append(strategy)

        delta_bs, gamma_bs = BSPricer.get_delta_gamma(x_test, self._derivative)

        if self._n_hedging_inst >= 2:
            strategies_stock = np.array(strategies)[:, :, 0].reshape((self._n_points, n_paths)).T
            delta_nn = strategies_stock
            gamma_nn = np.zeros((n_paths, self._n_points))

            i = 1
            for hedging_inst in self._hedging_inst:
                delta_hedging_inst, gamma_hedging_inst = BSPricer.get_delta_gamma(x_test, hedging_inst)

                strategies_derivative = np.array(strategies)[:, :, i].reshape((self._n_points, n_paths)).T
                delta_nn += strategies_derivative * delta_hedging_inst
                gamma_nn += strategies_derivative * gamma_hedging_inst
                i += 1

            fig = plt.figure(figsize=(15, 15))
            fig.suptitle('Gamma', fontsize=24)
            for i in range(n_paths):
                plt.subplot(5, 2, i + 1)
                plt.plot(gamma_nn[i, :], label='NN')
                plt.plot(gamma_bs[i, :], label='BS')
                plt.legend()
            plt.show()

        else:
            delta_nn = strategies
            delta_nn = np.array(delta_nn).reshape((self._n_points, n_paths)).T

        fig = plt.figure(figsize=(15, 15))
        title = 'Delta'
        if self._loss_name == 'exponential_utility':
            title = title + ', lambda = ' + str(self._lam)
        fig.suptitle(title, fontsize=24)
        for i in range(n_paths):
            plt.subplot(5, 2, i + 1)
            plt.plot(delta_nn[i, :], label='NN')
            plt.plot(delta_bs[i, :], label='BS')
            plt.legend()
        plt.show()

        if save:
            fig_name = 'RNN' + self._derivative_name+'_strategy_sample_paths_'+self._loss_name
            if self._loss_name == 'exponential_utility':
                fig_name += '_'+str(self._lam)
            fig_name += '.png'
            fig.savefig(fig_name)

    def build_delta(self, n=40, save=False):
        stock_price = Input(shape=(1,))
        time = Input(shape=(1,))
        input_layer = Concatenate(axis=1)([stock_price, time])
        for i in range(self._layers):
            layer = self._model.get_layer(str(i + 1))
            output = layer(input_layer)
            input_layer = output

        sub_model = Model([stock_price, time], outputs=output)

        stock_price = np.linspace(0.6, 1.4, n)
        sub_model_input = [stock_price.reshape((-1, 1)), np.zeros((n, 1))]
        strategies = np.array(sub_model.predict(sub_model_input))
        delta_nn = strategies[:, 0]

        delta_bs, gamma_bs = BSPricer.get_delta_gamma_distribution(self._derivative, stock_price)

        fig = plt.figure(figsize=(12, 8))
        title = 'Delta vs Stock Price'
        if self._loss_name == 'exponential_utility':
            title = title + ', lambda = ' + str(self._lam)
        plt.title(title)
        plt.plot(stock_price, delta_nn, label='NN')
        plt.plot(stock_price, delta_bs, label='BS')
        plt.legend()
        plt.show()

        if save:
            fig_name = 'RNN'+self._derivative_name+'_delta_'+self._loss_name
            if self._loss_name == 'exponential_utility':
                fig_name += '_'+str(self._lam)
            fig_name += '.png'
            fig.savefig(fig_name)

    def build_delta_gamma(self, n=100, save=False):
        stock_price = Input(shape=(self._n_hedging_inst,))
        strategy = Input(shape=(self._n_hedging_inst,))
        input_layer = Concatenate(axis=1)([stock_price, strategy])
        for i in range(self._layers):
            layer = self._model.get_layer(str(i + 1))
            output = layer(input_layer)
            input_layer = output

        sub_model = Model([stock_price, strategy], outputs=output)

        stock_price = np.linspace(0.6, 1.4, n)
        c = BSPricer.get_vanilla_prices(stock_price.reshape((-1, 1)), self._hedging_inst)
        sub_model_input = [np.concatenate((stock_price.reshape((-1, 1)), c), axis=1),
                           np.zeros((n, self._n_hedging_inst))]
        strategies = np.array(sub_model.predict(sub_model_input))
        strategies_stock = strategies[:, 0]

        i = 1
        delta_nn = strategies_stock
        gamma_nn = np.zeros(n)
        for hedging_inst in self._hedging_inst:
            strategies_derivative = strategies[:, i]

            strike = hedging_inst.strike
            volatility = hedging_inst.volatility
            time_to_maturity = hedging_inst.maturity

            d1 = (np.log(stock_price / strike) + time_to_maturity * (volatility ** 2 / 2)) / (
                    volatility * np.sqrt(time_to_maturity))

            delta_hedging_inst = norm.cdf(d1)
            gamma_hedging_inst = norm.pdf(d1) / (volatility * stock_price * np.sqrt(time_to_maturity))
            delta_nn += strategies_derivative * delta_hedging_inst

            gamma_nn += strategies_derivative * gamma_hedging_inst

            i += 1

        delta_bs, gamma_bs = BSPricer.get_delta_gamma_distribution(self._derivative, stock_price)

        fig = plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1)
        title = 'Delta vs Stock Price'
        if self._loss_name == 'exponential_utility':
            title = title + ', lambda = ' + str(self._lam)
        plt.title(title)
        plt.plot(stock_price, delta_nn, label='NN')
        plt.plot(stock_price, delta_bs, label='BS')
        plt.legend()
        plt.subplot(1, 2, 2)
        title = 'Gamma vs Stock Price'
        if self._loss_name == 'exponential_utility':
            title = title + ', lambda = ' + str(self._lam)
        plt.title(title)
        plt.plot(stock_price, gamma_nn, label='NN')
        plt.plot(stock_price, gamma_bs, label='BS')
        plt.legend()
        plt.show()

        if save:
            fig_name = 'RNN'+self._derivative_name+'_delta_gamma_'+self._loss_name
            if self._loss_name == 'exponential_utility':
                fig_name += '_'+str(self._lam)
            fig_name += '.png'
            fig.savefig(fig_name)
