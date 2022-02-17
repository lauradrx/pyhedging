import keras.backend as K


class ExponentialUtility:
    def __init__(self, lam=5):
        self.lam = lam

    def loss_function(self, y_true, y_pred):
        return 1 / self.lam * K.log(K.mean(K.exp(-self.lam * y_pred)))
