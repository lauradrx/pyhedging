
class Derivative:
    def __init__(self, payoff, exercise_type, strike=1):
        self.payoff = payoff
        self.exercise_type = exercise_type
        self.strike = strike

    initial_stock_price: float = 1
    strike: float
    volatility: float = 0.25
    rate: float = 0
    maturity: float = 30/360
    payoff: str  # call or put
    n_points: int = 30
    exercise_type: str  # european or asian


class BarrierOption(Derivative):
    def __init__(self, payoff, barrier, barrier_type, strike=1):
        super().__init__(payoff, 'european', strike)
        self.barrier = barrier
        self.barrier_type = barrier_type

    barrier: float = None
    barrier_type: str = None  # down_in, down_out, up_in, up_out


class DigitalOption(Derivative):
    def __init__(self, payoff, strike=1):
        super().__init__(payoff, 'european', strike)

    cash = 1


class AsianOption(Derivative):
    def __init__(self, payoff, strike=1):
        super().__init__(payoff, 'asian', strike)


class VanillaOption(Derivative):
    def __init__(self, payoff, exercise_type, strike=1):
        super().__init__(payoff, exercise_type, strike)
