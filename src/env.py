import numpy as np
import random

class Coinworld:
    """
    To create a mock trading environment.
    """

    def __init__(self, series, initial_balance=20, min_trading=10, trading_fee_rate=0.001, exploratory_reward=0.000, holding_penalty=0.001, profit_threshold=20, loss_threshold=5, min_cash_balance=1):
        """
        State = (
            Holding quantity,
            Average price,
            Cash balance,
            Market price,
            Other indicators,
        )
        """
        self.series = series
        self.initial_balance = initial_balance
        self.min_trading = min_trading
        self.trading_fee_rate = trading_fee_rate
        self.exploratory_reward = exploratory_reward
        self.holding_penalty = holding_penalty
        self.profit_threshold = profit_threshold
        self.loss_threshold = loss_threshold
        self.min_cash_balance = min_cash_balance
        self.reset()

    def reset(self):
        market_price_0 = self.series[0, 0]
        self.timestamp = 0

        buy_limit = self.initial_balance / market_price_0
        hold_qty = 0 #float(random.randint(0, 1) * buy_limit)
        avg_price = 0 if hold_qty == 0 else market_price_0
        cash_balance = self.initial_balance - hold_qty * avg_price
        hold_duration = 0

        self.acc_state = (hold_qty, avg_price, cash_balance, hold_duration)
        self.prev_acc_state = (hold_qty, avg_price, cash_balance, hold_duration)
        self.market_state = tuple(self.series[0])

        self.state = self.acc_state + tuple(self.series[0])
        self.prev_state = self.state

        self.goal = tuple(self.series[-1])
        self.done = len(self.series) <= 1
        return np.array(self.state)

    def update_state(self):
        hold_qty_0, avg_price_0, cash_balance_0, hold_duration_0 = self.prev_acc_state
        hold_qty, avg_price, cash_balance, hold_duration = self.acc_state

        if hold_qty_0 == hold_qty:
            hold_duration += 1
        else:
            hold_duration = 0

        self.acc_state = (hold_qty, avg_price, cash_balance, hold_duration)

        self.timestamp += 1
        self.prev_market_state = self.market_state
        self.market_state = tuple(self.series[self.timestamp])
        self.state = self.acc_state + self.market_state

    def Buy(self):
        hold_qty, avg_price, cash_balance, hold_duration = self.acc_state

        if cash_balance < self.min_cash_balance:
            print("Insufficient cash balance to perform buy action.")
            return False

        market_price = self.market_state[0]
        buy_price = market_price * (1 + self.trading_fee_rate)
        buy_limit = cash_balance / buy_price
        buy_qty = buy_limit  # Example: buy half of the limit

        avg_price = (hold_qty * avg_price + buy_price * buy_qty) / (hold_qty + buy_qty)
        hold_qty += buy_qty
        cash_balance -= buy_qty * buy_price

        self.acc_state = (hold_qty, avg_price, cash_balance, hold_duration)
        return True

    def Sell(self):
        hold_qty, avg_price, cash_balance, hold_duration = self.acc_state

        market_price = self.market_state[0]
        sell_price = market_price * (1 - self.trading_fee_rate)
        sell_qty = hold_qty  # Example: sell half of the holdings

        hold_qty -= sell_qty
        cash_balance += sell_qty * sell_price

        self.acc_state = (hold_qty, avg_price, cash_balance, hold_duration)

    def get_reward(self):
        hold_qty_1, avg_price_1, cash_balance_1, hold_duration_1 = self.acc_state
        hold_qty_0, avg_price_0, cash_balance_0, hold_duration_0 = self.prev_acc_state
        market_price = self.market_state[0]
        prev_market_price = self.prev_market_state[0]

        previous_value = hold_qty_0 * avg_price_0 + cash_balance_0
        current_value = hold_qty_1 * avg_price_1 + cash_balance_1
        reward = current_value - previous_value
        # reward += self.exploratory_reward * current_value
        # reward -= current_value * hold_duration_1 * self.holding_penalty

        return reward

    def step(self, action):
        if self.done:
            print('No more steps can be taken')
            return np.array(self.state), 0, self.done

        hold_qty, avg_price, cash_balance, *market_state = self.state
        self.prev_state = self.state

        if action == 1:
            if cash_balance >= self.min_cash_balance:
                success = self.Buy()
                if not success:
                    reward = -self.exploratory_reward  # Penalize for trying to buy with insufficient funds
            else:
                reward = -self.exploratory_reward
        elif action == 2 and hold_qty > 0:
            self.Sell()
        elif action == 0:
            reward = 0  # Penalize for invalid actions or doing nothing

        self.update_state()
        reward = self.get_reward() if action in [1, 2] else reward

        if self.timestamp >= len(self.series) - 1:
            self.done = True
            if self.state[0]*self.state[1]+self.state[2] >= self.initial_balance:
                reward = 10*(self.state[0]*self.state[1]+self.state[2]-self.initial_balance)

        current_balance = hold_qty * self.market_state[0] + cash_balance
        # if current_balance >= self.profit_threshold or current_balance <= self.loss_threshold:
        #     self.done = True

        self.prev_acc_state = self.acc_state

        return np.array(self.state), reward, self.done

    def get_state_space_size(self):
        return np.array(self.state).shape

    def get_action_space_size(self):
        return 3
