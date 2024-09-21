import numpy as np
class State:
    def __init__(
            self, 
            # Market state
            timestamp=0,
            series= np.zeros([1,len(mapping)]),
            mapping:dict=mapping
        ):
        market_state = series[timestamp]
        self.market_state = market_state
        self.timestamp=0,
        self.open = market_state[mapping['open']]
        self.high = market_state[mapping['high']]
        self.low = market_state[mapping['low']]
        self.close = market_state[mapping['close']]
        self.volume = market_state[mapping['volume']]

    def __repr__(self):
        fixed_width = 15
        message = '    |'.join(f'{i}: {j:.2f}'.ljust(fixed_width) for i, j in zip(mapping.keys(), self.market_state))
        return message
    
    def __call__(self):
        return {i: j for i,j in zip(mapping.keys(), self.market_state)}

class Account:
    
    def __init__(self, 
                 state={},
                 initial_balance=initial_balance,
                 min_trading=min_trading,
                 trading_fee_rate=trading_fee_rate
                 ):

        price                = state['close']

        self.initial_balance = initial_balance
        self.min_trading     = min_trading
        self.trading_fee_rate= trading_fee_rate
        self.coin_qty        = 0
        self.coin_cost       = 0
        self.cash_balance    = initial_balance 
        self.net_worth       = self.coin_qty * price + self.cash_balance

        # self.buy_limit    = initial_balance / price
        # self.coin_qty     = random.randint(0, 1) * self.buy_limit
        # self.coin_cost    = 0 if self.coin_qty == 0 else price
        # self.cash_balance = initial_balance - self.coin_qty * price
    
    def _buy(self, price):
        if self.cash_balance < self.min_trading:
            print("Insufficient cash balance to perform buy action.")
            return False

        else:
            hold_qty     = self.coin_qty
            avg_price    = self.coin_cost
            buy_price    = price * (1 + self.trading_fee_rate)
            buy_limit    = self.cash_balance / buy_price
            buy_qty      = buy_limit  # Example: buy half of the limit

            self.coin_cost    = (hold_qty * avg_price + buy_price * buy_qty) / (hold_qty + buy_qty)
            self.coin_qty    += buy_qty
            self.cash_balance-= buy_qty * buy_price
            self.net_worth    = self.coin_qty * price + self.cash_balance

            print(f'Buy {buy_qty:.5f} at {buy_price}')

            return True

    def _sell(self, price):
        # hold_qty     = self.coin_qty
        avg_price    = self.coin_cost 

        sell_price   = price * (1 - self.trading_fee_rate)
        sell_qty     = self.coin_qty  # Example: sell half of the holdings

        if sell_qty == 0:
            return False
        else:
            self.cash_balance += sell_qty * sell_price
            self.coin_qty     -= sell_qty
            self.coin_cost     = 0 
            self.net_worth     = self.coin_qty * self.coin_cost + self.cash_balance

            print(f'Sell {sell_qty:.5f} at {sell_price}')


    def __repr__(self):
        return f'coin_qty: {self.coin_qty:.5f} |coin_cost: {self.coin_cost:.2f} |cash_balance: {self.cash_balance:.2f} |net_worth: {self.net_worth:.2f}' 
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return [i for i in self.__dict__.values()]