import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

class PairsTradingSimulator:
    def __init__(self, symbol1, symbol2, start_date, end_date, entry_k=2, exit_k=0.5, tx_cost=0.0005):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.start_date = start_date
        self.end_date = end_date
        self.entry_k = entry_k        # Entry threshold (z-score multiples)
        self.exit_k = exit_k          # Exit threshold (z-score multiples)
        self.tx_cost = tx_cost        # Transaction cost per trade

        # Load data
        self.data = self.load_data()
        self.hedge_ratio = self.estimate_hedge_ratio()
        self.data['spread'] = self.data[self.symbol1] - self.hedge_ratio * self.data[self.symbol2]
        self.data['zscore'] = (self.data['spread'] - self.data['spread'].mean()) / self.data['spread'].std()

        # Strategy placeholders
        self.data['position'] = 0
        self.data['equity'] = 1.0

    def load_data(self):
        df1 = yf.download(self.symbol1, start=self.start_date, end=self.end_date)['Close']
        df2 = yf.download(self.symbol2, start=self.start_date, end=self.end_date)['Close']
        data = pd.concat([df1, df2], axis=1)
        data.columns = [self.symbol1, self.symbol2]
        return data.dropna()

    def estimate_hedge_ratio(self):
        # Linear regression hedge ratio
        y = self.data[self.symbol1]
        x = add_constant(self.data[self.symbol2])
        model = OLS(y, x).fit()
        return model.params[1]  # Slope = hedge ratio

    def simulate(self):
        position = 0
        equity = 1.0
        equity_curve = []

        for i in range(1, len(self.data)):
            z = self.data['zscore'].iloc[i]

            # Entry rules
            if z > self.entry_k and position == 0:
                position = -1  # Short spread
                equity -= self.tx_cost
            elif z < -self.entry_k and position == 0:
                position = 1   # Long spread
                equity -= self.tx_cost

            # Exit rules
            elif abs(z) < self.exit_k and position != 0:
                position = 0
                equity -= self.tx_cost

            # Daily PnL = position * spread change
            spread_change = self.data['spread'].iloc[i] - self.data['spread'].iloc[i-1]
            equity += position * spread_change / abs(self.data['spread'].iloc[i-1])

            equity_curve.append(equity)
            self.data['position'].iloc[i] = position
            self.data['equity'].iloc[i] = equity

        return pd.Series(equity_curve, index=self.data.index[1:])

    def performance_metrics(self, equity_curve):
        daily_returns = equity_curve.pct_change().dropna()
        cumulative_return = equity_curve.iloc[-1] - 1
        annualized_return = (1 + cumulative_return) ** (252 / len(equity_curve)) - 1
        annualized_vol = daily_returns.std() * np.sqrt(252)
        sharpe = (annualized_return - 0.02) / annualized_vol if annualized_vol > 0 else np.nan
        max_dd = ((equity_curve / equity_curve.cummax()) - 1).min()

        return {
            "Cumulative Return": cumulative_return,
            "Annualized Return": annualized_return,
            "Annualized Volatility": annualized_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd
        }

    def plot(self):
        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(self.data[self.symbol1], label=self.symbol1)
        plt.plot(self.data[self.symbol2], label=self.symbol2)
        plt.legend()
        plt.title("Prices")

        plt.subplot(3, 1, 2)
        plt.plot(self.data['spread'], label="Spread")
        plt.axhline(self.data['spread'].mean(), color='black', linestyle='--')
        plt.title("Spread")

        plt.subplot(3, 1, 3)
        plt.plot(self.data['equity'], label="Equity Curve")
        plt.title("Equity Curve")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    sim = PairsTradingSimulator("AAPL", "MSFT", "2019-01-01", "2020-01-01")
    equity_curve = sim.simulate()
    metrics = sim.performance_metrics(equity_curve)

    print("\nPerformance Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    sim.plot()
