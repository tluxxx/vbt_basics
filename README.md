The repo contains notebooks concerning the use of vectorbt (vbt) for backtesting, focussing on the vbt-internal class method portfolio.from_signals(). 

The notebooks, refer to a series of blogs on medium.com

CONTENT: 

Part 1: vbt_test_01.ipynb; Setting the scene, getting started with vbt, Overview on methods and parameters (for calculation of metrics, plotting, logging etc.)
Part 2: vbt_test_02.ipynb; trade-prices (default/modified), impact of parameters as size_type, size, min_size, max_size, fees, slippage, standard StopLoss and TakeProfit-behaviour
Part 3: vbt_test_03.ipynb; callback-functions in vbt, using callbacks for customizing Stops
Part 4: vbt_test_04.ipynb; multiple orders per position, impact of accumulation, accept_partial, StopLoss and TakeProfit for multiple orders, ShortOnly trades, LongShort trades
Part 5: vbt_test_05.ipynb; multi-ticker portfolio and standard outputs, cash distributions, cash-pooling (with and without order size limitations), groups of tickers, group-cash-pools (with and without order size-limitations), LongShortStrategies in multiticker-portfolios
Part 6: vbt_test_06.ipynb; some use-cases as batch-testing of different strategies, optimisation of parameters of trading strategies, sector rotation, etc…

