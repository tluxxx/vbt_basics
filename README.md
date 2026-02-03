The repository contains notebooks demonstrating the use of vectorbt (vbt) for backtesting, with a focus on the vbt-internal class method Portfolio.from_signals().

The notebooks accompany a series of blog posts published on Medium.

Content

- Part 1 — vbt_test_01.ipynb  -- Setting the scene and getting started with vectorbt. Overview of core methods and parameters for:
  - Metric calculation
  - Plotting
  - Logging

- Part 2 — vbt_test_02.ipynb -- Trade prices (default and modified) and the impact of parameters such as:
  - size_type, size, min_size, max_size
  - Fees and slippage
  - Standard Stop Loss and Take Profit behavior

- Part 3 — vbt_test_03.ipynb -- Callback functions in vectorbt, fosus on customizing stop logic

- Part 4 — vbt_test_04.ipynb -- Advanced order handling:
  - Multiple orders per position
  - Impact of accumulation and accept_partial
  - Stop Loss and Take Profit with multiple orders
  - Short-only and long–short trading

- Part 5 — vbt_test_05.ipynb -- Multi-ticker portfolios and advanced portfolio structures:
  - Standard outputs and metrics
  - Cash distribution and cash pooling
  - Cash pooling with and without order size limitations
  - Grouping of tickers
  - Group-level cash pools
  - Long–short strategies in multi-ticker portfolios

- Part 6 — vbt_test_06.ipynb -- Practical use cases:
  - Batch testing of multiple strategies
  - Parameter optimization
  - Sector rotation
  - And more
