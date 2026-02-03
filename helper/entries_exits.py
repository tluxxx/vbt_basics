import pandas as pd
import numpy as np


def classify_exit_types(trades: pd.DataFrame, take_profit: float,
                        exits: pd.Series, eps: float = 1e-08) -> pd.DataFrame:
    ''' Classify trade exits into take-profit, system, or stop-loss.
    Args
        trades: pf.trades.records_readable DataFrame
        take_profit: Take-profit threshold (e.g. 0.15)
        exits: Exit signal series indexed by datetime
        eps : optional, floating point tolerance
    Returns
        trades:  DataFrame with added 'exit_type' column
    '''
    # align exit signals to trade exit timestamps
    exit_signal_at_exit = (exits.reindex(trades['Exit Timestamp'].dt.normalize()).fillna(False).to_numpy())

    # find take-profit hit (float-safe)
    tp_hit = trades['Return'].to_numpy() >= (take_profit - eps)

    # classify exits into categories: TakeProfit, (trailing) StopLoss and System
    trades = trades.copy()
    trades['exit_type'] = np.select(
        [tp_hit, exit_signal_at_exit],
        ['take_profit','system'],
        default='stop_loss',
        )

    return trades
    

def lense_at_entry_exit(timeline: pd.DataFrame, entry_date: str, exit_date: str, n_rows: int):
    ''' Slices and displays the timeline around a given entry_date and exit_date (+-n_rows)
    Args:
        timeline: The DataFrame containing the timeline data.
        entry_date: The entry date in 'YYYY-MM-DD' format.
        exit_date: The exit date in 'YYYY-MM-DD' format.
        n_rows: The number of rows to display before and after the entry/exit dates.
    '''
    # Convert date strings to datetime objects for accurate index lookup
    entry_ts = pd.Timestamp(entry_date)
    exit_ts = pd.Timestamp(exit_date)

    # --- Entry Date Analysis ---
    print(f"### Timeline around Entry Date: {entry_date} ({n_rows} rows before and after)\n")
    entry_loc = timeline.index.get_loc(entry_ts)
    start_entry_idx = max(0, entry_loc - n_rows)
    end_entry_idx = min(len(timeline) - 1, entry_loc + n_rows)
    display(timeline.iloc[start_entry_idx : end_entry_idx + 1]) # +1 to include the end_entry_idx
    print("\n")

    # --- Exit Date Analysis ---
    print(f"### Timeline around Exit Date: {exit_date} ({n_rows} rows before and after)\n")
    exit_loc = timeline.index.get_loc(exit_ts)
    start_exit_idx = max(0, exit_loc - n_rows)
    end_exit_idx = min(len(timeline) - 1, exit_loc + n_rows)
    display(timeline.iloc[start_exit_idx : end_exit_idx + 1]) # +1 to include the end_exit_idx
    print("\n")