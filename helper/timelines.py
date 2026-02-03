import pandas as pd
import numpy as np

def get_timeline_basic(prices: pd.DataFrame, entries: pd.Series, exits: pd.Series, trades: pd.DataFrame) -> pd.DataFrame:
    ''' Generates a timeline from OHLC-data, entries, exits from strategy and true trade data.
    Args:
        prices: OHLC prices.
        entires: signals from strategy, indicating entry points.
        exits: signals from strategy, indicating exit points.
        trades: trade records.

    Returns:
        pd.DataFrame: DataFrame with plot parameters.

    '''
    # Flatten the column MultiIndex by dropping the 'Ticker' level (level 1)
    price_flat = prices.droplevel(1, axis=1)

    # Create the timeline DataFrame with the desired price columns
    timeline = pd.DataFrame({
        'open': price_flat['Open'],
        'high': price_flat['High'],
        'low': price_flat['Low'],
        'close': price_flat['Close'],
    }, index=prices.index)

    # Add system generated entries and exits signals
    timeline['sys_entries'] = entries
    timeline['sys_exits'] = exits

    # Generate eff_entries from Entry Timestamp
    eff_entries = pd.Series(False, index=prices.index)
    eff_entries.loc[trades['Entry Timestamp'].dropna()] = True

    # generates eff_exits from Exit Timestamp
    eff_exits = pd.Series(False, index=prices.index)
    eff_exits.loc[trades['Exit Timestamp'].dropna()] = True

    # Add the boolean series as columns to the plot_parameters DataFrame
    timeline['eff_entries'] = eff_entries
    timeline['eff_exits'] = eff_exits

    return timeline

def get_timeline_extended(prices: pd.DataFrame, entries: pd.Series, exits: pd.Series, trades: pd.DataFrame) -> pd.DataFrame:
    ''' Generates a timeline from OHLC-data, entries, exits from strategy and true trade data.
    Args:
        prices: OHLC prices.
        entires: signals from strategy, indicating entry points.
        exits: signals from strategy, indicating exit points.
        trades: trade records.

    Returns:
        pd.DataFrame: DataFrame with plot parameters.

    '''
    # Flatten the column MultiIndex by dropping the 'Ticker' level (level 1)
    price_flat = prices.droplevel(1, axis=1)

    # Create the timeline DataFrame with the desired price columns
    timeline = pd.DataFrame({
        'open': price_flat['Open'],
        'high': price_flat['High'],
        'low': price_flat['Low'],
        'close': price_flat['Close'],
    }, index=prices.index)

    # Add system generated entries and exits signals
    timeline['sys_entries'] = entries
    timeline['sys_exits'] = exits

    # Generate eff_entries from Entry Timestamp
    eff_entries = pd.Series(False, index=prices.index)
    eff_entries.loc[trades['Entry Timestamp'].dropna()] = True

    # Generate eff_exits_system from Exit Timestamp where exit_type is 'system'
    eff_exits_system = pd.Series(False, index=prices.index)
    system_exits = trades[trades['exit_type'] == 'system']['Exit Timestamp'].dropna()
    eff_exits_system.loc[system_exits] = True

    # Generate eff_exits_stop_loss from Exit Timestamp where exit_type is 'stop loss'
    eff_exits_stop_loss = pd.Series(False, index=prices.index)
    stop_loss_exits = trades[trades['exit_type'] == 'stop_loss']['Exit Timestamp'].dropna()
    eff_exits_stop_loss.loc[stop_loss_exits] = True

    # Generate eff_take_profits from Exit Timestamp where exit_type is 'take profit'
    eff_exits_take_profit = pd.Series(False, index=prices.index)
    take_profit_exits = trades[trades['exit_type'] == 'take_profit']['Exit Timestamp'].dropna()
    eff_exits_take_profit.loc[take_profit_exits] = True

    # Add the boolean series as columns to the plot_parameters DataFrame
    timeline['eff_entries'] = eff_entries
    timeline['eff_exits_system'] = eff_exits_system
    timeline['eff_exits_stop_loss'] = eff_exits_stop_loss
    timeline['eff_exits_take_profit'] = eff_exits_take_profit

    return timeline