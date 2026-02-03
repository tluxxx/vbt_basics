import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _is_boolean_series(s: pd.Series) -> bool:
    ''' # consider boolean dtype or series that only contains True/False (ignoring NaN) '''
    if s.dtype == bool:
        return True
    uniq = s.dropna().unique()
    return set(uniq).issubset({True, False})

def plot_multi_subplot_trading_data(
    df: pd.DataFrame,
    subplot_config: list[dict],
    size: tuple[int, int] = (1400, 800),
    shared_xaxis: bool = True,
    title: str = None,
    theme: str = "plotly",
    default_triangle_size: int = 18,
    connect_gaps: bool = False  # whether to connect gaps for line traces
):
    ''' Robust multi-subplot trading chart. Supports stepped position traces via 'line_shape' in subplot_config.
    - For scatter traces you can pass 'line_shape' (e.g. 'hv') in trace dict.
    - Boolean signal columns for scatter_markers are plotted at the 'close' price on dates where True.
    '''

    df = df.sort_index() # ensure df is sorted by index (very important for step plotting)

    n_subplots = len(subplot_config)
    row_heights = [cfg.get("height_ratio", 1) for cfg in subplot_config]
    subplot_titles = [cfg.get("title", "") for cfg in subplot_config]

    fig = make_subplots(
        rows=n_subplots,
        cols=1,
        shared_xaxes=shared_xaxis,
        vertical_spacing=0.02,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    for i, cfg in enumerate(subplot_config, start=1):
        traces = cfg.get("traces", [])

        # Add candlesticks first (if present)
        for trace_cfg in [t for t in traces if t.get("type") == "candlestick"]:
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df[trace_cfg["open"]],
                    high=df[trace_cfg["high"]],
                    low=df[trace_cfg["low"]],
                    close=df[trace_cfg["close"]],
                    name=trace_cfg.get("name", "Price"),
                ),
                row=i, col=1,
            )

        # Then add other traces (overlaid above candles)
        for trace_cfg in [t for t in traces if t.get("type") != "candlestick"]:
            ttype = trace_cfg.get("type")

            if ttype == "scatter":
                col = trace_cfg["y"]
                y_series = df[col]

                # Convert boolean -> numeric if necessary for clean plotting
                if _is_boolean_series(y_series):
                    # convert True->1, False->0 (or keep as int categories - depends on user)
                    y_vals = y_series.astype(float)
                else:
                    y_vals = y_series

                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=y_vals,
                        mode=trace_cfg.get("mode", "lines"),
                        name=trace_cfg.get("name", col),
                        line=trace_cfg.get("line", {}),
                        line_shape=trace_cfg.get("line_shape", "linear"),
                        connectgaps=connect_gaps,
                    ),
                    row=i, col=1,
                )

            elif ttype == "scatter_markers":
                col = trace_cfg["y"]
                marker = trace_cfg.get("marker", {}).copy()
                symbol = marker.get("symbol", "")
                if symbol.startswith("triangle"):
                    marker.setdefault("size", default_triangle_size)
                # small outline so triangles show against candles
                marker.setdefault("line", dict(width=1, color="black"))

                # If column is boolean -> plot only at close price for True rows
                if _is_boolean_series(df[col]):
                    mask = df[col] == True
                    x_vals = df.index[mask]
                    y_vals = df.loc[mask, "close"]
                else:
                    # numeric or other series: plot at series values
                    x_vals = df.index
                    y_vals = df[col]

                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="markers",
                        name=trace_cfg.get("name", col),
                        marker=marker,
                    ),
                    row=i, col=1,
                )

            else:
                raise ValueError(f"Unknown trace type: {ttype}")

    # Put legend outside on the right; add right margin for it
    fig.update_layout(
        height=size[1],
        width=size[0],
        title=title,
        template=theme,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            borderwidth=1,
        ),
        margin=dict(t=60, b=40, l=60, r=240),
    )

    return fig
    
def plot_signals_and_positions(trades, timeline, entries, exits, ticker, title):
    # Create boolean Series for all effective entries and exits
    effective_entries = pd.Series(False, index=timeline.index)
    effective_exits = pd.Series(False, index=timeline.index)

    entry_timestamps = trades['Entry Timestamp'].dropna()
    effective_entries.loc[entry_timestamps] = True
    exit_timestamps = trades['Exit Timestamp'].dropna()
    effective_exits.loc[exit_timestamps] = True

    # Plotting
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=[f'Original Strategy Signals', f'Effective Trading Signals', 'Position', 'Assets', 'Cash'])

    # Original Entry and Exit Signals
    fig.add_trace(
        go.Bar(x=entries.index, y=entries.astype(int),
               name='Original Entries', marker_color='lightgreen', showlegend=True),
        row=1, col=1)

    fig.add_trace(
        go.Bar(x=exits.index, y=(exits.astype(int) * -1),
               name='Original Exits', marker_color='salmon', showlegend=True),
        row=1, col=1)

    # Add plot for effective Entries and Exits
    fig.add_trace(
        go.Bar(x=effective_entries.index, y=effective_entries.astype(int),
               name='Effective Entries', marker_color='green', showlegend=True),
        row=2, col=1)

    fig.add_trace(
        go.Bar( x=effective_exits.index, y=(effective_exits.astype(int) * -1),
               name='Effective Exits', marker_color='red', showlegend=True),
        row=2, col=1)

    # Add specific markers for different effective exit types
    mask = timeline.eff_exits_system.astype(bool)
    fig.add_trace(
        go.Scatter(x=timeline.index[mask], y=[-0.5] * mask.sum(),
                   name='system-exit', mode='markers',
                   marker=dict(color='red', symbol='triangle-down', size=10), showlegend=True),
        row=2, col=1)

    mask = timeline.eff_exits_stop_loss.astype(bool)
    fig.add_trace(
        go.Scatter(x=timeline.index[mask], y=[-0.5] * mask.sum(),
                   name='stop-loss-exit', mode='markers',
                   marker=dict(color='blue', symbol='triangle-down', size=10), showlegend=True),
        row=2, col=1)

    mask = timeline.eff_exits_take_profit.astype(bool)
    fig.add_trace(
        go.Scatter(x=timeline.index[mask], y=[-0.5] * mask.sum(),
                   name='take-profit-exit', mode='markers',
                   marker=dict(color='gray', symbol='triangle-down', size=10), showlegend=True),
        row=2, col=1)

    # Add plots for position, assets and cash
    fig.add_trace(
        go.Scatter(x=timeline.index, y=timeline['position'],
                   name='Position', mode='lines', line=dict(color='lightgreen', width=2), fill='tozeroy'),
        row=3, col=1)

    fig.add_trace(
        go.Scatter(x=timeline.index, y=timeline['assets'],
                   name='Assets', mode='lines', line=dict(color='blue', width=2), fill='tozeroy'),
        row=4, col=1)

    fig.add_trace(
        go.Scatter(x=timeline.index, y=timeline['cash'],
                   name='Cash', mode='lines', line=dict(color='magenta', width=2), fill='tozeroy'),
        row=5, col=1)

    # finetuning
    fig.update_layout(
        title_text= title,
        yaxis_title='Signal (1=Entry, -1=Exit)',
        template='plotly',
        height=600,
        width=1200,
        bargap=0) 

    fig.show()
    
def plot_portfolio_positions(pos: pd.DataFrame) -> go.Figure:
    ''' Plots the positions for each portfolio member over time.
    Args:
        pos:    DataFrame, index is Date, columns are tickers, values representing position
                (e.g., 1 for long, -1 for short, 0 for flat).
    Returns:
        Figure: displaying the positions of all portfolio members.
    '''
    fig = make_subplots(rows=len(pos.columns), cols=1, subplot_titles=pos.columns, vertical_spacing=0.2, shared_xaxes=True)

    for i, col in enumerate(pos.columns):
        fig.add_trace(
            go.Scatter(x=pos.index, y=pos[col].clip(lower=0),
                       mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='lightgreen', showlegend=False),
            row=i+1, col=1)

        fig.add_trace(
            go.Scatter(x=pos.index, y=pos[col].clip(upper=0),
                       mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='lightcoral', showlegend=False),
            row=i+1, col=1)

        fig.add_trace(
            go.Scatter(x=pos.index, y=pos[col],
                       mode='lines', line=dict(width=2, color='blue', shape='hv'), name=col),
            row=i+1, col=1)

        fig.update_yaxes(tickvals=[-1, 0, 1], row=i+1, col=1)

        if i < len(pos.columns) - 1:
            fig.update_xaxes(showticklabels=False, row=i+1, col=1)
        else:
            fig.update_xaxes(showticklabels=True, row=i+1, col=1)

    fig.update_layout(height=100 * len(pos.columns),
                      width = 1000,
                      title_text="Portfolio Member Positions",
                      showlegend=False)
    return fig
    
def build_position_timeline(trades: pd.DataFrame, close: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    '''Build a timeline of positions (1, -1, or 0) for each ticker based on trade entry/exit timestamps
    Args:
        trades:         Dataframe with key datas of trades (from pf.trades.records_readable)
        close:          DataFrame with close-values for each ticker, index = Date
        tickers:        list of ticker
    Returns:
        pos_timeline:   DataFrame, index = Date, columns = tickers, values representing position
    '''

    # preparations
    tr = trades.copy()
    tr['Entry Timestamp'] = pd.to_datetime(tr['Entry Timestamp'])
    tr['Exit Timestamp'] = pd.to_datetime(tr['Exit Timestamp'])
    tr['Position'] = tr['Direction'].map({'Long': 1, 'Short': -1})
    pos_timeline = pd.DataFrame(np.nan, index=close.index, columns=tickers)

    # Handle entry timestamps: set position to 1 or -1
    entry_df = tr[['Column', 'Entry Timestamp', 'Position']].set_index('Entry Timestamp')
    pos_timeline.update(entry_df.pivot(columns='Column', values='Position')) # widening, each ticker gets a column with values = Positions (or NaN)

    # Handle exit timestamps: set position to 0
    exit_df = tr[['Column', 'Exit Timestamp']].set_index('Exit Timestamp')
    exit_df['Position'] = 0
    pos_timeline.update(exit_df.pivot(columns='Column', values='Position'))# widening, each ticker gets a column with values = Positions (or NaN)

    # final processing
    pos_timeline.ffill(inplace=True)
    pos_timeline.fillna(0, inplace=True)

    return pos_timeline
    
def plot_positions_stacked(pos: pd.DataFrame) -> None:
    ''' Plots the positions for each portfolio member over time.
     Args:
        pos: DataFrame, index is Date, columns are tickers, values representing position
             (e.g., 1 for long, -1 for short, 0 for flat).
    Returns:
        None: displays a stacked Plotly figure.
    '''
    # Clean small positions
    pos_clean = pos.copy()
    pos_clean[np.abs(pos_clean) < 1] = 0
    
    n_tickers = len(pos_clean.columns)
    fig = make_subplots(rows=n_tickers, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    
    for i, ticker in enumerate(pos_clean.columns, start=1):
        s = pos_clean[ticker]
       
        fig.add_trace(go.Scatter(x=s.index, y=s.clip(lower=0),
                                 fill='tozeroy', mode='none', line_shape='hv', fillcolor='green', name='Long'),
                      row=i, col=1)
        fig.add_trace(go.Scatter(x=s.index, y=s.clip(upper=0),
                                 fill='tozeroy', mode='none', line_shape='hv', fillcolor='salmon', name='Short'),
                      row=i, col=1)
        fig.add_trace(go.Scatter(x=s.index, y=[0]*len(s),
                                 mode='lines', line=dict(color='black', width=1), name='Zero'),
                      row=i, col=1)
        fig.update_yaxes(title_text=ticker, row=i, col=1, tickvals=[-1,0,1],
                         ticktext=['Short','Flat','Long'])
    
    fig.update_layout(height=80*n_tickers, width=1000, showlegend=False, title="Positions Over Time")
    fig.show()
    
