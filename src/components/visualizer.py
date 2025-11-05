# File: src/component/visualizer.py

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf as sm_acf


def plot_hourly_posts(posts_per_hour: pl.DataFrame):
    """Plots the raw posts-per-hour time series."""
    print("Creating plot: Posts Per Hour...")
    fig = px.line(posts_per_hour.to_pandas(),
                  x='post_timestamp',
                  y='post_count',
                  title='Posts Per Hour on Truth Social Sample')
    fig.show()


def plot_acf(posts_per_hour: pl.DataFrame):
    """Calculates and plots the Autocorrelation Function (ACF)."""
    print("Creating plot: Autocorrelation Function (ACF)...")
    fig, ax = plt.subplots(figsize=(12, 5))
    # Ensure we pass a pandas Series / 1-d array to statsmodels
    series = posts_per_hour.to_pandas()['post_count']
    # Clean series
    series_clean = series.dropna().astype(float)
    max_lags = 7 * 24
    # Compute ACF values using statsmodels.tsa.stattools.acf (robust across versions)
    acf_vals = sm_acf(series_clean, nlags=max_lags, fft=True)
    lags = np.arange(len(acf_vals))
    ax.stem(lags, acf_vals)
    ax.set_title('Autocorrelation Function (ACF) of Posts Per Hour')
    ax.set_xlabel('Lag (Hours)')
    ax.set_ylabel('Autocorrelation')
    ax.grid(True)
    plt.show()


def plot_transformation_comparison(posts_per_hour: pl.DataFrame, posts_per_hour_transformed: pl.DataFrame):
    """Shows the original and log-transformed plots side-by-side."""
    print("Creating plot: Variance Comparison...")
    fig_original = px.line(posts_per_hour.to_pandas(),
                           x='post_timestamp',
                           y='post_count',
                           title='Original Posts Per Hour (Raw Count)')

    fig_log = px.line(posts_per_hour_transformed.to_pandas(),
                      x='post_timestamp',
                      y='log_post_count',
                      title='Log-Transformed Posts Per Hour (log(1 + Count))')

    fig_original.show()
    fig_log.show()


def plot_burst_rectangles(posts_per_hour_transformed: pl.DataFrame, burst_list: list):
    """Plots the time series with burst periods highlighted as rectangles."""
    print("Creating plot: Burst Detection with Rectangles...")
    plot_df = posts_per_hour_transformed.to_pandas()

    fig = go.Figure()

    # 1. Add the baseline time series
    fig.add_trace(
        go.Scatter(
            x=plot_df['post_timestamp'],
            y=plot_df['log_post_count'],
            mode='lines+markers',
            name='Log(Posts per Hour)',
            line=dict(color='blue'),
            marker=dict(size=4)
        )
    )

    # 2. Add burst rectangles
    color_map = {
        1.0: 'rgba(255, 255, 0, 0.2)',  # Level 1: Transparent Yellow
        2.0: 'rgba(255, 165, 0, 0.3)',  # Level 2: Transparent Orange
        3.0: 'rgba(255, 0, 0, 0.4)'  # Level 3: Transparent Red
    }
    legend_added = {level: False for level in color_map.keys()}

    for burst in burst_list:
        level = burst['level']
        fill_color = color_map.get(level, 'rgba(128, 128, 128, 0.2)')
        show_legend = not legend_added.get(level, False)

        fig.add_vrect(
            x0=burst['start_time'],
            x1=burst['end_time'],
            fillcolor=fill_color,
            layer="below",
            line_width=0,
            name=f'Burst Level {level}',
            showlegend=show_legend
        )
        if level in legend_added:
            legend_added[level] = True

    # 3. Update layout
    fig.update_layout(
        title='Burst Detection: Posts per Hour (Log Transformed) with Burst Regions',
        xaxis_title='Time',
        yaxis_title='Log(Posts per Hour)',
        hovermode="x unified"
    )
    fig.show()


def plot_burst_gantt(burst_list: list):
    """Creates a Gantt chart of all detected burst periods."""
    print("Creating plot: Burst Gantt Chart...")
    gantt_df = pd.DataFrame(burst_list)
    gantt_df['burst_level_str'] = gantt_df['level'].apply(lambda x: f"Level {int(x)}")

    fig = px.timeline(
        gantt_df,
        x_start="start_time",
        x_end="end_time",
        y="burst_level_str",
        color="burst_level_str",
        title="Gantt Chart of Detected Burst Periods by Level",
        labels={"burst_level_str": "Burst Level"}
    )
    fig.update_yaxes(categoryorder='array', categoryarray=sorted(gantt_df['burst_level_str'].unique(), reverse=True))
    fig.show()