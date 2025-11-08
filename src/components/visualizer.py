# File: src/component/visualizer.py

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf as sm_acf
import os


def get_html_path(plot_name: str, plots_dir: str) -> str:
    """Get the full path for an HTML plot file"""
    return os.path.join(plots_dir, f"{plot_name}.html")


def plot_hourly_posts(posts_per_hour: pl.DataFrame, save_path: str = None):
    """Plots the raw posts-per-hour time series."""
    print("Creating plot: Posts Per Hour...")
    fig = px.line(posts_per_hour.to_pandas(),
                  x='post_timestamp',
                  y='post_count',
                  title='Posts Per Hour on Truth Social Sample')

    if save_path:
        # Save static version
        fig.write_image(save_path)
        # Save interactive HTML version
        html_path = get_html_path('hourly_posts', os.path.dirname(save_path))
        fig.write_html(html_path)
    else:
        fig.show()


def plot_acf(posts_per_hour: pl.DataFrame, save_path: str = None):
    """Calculates and plots the Autocorrelation Function (ACF)."""
    print("Creating plot: Autocorrelation Function (ACF)...")

    # Create plotly figure for ACF (instead of matplotlib)
    series = posts_per_hour.to_pandas()['post_count']
    series_clean = series.dropna().astype(float)
    max_lags = 7 * 24
    acf_vals = sm_acf(series_clean, nlags=max_lags, fft=True)
    lags = np.arange(len(acf_vals))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lags,
        y=acf_vals,
        mode='markers+lines',
        name='ACF'
    ))

    fig.update_layout(
        title='Autocorrelation Function (ACF) of Posts Per Hour',
        xaxis_title='Lag (Hours)',
        yaxis_title='Autocorrelation',
        showlegend=False
    )

    if save_path:
        # Save static version
        fig.write_image(save_path)
        # Save interactive HTML version
        html_path = get_html_path('acf_plot', os.path.dirname(save_path))
        fig.write_html(html_path)
    else:
        fig.show()


def plot_transformation_comparison(posts_per_hour: pl.DataFrame, posts_per_hour_transformed: pl.DataFrame, save_path: str = None):
    """Shows the original and log-transformed plots side-by-side."""
    print("Creating plot: Variance Comparison...")

    fig = go.Figure()

    # Add original data subplot
    fig.add_trace(
        go.Scatter(
            x=posts_per_hour.to_pandas()['post_timestamp'],
            y=posts_per_hour.to_pandas()['post_count'],
            name='Original',
            line=dict(color='blue')
        )
    )

    # Add transformed data subplot
    fig.add_trace(
        go.Scatter(
            x=posts_per_hour_transformed.to_pandas()['post_timestamp'],
            y=posts_per_hour_transformed.to_pandas()['log_post_count'],
            name='Log-Transformed',
            yaxis='y2',
            line=dict(color='red')
        )
    )

    # Update layout to show plots side by side
    fig.update_layout(
        title='Original vs Log-Transformed Posts Per Hour',
        yaxis=dict(
            title='Post Count',
            title_font=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title='Log(Post Count)',
            title_font=dict(color='red'),
            tickfont=dict(color='red'),
            anchor='x',
            overlaying='y',
            side='right'
        )
    )

    if save_path:
        # Save static version
        fig.write_image(save_path)
        # Save interactive HTML version
        html_path = get_html_path('transformation_comparison', os.path.dirname(save_path))
        fig.write_html(html_path)
    else:
        fig.show()


def plot_burst_rectangles(posts_per_hour_transformed: pl.DataFrame, burst_list: list, save_path: str = None):
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

    if save_path:
        # Save static version
        fig.write_image(save_path)
        # Save interactive HTML version
        html_path = get_html_path('burst_rectangles', os.path.dirname(save_path))
        fig.write_html(html_path)
    else:
        fig.show()


def plot_burst_gantt(burst_list: list, save_path: str = None):
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

    if save_path:
        # Save static version
        fig.write_image(save_path)
        # Save interactive HTML version
        html_path = get_html_path('burst_gantt', os.path.dirname(save_path))
        fig.write_html(html_path)
    else:
        fig.show()
