import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import polars as pl
import numpy as np
from statsmodels.tsa.stattools import acf, adfuller
import tempfile
import sys
import os
from datetime import datetime, timedelta
from collections import Counter, defaultdict

# Add project root to path (demo folder is one level down from root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import your existing components
try:
    from components.data_analyzer import DataAnalyzer
    from components.burst_detector_enhanced import BurstDetectorEnhanced
except ImportError:
    st.error("Could not import burst detection components. Make sure you're running from the project root.")

st.set_page_config(
    page_title="Interactive Burst Detection",
    page_icon="📊",
    layout="wide"
)

def validate_csv_columns(df):
    """Validate that uploaded CSV has required columns"""
    required_cols = ['created_at']
    optional_cols = ['account.username', 'content_cleaned', 'account.display_name', 'id']

    missing_required = [col for col in required_cols if col not in df.columns]

    if missing_required:
        st.error(f"❌ Missing required columns: {missing_required}")
        return False

    # Check for at least some optional columns
    present_optional = [col for col in optional_cols if col in df.columns]
    if not present_optional:
        st.warning("⚠️ No optional columns found. Limited functionality available.")

    return True

def normalize_datetime(dt_obj):
    """Convert datetime object to timezone-naive pandas Timestamp"""
    try:
        if dt_obj is None:
            return None

        # Convert to pandas datetime if it's a string
        if isinstance(dt_obj, str):
            dt_obj = pd.to_datetime(dt_obj)

        # Handle timezone-aware datetime
        if hasattr(dt_obj, 'tz') and dt_obj.tz is not None:
            # Convert to UTC first, then make naive
            dt_obj = dt_obj.tz_convert('UTC').tz_localize(None)
        elif hasattr(dt_obj, 'tzinfo') and dt_obj.tzinfo is not None:
            # Handle Python datetime objects with timezone
            dt_obj = dt_obj.replace(tzinfo=None)

        return pd.to_datetime(dt_obj)

    except Exception as e:
        st.error(f"Error normalizing datetime: {e}")
        return None

def check_data_stationarity(data_series):
    """Check if the data is stationary using Augmented Dickey-Fuller test"""
    try:
        # Remove any NaN values
        clean_series = data_series.dropna()

        if len(clean_series) < 10:
            return False, "Not enough data points for stationarity test"

        # Perform Augmented Dickey-Fuller test
        result = adfuller(clean_series)

        # ADF test: null hypothesis is that the series has a unit root (non-stationary)
        # If p-value < 0.05, we reject null hypothesis (data is stationary)
        is_stationary = result[1] < 0.05

        return is_stationary, {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4]
        }

    except Exception as e:
        return False, f"Error in stationarity test: {e}"

def calculate_acf(data_series, max_lags=168):  # 7 days * 24 hours to show weekly patterns
    """Calculate Autocorrelation Function"""
    try:
        clean_series = data_series.dropna()

        if len(clean_series) < max_lags:
            max_lags = len(clean_series) - 1

        acf_vals = acf(clean_series, nlags=max_lags, fft=True)
        return acf_vals, np.arange(len(acf_vals))

    except Exception as e:
        st.error(f"Error calculating ACF: {e}")
        return None, None

def create_hourly_plot_with_bursts(posts_per_hour_df, burst_list=None):
    """Create interactive hourly posts plot with burst rectangles"""

    fig = go.Figure()

    # Add the main time series line
    fig.add_trace(go.Scatter(
        x=posts_per_hour_df['post_timestamp'],
        y=posts_per_hour_df['post_count'],
        mode='lines+markers',
        name='Posts per Hour',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4),
        hovertemplate='<b>Time:</b> %{x}<br><b>Posts:</b> %{y}<extra></extra>'
    ))

    # Add burst rectangles if provided
    if burst_list:
        colors = {1.0: 'rgba(255, 255, 0, 0.3)', 2.0: 'rgba(255, 165, 0, 0.4)', 3.0: 'rgba(255, 0, 0, 0.5)'}

        for i, burst in enumerate(burst_list):
            level = burst['level']
            start_time = burst['start_time']
            end_time = burst['end_time']

            # Find y-range for the rectangle
            mask = ((posts_per_hour_df['post_timestamp'] >= start_time) &
                   (posts_per_hour_df['post_timestamp'] <= end_time))

            if mask.any():
                y_vals = posts_per_hour_df[mask]['post_count']
                y_min = 0
                y_max = y_vals.max() if len(y_vals) > 0 else 100
            else:
                y_max = posts_per_hour_df['post_count'].max()
                y_min = 0

            fig.add_shape(
                type="rect",
                x0=start_time,
                y0=y_min,
                x1=end_time,
                y1=y_max * 1.1,
                fillcolor=colors.get(level, 'rgba(128, 128, 128, 0.3)'),
                line=dict(color=colors.get(level, 'gray'), width=1),
                layer="below"
            )

            # Add burst label
            fig.add_annotation(
                x=start_time + (end_time - start_time) / 2,
                y=y_max * 1.05,
                text=f"L{int(level)}",
                showarrow=False,
                font=dict(size=10, color='black'),
                bgcolor='white',
                bordercolor='black',
                borderwidth=1
            )

    fig.update_layout(
        title='Hourly Posts with Burst Detection',
        xaxis_title='Time',
        yaxis_title='Number of Posts',
        hovermode='closest',
        height=500,
        showlegend=True
    )

    return fig

def create_interactive_burst_plot(full_df, burst_list=None):
    """Create interactive plot showing exact post timestamps with clickable burst regions only"""
    fig = go.Figure()

    # Create a time series from exact post timestamps
    if 'created_at' in full_df.columns:
        # Convert to datetime if needed
        df_copy = full_df.copy()
        df_copy['created_at'] = pd.to_datetime(df_copy['created_at'], errors='coerce')

        # Normalize to timezone-naive if timezone-aware
        if hasattr(df_copy['created_at'].dtype, 'tz') and df_copy['created_at'].dtype.tz is not None:
            df_copy['created_at'] = df_copy['created_at'].dt.tz_convert('UTC').dt.tz_localize(None)

        # Count posts per exact timestamp (minute-level granularity)
        df_copy['timestamp_minute'] = df_copy['created_at'].dt.floor('T')  # Floor to minute
        post_counts = df_copy.groupby('timestamp_minute').size().reset_index(name='post_count')
        post_counts = post_counts.sort_values('timestamp_minute')
    else:
        st.error("No timestamp column found")
        return go.Figure()

    # Add the main time series line (NOT clickable outside burst regions)
    fig.add_trace(go.Scatter(
        x=post_counts['timestamp_minute'],
        y=post_counts['post_count'],
        mode='lines+markers',
        name='Posts (Exact Times)',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=3),
        hovertemplate='<b>Time:</b> %{x}<br><b>Posts:</b> %{y}<extra></extra>',
        showlegend=True
    ))

    # Add clickable burst regions as separate traces
    if burst_list:
        colors = {1.0: 'gold', 2.0: 'orange', 3.0: 'red'}
        fill_colors = {1.0: 'rgba(255, 215, 0, 0.3)', 2.0: 'rgba(255, 165, 0, 0.4)', 3.0: 'rgba(255, 0, 0, 0.5)'}

        for i, burst in enumerate(burst_list):
            level = burst['level']
            start_time = burst['start_time']
            end_time = burst['end_time']

            # Normalize burst times to be timezone-naive to match post_counts
            if hasattr(start_time, 'tz') and start_time.tz is not None:
                start_time = start_time.tz_convert('UTC').tz_localize(None)
            elif hasattr(start_time, 'tzinfo') and start_time.tzinfo is not None:
                start_time = start_time.replace(tzinfo=None)

            if hasattr(end_time, 'tz') and end_time.tz is not None:
                end_time = end_time.tz_convert('UTC').tz_localize(None)
            elif hasattr(end_time, 'tzinfo') and end_time.tzinfo is not None:
                end_time = end_time.replace(tzinfo=None)

            # Convert to pandas datetime to ensure compatibility
            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)

            duration = (end_time - start_time).total_seconds() / 3600

            # Get posts within this burst period
            burst_mask = ((post_counts['timestamp_minute'] >= start_time) &
                         (post_counts['timestamp_minute'] <= end_time))

            burst_posts = post_counts[burst_mask]

            if len(burst_posts) > 0:
                y_max = burst_posts['post_count'].max()

                # Create custom data for this entire burst region
                burst_info = f"BURST_{i}_{level}_{start_time.isoformat()}_{end_time.isoformat()}_{duration:.2f}"
                custom_data = [burst_info] * len(burst_posts)

                # Add clickable scatter trace for this burst region
                fig.add_trace(go.Scatter(
                    x=burst_posts['timestamp_minute'],
                    y=burst_posts['post_count'],
                    mode='lines+markers',
                    name=f'Level {int(level)} Burst (CLICKABLE)',
                    line=dict(color=colors.get(level, 'gray'), width=4),
                    marker=dict(size=8, color=colors.get(level, 'gray')),
                    customdata=custom_data,
                    hovertemplate=f'<b>🎯 LEVEL {int(level)} BURST</b><br>' +
                                 f'Time: %{{x}}<br>Posts: %{{y}}<br>' +
                                 f'Duration: {duration:.1f}h<br>' +
                                 '<b>CLICK TO SEE CONTRIBUTORS!</b><extra></extra>',
                    showlegend=True,
                    legendgroup=f'burst_level_{level}'
                ))
            else:
                y_max = post_counts['post_count'].max()

            # Add background rectangle for visual indication
            fig.add_shape(
                type="rect",
                x0=start_time,
                y0=0,
                x1=end_time,
                y1=y_max * 1.2,
                fillcolor=fill_colors.get(level, 'rgba(128, 128, 128, 0.3)'),
                line=dict(color=colors.get(level, 'gray'), width=2),
                layer="below"
            )

            # Add level label
            center_time = start_time + (end_time - start_time) / 2
            fig.add_annotation(
                x=center_time,
                y=y_max * 1.15,
                text=f"Level {int(level)}",
                showarrow=False,
                font=dict(size=12, color='white', family='Arial Black'),
                bgcolor=colors.get(level, 'gray'),
                bordercolor='white',
                borderwidth=2,
                borderpad=4
            )

    fig.update_layout(
        title='📊 Posts at Exact Times - Click on Colored Burst Regions Only!',
        xaxis_title='Time',
        yaxis_title='Number of Posts',
        hovermode='closest',
        height=600,
        showlegend=True
    )

    return fig

def get_burst_contributors(burst_index, contributors_list, full_df):
    """Extract top contributors for a specific burst"""
    try:
        if burst_index >= len(contributors_list):
            st.warning(f"Burst index {burst_index} is out of range (contributors list has {len(contributors_list)} items)")
            return pd.DataFrame(), {}

        burst_contrib = contributors_list[burst_index]

        # Check if we have the expected structure
        if not burst_contrib or not isinstance(burst_contrib, dict):
            st.warning("No contributor data found for this burst")
            return pd.DataFrame(), {}

        # Get the top_accounts list
        top_accounts = burst_contrib.get('top_accounts', [])

        if not top_accounts:
            st.warning("No top accounts found for this burst")
            return pd.DataFrame(), {}

        # Create account groups with contribution data
        account_groups = {}

        # Process top accounts (they're already sorted by count)
        for rank, account_info in enumerate(top_accounts[:10], 1):
            username = account_info.get('account.username', '')
            contrib_count = account_info.get('count', 0)

            if not username:
                continue

            # Calculate percentage
            total_posts_in_burst = burst_contrib.get('post_count', 1)
            contrib_percentage = (contrib_count / total_posts_in_burst) * 100 if total_posts_in_burst > 0 else 0

            # Get posts from this user in the entire dataset
            if 'account.username' in full_df.columns:
                user_posts = full_df[full_df['account.username'] == username].copy()

                # Sort by time if possible
                if 'created_at' in user_posts.columns:
                    user_posts = user_posts.sort_values('created_at')

                display_name = user_posts.iloc[0].get('account.display_name', username) if len(user_posts) > 0 else username

                account_groups[username] = {
                    'rank': rank,
                    'contribution_count': contrib_count,
                    'contribution_percentage': contrib_percentage,
                    'total_posts': len(user_posts),
                    'posts': user_posts.head(20),  # Limit for performance
                    'display_name': display_name
                }

        return full_df, account_groups

    except Exception as e:
        st.error(f"Error getting burst contributors: {e}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame(), {}

def display_burst_analysis(burst_info, contributors_data, full_df):
    """Display detailed analysis for a selected burst"""
    try:
        # Parse burst info from custom data
        parts = burst_info.split('_')
        if len(parts) < 6:
            st.error("Invalid burst data")
            return

        burst_index = int(parts[1])
        level = float(parts[2])
        start_time = pd.to_datetime(parts[3])
        end_time = pd.to_datetime(parts[4])
        duration = float(parts[5])

        # Normalize times to be timezone-naive
        if hasattr(start_time, 'tz') and start_time.tz is not None:
            start_time = start_time.tz_convert('UTC').tz_localize(None)
        if hasattr(end_time, 'tz') and end_time.tz is not None:
            end_time = end_time.tz_convert('UTC').tz_localize(None)

        # Display burst header
        st.markdown("---")
        st.subheader(f"💥 Level {int(level)} Burst Analysis")

        # Burst metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Level", int(level))
        with col2:
            st.metric("⏱️ Duration", f"{duration:.1f}h")
        with col3:
            st.metric("🕐 Start", start_time.strftime("%m/%d %H:%M"))
        with col4:
            st.metric("🕐 End", end_time.strftime("%m/%d %H:%M"))

        st.info(f"📅 **Full Period:** {start_time} → {end_time}")

        # Get contributors
        burst_df, account_groups = get_burst_contributors(burst_index, contributors_data, full_df)

        if account_groups:
            st.success(f"✅ Found {len(account_groups)} contributing accounts for this burst!")

            # Contributors summary table
            st.subheader("🏆 Top Contributors")

            summary_data = []
            for username, data in account_groups.items():
                summary_data.append({
                    'Rank': data['rank'],
                    'Username': f"@{username}",
                    'Display Name': data['display_name'],
                    'Contributions': data['contribution_count'],
                    'Contribution %': f"{data['contribution_percentage']:.1f}%",
                    'Total Posts': data['total_posts']
                })

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # Detailed contributor posts
            st.subheader("📝 Posts from Top Contributors")

            for username, data in list(account_groups.items())[:5]:  # Show top 5
                rank = data['rank']
                contributions = data['contribution_count']
                percentage = data['contribution_percentage']

                with st.expander(f"🥇 #{rank} @{username} ({data['display_name']}) - {contributions} contributions ({percentage:.1f}%)",
                               expanded=(rank <= 2)):

                    # Account summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Contributions", contributions)
                    with col2:
                        st.metric("Total Posts", data['total_posts'])
                    with col3:
                        st.metric("Percentage", f"{percentage:.1f}%")

                    st.markdown("---")

                    # Show sample posts
                    posts = data['posts']
                    if len(posts) > 0:
                        st.write(f"**Showing {min(len(posts), 10)} recent posts:**")

                        for idx, (_, post) in enumerate(posts.head(10).iterrows()):
                            timestamp = post.get('created_at', 'No timestamp')
                            content = post.get('content_cleaned', 'No content')
                            post_id = post.get('id', 'N/A')

                            st.markdown(f"**⏰ {timestamp}**")

                            if len(content) > 300:
                                with st.expander(f"📄 View full post (ID: {post_id})"):
                                    st.write(content)
                            else:
                                st.write(f"💬 {content}")

                            if idx < 9:  # Add separator except for last
                                st.write("")
                    else:
                        st.info("No posts available for this contributor")

            if len(account_groups) > 5:
                st.info(f"Showing top 5 contributors. Total contributors: {len(account_groups)}")

        else:
            st.warning("No contributors found for this burst")

    except Exception as e:
        st.error(f"Error displaying burst analysis: {e}")

@st.cache_data
def process_uploaded_csv(uploaded_file):
    """Process the uploaded CSV file"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Load and process data
        analyzer = DataAnalyzer(data_file_path=tmp_path)
        analyzer.run_all()

        # Convert to pandas for easier handling
        posts_per_hour_df = analyzer.posts_per_hour.to_pandas()
        full_df = analyzer.df.to_pandas()

        # Clean up temp file
        os.unlink(tmp_path)

        return analyzer, posts_per_hour_df, full_df

    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

def get_posts_at_timepoint(full_df, selected_time, time_window_hours=1):
    """Get posts within a time window around the selected point, grouped by account"""
    try:
        # Normalize selected_time to timezone-naive datetime
        selected_time = normalize_datetime(selected_time)
        if selected_time is None:
            return pd.DataFrame(), {}

        # Create time window
        start_time = selected_time - timedelta(hours=time_window_hours/2)
        end_time = selected_time + timedelta(hours=time_window_hours/2)

        # Filter posts within time window
        if 'created_at' in full_df.columns:
            # Make a copy to avoid modifying the original dataframe
            df_copy = full_df.copy()

            # Normalize the created_at column
            if df_copy['created_at'].dtype == 'object':
                df_copy['created_at'] = pd.to_datetime(df_copy['created_at'], errors='coerce')

            # Handle timezone-aware columns
            if hasattr(df_copy['created_at'].dtype, 'tz') and df_copy['created_at'].dtype.tz is not None:
                df_copy['created_at'] = df_copy['created_at'].dt.tz_convert('UTC').dt.tz_localize(None)

            # Filter posts within time window
            mask = ((df_copy['created_at'] >= start_time) &
                   (df_copy['created_at'] <= end_time))

            filtered_posts = df_copy[mask]

            # Group posts by account and get top 10 accounts by post count
            account_groups = {}
            if len(filtered_posts) > 0 and 'account.username' in filtered_posts.columns:
                # Count posts per account
                account_counts = filtered_posts.groupby('account.username').size().reset_index(name='post_count')
                account_counts = account_counts.sort_values('post_count', ascending=False).head(10)

                # Get posts for each top account
                for _, row in account_counts.iterrows():
                    username = row['account.username']
                    user_posts = filtered_posts[filtered_posts['account.username'] == username].copy()
                    # Sort posts by time
                    user_posts = user_posts.sort_values('created_at')
                    account_groups[username] = {
                        'post_count': row['post_count'],
                        'posts': user_posts,
                        'display_name': user_posts.iloc[0].get('account.display_name', username) if len(user_posts) > 0 else username
                    }

            return filtered_posts, account_groups

        return pd.DataFrame(), {}  # Return empty if no timestamp column

    except Exception as e:
        st.error(f"Error filtering posts: {e}")
        return pd.DataFrame(), {}

def main():
    st.title("🚀 Interactive Social Media Burst Detection")
    st.markdown("*Upload your data, analyze patterns, and explore bursts interactively*")

    # Initialize session state for burst selection
    if 'selected_burst' not in st.session_state:
        st.session_state.selected_burst = None

    # File upload
    uploaded_file = st.file_uploader(
        "📁 Choose a CSV file",
        type="csv",
        help="Upload your social media data CSV file with timestamp and content information"
    )

    if uploaded_file is None:
        st.info("👆 Please upload a CSV file to begin analysis")
        st.markdown("""
        ### 📋 Required CSV Format:
        - **`created_at`**: Timestamp column (required)
        - **`account.username`**: Account identifier (optional but recommended)
        - **`content_cleaned`**: Post content (optional but recommended) 
        - **`account.display_name`**: Display name (optional)
        - **`id`**: Post ID (optional)
        """)
        return

    # Preview data
    st.subheader("📋 Data Preview")
    preview_df = pd.read_csv(uploaded_file, nrows=10)
    st.dataframe(preview_df.head())

    # Validate columns
    if not validate_csv_columns(preview_df):
        return

    # Reset file pointer and process
    uploaded_file.seek(0)

    # Process the full dataset
    with st.spinner("🔄 Processing your data..."):
        analyzer, posts_per_hour_df, full_df = process_uploaded_csv(uploaded_file)

    if analyzer is None:
        st.error("❌ Failed to process the uploaded file")
        return

    st.success(f"✅ Data processed successfully! Found {len(full_df)} total posts")

    # Step 1: Show hourly plot
    st.subheader("📊 Step 1: Hourly Posts Analysis")

    # Create basic hourly plot
    fig_basic = create_hourly_plot_with_bursts(posts_per_hour_df)
    st.plotly_chart(fig_basic, use_container_width=True)

    # Show basic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Posts", f"{len(full_df):,}")
    with col2:
        st.metric("Time Range", f"{(posts_per_hour_df['post_timestamp'].max() - posts_per_hour_df['post_timestamp'].min()).days} days")
    with col3:
        st.metric("Avg Posts/Hour", f"{posts_per_hour_df['post_count'].mean():.1f}")

    # Step 2: Stationarity and ACF checks
    st.subheader("🔍 Step 2: Data Quality Checks")

    # Check stationarity
    is_stationary, stat_results = check_data_stationarity(posts_per_hour_df['post_count'])

    col1, col2 = st.columns(2)

    with col1:
        if is_stationary:
            st.success("✅ Data is stationary (suitable for burst detection)")
            # Format p-value: use scientific notation if very small
            p_val = stat_results['p_value']
            if p_val < 0.0001:
                st.write(f"**ADF p-value:** {p_val:.2e} (< 0.0001)")
            else:
                st.write(f"**ADF p-value:** {p_val:.6f}")
            st.caption("p < 0.05 means stationary (rejects unit root hypothesis)")
        else:
            st.warning("⚠️ Data may not be stationary")
            p_val = stat_results['p_value']
            st.write(f"**ADF p-value:** {p_val:.6f}")
            st.info("Burst detection may still work but results should be interpreted carefully")
            st.caption("p ≥ 0.05 suggests possible non-stationarity")

    with col2:
        # Calculate and show ACF
        acf_vals, lags = calculate_acf(posts_per_hour_df['post_count'])
        if acf_vals is not None:
            fig_acf = go.Figure()
            fig_acf.add_trace(go.Scatter(
                x=lags,
                y=acf_vals,
                mode='lines+markers',
                name='ACF',
                line=dict(color='blue'),
                marker=dict(size=4)
            ))
            fig_acf.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)

            # Calculate confidence interval (approximately ±1.96/sqrt(n))
            n = len(posts_per_hour_df['post_count'])
            confidence_interval = 1.96 / np.sqrt(n)
            fig_acf.add_hline(y=confidence_interval, line_dash="dash", line_color="red",
                             annotation_text="95% threshold", annotation_position="right")
            fig_acf.add_hline(y=-confidence_interval, line_dash="dash", line_color="red")

            fig_acf.update_layout(
                title="Autocorrelation Function (ACF) of Posts Per Hour",
                xaxis_title="Lag (Hours)",
                yaxis_title="Autocorrelation",
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig_acf, use_container_width=True)

    # Step 3: Burst Detection
    st.subheader("🎯 Step 3: Burst Detection")

    # Burst detection parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        s_param = st.slider("S Parameter", 1.0, 5.0, 2.0, 0.1, help="Controls burst sensitivity")
    with col2:
        gamma_param = st.slider("Gamma Parameter", 0.1, 2.0, 1.0, 0.1, help="Controls state transition costs")
    with col3:
        run_burst = st.button("🚀 Run Burst Detection", type="primary")

    if run_burst:
        with st.spinner("🔍 Detecting bursts..."):
            # Run burst detection
            detector = BurstDetectorEnhanced(s=s_param, gamma=gamma_param)
            burst_list, posts_with_bursts, contributors = detector.detect_bursts(
                ts_df=analyzer.ts_df,
                posts_per_hour_transformed=analyzer.posts_per_hour_transformed,
                posts_df=analyzer.df
            )

            # Store results in session state
            st.session_state.burst_results = {
                'burst_list': burst_list,
                'contributors': contributors,
                'posts_per_hour_df': posts_per_hour_df,
                'full_df': full_df
            }

            # Reset selected burst when new detection is run
            st.session_state.selected_burst = None

    # Show burst results if available
    if 'burst_results' in st.session_state:
        results = st.session_state.burst_results
        burst_list = results['burst_list']
        contributors = results['contributors']

        if burst_list:
            st.success(f"✅ Detected {len(burst_list)} bursts!")

            # Show burst summary
            levels = [b['level'] for b in burst_list]
            level_counts = Counter(levels)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Level 1 Bursts", level_counts.get(1.0, 0))
            with col2:
                st.metric("Level 2 Bursts", level_counts.get(2.0, 0))
            with col3:
                st.metric("Level 3 Bursts", level_counts.get(3.0, 0))

            # Step 4: Interactive plot with bursts
            st.subheader("🎯 Step 4: Interactive Burst Exploration")
            st.markdown("**🎯 Click ONLY on colored burst regions! Shows contributors for the entire burst period.**")

            # Create interactive plot with exact post times
            fig_interactive = create_interactive_burst_plot(full_df, burst_list)

            # Display the plot and capture selections
            selected_data = st.plotly_chart(fig_interactive, use_container_width=True, on_select="rerun", key="burst_plot")

            # Handle clicks only on burst regions
            if selected_data and hasattr(selected_data, 'selection') and selected_data.selection:
                if 'points' in selected_data.selection and selected_data.selection['points']:
                    point = selected_data.selection['points'][0]

                    # Check if this point has customdata indicating a burst
                    if 'customdata' in point and point['customdata']:
                        burst_info = point['customdata']

                        # Only respond to burst region clicks
                        if isinstance(burst_info, str) and burst_info.startswith('BURST_'):
                            st.session_state.selected_burst = burst_info
                        # Ignore all other clicks (do nothing)

            # Display burst analysis only if a burst region was clicked
            if st.session_state.get('selected_burst'):
                display_burst_analysis(st.session_state.selected_burst, contributors, full_df)
            else:
                st.info("👆 Click on any colored burst region in the chart above to see contributors for that entire burst!")

            # Step 5: Network Coordination Analysis
            st.markdown("---")
            st.subheader("🕸️ Step 5: Network Coordination Analysis")
            st.markdown("**Detect coordinated account networks using NetworkX**")

            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("💡 This analysis detects coordinated behavior through content similarity, retweet patterns, and temporal synchronization.")
            with col2:
                run_coordination = st.button("🚀 Run Network Analysis", type="primary", key="run_coordination")

            if run_coordination:
                with st.spinner("🕵️ Analyzing coordination patterns..."):
                    try:
                        # Import the content coordination detector
                        from components.content_coordination_detector import ContentCoordinationDetector

                        # Run coordination analysis using the correct method
                        detector = ContentCoordinationDetector(
                            identical_threshold=0.95,
                            high_similarity_threshold=0.85,
                            min_content_length=20
                        )

                        # Convert pandas DataFrame to polars if needed
                        if isinstance(full_df, pd.DataFrame):
                            import polars as pl
                            posts_pl = pl.from_pandas(full_df)
                        else:
                            posts_pl = full_df

                        # Run the coordination detection
                        coordination_results = detector.detect_coordination(
                            burst_contributors=contributors,  # Pass contributors, not burst_list
                            full_posts_df=posts_pl  # Pass the full posts dataframe
                        )

                        # Count RT temporal coordination from retweet_amplification
                        rt_temporal_count = 0
                        retweet_amp = coordination_results.get('retweet_amplification', [])
                        for rt in retweet_amp:
                            if rt.get('has_temporal_coordination', False):
                                rt_temporal_count += 1

                        # Process results for display
                        processed_results = {
                            'num_networks': coordination_results.get('summary_stats', {}).get('total_networks', 0),
                            'total_network_accounts': coordination_results.get('summary_stats', {}).get('total_network_accounts', 0),
                            'overall_confidence': coordination_results.get('confidence_level', 'UNKNOWN'),
                            'total_coordination_pairs': coordination_results.get('summary_stats', {}).get('total_coordination_pairs', 0),
                            'evidence_summary': {
                                'identical_content': coordination_results.get('summary_stats', {}).get('identical_content_instances', 0),
                                'hashtag_coordination': coordination_results.get('summary_stats', {}).get('hashtag_coordination_instances', 0),
                                'url_coordination': coordination_results.get('summary_stats', {}).get('url_coordination_instances', 0),
                                'retweet_coordination': coordination_results.get('summary_stats', {}).get('retweet_amplification_instances', 0),
                                'rt_temporal_coordination': rt_temporal_count,
                                'behavioral_patterns': coordination_results.get('summary_stats', {}).get('behavioral_pattern_instances', 0)
                            },
                            'networks': [],
                            'hubs': []
                        }

                        # Process networks
                        for network in coordination_results.get('coordination_networks', []):
                            # Extract network metrics from the nested structure
                            net_metrics = network.get('network_metrics', {})
                            net_structure = network.get('network_structure', {})

                            # Extract hub accounts from degree centrality with scores
                            degree_centrality = net_metrics.get('degree_centrality', {})
                            if degree_centrality:
                                # Sort by centrality score and get top 3 with scores
                                sorted_hubs = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
                                hub_accounts = [{'account': acc, 'centrality': score} for acc, score in sorted_hubs[:3]]
                            else:
                                # Fallback: if degree_centrality not available, use most_central as top hub
                                most_central = net_metrics.get('most_central_account', '')
                                most_central_score = net_metrics.get('most_central_score', 0)
                                hub_accounts = [{'account': most_central, 'centrality': most_central_score}] if most_central else []

                            processed_results['networks'].append({
                                'size': network.get('size', 0),
                                'risk_level': network.get('risk_level', 'UNKNOWN'),
                                'confidence': network.get('avg_confidence', 0),
                                'evidence_types': network.get('evidence_types', []),
                                'metrics': {
                                    'density': net_metrics.get('density', 0),
                                    'clustering': net_metrics.get('avg_clustering', 0),
                                    'structure_type': net_structure.get('type', 'UNKNOWN')
                                },
                                'most_central': net_metrics.get('most_central_account', ''),
                                'hub_accounts': hub_accounts,
                                'accounts': network.get('accounts', [])
                            })

                        # Extract retweet hubs from retweet coordination data
                        retweet_data = coordination_results.get('retweet_amplification', [])
                        if retweet_data:
                            # Build hub data from retweet patterns
                            hub_tracker = defaultdict(lambda: {'retweeters': set(), 'bursts': set(), 'amplification_count': 0, 'confidence': 0})

                            for rt in retweet_data:
                                # Each rt item has: rt_source, retweeters (list), burst_index
                                source = rt.get('rt_source', '')
                                retweeters = rt.get('retweeters', [])
                                burst_idx = rt.get('burst_index', 0)
                                confidence = rt.get('confidence', 0)

                                if source and retweeters:
                                    # Add all retweeters for this source
                                    for retweeter in retweeters:
                                        hub_tracker[source]['retweeters'].add(retweeter)
                                    hub_tracker[source]['bursts'].add(burst_idx)
                                    hub_tracker[source]['amplification_count'] = max(
                                        hub_tracker[source]['amplification_count'],
                                        len(retweeters)
                                    )
                                    hub_tracker[source]['confidence'] = max(
                                        hub_tracker[source]['confidence'],
                                        confidence
                                    )

                            # Convert to hub list
                            for account, data in hub_tracker.items():
                                retweeter_count = len(data['retweeters'])
                                if retweeter_count >= 2:  # At least 2 different retweeters
                                    processed_results['hubs'].append({
                                        'account': account,
                                        'retweeter_count': retweeter_count,
                                        'burst_count': len(data['bursts']),
                                        'confidence': data['confidence'],
                                        'retweeters': list(data['retweeters'])
                                    })

                            # Sort by retweeter count
                            processed_results['hubs'].sort(key=lambda x: x['retweeter_count'], reverse=True)

                        # Store processed results in session state
                        st.session_state.coordination_results = processed_results

                        st.success(f"✅ Analysis complete! Found {processed_results['num_networks']} networks with {processed_results['total_network_accounts']} accounts.")

                    except Exception as e:
                        st.error(f"Error running coordination analysis: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            # Display coordination results if available
            if 'coordination_results' in st.session_state:
                coord_results = st.session_state.coordination_results

                # Overall Summary
                st.markdown("### 📊 Coordination Summary")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🕸️ Networks", coord_results.get('num_networks', 0))
                with col2:
                    st.metric("👥 Accounts", coord_results.get('total_network_accounts', 0))
                with col3:
                    confidence = coord_results.get('overall_confidence', 'UNKNOWN')
                    st.metric("📊 Confidence", confidence)
                with col4:
                    st.metric("🔗 Coordination Pairs", coord_results.get('total_coordination_pairs', 0))

                # Evidence breakdown
                st.markdown("### 📋 Evidence Breakdown")
                evidence = coord_results.get('evidence_summary', {})

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📝 Identical Content", evidence.get('identical_content', 0))
                    st.metric("🏷️ Hashtag Coordination", evidence.get('hashtag_coordination', 0))
                with col2:
                    st.metric("🔗 URL Coordination", evidence.get('url_coordination', 0))
                    st.metric("🔄 Retweet Amplification", evidence.get('retweet_coordination', 0))
                with col3:
                    st.metric("⏱️ Temporal RT Coordination", evidence.get('rt_temporal_coordination', 0))
                    st.metric("🤖 Behavioral Patterns", evidence.get('behavioral_patterns', 0))

                # Top Networks
                networks = coord_results.get('networks', [])
                if networks:
                    st.markdown("### 🕸️ Top Coordination Networks")
                    st.info(f"Showing top {min(5, len(networks))} networks (out of {len(networks)} total)")

                    for i, network in enumerate(networks[:5], 1):
                        risk_level = network.get('risk_level', 'UNKNOWN')
                        risk_emoji = "🔴" if risk_level == "HIGH" else "🟡" if risk_level == "MEDIUM" else "🟢"

                        with st.expander(f"{risk_emoji} Network {i} - {network.get('size', 0)} accounts ({risk_level} risk)", expanded=(i <= 2)):

                            # Network metrics with explanations
                            metrics = network.get('metrics', {})
                            density = metrics.get('density', 0)
                            clustering = metrics.get('clustering', 0)
                            structure = metrics.get('structure_type', 'UNKNOWN')

                            st.markdown("#### 📊 Network Metrics")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Network Density", f"{density:.3f}")
                                if density > 0.7:
                                    st.caption("🔴 Very High - Tight coordination")
                                elif density > 0.3:
                                    st.caption("🟡 Medium - Moderate coordination")
                                else:
                                    st.caption("🟢 Low - Loose connections")

                            with col2:
                                st.metric("Clustering Coefficient", f"{clustering:.3f}")
                                if clustering > 0.7:
                                    st.caption("🔴 Very High - Tight groups")
                                elif clustering > 0.3:
                                    st.caption("🟡 Medium - Some groups")
                                else:
                                    st.caption("🟢 Low - Star pattern")

                            with col3:
                                st.metric("Structure Type", structure)
                                if structure == "HIERARCHICAL":
                                    st.caption("🎯 Hub-and-spoke pattern")
                                elif structure == "DISTRIBUTED":
                                    st.caption("🕸️ Peer-to-peer pattern")
                                else:
                                    st.caption("🔀 Mixed pattern")

                            # Explanation section
                            with st.expander("📖 What do these metrics mean?"):
                                st.markdown("""
                                **Network Density** measures how interconnected the accounts are:
                                - **High (0.7-1.0):** Most accounts coordinate with each other → Very suspicious
                                - **Medium (0.3-0.7):** Moderate connections → Organized campaign
                                - **Low (0.0-0.3):** Sparse connections → Less suspicious, might be organic
                                
                                **Clustering Coefficient** measures how "cliquish" the network is:
                                - **High (0.7-1.0):** Tight groups where everyone knows everyone → Echo chambers
                                - **Medium (0.3-0.7):** Some group formation → Mixed coordination
                                - **Low (0.0-0.3):** Star pattern, central hub → Influencer pattern
                                
                                **Structure Type** shows the network organization:
                                - **HIERARCHICAL:** Few central hubs with many followers (could be legitimate influencer)
                                - **DISTRIBUTED:** Evenly spread connections (typical bot network pattern)
                                - **MIXED:** Combination of hubs and peer connections (organized campaign)
                                """)

                            st.markdown("---")

                            # Basic info
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.markdown(f"**Confidence Score:** {network.get('confidence', 0):.2f}")
                                st.markdown(f"**Evidence Types:** {', '.join(network.get('evidence_types', []))}")


                            with col2:
                                # Most central account
                                most_central = network.get('most_central', '')
                                if most_central:
                                    st.markdown(f"**🎯 Most Central Account:**")
                                    st.markdown(f"`{most_central}`")
                                    st.caption("Top coordinator in network")

                            # Hub accounts section
                            st.markdown("#### 🌟 Hub Accounts (Top Coordinators)")
                            hubs = network.get('hub_accounts', [])
                            if hubs:
                                st.info(f"These {len(hubs)} accounts have the most coordination connections")

                                # Display hub accounts with centrality scores
                                for idx, hub_data in enumerate(hubs, 1):
                                    if isinstance(hub_data, dict):
                                        account = hub_data.get('account', '')
                                        centrality = hub_data.get('centrality', 0)
                                        st.markdown(f"**#{idx}:** `{account}` (centrality: **{centrality:.3f}**)")
                                    else:
                                        # Fallback for old format (just account name)
                                        st.markdown(f"**#{idx}:** `{hub_data}`")
                            else:
                                st.caption("No hub accounts identified")

                            # Account list
                            st.markdown("#### 👥 Network Members")
                            accounts = network.get('accounts', [])
                            if accounts:
                                st.markdown(f"**Total Accounts:** {len(accounts)}")
                                if len(accounts) <= 10:
                                    st.markdown(", ".join([f"`{acc}`" for acc in accounts]))
                                else:
                                    shown = accounts[:10]
                                    st.markdown(", ".join([f"`{acc}`" for acc in shown]) + f" ... (+{len(accounts)-10} more)")
                                    with st.expander(f"View all {len(accounts)} accounts"):
                                        # Display in columns for better readability
                                        accounts_per_col = 20
                                        num_cols = min(3, (len(accounts) + accounts_per_col - 1) // accounts_per_col)
                                        account_cols = st.columns(num_cols)
                                        for idx, account in enumerate(accounts):
                                            col_idx = idx % num_cols
                                            with account_cols[col_idx]:
                                                st.markdown(f"• `{account}`")

                # Coordination Hubs Summary Section
                hubs = coord_results.get('hubs', [])
                if hubs:
                    st.markdown("---")
                    st.markdown("### 🎯 Coordination Hubs Analysis")

                    # Summary metrics
                    total_hubs = len(hubs)
                    total_retweeters = sum(hub.get('retweeter_count', 0) for hub in hubs)
                    avg_retweeters = total_retweeters / total_hubs if total_hubs > 0 else 0
                    high_confidence_hubs = len([h for h in hubs if h.get('confidence', 0) > 0.5])

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📍 Total Hubs", total_hubs)
                        st.caption("Accounts being amplified")
                    with col2:
                        st.metric("🔄 Total Retweeters", total_retweeters)
                        st.caption("Amplification network size")
                    with col3:
                        st.metric("📊 Avg Retweeters/Hub", f"{avg_retweeters:.1f}")
                        st.caption("Average amplification")
                    with col4:
                        st.metric("🔴 High Confidence", high_confidence_hubs)
                        st.caption("Significant amplification")

                    # What are Coordination Hubs explanation
                    with st.expander("📖 What are Coordination Hubs?"):
                        st.markdown("""
                        **Coordination Hubs** are accounts that are frequently retweeted by multiple others during bursts.
                        
                        **Key Differences from Hub Accounts:**
                        - **Hub Accounts (Network)** = WHO coordinates (accounts with most connections)
                        - **Coordination Hubs** = WHAT gets amplified (accounts being retweeted)
                        
                        **What this reveals:**
                        - 🎯 **Content originators** being amplified by the coordination network
                        - 📢 **Influencers** whose content gets coordinated retweets
                        - 🎭 **Potential targets** of amplification campaigns
                        - 📰 **Key narratives** being pushed through coordination
                        
                        **Interpretation:**
                        - An account can be BOTH a Hub Account AND Coordination Hub (dual role - very suspicious!)
                        - Coordination Hubs might be:
                          - Politicians/public figures being amplified
                          - News sources being promoted
                          - Influencers in the coordination campaign
                          - Legitimate accounts being exploited
                        
                        **Confidence Levels:**
                        - 🔴 **HIGH (>0.5):** 5+ retweeters - significant coordinated amplification
                        - 🟡 **MEDIUM (0.3-0.5):** 3-4 retweeters - moderate amplification
                        - 🟢 **LOW (<0.3):** 2 retweeters - minimal amplification
                        """)

                    st.markdown("---")

                # Top Coordination Hubs (Detailed List)
                if hubs:
                    st.markdown("### 📋 Top Coordination Hubs (Detailed View)")
                    st.info(f"Showing top {min(5, len(hubs))} hubs (out of {len(hubs)} total)")

                    # Remove the old expander since we have one above
                    for i, hub in enumerate(hubs[:5], 1):
                        confidence = hub.get('confidence', 0)
                        retweeter_count = hub.get('retweeter_count', 0)
                        confidence_label = "HIGH" if confidence > 0.5 else "MEDIUM" if confidence > 0.3 else "LOW"
                        emoji = "🔴" if confidence > 0.5 else "🟡" if confidence > 0.3 else "🟢"

                        with st.expander(f"{emoji} {i}. `{hub.get('account', 'Unknown')}` - {retweeter_count} retweeters ({confidence_label})", expanded=(i <= 2)):

                            st.markdown("#### 📊 Amplification Metrics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Retweeters", retweeter_count)
                                st.caption("Different accounts retweeting")
                            with col2:
                                st.metric("Active Bursts", hub.get('burst_count', 0))
                                st.caption("Bursts where amplified")
                            with col3:
                                st.metric("Confidence", f"{confidence:.2f}")
                                if confidence > 0.5:
                                    st.caption("🔴 High amplification")
                                elif confidence > 0.3:
                                    st.caption("🟡 Moderate amplification")
                                else:
                                    st.caption("🟢 Low amplification")

                            # Retweeter network
                            st.markdown("#### 🕸️ Retweeter Network")
                            retweeters = hub.get('retweeters', [])
                            if retweeters:
                                st.markdown(f"**{len(retweeters)} accounts retweeting this hub:**")
                                if len(retweeters) <= 10:
                                    for rt in retweeters:
                                        st.markdown(f"• `{rt}`")
                                else:
                                    for rt in retweeters[:10]:
                                        st.markdown(f"• `{rt}`")
                                    with st.expander(f"View all {len(retweeters)} retweeters"):
                                        ret_cols = st.columns(3)
                                        for idx, rt in enumerate(retweeters):
                                            with ret_cols[idx % 3]:
                                                st.markdown(f"• `{rt}`")

        else:
            st.warning("⚠️ No bursts detected with current parameters. Try adjusting S and Gamma values.")

if __name__ == "__main__":
    main()
