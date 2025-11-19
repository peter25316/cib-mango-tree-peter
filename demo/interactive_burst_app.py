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
from collections import Counter

# Add project root to path (demo folder is one level down from root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import your existing components
try:
    from components.data_analyzer import DataAnalyzer
    from components.burst_detector import BurstDetector
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

def calculate_acf(data_series, max_lags=50):
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

def main():
    st.title("🚀 Interactive Social Media Burst Detection")
    st.markdown("*Upload your data, analyze patterns, and explore bursts interactively*")

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
            st.write(f"**ADF p-value:** {stat_results['p_value']:.6f}")
        else:
            st.warning("⚠️ Data may not be stationary")
            st.write(f"**ADF p-value:** {stat_results['p_value']:.6f}")
            st.info("Burst detection may still work but results should be interpreted carefully")

    with col2:
        # Calculate and show ACF
        acf_vals, lags = calculate_acf(posts_per_hour_df['post_count'])
        if acf_vals is not None:
            fig_acf = go.Figure()
            fig_acf.add_trace(go.Scatter(x=lags, y=acf_vals, mode='lines+markers', name='ACF'))
            fig_acf.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_acf.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="5% threshold")
            fig_acf.add_hline(y=-0.05, line_dash="dash", line_color="red")
            fig_acf.update_layout(title="Autocorrelation Function", xaxis_title="Lag (hours)", yaxis_title="ACF", height=300)
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
            detector = BurstDetector(s=s_param, gamma=gamma_param)
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
            st.subheader("📈 Step 4: Interactive Burst Visualization")
            st.markdown("**Click on any point in the plot to see posts and accounts at that time!**")

            # Create plot with burst rectangles
            fig_bursts = create_hourly_plot_with_bursts(posts_per_hour_df, burst_list)

            # Make the plot interactive - capture click events
            selected_points = st.plotly_chart(fig_bursts, use_container_width=True, on_select="rerun")

            # Handle point selection
            if selected_points and hasattr(selected_points, 'selection') and selected_points.selection:
                if selected_points.selection['points']:
                    # Get the selected point
                    point = selected_points.selection['points'][0]
                    selected_time = point['x']
                    selected_posts = point['y']

                    # Normalize selected_time to handle timezone issues
                    selected_time = normalize_datetime(selected_time)

                    if selected_time is not None:
                        st.subheader(f"📝 Posts at {selected_time}")
                        st.markdown(f"**Selected time:** {selected_time} | **Posts count:** {selected_posts}")

                        # Get posts around this time point, grouped by account
                        posts_at_time, account_groups = get_posts_at_timepoint(full_df, selected_time, time_window_hours=1)

                        if len(posts_at_time) > 0:
                            st.write(f"**Found {len(posts_at_time)} posts within ±30 minutes from {len(account_groups)} accounts**")

                            if account_groups:
                                # Show top 10 accounts summary
                                st.subheader("📊 Top 10 Most Active Accounts")

                                # Create a summary table
                                summary_data = []
                                for username, data in account_groups.items():
                                    summary_data.append({
                                        'Username': f"@{username}",
                                        'Display Name': data['display_name'],
                                        'Post Count': data['post_count']
                                    })

                                summary_df = pd.DataFrame(summary_data)
                                st.dataframe(summary_df, use_container_width=True)

                                # Show posts grouped by account
                                st.subheader("📝 Posts by Account")

                                for rank, (username, data) in enumerate(account_groups.items(), 1):
                                    with st.expander(f"#{rank} @{username} ({data['display_name']}) - {data['post_count']} posts", expanded=(rank <= 3)):

                                        # Show account info
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Posts", data['post_count'])
                                        with col2:
                                            st.write(f"**Username:** @{username}")
                                        with col3:
                                            st.write(f"**Display Name:** {data['display_name']}")

                                        st.write("---")

                                        # Show individual posts for this account
                                        for idx, post in data['posts'].iterrows():
                                            st.write(f"**🕒 {post.get('created_at', 'No timestamp')}**")

                                            content = post.get('content_cleaned', 'No content available')
                                            if len(content) > 500:
                                                # Truncate very long posts
                                                with st.expander(f"📄 Post content (ID: {post.get('id', 'N/A')})"):
                                                    st.write(content)
                                            else:
                                                st.write(f"💬 {content}")

                                            st.write("")  # Add spacing between posts
                        else:
                            st.info("No posts found in the selected time window")
            else:
                st.info("👆 Click on any point in the chart above to see posts and accounts at that time")

        else:
            st.warning("⚠️ No bursts detected with current parameters. Try adjusting S and Gamma values.")

if __name__ == "__main__":
    main()
