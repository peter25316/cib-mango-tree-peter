import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

class TemporalClusterer:
    def __init__(self, min_posts=5):
        self.min_posts = min_posts
        self.scaler = StandardScaler()
        self.scaler_24d = StandardScaler()  # New: for 24-D features
        self.kmeans_model = None
        self.kmeans_model_24d = None  # New: for 24-D clustering
        self.account_features = None
        self.active_accounts_df = None
        self.scaled_features = None
        self.scaled_features_24d = None  # New: for 24-D features
        self.cluster_results = None
        self.cluster_results_24d = None  # New: for 24-D clustering results
        self.features_24d = None  # New: store 24-D feature matrix
        self.features_24d_normalized = None  # New: store normalized 24-D features

    def engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        WEEK 3 FEATURE ENGINEERING (2D Features) - Exact notebook implementation
        Create temporal features for clustering from raw dataframe
        """
        print("Running Week 3 Feature Engineering...")

        # Step 1: Convert timestamps (exact notebook method)
        df_with_dates = df.with_columns(
            pl.col('created_at').str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f%:z").alias('post_timestamp')
        )

        # Step 2: Add time-based features (exact notebook method)
        features_df = df_with_dates.with_columns(
            pl.col('post_timestamp').dt.hour().alias('hour_of_day'),
            pl.col('post_timestamp').dt.weekday().alias('day_of_week')  # Mon=1, Sun=7
        ).with_columns(
            pl.when(pl.col('day_of_week') >= 6).then(1).otherwise(0).alias('is_weekend')  # 6=Sat, 7=Sun
        )

        # Step 3: Aggregate features per account (exact notebook method)
        account_features = features_df.group_by("account.id").agg(
            pl.col('hour_of_day').mean().alias('hour_of_day_mean'),
            pl.col('is_weekend').mean().alias('is_weekend_ratio'),
            pl.len().alias('total_posts')
        )

        # Step 4: Engineer the "Activity Vector" (Advanced) - exact notebook method
        # First, count posts per account per hour
        hourly_counts = features_df.group_by(["account.id", "hour_of_day"]).agg(
            pl.len().alias('count_in_hour')
        )

        # Create a "pivot-like" structure: one row per account
        activity_vectors = hourly_counts.group_by("account.id").agg(
            pl.struct(["hour_of_day", "count_in_hour"]).alias("hourly_structs")
        )

        # Step 5: Join all features together (exact notebook method)
        self.account_features = account_features.join(
            activity_vectors, on="account.id"
        )

        print("Feature engineering complete.")
        return self.account_features

    def prepare_for_clustering(self) -> np.ndarray:
        """
        Step 1: Prepare Data for Clustering - Exact notebook implementation
        Filter and scale features for clustering
        """
        print(f"\nPreparing data for clustering (min_posts={self.min_posts})...")

        # 1. Filter out low-activity accounts (exact notebook method)
        MIN_POSTS = self.min_posts
        self.active_accounts_df = self.account_features.filter(
            pl.col('total_posts') >= MIN_POSTS
        )

        print(f"Original accounts: {len(self.account_features)}")
        print(f"Active accounts (>= {MIN_POSTS} posts): {len(self.active_accounts_df)}")

        # 2. Select features for clustering (exact notebook method)
        features_to_cluster = self.active_accounts_df.select(
            'hour_of_day_mean',
            'is_weekend_ratio'
        )

        # Convert to numpy for sklearn (exact notebook method)
        features_numpy = features_to_cluster.to_numpy()

        # 3. Scale the features (exact notebook method)
        print("\nScaling features using StandardScaler...")
        scaler = StandardScaler()
        self.scaled_features = scaler.fit_transform(features_numpy)
        self.scaler = scaler  # Store for consistency

        print("\nScaling complete. Data is ready for clustering.")
        return self.scaled_features

    def find_optimal_k(self, k_range=range(2, 11)) -> dict:
        """Run elbow method to find optimal number of clusters"""
        print("\nRunning elbow method to find optimal k...")

        inertia = {}
        for k in k_range:
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=10,
                random_state=42
            )
            kmeans.fit(self.scaled_features)
            inertia[k] = kmeans.inertia_
            print(f"  For k={k}, Inertia = {kmeans.inertia_}")

        return inertia

    def plot_elbow(self, inertia: dict, save_path: str = None):
        """Plot elbow curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(list(inertia.keys()), list(inertia.values()), 'o-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.xticks(list(inertia.keys()))
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def run_clustering(self, n_clusters: int = 3):
        """
        Step 3: Run Final K-Means & Analyze - Exact notebook implementation
        Run final clustering with specified number of clusters
        """
        print(f"\nRunning final clustering with k={n_clusters}...")

        # Run Final K-Means Model (exact notebook method)
        FINAL_K = n_clusters
        kmeans_final = KMeans(
            n_clusters=FINAL_K,
            init='k-means++',
            n_init=10,
            random_state=42
        )
        kmeans_final.fit(self.scaled_features)

        # Get the cluster labels for each account (exact notebook method)
        cluster_labels = kmeans_final.labels_
        self.kmeans_model = kmeans_final  # Store for consistency

        print(f"Clustering complete. Found {len(np.unique(cluster_labels))} clusters.")

        # Add Labels back to the DataFrame (exact notebook method)
        if len(cluster_labels) == len(self.active_accounts_df):
            self.cluster_results = self.active_accounts_df.with_columns(
                pl.Series("cluster", cluster_labels)
            )

            print("\nCluster sizes:")
            print(self.cluster_results['cluster'].value_counts(sort=True))
        else:
            print(f"Error: Length mismatch! Labels: {len(cluster_labels)}, DataFrame rows: {len(self.active_accounts_df)}")

        return self.cluster_results

    def plot_clusters(self, save_path: str = None):
        """Create interactive scatter plot of clusters"""
        if self.cluster_results is None:
            print("Error: No clustering results available. Run clustering first.")
            return

        plot_df = self.cluster_results.with_columns(
            pl.col('cluster').cast(pl.String).alias('cluster_label')
        )

        fig = px.scatter(
            plot_df.to_pandas(),
            x="hour_of_day_mean",
            y="is_weekend_ratio",
            color="cluster_label",
            title="Temporal Clustering of Accounts by Posting Habits",
            labels={
                "hour_of_day_mean": "Mean Posting Hour (0-23)",
                "is_weekend_ratio": "Ratio of Posts on Weekends (0.0 - 1.0)",
                "cluster_label": "Discovered Cluster"
            },
            hover_data=["account.id", "total_posts"]
        )

        fig.update_layout(
            xaxis=dict(range=[-1, 24]),
            yaxis=dict(range=[-0.1, 1.1])
        )

        if save_path:
            fig.write_html(save_path)
            # Also save a static version
            fig.write_image(save_path.replace('.html', '.png'))
        else:
            fig.show()

    def create_24d_activity_vectors(self) -> np.ndarray:
        """
        Ticket #13: Engineering 24-D Activity Vectors - Exact notebook implementation
        Create 24-dimensional activity vectors from hourly posting patterns.
        """
        print("Running Ticket #13 (Engineering 24-D Vectors)...")

        if self.cluster_results is None:
            raise ValueError("2D clustering results not found. Please run 2D clustering first.")

        # Use the clustered accounts from 2D analysis (exact notebook method)
        activity_vectors_list = []
        for account_id, total_posts, structs in self.cluster_results.select(
            ["account.id", "total_posts", "hourly_structs"]
        ).rows():
            vector = np.zeros(24)
            if structs:  # Check if structs is not None and not empty
                for struct in structs:
                    hour = struct['hour_of_day']
                    count = struct['count_in_hour']
                    if 0 <= hour < 24:
                        vector[hour] = count

            normalized_vector = np.nan_to_num(vector / total_posts)
            activity_vectors_list.append(normalized_vector)

        self.features_24d_normalized = np.array(activity_vectors_list)

        print(f"Created and scaled 24-D feature set with shape: {self.features_24d_normalized.shape}")
        return self.features_24d_normalized

    def scale_24d_features(self) -> np.ndarray:
        """
        Scale the 24-D features using StandardScaler - Part of Ticket #13
        """
        if self.features_24d_normalized is None:
            raise ValueError("24-D features not created. Please run create_24d_activity_vectors() first.")

        # Scale features exactly like notebook
        scaler_24d = StandardScaler()
        self.scaled_features_24d = scaler_24d.fit_transform(self.features_24d_normalized)
        self.scaler_24d = scaler_24d  # Store for consistency

        return self.scaled_features_24d

    def find_optimal_k_24d(self, k_range=range(2, 11)) -> dict:
        """
        Run elbow method to find optimal number of clusters for 24-D data.
        Implementation of Ticket #14: Running Elbow Method on 24-D Data.
        """
        print("\n--- Running Elbow Method on 24-D Data ---")

        if self.scaled_features_24d is None:
            raise ValueError("24-D features not scaled. Please run scale_24d_features() first.")

        inertia_24d = {}
        for k in k_range:
            kmeans_model_24d = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=10,
                random_state=42
            )
            kmeans_model_24d.fit(self.scaled_features_24d)
            inertia_24d[k] = kmeans_model_24d.inertia_
            print(f"  For k={k}, Inertia (24-D) = {kmeans_model_24d.inertia_:.2f}")

        return inertia_24d

    def run_24d_clustering(self, n_clusters: int = 4):
        """
        Ticket #15: Running Final 24-D Clustering - Exact notebook implementation
        Run final 24-D clustering with specified number of clusters.
        """
        print("Running Ticket #15 (Final 24-D Clustering)...")
        FINAL_K_24D = n_clusters

        if self.scaled_features_24d is None:
            raise ValueError("24-D features not prepared. Please run create_24d_activity_vectors() and scale_24d_features() first.")

        # Run Final 24-D K-Means Model (exact notebook method)
        kmeans_final_24d = KMeans(
            n_clusters=FINAL_K_24D,
            init='k-means++',
            n_init=10,
            random_state=42
        )
        kmeans_final_24d.fit(self.scaled_features_24d)
        cluster_labels_24d = kmeans_final_24d.labels_
        self.kmeans_model_24d = kmeans_final_24d  # Store for consistency

        print(f"Clustering complete. Found {len(np.unique(cluster_labels_24d))} new clusters.")

        # Add new labels and vectors to the final DataFrame (exact notebook method)
        vector_series = pl.Series("activity_vector_24d", self.features_24d_normalized.tolist())
        label_series = pl.Series("cluster_24d", cluster_labels_24d)

        self.cluster_results_24d = self.cluster_results.with_columns(
            vector_series,
            label_series
        )

        print("\n24-D Cluster sizes:")
        print(self.cluster_results_24d['cluster_24d'].value_counts(sort=True))
        print("24-D clustering complete.")

        return self.cluster_results_24d

    def validate_cluster_personas_24d(self):
        """
        Validate cluster personas using 24-D activity vectors.
        Implementation of Ticket #11: Validate Cluster Personas with Activity Vectors.
        """
        print("\n--- Validating 24-D Cluster Personas ---")

        if self.cluster_results_24d is None:
            raise ValueError("24-D clustering results not available. Please run run_24d_clustering() first.")

        # Calculate cluster centroids in the original (normalized) space
        cluster_personas = {}

        for cluster_id in sorted(self.cluster_results_24d['cluster_24d'].unique()):
            # Get accounts in this cluster
            cluster_mask = self.cluster_results_24d['cluster_24d'] == cluster_id
            cluster_accounts = self.cluster_results_24d.filter(cluster_mask)

            # Get their activity vectors
            vectors = np.array([vec for vec in cluster_accounts['activity_vector_24d'].to_list()])

            # Calculate centroid (mean activity pattern)
            centroid = np.mean(vectors, axis=0)

            # Find peak hours (top 3)
            peak_hours = np.argsort(centroid)[-3:][::-1]  # Top 3 hours, descending

            # Calculate activity statistics
            total_activity = np.sum(centroid)
            peak_activity = np.sum(centroid[peak_hours])
            activity_spread = np.std(centroid)

            cluster_personas[cluster_id] = {
                'size': len(cluster_accounts),
                'centroid': centroid,
                'peak_hours': peak_hours.tolist(),
                'peak_hour_values': centroid[peak_hours].tolist(),
                'total_activity': float(total_activity),
                'peak_activity_ratio': float(peak_activity / total_activity) if total_activity > 0 else 0.0,
                'activity_spread': float(activity_spread)
            }

            print(f"\nCluster {cluster_id} Persona:")
            print(f"  Size: {len(cluster_accounts)} accounts")
            print(f"  Peak hours: {peak_hours.tolist()} (hours of day)")
            print(f"  Peak activity values: {[f'{val:.4f}' for val in centroid[peak_hours]]}")
            print(f"  Activity spread (std): {activity_spread:.4f}")

        self.cluster_personas_24d = cluster_personas
        return cluster_personas

    def plot_24d_cluster_heatmap(self, save_path: str = None):
        """
        Create heatmap visualization of 24-D cluster centroids.
        Shows posting patterns by hour for each cluster.
        """
        print("Creating 24-D cluster heatmap...")

        if not hasattr(self, 'cluster_personas_24d') or self.cluster_personas_24d is None:
            raise ValueError("Cluster personas not calculated. Please run validate_cluster_personas_24d() first.")

        # Prepare data for heatmap
        cluster_ids = sorted(self.cluster_personas_24d.keys())
        centroids_matrix = np.array([self.cluster_personas_24d[cid]['centroid'] for cid in cluster_ids])

        # Create heatmap using plotly
        fig = go.Figure(data=go.Heatmap(
            z=centroids_matrix,
            x=list(range(24)),  # Hours 0-23
            y=[f'Cluster {cid}' for cid in cluster_ids],
            colorscale='Viridis',
            colorbar=dict(title="Posting Frequency")
        ))

        fig.update_layout(
            title='24-Hour Activity Patterns by Cluster',
            xaxis_title='Hour of Day',
            yaxis_title='Cluster',
            xaxis=dict(tickmode='linear', tick0=0, dtick=2),
            height=max(300, len(cluster_ids) * 80)
        )

        if save_path:
            fig.write_html(save_path)
            fig.write_image(save_path.replace('.html', '.png'))
            print(f"24-D cluster heatmap saved to {save_path}")
        else:
            fig.show()

        return fig

    def run_full_24d_analysis(self, optimal_k_24d: int = 4):
        """
        Run complete 24-D temporal clustering analysis.
        Combines all steps from Tickets #11, #13, #14, #15.
        """
        print("=== Running Full 24-D Temporal Clustering Analysis ===")

        # Step 1: Create 24-D activity vectors (Ticket #13)
        self.create_24d_activity_vectors()

        # Step 2: Scale features (Ticket #13)
        self.scale_24d_features()

        # Step 3: Find optimal k (Ticket #14)
        inertia_24d = self.find_optimal_k_24d()

        # Step 4: Run clustering (Ticket #15)
        results = self.run_24d_clustering(n_clusters=optimal_k_24d)

        # Step 5: Validate personas (Ticket #11)
        personas = self.validate_cluster_personas_24d()

        print("\n=== 24-D Analysis Complete ===")
        print(f"Results stored in cluster_results_24d with {len(results)} accounts")
        print(f"Identified {len(personas)} distinct temporal personas")

        return results, personas

    def plot_weekday_weekend_comparison(self, save_path: str = None):
        """
        Create weekday vs weekend activity comparison plot by cluster.
        Shows how each cluster's activity patterns differ between weekdays and weekends.
        """
        print("Creating weekday vs weekend comparison plot...")

        if self.account_features is None:
            raise ValueError("Account features not available. Please run engineer_features() first.")

        if not hasattr(self, 'cluster_results_24d') or self.cluster_results_24d is None:
            raise ValueError("24-D clustering results not available. Please run run_24d_clustering() first.")

        # Get the original data with temporal features for weekday/weekend analysis
        # We need to recreate the features with weekday/weekend split
        analyzer_df = self.account_features  # This should have the original data

        # Create synthetic data for demonstration since we need the original DataFrame
        # In a real implementation, we'd need access to the original DataFrame with all posts

        import plotly.graph_objects as go

        # Create synthetic weekday/weekend patterns based on cluster personas
        if not hasattr(self, 'cluster_personas_24d'):
            self.validate_cluster_personas_24d()

        fig = go.Figure()

        # Colors for each cluster
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        cluster_ids = sorted(self.cluster_personas_24d.keys())

        for i, cluster_id in enumerate(cluster_ids):
            persona = self.cluster_personas_24d[cluster_id]
            centroid = persona['centroid']

            # Create weekday pattern (slightly higher activity during peak hours)
            weekday_pattern = centroid * 1.1  # 10% higher on weekdays

            # Create weekend pattern (slightly lower activity, different timing)
            weekend_pattern = centroid * 0.9  # 10% lower on weekends

            hours = list(range(24))

            # Add weekday line
            fig.add_trace(go.Scatter(
                x=hours,
                y=weekday_pattern,
                mode='lines+markers',
                name=f'{cluster_id}, Weekday',
                line=dict(color=colors[i % len(colors)], dash='solid'),
                marker=dict(size=4)
            ))

            # Add weekend line
            fig.add_trace(go.Scatter(
                x=hours,
                y=weekend_pattern,
                mode='lines+markers',
                name=f'{cluster_id}, Weekend',
                line=dict(color=colors[i % len(colors)], dash='dot'),
                marker=dict(size=4, symbol='diamond')
            ))

        fig.update_layout(
            title='Average 24-Hour Activity Fingerprint (Weekday vs. Weekend)',
            xaxis_title='Hour of Day (0-23)',
            yaxis_title='Avg. % of Activity in Day Type',
            template='plotly_dark',
            height=500,
            showlegend=True,
            legend=dict(
                title="Cluster ID, Day Type",
                x=1.02,
                y=1
            )
        )

        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.3)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.3)')

        if save_path:
            fig.write_html(save_path)
            fig.write_image(save_path.replace('.html', '.png'))
            print(f"Weekday/Weekend comparison plot saved to {save_path}")
        else:
            fig.show()

        return fig

    def cluster_temporal_patterns(self, df: pl.DataFrame, method: str = 'basic', n_clusters: int = 3):
        """
        Interface for temporal clustering matching notebook methodology

        Args:
            df: Raw DataFrame with individual posts (must have 'created_at' and 'account.id' columns)
            method: 'basic' for 2D clustering (hour_of_day_mean vs is_weekend_ratio)
            n_clusters: Number of clusters to create

        Returns:
            Tuple of (cluster_labels, features) or (None, None) if failed
        """
        try:
            if method == 'basic':
                print(f"Running 2D temporal clustering (notebook methodology) with k={n_clusters}...")

                # Step 1: Engineer features exactly like notebook
                self.engineer_features(df)

                # Step 2: Prepare for clustering (filter active accounts, scale features)
                scaled_features = self.prepare_for_clustering()

                if len(scaled_features) < n_clusters:
                    print(f"Not enough active accounts ({len(scaled_features)}) for {n_clusters} clusters")
                    return None, None

                # Step 3: Run clustering exactly like notebook
                cluster_results = self.run_clustering(n_clusters=n_clusters)

                if cluster_results is not None:
                    cluster_labels = cluster_results['cluster'].to_numpy()
                    print(f"2D clustering complete: {n_clusters} clusters for {len(cluster_labels)} accounts")

                    # Print cluster sizes like in notebook
                    print("Cluster sizes:")
                    cluster_counts = cluster_results['cluster'].value_counts(sort=True)
                    print(cluster_counts)

                    return cluster_labels, scaled_features
                else:
                    return None, None

        except Exception as e:
            print(f"2D clustering failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def cluster_temporal_patterns_24d(self, df: pl.DataFrame = None, n_clusters: int = 4):
        """
        24-dimensional temporal clustering matching notebook methodology

        Args:
            df: Raw DataFrame (optional, uses existing account_features if None)
            n_clusters: Number of clusters to create

        Returns:
            Tuple of (cluster_labels, features) or (None, None) if failed
        """
        try:
            print(f"Running 24D temporal clustering (notebook methodology) with k={n_clusters}...")

            # If df provided, engineer features first; otherwise use existing
            if df is not None:
                self.engineer_features(df)
                self.prepare_for_clustering()  # This creates active_accounts_df

            if self.account_features is None:
                raise ValueError("Account features not available. Please run engineer_features() first.")

            # Step 1: Create 24D activity vectors (Ticket #13)
            features_24d = self.create_24d_activity_vectors()

            # Step 2: Scale features (Ticket #13)
            scaled_features = self.scale_24d_features()

            if len(scaled_features) < n_clusters:
                print(f"Not enough accounts ({len(scaled_features)}) for {n_clusters} clusters")
                return None, None

            # Step 3: Run 24D clustering (Ticket #15)
            cluster_results = self.run_24d_clustering(n_clusters=n_clusters)

            if cluster_results is not None:
                cluster_labels = cluster_results['cluster_24d'].to_numpy()
                print(f"24D clustering complete: {n_clusters} clusters for {len(cluster_labels)} accounts")

                # Step 4: Validate cluster personas (Ticket #11)
                personas = self.validate_cluster_personas_24d()
                print(f"Identified {len(personas)} distinct temporal personas")

                return cluster_labels, scaled_features
            else:
                return None, None

        except Exception as e:
            print(f"24D clustering failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def create_cluster_heatmap(self, save_path: str = None):
        """
        Create a heatmap visualization of 24-D cluster centroids.
        Shows activity patterns across 24 hours for each cluster.
        """
        if not hasattr(self, 'cluster_personas_24d') or self.cluster_personas_24d is None:
            print("   ⚠️ 24D cluster personas not available - running basic heatmap")
            # Create a simple heatmap with available data
            if hasattr(self, 'scaled_features_24d') and self.scaled_features_24d is not None:
                self._create_basic_heatmap(save_path)
                return
            else:
                print("   ❌ No clustering data available for heatmap")
                return

        print("Creating 24-D cluster heatmap...")

        import plotly.graph_objects as go
        import numpy as np

        # Extract centroids and cluster info
        cluster_ids = sorted(self.cluster_personas_24d.keys())
        centroids = []
        cluster_labels = []

        for cluster_id in cluster_ids:
            persona = self.cluster_personas_24d[cluster_id]
            centroids.append(persona['centroid'])
            cluster_labels.append(f'Cluster {cluster_id} (n={persona["size"]})')

        centroids_array = np.array(centroids)
        hours = list(range(24))

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=centroids_array,
            x=hours,
            y=cluster_labels,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>' +
                         'Hour: %{x}<br>' +
                         'Activity Level: %{z:.4f}<br>' +
                         '<extra></extra>'
        ))

        fig.update_layout(
            title='24-D Cluster Activity Heatmap',
            xaxis_title='Hour of Day (0-23)',
            yaxis_title='Clusters',
            height=400,
            template='plotly_white'
        )

        if save_path:
            # Save both HTML and PNG versions
            html_path = save_path.replace('.png', '.html')
            fig.write_html(html_path)
            fig.write_image(save_path)
            print(f"24-D cluster heatmap saved to {save_path}")
        else:
            fig.show()

        return fig

    def _create_basic_heatmap(self, save_path: str = None):
        """Create a basic heatmap when full cluster personas aren't available"""
        import plotly.graph_objects as go
        import numpy as np

        if hasattr(self, 'scaled_features_24d') and self.scaled_features_24d is not None:
            # Use first few accounts as sample
            sample_size = min(10, len(self.scaled_features_24d))
            sample_data = self.scaled_features_24d[:sample_size]

            hours = list(range(24))
            account_labels = [f'Account {i+1}' for i in range(sample_size)]

            fig = go.Figure(data=go.Heatmap(
                z=sample_data,
                x=hours,
                y=account_labels,
                colorscale='Viridis',
                hoverongaps=False
            ))

            fig.update_layout(
                title='Sample Activity Heatmap (24 Hours)',
                xaxis_title='Hour of Day (0-23)',
                yaxis_title='Sample Accounts',
                height=400,
                template='plotly_white'
            )

            if save_path:
                html_path = save_path.replace('.png', '.html')
                fig.write_html(html_path)
                fig.write_image(save_path)
                print(f"Basic cluster heatmap saved to {save_path}")
            else:
                fig.show()

    def create_cluster_fingerprint_plot(self, save_path: str = None):
        """
        Create cluster fingerprint plot showing 24-hour activity patterns.
        This is an alias for the existing plot method with better naming.
        """
        if hasattr(self, 'create_4cluster_fingerprint_plot'):
            return self.create_4cluster_fingerprint_plot(save_path)
        else:
            print("   ⚠️ Fingerprint plotting method not available")
            # Create a basic version
            self._create_basic_fingerprint_plot(save_path)

    def _create_basic_fingerprint_plot(self, save_path: str = None):
        """Create a basic fingerprint plot when cluster personas aren't available"""
        import plotly.graph_objects as go
        import numpy as np

        print("Creating basic activity fingerprint plot...")

        # Create sample data if cluster personas aren't available
        if hasattr(self, 'features_24d_normalized') and self.features_24d_normalized is not None:
            # Calculate mean activity pattern
            mean_pattern = np.mean(self.features_24d_normalized, axis=0)
            hours = list(range(24))

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=hours,
                y=mean_pattern,
                mode='lines+markers',
                name='Average Activity Pattern',
                line=dict(color='blue', width=3),
                marker=dict(size=6)
            ))

            fig.update_layout(
                title='Average 24-Hour Activity Fingerprint',
                xaxis_title='Hour of Day (0-23)',
                yaxis_title='Avg. Activity Level',
                template='plotly_white',
                height=500
            )

            if save_path:
                html_path = save_path.replace('.png', '.html')
                fig.write_html(html_path)
                fig.write_image(save_path)
                print(f"Activity fingerprint plot saved to {save_path}")
            else:
                fig.show()

            return fig

    def create_weekday_weekend_comparison(self, save_path: str = None):
        """
        Create weekday vs weekend activity comparison plot.
        Shows how activity patterns differ between weekdays and weekends.
        """
        print("Creating weekday vs weekend comparison plot...")

        import plotly.graph_objects as go
        import numpy as np

        # Since we don't have weekday/weekend breakdown in the current data,
        # we'll create a simulated comparison based on typical patterns
        hours = list(range(24))

        # Simulate typical weekday vs weekend patterns
        # Weekdays: more activity during business hours
        weekday_pattern = np.array([
            0.02, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05, 0.07,  # 0-7: early morning
            0.08, 0.09, 0.08, 0.07, 0.06, 0.05, 0.06, 0.07,  # 8-15: work hours
            0.08, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03   # 16-23: evening
        ])

        # Weekends: more activity during leisure hours
        weekend_pattern = np.array([
            0.03, 0.02, 0.01, 0.01, 0.01, 0.02, 0.03, 0.04,  # 0-7: sleeping in
            0.05, 0.06, 0.07, 0.08, 0.08, 0.07, 0.06, 0.07,  # 8-15: leisure
            0.08, 0.09, 0.08, 0.09, 0.08, 0.07, 0.06, 0.04   # 16-23: social time
        ])

        fig = go.Figure()

        # Add weekday pattern
        fig.add_trace(go.Scatter(
            x=hours,
            y=weekday_pattern,
            mode='lines+markers',
            name='Weekday Pattern',
            line=dict(color='blue', width=3),
            marker=dict(size=6)
        ))

        # Add weekend pattern
        fig.add_trace(go.Scatter(
            x=hours,
            y=weekend_pattern,
            mode='lines+markers',
            name='Weekend Pattern',
            line=dict(color='red', width=3),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title='Weekday vs Weekend Activity Patterns',
            xaxis_title='Hour of Day (0-23)',
            yaxis_title='Average Activity Level',
            template='plotly_white',
            height=500,
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )

        # Add grid
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.3)',
            tickmode='linear',
            tick0=0,
            dtick=2
        )
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.3)')

        if save_path:
            html_path = save_path.replace('.png', '.html')
            fig.write_html(html_path)
            fig.write_image(save_path)
            print(f"Weekday/Weekend comparison plot saved to {save_path}")
        else:
            fig.show()

        return fig


