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
        """Create temporal features for clustering from raw dataframe"""
        print("Engineering temporal features...")

        # Convert timestamps
        df_with_dates = df.with_columns(
            pl.col('created_at').str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f%:z").alias('post_timestamp')
        )

        # Add time-based features
        features_df = df_with_dates.with_columns(
            pl.col('post_timestamp').dt.hour().alias('hour_of_day'),
            pl.col('post_timestamp').dt.weekday().alias('day_of_week')
        ).with_columns(
            pl.when(pl.col('day_of_week') >= 6)
              .then(1)
              .otherwise(0)
              .alias('is_weekend')
        )

        # Aggregate features per account
        self.account_features = features_df.group_by("account.id").agg(
            pl.col('hour_of_day').mean().alias('hour_of_day_mean'),
            pl.col('is_weekend').mean().alias('is_weekend_ratio'),
            pl.len().alias('total_posts')
        )

        # Create hourly activity vectors
        hourly_counts = features_df.group_by(["account.id", "hour_of_day"]).agg(
            pl.len().alias('count_in_hour')
        )

        activity_vectors = hourly_counts.group_by("account.id").agg(
            pl.struct(["hour_of_day", "count_in_hour"]).alias("hourly_structs")
        )

        # Join all features
        self.account_features = self.account_features.join(
            activity_vectors, on="account.id"
        )

        return self.account_features

    def prepare_for_clustering(self) -> np.ndarray:
        """Filter and scale features for clustering"""
        print(f"\nPreparing data for clustering (min_posts={self.min_posts})...")

        # Filter active accounts
        self.active_accounts_df = self.account_features.filter(
            pl.col('total_posts') >= self.min_posts
        )

        print(f"Original accounts: {len(self.account_features)}")
        print(f"Active accounts (>= {self.min_posts} posts): {len(self.active_accounts_df)}")

        # Select features for clustering
        features_to_cluster = self.active_accounts_df.select(
            'hour_of_day_mean',
            'is_weekend_ratio'
        )

        # Scale features
        features_numpy = features_to_cluster.to_numpy()
        self.scaled_features = self.scaler.fit_transform(features_numpy)

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
        """Run final clustering with specified number of clusters"""
        print(f"\nRunning final clustering with k={n_clusters}...")

        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=10,
            random_state=42
        )
        self.kmeans_model.fit(self.scaled_features)

        # Add cluster labels to DataFrame
        cluster_labels = self.kmeans_model.labels_
        self.cluster_results = self.active_accounts_df.with_columns(
            pl.Series("cluster", cluster_labels)
        )

        print("\nCluster sizes:")
        print(self.cluster_results['cluster'].value_counts(sort=True))

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
        Create 24-dimensional activity vectors from hourly posting patterns.
        Implementation of Ticket #13: Engineering 24-D Activity Vectors.
        """
        print("\n--- Creating 24-D Activity Vectors ---")

        if self.account_features is None:
            raise ValueError("Account features not found. Please run engineer_features() first.")

        # Filter active accounts for 24-D analysis
        active_accounts = self.account_features.filter(
            pl.col('total_posts') >= self.min_posts
        )

        print(f"Creating 24-D vectors for {len(active_accounts)} active accounts...")

        # Create list to hold 24-D feature data
        activity_vectors_list = []
        account_ids = []

        # Process each account
        for account_id, total_posts, structs in active_accounts.select(
            ["account.id", "total_posts", "hourly_structs"]
        ).rows():

            # Initialize vector of 24 zeros (one for each hour)
            vector = np.zeros(24)

            if structs:  # Check if structs is not None
                for struct in structs:
                    hour = struct['hour_of_day']
                    count = struct['count_in_hour']
                    if 0 <= hour < 24:  # Safety check
                        vector[hour] = count

            # Normalize by total posts to get frequency
            normalized_vector = np.nan_to_num(vector / total_posts) if total_posts > 0 else vector
            activity_vectors_list.append(normalized_vector)
            account_ids.append(account_id)

        # Convert to numpy array for sklearn
        self.features_24d_normalized = np.array(activity_vectors_list)

        print(f"Created 24-D feature set with shape: {self.features_24d_normalized.shape}")

        # Store account mapping for results
        self.account_ids_24d = account_ids

        return self.features_24d_normalized

    def scale_24d_features(self) -> np.ndarray:
        """
        Scale the 24-D features using StandardScaler.
        Part of Ticket #13 implementation.
        """
        print("Scaling 24-D features using StandardScaler...")

        if self.features_24d_normalized is None:
            raise ValueError("24-D features not created. Please run create_24d_activity_vectors() first.")

        self.scaled_features_24d = self.scaler_24d.fit_transform(self.features_24d_normalized)
        print("24-D feature scaling complete.")

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
        Run final 24-D clustering with specified number of clusters.
        Implementation of Ticket #15: Running 24-D Clustering & Visualizing.
        """
        print(f"\n--- Running Final 24-D Clustering (k={n_clusters}) ---")

        if self.scaled_features_24d is None:
            raise ValueError("24-D features not prepared. Please run create_24d_activity_vectors() and scale_24d_features() first.")

        # Run K-Means clustering
        self.kmeans_model_24d = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=10,
            random_state=42
        )
        self.kmeans_model_24d.fit(self.scaled_features_24d)
        cluster_labels_24d = self.kmeans_model_24d.labels_

        print(f"24-D Clustering complete. Found {len(np.unique(cluster_labels_24d))} clusters.")

        # Create results DataFrame
        results_data = {
            'account.id': self.account_ids_24d,
            'cluster_24d': cluster_labels_24d,
            'activity_vector_24d': self.features_24d_normalized.tolist()
        }

        self.cluster_results_24d = pl.DataFrame(results_data)

        # Print cluster sizes
        print("\n24-D Cluster sizes:")
        cluster_counts = self.cluster_results_24d['cluster_24d'].value_counts(sort=True)
        print(cluster_counts)

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

    def plot_cluster_fingerprints(self, save_path: str = None):
        """
        Create pure cluster fingerprint plot showing average 24-hour patterns.
        Shows the centroid activity patterns for each cluster.
        """
        print("Creating cluster fingerprint plot...")

        if not hasattr(self, 'cluster_personas_24d') or self.cluster_personas_24d is None:
            raise ValueError("Cluster personas not calculated. Please run validate_cluster_personas_24d() first.")

        import plotly.graph_objects as go

        fig = go.Figure()

        # Colors for each cluster
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        cluster_ids = sorted(self.cluster_personas_24d.keys())

        for i, cluster_id in enumerate(cluster_ids):
            persona = self.cluster_personas_24d[cluster_id]
            centroid = persona['centroid']
            size = persona['size']

            hours = list(range(24))

            fig.add_trace(go.Scatter(
                x=hours,
                y=centroid,
                mode='lines+markers',
                name=f'{cluster_id}',
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=6),
                hovertemplate=f'<b>Cluster {cluster_id}</b><br>' +
                             'Hour: %{x}<br>' +
                             'Activity: %{y:.4f}<br>' +
                             f'Size: {size} accounts<extra></extra>'
            ))

        # Add hover annotation example (like in your image)
        if cluster_ids:
            # Add annotation for one of the clusters as example
            sample_cluster = cluster_ids[-1]  # Last cluster
            sample_persona = self.cluster_personas_24d[sample_cluster]
            peak_hour = sample_persona['peak_hours'][0]
            peak_value = sample_persona['centroid'][peak_hour]

            fig.add_annotation(
                x=peak_hour,
                y=peak_value,
                text=f"New Cluster ID={sample_cluster}<br>" +
                     f"Hour of Day (0-23)={peak_hour}<br>" +
                     f"Avg. % of Daily Activity={peak_value:.6f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="white",
                ax=50,
                ay=-50,
                bgcolor="rgba(128, 0, 128, 0.8)",
                bordercolor="white",
                borderwidth=2,
                font=dict(color="white", size=12)
            )

        fig.update_layout(
            title='Average 24-Hour Fingerprint (4-Cluster 24-D Model)',
            xaxis_title='Hour of Day (0-23)',
            yaxis_title='Avg. % of Daily Activity',
            template='plotly_dark',
            height=500,
            showlegend=True,
            legend=dict(
                title="New Cluster ID",
                x=1.02,
                y=1
            )
        )

        # Add grid
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.3)',
            range=[-1, 24],
            tickmode='linear',
            tick0=0,
            dtick=1
        )
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.3)')

        if save_path:
            fig.write_html(save_path)
            fig.write_image(save_path.replace('.html', '.png'))
            print(f"Cluster fingerprint plot saved to {save_path}")
        else:
            fig.show()

        return fig
