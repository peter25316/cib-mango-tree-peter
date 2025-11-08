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
        self.kmeans_model = None
        self.account_features = None
        self.active_accounts_df = None
        self.scaled_features = None
        self.cluster_results = None

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
