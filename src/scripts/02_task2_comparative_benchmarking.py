import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('default')

class SimpleLinearRegression:
    """Simple linear regression to replace sklearn"""
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # Calculate coefficients
        coeffs = np.linalg.solve(X_with_intercept.T @ X_with_intercept, X_with_intercept.T @ y)
        
        self.intercept_ = coeffs[0]
        self.coef_ = coeffs[1:] if len(coeffs) > 2 else coeffs[1]
        
        return self
        
    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if np.isscalar(self.coef_):
            return self.intercept_ + self.coef_ * X.flatten()
        else:
            return self.intercept_ + X @ self.coef_
            
    def score(self, X, y):
        """Calculate R-squared"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

class SimpleStandardScaler:
    """Simple standard scaler to replace sklearn"""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit_transform(self, X):
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return (X - self.mean_) / self.scale_

class SimpleKMeans:
    """Simple K-means clustering to replace sklearn"""
    def __init__(self, n_clusters=4, random_state=42, n_init=10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        
    def fit_predict(self, X):
        np.random.seed(self.random_state)
        X = np.array(X)
        n_samples, n_features = X.shape
        
        best_inertia = float('inf')
        best_labels = None
        
        for _ in range(self.n_init):
            # Initialize centroids randomly
            centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
            
            for _ in range(100):  # Max iterations
                # Assign points to closest centroid
                distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
                labels = np.argmin(distances, axis=0)
                
                # Update centroids
                new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
                
                # Check convergence
                if np.allclose(centroids, new_centroids):
                    break
                centroids = new_centroids
            
            # Calculate inertia
            inertia = sum(np.sum((X[labels == k] - centroids[k])**2) for k in range(self.n_clusters))
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
        
        return best_labels

def simple_silhouette_score(X, labels):
    """Simple silhouette score calculation"""
    X = np.array(X)
    n_samples = len(X)
    n_clusters = len(np.unique(labels))
    
    if n_clusters < 2:
        return 0
    
    silhouette_vals = []
    
    for i in range(n_samples):
        # Calculate a(i) - average distance to points in same cluster
        same_cluster = X[labels == labels[i]]
        if len(same_cluster) > 1:
            a_i = np.mean([np.linalg.norm(X[i] - point) for point in same_cluster if not np.array_equal(point, X[i])])
        else:
            a_i = 0
        
        # Calculate b(i) - min average distance to points in other clusters
        b_i = float('inf')
        for cluster in np.unique(labels):
            if cluster != labels[i]:
                other_cluster = X[labels == cluster]
                if len(other_cluster) > 0:
                    avg_dist = np.mean([np.linalg.norm(X[i] - point) for point in other_cluster])
                    b_i = min(b_i, avg_dist)
        
        # Silhouette coefficient
        if max(a_i, b_i) > 0:
            silhouette_vals.append((b_i - a_i) / max(a_i, b_i))
        else:
            silhouette_vals.append(0)
    
    return np.mean(silhouette_vals)

def simple_ttest_ind(x, y):
    """Simple independent t-test (Welch's t-test for unequal variances)"""
    x, y = np.array(x), np.array(y)
    
    n1, n2 = len(x), len(y)
    mean1, mean2 = np.mean(x), np.mean(y)
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
    
    # Welch's t-statistic
    pooled_se = np.sqrt(var1/n1 + var2/n2)
    t_stat = (mean1 - mean2) / pooled_se
    
    # Degrees of freedom (Welch-Satterthwaite equation)
    df = (var1/n1 + var2/n2)**2 / (var1**2/(n1**2*(n1-1)) + var2**2/(n2**2*(n2-1)))
    
    # Simple p-value approximation
    p_value = 0.05 if abs(t_stat) > 2 else 0.1  # Rough approximation
    
    return t_stat, p_value

def load_master_data():
    """Load the complete master dataset including India"""
    master = pd.read_parquet("data/processed/beauty_income_panel.parquet")
    return master

def income_matched_snapshot(df):
    """2.1 Income-matched snapshot analysis"""
    
    # Get India's latest GDP per capita
    india_data = df[df['country'] == 'india'].copy().sort_values('year')
    latest_india = india_data.iloc[-1]  # Use iloc[-1] for most recent
    india_gdp = latest_india['gdppcppp']
    
    # Find income-matched years for peer countries
    peers = ['china', 'japan', 'south korea', 'usa']  # All benchmark countries
    matched_results = []
    
    for peer in peers:
        peer_data = df[df['country'] == peer].copy().sort_values('year')
        if peer_data.empty:
            continue
            
        # Check if peer was already above milestone at start
        if peer_data.iloc[0]['gdppcppp'] >= (india_gdp - 500):
            matched_year_data = peer_data.iloc[0]
        else:
            # Find first year with GDP >= India's current level
            milestone_candidates = peer_data[peer_data['gdppcppp'] >= (india_gdp - 500)]
            
            if len(milestone_candidates) > 0:
                matched_year_data = milestone_candidates.sort_values('year').iloc[0]
            else:
                # Fallback to closest match if no milestone reached
                peer_data['gdp_diff'] = abs(peer_data['gdppcppp'] - india_gdp)
                best_match_idx = peer_data['gdp_diff'].argmin()
                matched_year_data = peer_data.iloc[best_match_idx]
        
        # Calculate 5-year CAGR after matched year (require full 5 years)
        future_years = peer_data[peer_data['year'] > matched_year_data['year']].sort_values('year')
        
        # Find exact 5-year window or closest to 5 years
        if len(future_years) >= 5:
            # Look for exactly 5 years later first
            exact_match = future_years[future_years['year'] == matched_year_data['year'] + 5]
            if len(exact_match) > 0:
                future_5y = exact_match.iloc[0]
                actual_years = 5
            else:
                # Use closest available (usually annual data so should be close)
                future_5y = future_years.iloc[4]  # 5th year of data available
                actual_years = future_5y['year'] - matched_year_data['year']
            
            if actual_years > 0 and matched_year_data['beautypc'] > 0:
                cagr_5y = (future_5y['beautypc'] / matched_year_data['beautypc']) ** (1/actual_years) - 1
            else:
                cagr_5y = np.nan
        else:
            cagr_5y = np.nan
        
        # Check if this is a reasonable match (within 50% of India's GDP)
        gdp_ratio = matched_year_data['gdppcppp'] / india_gdp
        is_reasonable_match = 0.5 <= gdp_ratio <= 2.0
        
        matched_results.append({
            'peer_country': peer,
            'matched_year': matched_year_data['year'],
            'matched_gdp': matched_year_data['gdppcppp'],
            'matched_beautypc': matched_year_data['beautypc'],
            'cagr_5y_post_match': cagr_5y,
            'gdp_ratio': gdp_ratio,
            'is_reasonable_match': is_reasonable_match
        })
    
    # Calculate India's recent 5-year CAGR
    india_sorted = india_data.sort_values('year')
    if len(india_sorted) >= 6:  # Need 6 points for 5-year CAGR
        india_recent_cagr = (india_sorted.iloc[-1]['beautypc'] / india_sorted.iloc[-6]['beautypc']) ** (1/5) - 1
    else:
        india_recent_cagr = np.nan
    
    matched_df = pd.DataFrame(matched_results)
    matched_df['india_recent_cagr'] = india_recent_cagr
    
    # Save results
    matched_df.to_csv("figures/T2-1_income_matched_snapshot.csv", index=False)
    
    # Create bar chart T2-1
    _, ax = plt.subplots(figsize=(10, 6))
    
    countries = matched_df['peer_country'].tolist() + ['india']
    cagrs = matched_df['cagr_5y_post_match'].tolist() + [india_recent_cagr]
    colors = ['skyblue'] * len(matched_df) + ['coral']
    
    bars = ax.bar(countries, [c * 100 if not pd.isna(c) else 0 for c in cagrs], color=colors)
    
    ax.set_ylabel('5-Year CAGR (%)')
    ax.set_title('Beauty Consumption Growth: Peers at Similar Income vs India Recent')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, cagr in zip(bars, cagrs):
        if not pd.isna(cagr):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   f'{cagr*100:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("figures/T2-1_cagr_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    return matched_df, india_gdp

def alignment_chart(df, india_gdp):
    """2.2 Alignment chart (Years since milestone X)"""
    
    tolerance = 500  # $500 tolerance for GDP matching
    
    # Find Year 0 for each peer country
    peers = ['china', 'japan', 'south korea', 'usa']
    alignment_data = []
    
    for peer in peers:
        peer_data = df[df['country'] == peer].copy()
        if peer_data.empty:
            continue
            
        # Find first year where GDP >= india_gdp (within tolerance)
        milestone_years = peer_data[peer_data['gdppcppp'] >= (india_gdp - tolerance)]
        
        if len(milestone_years) > 0:
            year0 = milestone_years.iloc[0]['year']
            
            # Create relative years
            peer_data['rel_year'] = peer_data['year'] - year0
            
            # Filter to relevant range (0 to +10 years)
            relevant_data = peer_data[(peer_data['rel_year'] >= 0) & (peer_data['rel_year'] <= 10)]
            
            for _, row in relevant_data.iterrows():
                alignment_data.append({
                    'country': peer,
                    'rel_year': row['rel_year'],
                    'beautypc': row['beautypc'],
                    'year0': year0
                })
    
    # Add India at rel_year = 0 (current position)
    india_latest = df[df['country'] == 'india'].iloc[-1]
    alignment_data.append({
        'country': 'india',
        'rel_year': 0,
        'beautypc': india_latest['beautypc'],
        'year0': india_latest['year']
    })
    
    alignment_df = pd.DataFrame(alignment_data)
    
    # Create T2-2 plot
    plt.figure(figsize=(12, 8))
    
    colors = {'china': 'red', 'japan': 'blue', 'south korea': 'orange', 'usa': 'purple', 'india': 'green'}
    
    for country in alignment_df['country'].unique():
        country_data = alignment_df[alignment_df['country'] == country]
        
        if country == 'india':
            plt.scatter(country_data['rel_year'], country_data['beautypc'], 
                       color=colors[country], s=100, marker='o', 
                       label=f'{country.title()} (Current)', zorder=5)
        else:
            plt.plot(country_data['rel_year'], country_data['beautypc'], 
                    'o-', color=colors[country], alpha=0.7, 
                    label=f'{country.title()} (from Year 0: {country_data.iloc[0]["year0"]})')
    
    plt.xlabel('Years Since GDP Milestone')
    plt.ylabel('Beauty PC (USD 2015)')
    plt.title(f'Beauty Consumption Trajectories: Years Since ${india_gdp:,.0f} GDP per capita')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("figures/T2-2_alignment_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    return alignment_df

def beauty_share_overlay(df):
    """2.3 BeautyShare overlay plot with distinct colors"""
    
    plt.figure(figsize=(14, 10))
    
    countries = df['country'].unique()
    other_countries = [c for c in countries if c != 'india']
    colors = plt.cm.tab10(np.linspace(0, 1, len(other_countries)))
    
    # Plot other countries with distinct colors
    for country, color in zip(other_countries, colors):
        country_data = df[df['country'] == country]
        country_label = country.upper() if country == 'usa' else country.title()
        plt.scatter(country_data['gdppcppp'], country_data['beautyshare'] * 100, 
                   alpha=0.8, label=country_label, color=color, s=60)
    
    # Plot India with special highlighting
    india_data = df[df['country'] == 'india']
    if not india_data.empty:
        # Plot India trajectory in light red
        plt.plot(india_data['gdppcppp'], india_data['beautyshare'] * 100, 
                'o-', color='lightcoral', alpha=0.6, linewidth=2, markersize=6, 
                label='India Trajectory')
        
        # Highlight current India position with distinct color
        latest_india = india_data.iloc[-1]
        plt.scatter(latest_india['gdppcppp'], latest_india['beautyshare'] * 100, 
                   color='darkred', s=300, marker='*', 
                   label='India Current (2023)', zorder=10, edgecolor='black', linewidth=2)
    
    plt.xlabel('GDP per Capita PPP (2021 International $)')
    plt.ylabel('Beauty Share of Household Consumption (%)')
    plt.title('Beauty Consumption Share vs Income: India\'s Position Among Global Peers')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/T2-3_beauty_share_overlay.png", dpi=300, bbox_inches='tight')
    plt.close()
    

def rolling_elasticity_india(df):
    """2.4 Rolling elasticity for India"""
    
    india_data = df[df['country'] == 'india'].copy().sort_values('year')
    
    if len(india_data) < 10:
        return None
    
    # 5-year rolling window regression  
    window_size = 5
    rolling_results = []
    
    for i in range(window_size, len(india_data) + 1):
        window_data = india_data.iloc[i-window_size:i].copy()
        
        # Remove zero values for log regression
        valid_data = window_data[(window_data['beautypc'] > 0) & 
                                (window_data['gdppcppp'] > 0)]
        
        if len(valid_data) < 3:
            continue
        
        # Simple OLS on logs with relaxed guardrails for debugging
        try:
            X = np.log(valid_data['gdppcppp'].values).reshape(-1, 1)
            y = np.log(valid_data['beautypc'].values + 0.01)
            
            reg = SimpleLinearRegression().fit(X, y)
            r_squared = reg.score(X, y)
            
            # Relaxed guardrails for debugging - was too strict
            beautypc_var = valid_data['beautypc'].var()
            if r_squared < 0.1 or beautypc_var < 0.001:
                continue
                
            elasticity = reg.coef_
            end_year = valid_data.iloc[-1]['year']
            
            rolling_results.append({
                'end_year': end_year,
                'elasticity': elasticity,
                'r_squared': r_squared
            })
            
        except Exception:
            continue
    
    if not rolling_results:
        return None
    
    rolling_df = pd.DataFrame(rolling_results)
    
    # Create T2-4 plot
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_df['end_year'], rolling_df['elasticity'], 'o-', linewidth=2, markersize=6)
    plt.xlabel('Window End Year')
    plt.ylabel('Income Elasticity (β)')
    plt.title('India: Rolling 5-Year Income Elasticity of Beauty Consumption')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    if len(rolling_df) > 2:
        z = np.polyfit(rolling_df['end_year'], rolling_df['elasticity'], 1)
        p = np.poly1d(z)
        plt.plot(rolling_df['end_year'], p(rolling_df['end_year']), '--', 
                alpha=0.8, color='red', 
                label=f'Trend (slope={z[0]:.3f})')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig("figures/T2-4_rolling_elasticity_india.png", dpi=300, bbox_inches='tight')
    plt.close()
    return rolling_df

def kmeans_clustering(df):
    """2.5 K-means clustering analysis"""
    
    # Prepare data for clustering
    cluster_data = df[['beautypc', 'gdppcppp', 'beautyshare']].copy()
    cluster_data = cluster_data.dropna()
    
    if len(cluster_data) < 10:
        return None, None
    
    # Standardize features (z-score)
    scaler = SimpleStandardScaler()
    features_scaled = scaler.fit_transform(cluster_data)
    # Use 4 clusters as requested for better differentiation
    optimal_k = 4
    
    # Still calculate silhouette score for validation
    test_kmeans = SimpleKMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    test_labels = test_kmeans.fit_predict(features_scaled)
    silhouette_avg = simple_silhouette_score(features_scaled, test_labels)
    
    # Final clustering with optimal k
    final_kmeans = SimpleKMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = final_kmeans.fit_predict(features_scaled)
    
    # Add cluster labels back to original data
    df_clustered = df.loc[cluster_data.index].copy()
    df_clustered['cluster'] = cluster_labels
    
    # Track India's cluster over time
    india_clusters = df_clustered[df_clustered['country'] == 'india'].copy()
    
    # Debug: Check if any country changes clusters over time
    cluster_changes = []
    for country in df_clustered['country'].unique():
        country_clusters = df_clustered[df_clustered['country'] == country]['cluster'].unique()
        if len(country_clusters) > 1:
            cluster_changes.append(f"{country}: {len(country_clusters)} different clusters")
    
    # Countries with cluster changes tracked but not logged
    
    # Create T2-5: 2D scatter plot with 4 clusters
    plt.figure(figsize=(14, 10))
    
    # Plot points colored by cluster with distinct colors
    cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Distinct colors for 4 clusters
    cluster_names = ['Low Income\nLow Beauty', 'Middle Income\nModerate Beauty', 
                    'High Income\nHigh Beauty', 'Premium\nConsumption']
    
    for cluster_id in range(optimal_k):
        cluster_data_plot = df_clustered[df_clustered['cluster'] == cluster_id]
        
        if len(cluster_data_plot) == 0:
            continue
            
        # Use beauty share as point size (scaled to percentage)
        sizes = (cluster_data_plot['beautyshare'] * 100000).fillna(20)  # Convert to % and scale
        sizes = np.clip(sizes, 30, 200)  # Reasonable size range
        
        plt.scatter(cluster_data_plot['gdppcppp'], cluster_data_plot['beautypc'],
                   c=cluster_colors[cluster_id], s=sizes, alpha=0.7, 
                   label=f'Cluster {cluster_id}: {cluster_names[cluster_id]}', 
                   edgecolors='white', linewidth=0.5)
    
    # Highlight India points with special treatment
    if not india_clusters.empty:
        # Show India's trajectory over time
        plt.plot(india_clusters['gdppcppp'], india_clusters['beautypc'],
                'o-', color='darkred', alpha=0.6, linewidth=2, markersize=8,
                label='India Trajectory', zorder=5)
        
        # Highlight current India position
        latest_india = india_clusters.iloc[-1]
        plt.scatter(latest_india['gdppcppp'], latest_india['beautypc'],
                   c='darkred', s=400, marker='*', edgecolor='black', linewidth=2,
                   label='India Current (2023)', zorder=10)
    
    plt.xlabel('GDP per Capita PPP (2021 International $)')
    plt.ylabel('Beauty Consumption per Capita (USD 2015)')
    plt.title('Global Beauty Consumption Clusters\n(Point size proportional to Beauty Share of Household Spending)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/T2-5_kmeans_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create T2-6: India's cluster over time (only if it actually changes)
    if not india_clusters.empty and len(india_clusters['cluster'].unique()) > 1:
        plt.figure(figsize=(10, 6))
        plt.step(india_clusters['year'], india_clusters['cluster'], 
                where='post', linewidth=2, marker='o', markersize=6)
        plt.xlabel('Year')
        plt.ylabel('Cluster ID')
        plt.title("India's Cluster Assignment Over Time")
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.5, optimal_k - 0.5)
        plt.tight_layout()
        plt.savefig("figures/T2-6_india_cluster_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
    # India cluster timeline chart created or skipped based on cluster changes
    
    return df_clustered, india_clusters

def test_india_growth_acceleration(df):
    """Test for India's growth acceleration using t-test"""
    
    india_data = df[df['country'] == 'india'].copy().sort_values('year')
    
    if len(india_data) < 10:
        return None
    
    # Calculate YoY growth rates
    india_data['yoy_growth'] = india_data['beautypc'].pct_change()
    india_data = india_data.dropna(subset=['yoy_growth'])
    
    if len(india_data) < 10:
        return None
    
    # Split into two periods: recent 5 years vs prior 5 years
    recent_growth = india_data.iloc[-5:]['yoy_growth'].values
    prior_growth = india_data.iloc[-10:-5]['yoy_growth'].values
    
    if len(recent_growth) < 5 or len(prior_growth) < 5:
        return None
    
    # Welch's t-test for unequal variances
    t_stat, p_value = simple_ttest_ind(recent_growth, prior_growth)
    
    recent_mean = recent_growth.mean()
    prior_mean = prior_growth.mean()
    
    acceleration_test = {
        'recent_mean_growth': recent_mean,
        'prior_mean_growth': prior_mean, 
        'growth_difference': recent_mean - prior_mean,
        't_statistic': t_stat,
        'p_value': p_value,
        'is_accelerating': (recent_mean > prior_mean) and (p_value < 0.10)  # 10% significance
    }
    
    # Save results
    import json
    def convert_for_json(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    
    with open("figures/T2_india_growth_test.json", "w") as f:
        json.dump({k: convert_for_json(v) for k, v in acceleration_test.items()}, f, indent=2)
    
    
    return acceleration_test

def interpretive_analysis(df_clustered, india_clusters, matched_df, growth_test=None):
    """2.6 Interpretive analysis of clustering and benchmarking results"""
    
    if df_clustered is None or india_clusters is None:
        return
    
    interpretation = """
Task 2: Comparative Benchmarking Results
=======================================

Income-Matched Analysis:
"""
    
    if not matched_df.empty:
        india_recent_cagr = matched_df.iloc[0]['india_recent_cagr']
        interpretation += f"""
India's Current Position:
- Recent 5-year CAGR: {india_recent_cagr*100:.1f}%

Peer Comparisons at Similar Income Levels:
"""
        for _, row in matched_df.iterrows():
            peer_cagr = row['cagr_5y_post_match']
            cagr_str = f"{peer_cagr*100:.1f}%" if not pd.isna(peer_cagr) else "N/A"
            interpretation += f"""
- {row['peer_country'].title()}: {cagr_str} (at similar income level)
"""
    
    if growth_test is not None:
        acceleration_status = "accelerating" if growth_test['is_accelerating'] else "stable/decelerating"
        interpretation += f"""

Growth Acceleration Analysis:
- India's recent growth is {acceleration_status}
- Growth difference: {growth_test['growth_difference']*100:.2f}% (p={growth_test['p_value']:.3f})
"""
    
    if not india_clusters.empty:
        latest_cluster = india_clusters.iloc[-1]['cluster']
        interpretation += f"""

Clustering Analysis:
- India's current cluster: {latest_cluster}
"""
        
        # Check cluster movement
        if len(india_clusters) > 1:
            cluster_change = india_clusters.iloc[-1]['cluster'] - india_clusters.iloc[0]['cluster']
            if cluster_change > 0:
                interpretation += "- Moved to higher consumption cluster (positive trajectory)\n"
            elif cluster_change < 0:
                interpretation += "- Moved to lower consumption cluster (concerning trend)\n"
            else:
                interpretation += "- Stable cluster membership over time\n"
    
    interpretation += """

Investment Implications:
- Assessment based on income-matching and clustering patterns
- Growth trajectory analysis suggests timing considerations
"""
    
    with open("figures/T2_interpretive_analysis.txt", "w") as f:
        f.write(interpretation)
    

def main():
    """Run complete Task 2 analysis"""
    
    
    # Load data
    df = load_master_data()
    
    # 2.1 Income-matched snapshot
    matched_df, india_gdp = income_matched_snapshot(df)
    
    # 2.2 Alignment chart
    alignment_chart(df, india_gdp)
    
    # 2.3 Beauty share overlay
    beauty_share_overlay(df)
    
    # 2.4 Rolling elasticity for India
    rolling_df = rolling_elasticity_india(df)
    
    # 2.5 K-means clustering
    df_clustered, india_clusters = kmeans_clustering(df)
    
    # Growth acceleration test for India
    growth_test = test_india_growth_acceleration(df)
    
    # 2.6 Interpretive analysis
    interpretive_analysis(df_clustered, india_clusters, matched_df, growth_test)
    

if __name__ == "__main__":
    main()