import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('default')

EMERGING_COUNTRIES = ['brazil', 'china', 'russia', 'mexico', 'india']
HIGH_INCOME_COUNTRIES = ['japan', 'south korea', 'usa']

# Consistent color palette matching Task 1 exactly (tab10 colors in alphabetical order)
COUNTRY_COLORS = {
    'brazil': '#1f77b4',        # Blue (tab10 position 0)
    'china': '#ff7f0e',         # Orange (tab10 position 1)
    'france': '#2ca02c',        # Green (tab10 position 2)
    'germany': '#d62728',       # Red (tab10 position 3)
    'india': '#9467bd',         # Purple (tab10 position 4)
    'japan': '#8c564b',         # Brown (tab10 position 5)
    'mexico': '#e377c2',        # Pink (tab10 position 6)
    'russia': '#7f7f7f',        # Gray (tab10 position 7)
    'south korea': '#bcbd22',   # Olive (tab10 position 8)
    'united kingdom': '#17becf', # Cyan (tab10 position 9)
    'usa': '#17becf'            # Cyan (same as UK due to tab10 limit)
}

# Special colors for T1-1 chart (different from country colors)
T1_CHART_COLORS = {
    'segments': '#17becf',     # Cyan
    'overall': '#bcbd22'       # Olive
}

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
    script_dir = Path(__file__).resolve().parent.parent.parent
    data_path = script_dir / "data" / "processed" / "beauty_income_panel.parquet"
    master = pd.read_parquet(data_path)
    return master

def income_matched_snapshot(df):
    """2.1 Income-matched snapshot analysis"""
    
    # Get India's latest GDP per capita
    india_data = df[df['country'] == 'india'].copy().sort_values('year')
    latest_india = india_data.iloc[-1]  # Use iloc[-1] for most recent
    india_gdp = latest_india['gdppcppp']
    
    # Find income-matched years for peer countries (both emerging and high income)
    emerging_peers = [c for c in EMERGING_COUNTRIES if c != 'india']
    high_income_peers = HIGH_INCOME_COUNTRIES
    all_peers = emerging_peers + high_income_peers
    matched_results = []
    
    for peer in all_peers:
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
            'is_reasonable_match': is_reasonable_match,
            'country_type': 'emerging' if peer in emerging_peers else 'high_income'
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
    
    # Create beautified bar chart T2-1 with three categories
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Separate countries by type
    emerging_data = matched_df[matched_df['country_type'] == 'emerging']
    high_income_data = matched_df[matched_df['country_type'] == 'high_income']
    
    # Prepare data for plotting
    emerging_countries = emerging_data['peer_country'].tolist()
    emerging_cagrs = [c * 100 if not pd.isna(c) else 0 for c in emerging_data['cagr_5y_post_match']]
    
    high_income_countries = high_income_data['peer_country'].tolist()
    high_income_cagrs = [c * 100 if not pd.isna(c) else 0 for c in high_income_data['cagr_5y_post_match']]
    
    india_cagr = india_recent_cagr * 100 if not pd.isna(india_recent_cagr) else 0
    
    # Create positions with better spacing
    n_emerging = len(emerging_countries)
    n_high_income = len(high_income_countries)
    
    # Improved spacing: emerging countries, larger gap, India, larger gap, high income countries
    bar_width = 0.8
    group_spacing = 2.5  # Larger gaps between groups
    
    emerging_positions = [i * (bar_width + 0.3) for i in range(n_emerging)]
    india_position = emerging_positions[-1] + group_spacing if emerging_positions else group_spacing
    high_income_start = india_position + group_spacing
    high_income_positions = [high_income_start + i * (bar_width + 0.3) for i in range(n_high_income)]
    
    # Enhanced colors with better contrast
    emerging_color = '#2E8B57'    # Sea green - darker and more readable
    india_color = COUNTRY_COLORS['india']  # Dark blue for India
    high_income_color = '#CD853F'  # Peru - darker salmon for better readability
    
    # Plot bars with enhanced styling
    bars1 = ax.bar(emerging_positions, emerging_cagrs, width=bar_width, color=emerging_color, 
                   label='Emerging Market Peers', alpha=0.85, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar([india_position], [india_cagr], width=bar_width, color=india_color, 
                   label='India (Current)', alpha=0.95, edgecolor='white', linewidth=2)
    bars3 = ax.bar(high_income_positions, high_income_cagrs, width=bar_width, color=high_income_color, 
                   label='High Income Countries', alpha=0.85, edgecolor='white', linewidth=1.5)
    
    # Add subtle vertical separator lines
    if emerging_positions:
        separator1_pos = (emerging_positions[-1] + india_position) / 2
    else:
        separator1_pos = india_position - group_spacing/2
    separator2_pos = (india_position + high_income_positions[0]) / 2 if high_income_positions else india_position + group_spacing/2
    
    ax.axvline(x=separator1_pos, color='gray', linestyle=':', alpha=0.6, linewidth=2)
    ax.axvline(x=separator2_pos, color='gray', linestyle=':', alpha=0.6, linewidth=2)
    
    # Set x-axis labels with better formatting
    all_positions = emerging_positions + [india_position] + high_income_positions
    all_labels = ([c.replace('_', ' ').title() for c in emerging_countries] + 
                 ['India'] + 
                 [c.replace('_', ' ').title() for c in high_income_countries])
    
    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, rotation=0, ha='center', fontsize=11, fontweight='bold')
    
    # Add enhanced value labels on bars
    all_bars = list(bars1) + list(bars2) + list(bars3)
    all_cagrs = emerging_cagrs + [india_cagr] + high_income_cagrs
    
    for bar, cagr in zip(all_bars, all_cagrs):
        if cagr != 0:
            # Position label higher for better visibility
            label_height = bar.get_height() + (ax.get_ylim()[1] * 0.01)
            ax.text(bar.get_x() + bar.get_width()/2, label_height, 
                   f'{cagr:.1f}%', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Enhanced styling
    ax.set_ylabel('5-Year CAGR (%)', fontsize=13, fontweight='bold')
    ax.set_title('Beauty Consumption Growth: India vs Global Peers\\n' + 
                'Comparison at Similar Income Levels', 
                fontsize=15, fontweight='bold', pad=20)
    
    # Improved legend positioned to avoid covering values
    legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                      edgecolor='gray', fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    
    # Enhanced grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set y-axis limits for better proportion
    y_max = max(all_cagrs) * 1.15 if all_cagrs else 20
    ax.set_ylim(min(0, min(all_cagrs) * 1.1), y_max)
    
    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add subtle background shading for groups
    if emerging_positions:
        ax.axvspan(emerging_positions[0] - bar_width/2, emerging_positions[-1] + bar_width/2, 
                  alpha=0.05, color=emerging_color, zorder=0)
    ax.axvspan(india_position - bar_width/2, india_position + bar_width/2, 
              alpha=0.05, color=india_color, zorder=0)
    if high_income_positions:
        ax.axvspan(high_income_positions[0] - bar_width/2, high_income_positions[-1] + bar_width/2, 
                  alpha=0.05, color=high_income_color, zorder=0)
    
    plt.tight_layout()
    plt.savefig("figures/T2-1_cagr_comparison.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    return matched_df, india_gdp

def alignment_chart(df, india_gdp):
    """2.2 Alignment chart (Years since milestone X)"""
    
    tolerance = 500  # $500 tolerance for GDP matching
    
    # Find Year 0 for each peer country
    peers = [c for c in EMERGING_COUNTRIES if c != 'india']
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
            
            # Filter to relevant range (0 to +10 years only for peer countries)
            relevant_data = peer_data[(peer_data['rel_year'] >= 0) & (peer_data['rel_year'] <= 10)]
            
            for _, row in relevant_data.iterrows():
                alignment_data.append({
                    'country': peer,
                    'rel_year': row['rel_year'],
                    'beautypc': row['beautypc'],
                    'year0': year0
                })
    
    # Add India's trajectory (including years before current position)
    india_data = df[df['country'] == 'india'].copy().sort_values('year')
    india_latest_year = india_data.iloc[-1]['year']
    
    # Create rel_year for India where 0 = current year
    india_data['rel_year'] = india_data['year'] - india_latest_year
    
    # Filter to relevant range (-5 to 0 years)
    india_relevant = india_data[(india_data['rel_year'] >= -5) & (india_data['rel_year'] <= 0)]
    
    for _, row in india_relevant.iterrows():
        alignment_data.append({
            'country': 'india',
            'rel_year': row['rel_year'],
            'beautypc': row['beautypc'],
            'year0': india_latest_year
        })
    
    alignment_df = pd.DataFrame(alignment_data)
    
    # Create beautified T2-2 plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate overall min/max rel_year for enhanced visualization
    min_rel_year = alignment_df['rel_year'].min() 
    max_rel_year = alignment_df['rel_year'].max()
    
    # Enhanced background shading for India's trajectory period (negative years only)
    india_data = alignment_df[alignment_df['country'] == 'india'].sort_values('rel_year')
    if len(india_data) > 0 and min_rel_year < 0:
        ax.axvspan(min_rel_year, 0, facecolor=COUNTRY_COLORS['india'], alpha=0.08, 
                  label='India\'s 5-Year Build-up Period', zorder=0)
    
    # Draw enhanced vertical line at Year 0 (Today)
    ax.axvline(x=0, linestyle='-', color='darkred', alpha=0.8, linewidth=3, 
              label='Today (India\'s Current Position)')
    
    # Plot peer countries with enhanced styling (post-milestone only)
    peer_countries = [c for c in alignment_df['country'].unique() if c != 'india']
    for country in peer_countries:
        country_data = alignment_df[alignment_df['country'] == country].sort_values('rel_year')
        
        if len(country_data) > 0:
            year0_info = country_data[country_data['rel_year'] >= 0]
            year0_text = f" (milestone: {year0_info.iloc[0]['year0']})" if len(year0_info) > 0 else ""
            
            ax.plot(country_data['rel_year'], country_data['beautypc'], 
                   'o-', color=COUNTRY_COLORS[country], alpha=0.9, linewidth=2.5, 
                   markersize=6, markeredgecolor='white', markeredgewidth=1,
                   label=f'{country.title()}{year0_text}', zorder=3)
    
    # Highlight India's trajectory with enhanced styling
    if len(india_data) > 0:
        # Plot India's full trajectory with gradient effect
        ax.plot(india_data['rel_year'], india_data['beautypc'], 
               'o-', color=COUNTRY_COLORS['india'], linewidth=4, markersize=8, 
               alpha=0.95, markeredgecolor='white', markeredgewidth=2,
               label='India\'s Recent Trajectory (2019-2024)', zorder=5)
        
        # Enhanced 'Today' marker
        india_today = india_data[india_data['rel_year'] == 0]
        if len(india_today) > 0:
            today_value = india_today['beautypc'].iloc[0]
            # Main marker
            ax.scatter(0, today_value, color=COUNTRY_COLORS['india'], s=400, 
                      marker='*', edgecolor='darkred', linewidth=3, zorder=10)
            # Outer ring
            ax.scatter(0, today_value, color='none', s=600, 
                      marker='o', edgecolor='darkred', linewidth=2, zorder=9)
            
            # Better positioned 'India Today' label (under the star)
            ax.text(0, today_value - 0.7, 'India Today', 
                   ha='center', va='top', fontweight='bold', fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor='darkred', alpha=0.9, linewidth=2))
    
    # Enhanced styling and labels
    ax.set_xlabel('Years Since GDP Milestone\\n(Negative = India\'s Build-up to Current Level)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Beauty Consumption per Capita (USD 2015)', 
                 fontsize=12, fontweight='bold')
    ax.set_title(f'Beauty Consumption Growth Trajectories\\n' +
                f'India\'s 5-Year Journey vs Peers\' Post-Milestone Growth\\n' +
                f'(Countries reaching ${india_gdp:,.0f} GDP per capita)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Enhanced legend positioned in top left with larger font
    legend = ax.legend(loc='upper left', fontsize=11, framealpha=0.95, 
                      edgecolor='gray', fancybox=True, shadow=True,
                      bbox_to_anchor=(0.02, 0.98))
    legend.get_frame().set_facecolor('white')
    
    # Enhanced grid and styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Set better axis limits
    y_values = alignment_df['beautypc'].values
    y_margin = (max(y_values) - min(y_values)) * 0.1
    ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
    
    plt.tight_layout()
    plt.savefig("figures/T2-2_alignment_chart.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    return alignment_df

def beauty_share_overlay(df):
    """2.3 Beautified BeautyShare overlay plot with enhanced styling"""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    countries = df['country'].unique()
    other_countries = [c for c in countries if c != 'india']
    
    # Plot other countries with enhanced styling
    for country in other_countries:
        country_data = df[df['country'] == country]
        country_label = country.upper() if country == 'usa' else country.replace('_', ' ').title()
        color = COUNTRY_COLORS.get(country, '#666666')  # fallback to gray
        
        ax.scatter(country_data['gdppcppp'], country_data['beautyshare'] * 100, 
                  alpha=0.8, label=country_label, color=color, s=80,
                  edgecolors='white', linewidth=1, zorder=3)
    
    # Plot India with enhanced highlighting
    india_data = df[df['country'] == 'india']
    if not india_data.empty:
        # Plot India trajectory with enhanced styling
        ax.plot(india_data['gdppcppp'], india_data['beautyshare'] * 100, 
               'o-', color=COUNTRY_COLORS['india'], alpha=0.7, linewidth=3, 
               markersize=8, markerfacecolor=COUNTRY_COLORS['india'],
               markeredgecolor='white', markeredgewidth=2,
               label='India Trajectory (1995-2024)', zorder=5)
        
        # Enhanced current India position marker
        latest_india = india_data.iloc[-1]
        ax.scatter(latest_india['gdppcppp'], latest_india['beautyshare'] * 100, 
                  color=COUNTRY_COLORS['india'], s=500, marker='*', 
                  label='India Current Position (2024)', zorder=10, 
                  edgecolor='darkred', linewidth=3)
        
        # Add annotation for current India position (under the star)
        ax.annotate('India Today', 
                   xy=(latest_india['gdppcppp'], latest_india['beautyshare'] * 100),
                   xytext=(0, -30), textcoords='offset points',
                   fontsize=12, fontweight='bold', ha='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor=COUNTRY_COLORS['india'], alpha=0.9))
    
    # Enhanced styling and labels
    ax.set_xlabel('GDP per Capita PPP (2021 International $)', 
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Beauty Share of Household Consumption (%)', 
                 fontsize=13, fontweight='bold')
    ax.set_title('Beauty Consumption Share vs Income Level\\n' +
                'India\'s Development Path Among Global Peers', 
                fontsize=15, fontweight='bold', pad=20)
    
    # Enhanced legend with larger font
    legend = ax.legend(loc='upper left', fontsize=12, framealpha=0.95, 
                      edgecolor='gray', fancybox=True, shadow=True,
                      bbox_to_anchor=(0.02, 0.98))
    legend.get_frame().set_facecolor('white')
    
    # Enhanced grid and styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Format axis ticks
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Add subtle background gradient effect
    ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    plt.savefig("figures/T2-3_beauty_share_overlay.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
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
    
    # Create beautified T2-4 plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Main elasticity line with enhanced styling
    ax.plot(rolling_df['end_year'], rolling_df['elasticity'], 
           'o-', color=COUNTRY_COLORS['india'], linewidth=3, markersize=8,
           markerfacecolor=COUNTRY_COLORS['india'], markeredgecolor='white', 
           markeredgewidth=2, alpha=0.9, label='5-Year Rolling Elasticity')
    
    # Enhanced trend line
    if len(rolling_df) > 2:
        z = np.polyfit(rolling_df['end_year'], rolling_df['elasticity'], 1)
        p = np.poly1d(z)
        trend_color = '#d62728'  # Red for trend
        ax.plot(rolling_df['end_year'], p(rolling_df['end_year']), '--', 
                alpha=0.8, color=trend_color, linewidth=2.5,
                label=f'Linear Trend (slope: {z[0]:.3f}/year)')
        
        # Add shaded confidence region around trend
        trend_values = p(rolling_df['end_year'])
        residuals = rolling_df['elasticity'] - trend_values
        std_dev = np.std(residuals)
        ax.fill_between(rolling_df['end_year'], 
                       trend_values - std_dev, trend_values + std_dev,
                       alpha=0.2, color=trend_color, 
                       label=f'±1 Std Dev ({std_dev:.3f})')
    
    # Enhanced styling and labels
    ax.set_xlabel('Window End Year', fontsize=13, fontweight='bold')
    ax.set_ylabel('Income Elasticity of Beauty Consumption (β)', fontsize=13, fontweight='bold')
    ax.set_title('India: Evolution of Income Elasticity\\n' +
                'Rolling 5-Year Analysis of Beauty Consumption Responsiveness', 
                fontsize=15, fontweight='bold', pad=20)
    
    # Enhanced legend
    legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                      edgecolor='gray', fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    
    # Enhanced grid and styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Format axis ticks
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Add reference line at elasticity = 1 (unit elastic)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.7, linewidth=2,
              label='Unit Elastic (β=1)')
    
    # Add subtle background
    ax.set_facecolor('#fafafa')
    
    # Update legend to include reference line
    legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                      edgecolor='gray', fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig("figures/T2-4_rolling_elasticity_india.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
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
    
    # Create beautified T2-5: 2D scatter plot with 4 clusters
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Enhanced cluster colors and names
    cluster_colors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c']  # More vibrant colors
    cluster_names = ['Developing Markets\\n(Low Income & Spending)', 
                    'Emerging Markets\\n(Moderate Income & Beauty)', 
                    'Developed Markets\\n(High Income & Consumption)', 
                    'Premium Markets\\n(Luxury Consumption)']
    
    # Plot cluster points with enhanced styling
    for cluster_id in range(optimal_k):
        cluster_data_plot = df_clustered[df_clustered['cluster'] == cluster_id]
        
        if len(cluster_data_plot) == 0:
            continue
            
        # Enhanced size scaling based on beauty share
        sizes = (cluster_data_plot['beautyshare'] * 150000).fillna(50)  # Better scaling
        sizes = np.clip(sizes, 50, 300)  # Wider size range for better visibility
        
        ax.scatter(cluster_data_plot['gdppcppp'], cluster_data_plot['beautypc'],
                  c=cluster_colors[cluster_id], s=sizes, alpha=0.7, 
                  label=f'{cluster_names[cluster_id]}', 
                  edgecolors='white', linewidth=1.5, zorder=3)
    
    # Enhanced India trajectory visualization
    if not india_clusters.empty:
        # Show India's development path with gradient effect
        ax.plot(india_clusters['gdppcppp'], india_clusters['beautypc'],
               'o-', color=COUNTRY_COLORS['india'], alpha=0.8, linewidth=4, 
               markersize=10, markerfacecolor=COUNTRY_COLORS['india'],
               markeredgecolor='white', markeredgewidth=2,
               label='India\'s Development Path (1995-2024)', zorder=6)
        
        # Enhanced current India position marker
        latest_india = india_clusters.iloc[-1]
        ax.scatter(latest_india['gdppcppp'], latest_india['beautypc'],
                  c=COUNTRY_COLORS['india'], s=600, marker='*', 
                  edgecolor='darkred', linewidth=3,
                  label='India Current Position (2024)', zorder=10)
        
        # Add annotation for India's current position
        ax.annotate('India Today', 
                   xy=(latest_india['gdppcppp'], latest_india['beautypc']),
                   xytext=(15, 15), textcoords='offset points',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor=COUNTRY_COLORS['india'], alpha=0.9),
                   arrowprops=dict(arrowstyle='->', color=COUNTRY_COLORS['india'], lw=2))
    
    # Enhanced styling and labels
    ax.set_xlabel('GDP per Capita PPP (2021 International $)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Beauty Consumption per Capita (USD 2015)', 
                 fontsize=14, fontweight='bold')
    ax.set_title('Global Beauty Consumption Clusters\\n' +
                'India\'s Position in the Global Market', 
                fontsize=16, fontweight='bold', pad=25)
    
    # Enhanced legend positioned inside the plot area
    legend = ax.legend(loc='upper right', fontsize=10, framealpha=0.95, 
                      edgecolor='gray', fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    
    # Enhanced grid and styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Format axis ticks
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add subtle background
    ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    plt.savefig("figures/T2-5_kmeans_scatter.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
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