import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import scipy.stats
import statsmodels.api as sm

# Load master dataset
def load_master_data():
    """Load the master dataset with all categories"""
    script_dir = Path(__file__).resolve().parent.parent.parent
    data_path = script_dir / "data" / "processed" / "beauty_income_panel.parquet"
    df = pd.read_parquet(data_path)

    # Ensure we have the required columns
    required_cols = ['country', 'year', 'gdppcppp', 'BeautyPC', 'SkincarePC', 'MenPC', 'WomenPC']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}. Using available data.")

    return df

def log_log_elasticity(x, y, country_name, series_name):
    """Calculate log-log elasticity with HC3 robust standard errors"""
    # Remove zero and negative values from x, allow zero in y
    mask = (x > 0) & ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 5:  # Need minimum observations
        return {'beta': np.nan, 'se': np.nan, 'pvalue': np.nan, 'n_obs': mask.sum(),
                'intercept': np.nan, 'intercept_se': np.nan, 'intercept_ci': (np.nan, np.nan)}
    
    x_clean, y_clean = x[mask], y[mask]
    
    # Handle y values: add small constant for log transformation
    y_log_safe = np.log(y_clean + 1)  # log(y+1) transformation
    log_x = np.log(x_clean)
    
    # Simple OLS regression
    X = np.column_stack([np.ones(len(log_x)), log_x])
    try:
        model = sm.OLS(y_log_safe, sm.add_constant(log_x))
        results = model.fit(cov_type='HC3')

        return {
            'beta': results.params[1],
            'se': results.bse[1], 
            'pvalue': results.pvalues[1],
            'n_obs': results.nobs,
            'intercept': results.params[0],
            'intercept_se': results.bse[0],
            'intercept_ci': results.conf_int()[0]
        }
    except Exception:
        return {'beta': np.nan, 'se': np.nan, 'pvalue': np.nan, 'n_obs': mask.sum(),
                'intercept': np.nan, 'intercept_se': np.nan, 'intercept_ci': (np.nan, np.nan)}

def piecewise_regression(x, y, country_name, series_name):
    """
    Perform piecewise linear regression with grid search for breakpoint
    Returns optimal breakpoint, slopes, and F-test results
    """
    # Clean data - only require x > 0, allow y = 0
    mask = ~(np.isnan(x) | np.isnan(y)) & (x > 0)
    if mask.sum() < 8:  # Need minimum observations for two segments
        return None
    
    x_clean, y_clean = x[mask], y[mask]
    n = len(x_clean)
    
    # Minimum observation guard
    if len(x_clean) < 10:
        return None
    
    # Calculate R² for initial screening
    try:
        X_single = sm.add_constant(x_clean)
        model_single = sm.OLS(y_clean, X_single).fit()
        r_squared = model_single.rsquared
        if r_squared < 0.3:
            return None
    except Exception:
        return None
    
    # Define grid search range (10th to 90th percentile, $1K steps)
    x_min, x_max = np.percentile(x_clean, [10, 90])
    breakpoint_grid = np.arange(x_min, x_max, 1000)
    
    if len(breakpoint_grid) < 3:
        return None
    
    best_sse = np.inf
    best_breakpoint = None
    best_model1_params = None
    best_model2_params = None
    
    # Grid search for optimal breakpoint
    for c in breakpoint_grid:
        # Split data at breakpoint
        mask1 = x_clean <= c
        mask2 = x_clean > c
        
        # Need minimum observations in each segment
        if mask1.sum() < 10 or mask2.sum() < 10 or mask1.sum() < 0.2 * n or mask2.sum() < 0.2 * n:
            continue
        
        try:
            # Fit segment 1: y = a1 + b1*x
            X1 = sm.add_constant(x_clean[mask1])
            model1 = sm.OLS(y_clean[mask1], X1).fit()
            sse1 = model1.ssr
            
            # Fit segment 2: y = a2 + b2*x  
            X2 = sm.add_constant(x_clean[mask2])
            model2 = sm.OLS(y_clean[mask2], X2).fit()
            sse2 = model2.ssr
            
            total_sse = sse1 + sse2
            
            if total_sse < best_sse:
                best_sse = total_sse
                best_breakpoint = c
                best_model1_params = model1.params
                best_model2_params = model2.params
                
        except Exception:
            continue
    
    if best_breakpoint is None:
        return None
    
    # Calculate F-test for best breakpoint
    try:
        # Single line regression (null model)
        X_single = sm.add_constant(x_clean)
        model_single = sm.OLS(y_clean, X_single).fit()
        sse_single = model_single.ssr
        
        # F-test: F = ((SSE_single - SSE_two)/1) / (SSE_two/(N-4))
        f_stat = ((sse_single - best_sse) / 1) / (best_sse / (n - 4))
        p_value = scipy.stats.f.sf(f_stat, 1, n - 4)
        
        return {
            'country': country_name,
            'series': series_name,
            'breakpoint': best_breakpoint,
            'b1': best_model1_params[1],
            'b2': best_model2_params[1],
            'intercept1': best_model1_params[0],
            'intercept2': best_model2_params[0],
            'delta_b': best_model2_params[1] - best_model1_params[1],
            'f_stat': f_stat,
            'p_value': p_value,
            'n_obs': n,
            'sse_single': sse_single,
            'sse_two': best_sse
        }
    except Exception as e:
        print(f"Error in piecewise_regression for {country_name}-{series_name}: {e}")
        return None

def bootstrap_confidence_interval(x, y, country_name, series_name, n_bootstrap=1000):
    """
    Bootstrap confidence interval for breakpoint
    Resample country-year rows with replacement
    """
    # Clean data first - only require x > 0, allow y = 0
    mask = ~(np.isnan(x) | np.isnan(y)) & (x > 0)
    if mask.sum() < 8:
        return None
    
    x_clean, y_clean = x[mask], y[mask]
    n = len(x_clean)
    
    # Only bootstrap if sufficient observations
    if n < 20:
        return None
    
    # Use fewer bootstrap samples for smaller datasets
    if n < 30:
        n_bootstrap = 500
    
    breakpoints = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        x_boot, y_boot = x_clean[indices], y_clean[indices]
        
        # Run piecewise regression on bootstrap sample
        result = piecewise_regression(x_boot, y_boot, country_name, series_name)
        if result and not np.isnan(result['breakpoint']):
            breakpoints.append(result['breakpoint'])
    
    # Need at least 200 valid bootstrap samples for reliable CI
    if len(breakpoints) < 200:  
        return None
    
    # Calculate 95% confidence interval (2.5th and 97.5th percentiles)
    ci_lower = np.percentile(breakpoints, 2.5)
    ci_upper = np.percentile(breakpoints, 97.5)
    
    return {'ci_lower': ci_lower, 'ci_upper': ci_upper, 'bootstrap_samples': len(breakpoints)}

def run_hypothesis_testing():
    """
    Main function to run hypothesis testing across all countries and series
    """
    print("Loading master dataset...")
    df = load_master_data()
    
    # Define series to analyze
    series_mapping = {
        'BeautyPC': 'Beauty',
        'SkincarePC': 'Skincare', 
        'MenPC': 'Mens Wear',
        'WomenPC': 'Womens Wear'
    }
    
    # Filter to countries with sufficient data
    countries = df['country'].unique()
    print(f"Analyzing {len(countries)} countries: {sorted(countries)}")
    
    results = []
    elasticity_results = []
    
    print("\nRunning piecewise regression analysis...")
    
    for country in countries:
        country_data = df[df['country'] == country].copy()
        
        if len(country_data) < 10:  # Need minimum observations
            continue
            
        print(f"  Processing {country}...")
        
        for series_col, series_name in series_mapping.items():
            if series_col not in country_data.columns:
                continue
                
            x = country_data['gdppcppp'].values
            y = country_data[series_col].values
            
            # Log-log elasticity
            elasticity = log_log_elasticity(x, y, country, series_name)
            elasticity['country'] = country
            elasticity['series'] = series_name
            elasticity_results.append(elasticity)
            
            # Piecewise regression
            pw_result = piecewise_regression(x, y, country, series_name)
            if pw_result:
                # Bootstrap confidence interval
                ci_result = bootstrap_confidence_interval(x, y, country, series_name)
                if ci_result:
                    pw_result.update(ci_result)
                else:
                    pw_result.update({'ci_lower': np.nan, 'ci_upper': np.nan, 'bootstrap_samples': 0})
                
                results.append(pw_result)
    
    # Convert to DataFrames for easy analysis
    results_df = pd.DataFrame(results)
    elasticity_df = pd.DataFrame(elasticity_results)
    
    # Save results
    script_dir = Path(__file__).resolve().parent.parent.parent
    figures_dir = script_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    results_df.to_csv(figures_dir / 'T3_piecewise_results.csv', index=False)
    elasticity_df.to_csv(figures_dir / 'T3_elasticity_results.csv', index=False)

    print(f"\nPiecewise regression completed:")
    print(f"  - {len(results_df)} successful analyses")
    print(f"  - Results saved to {figures_dir / 'T3_piecewise_results.csv'}")
    print(f"  - Elasticity results saved to {figures_dir / 'T3_elasticity_results.csv'}")

    return results_df, elasticity_df, df

def create_visualizations(results_df, elasticity_df, master_df):
    """Create all required visualizations"""
    
    # Set style
    plt.style.use('default')
    
    # HT-1: Small multiples of piecewise fits
    create_small_multiples_chart(results_df, master_df)
    
    # HT-2: Slope change bar chart  
    create_slope_change_chart(results_df)
    
    # HT-3: Breakpoint histogram
    create_breakpoint_histogram(results_df)
    
    # HT-4: F-test heatmap
    create_ftest_heatmap(results_df)
    
    # HT-5: India vs emerging peers overlay
    create_india_overlay_chart(results_df, master_df)

def create_small_multiples_chart(results_df, master_df):
    """HT-1: Small multiples showing piecewise fits by country/category"""
    
    series_list = results_df['series'].unique()
    countries = results_df['country'].unique()
    
    fig, axes = plt.subplots(len(series_list), len(countries), 
                            figsize=(12, 8), dpi=200)
    
    if len(countries) == 1:
        axes = axes.reshape(-1, 1)
    if len(series_list) == 1:
        axes = axes.reshape(1, -1)
    
    for i, series in enumerate(series_list):
        for j, country in enumerate(countries):
            ax = axes[i, j]
            
            # Get country data
            country_data = master_df[master_df['country'] == country]
            series_col = {'Beauty': 'BeautyPC', 'Skincare': 'SkincarePC', 
                         'Mens Wear': 'MenPC', 'Womens Wear': 'WomenPC'}[series]
            
            if series_col not in country_data.columns:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{country}\n{series}')
                continue
            
            x = country_data['gdppcppp']
            y = country_data[series_col]
            
            # Remove NaN values - allow y = 0
            mask = ~(np.isnan(x) | np.isnan(y)) & (x > 0)
            if mask.sum() < 3:
                ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{country}\n{series}')
                continue
                
            x_clean, y_clean = x[mask], y[mask]
            
            # Plot scatter
            ax.scatter(x_clean, y_clean, alpha=0.6, s=30)
            
            # Get piecewise result for this country-series
            pw_result = results_df[(results_df['country'] == country) & 
                                  (results_df['series'] == series)]
            
            if not pw_result.empty:
                breakpoint = pw_result.iloc[0]['breakpoint']
                b1 = pw_result.iloc[0]['b1']
                b2 = pw_result.iloc[0]['b2']
                intercept1 = pw_result.iloc[0]['intercept1']
                intercept2 = pw_result.iloc[0]['intercept2']
                
                # Plot piecewise lines
                x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
                
                # Segment 1
                x1_range = x_range[x_range <= breakpoint]
                if len(x1_range) > 0:
                    # Need intercept for line plotting - estimate from data
                    mask1 = x_clean <= breakpoint
                    if mask1.sum() > 0:
                        y1_range = intercept1 + b1 * x1_range
                        ax.plot(x1_range, y1_range, 'r-', linewidth=2, alpha=0.8)
                
                # Segment 2
                x2_range = x_range[x_range > breakpoint]
                if len(x2_range) > 0:
                    mask2 = x_clean > breakpoint
                    if mask2.sum() > 0:
                        y2_range = intercept2 + b2 * x2_range
                        ax.plot(x2_range, y2_range, 'b-', linewidth=2, alpha=0.8)
                
                # Vertical line at breakpoint
                ax.axvline(breakpoint, color='green', linestyle='--', alpha=0.7)
                
                # Add p-value annotation
                p_val = pw_result.iloc[0]['p_value']
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                ax.text(0.05, 0.95, f'p={p_val:.3f}{significance}', 
                       transform=ax.transAxes, fontsize=8, verticalalignment='top')
            
            ax.set_title(f'{country}\n{series}', fontsize=10)
            ax.set_xlabel('GDP per capita PPP', fontsize=8)
            ax.set_ylabel(f'{series} per capita', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.set_xscale('log')
            ax.set_yscale('log')
    
    plt.tight_layout()
    script_dir = Path(__file__).resolve().parent.parent.parent
    figures_dir = script_dir / "figures"
    plt.savefig(figures_dir / 'HT-1_piecewise_fits_small_multiples.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created HT-1: Small multiples piecewise fits")

def create_slope_change_chart(results_df):
    """HT-2: Bar chart of slope changes (Δb) across countries"""
    
    if results_df.empty:
        print("  Skipping HT-2: No piecewise results available")
        return
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Prepare data for plotting
    plot_data = results_df[['country', 'series', 'delta_b', 'p_value']].copy()
    plot_data['significant'] = plot_data['p_value'] < 0.05
    
    # Create grouped bar chart
    series_list = plot_data['series'].unique()
    countries = plot_data['country'].unique()
    
    x_pos = np.arange(len(countries))
    bar_width = 0.8 / len(series_list)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(series_list)))
    
    for i, series in enumerate(series_list):
        series_data = plot_data[plot_data['series'] == series]

        values = []
        significance = []
        for country in countries:
            country_data = series_data[series_data['country'] == country]
            if not country_data.empty:
                values.append(country_data.iloc[0]['delta_b'])
                significance.append(country_data.iloc[0]['significant'])
            else:
                values.append(0)
                significance.append(False)

        # Plot bars with different alpha for significance
        bars = ax.bar(x_pos + i * bar_width, values, bar_width, 
                     label=series, color=colors[i], alpha=0.8)

        # Add significance markers (asterisks) just above or below the bar, inside the chart area
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        offset = 0.01 * y_range  # 1% of y-range as offset
        for j, (bar, sig) in enumerate(zip(bars, significance)):
            if sig and values[j] != 0:
                height = bar.get_height()
                # Place asterisk just inside the chart area
                if height > 0:
                    y = min(height + offset, ylim[1] - 0.02 * y_range)
                    va = 'bottom'
                else:
                    y = max(height - offset, ylim[0] + 0.02 * y_range)
                    va = 'top'
                ax.text(bar.get_x() + bar.get_width()/2., y, '*', ha='center', va=va, fontweight='bold', fontsize=14, clip_on=True)
    
    ax.set_xlabel('Country', fontsize=12)
    ax.set_ylabel('Slope Change (Δb)', fontsize=12)
    ax.set_title('Slope Change at Income Threshold by Country and Category\n(* indicates p < 0.05)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos + bar_width * (len(series_list) - 1) / 2)
    ax.set_xticklabels([c.title() for c in countries], rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    script_dir = Path(__file__).resolve().parent.parent.parent
    figures_dir = script_dir / "figures"
    plt.savefig(figures_dir / 'HT-2_slope_change_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created HT-2: Slope change bar chart")

def create_breakpoint_histogram(results_df):
    """HT-3: Histogram of breakpoints by category"""
    
    if results_df.empty:
        print("  Skipping HT-3: No piecewise results available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    series_list = results_df['series'].unique()
    
    for i, series in enumerate(series_list):
        if i >= 4:
            break
            
        ax = axes[i]
        series_data = results_df[results_df['series'] == series]
        
        if not series_data.empty:
            breakpoints = series_data['breakpoint'].dropna()
            
            if len(breakpoints) > 0:
                ax.hist(breakpoints, bins=15, alpha=0.7, edgecolor='black')
                ax.axvline(breakpoints.median(), color='red', linestyle='--', 
                          label=f'Median: ${breakpoints.median():,.0f}')
                ax.set_xlabel('GDP per capita PPP (USD)', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.set_title(f'{series} Breakpoints\n({len(breakpoints)} countries)', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No significant\nbreakpoints found', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{series} Breakpoints', fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{series} Breakpoints', fontsize=12)
    
    # Hide unused subplots
    for i in range(len(series_list), 4):
        axes[i].set_visible(False)
    
    plt.suptitle('Distribution of Income Thresholds by Category\n(Breakpoints where consumption accelerates)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    script_dir = Path(__file__).resolve().parent.parent.parent
    figures_dir = script_dir / "figures"
    plt.savefig(figures_dir / 'HT-3_breakpoint_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created HT-3: Breakpoint histogram")

def create_ftest_heatmap(results_df):
    """HT-4: Heatmap of F-test p-values"""
    
    if results_df.empty:
        print("  Skipping HT-4: No piecewise results available")
        return
    
    # Create pivot table for heatmap
    heatmap_data = results_df.pivot(index='country', columns='series', values='p_value')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap with custom colormap (lower p-values = more significant = darker)
    mask = heatmap_data.isnull()
    # Replace seaborn heatmap with matplotlib
    import matplotlib.colors as colors
    masked_data = np.ma.masked_where(heatmap_data.isnull(), heatmap_data)
    im = ax.imshow(masked_data, cmap='RdYlBu_r', aspect='auto', 
                   norm=colors.TwoSlopeNorm(vcenter=0.05, vmin=0, vmax=0.2))
    
    # Add text annotations
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            if not heatmap_data.iloc[i, j] is np.nan:
                ax.text(j, i, f'{heatmap_data.iloc[i, j]:.3f}', 
                       ha='center', va='center', fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('p-value')
    
    # Set tick labels
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticklabels(heatmap_data.index)
    
    ax.set_title('Statistical Significance of Income Thresholds\n(F-test p-values, darker = more significant)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Country', fontsize=12)
    
    # Add significance threshold line in colorbar
    cbar.ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2)
    cbar.ax.text(0.5, 0.05, 'p=0.05', ha='center', va='bottom', 
                transform=cbar.ax.transData, color='red', fontweight='bold')
    
    plt.tight_layout()
    script_dir = Path(__file__).resolve().parent.parent.parent
    figures_dir = script_dir / "figures"
    plt.savefig(figures_dir / 'HT-4_ftest_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created HT-4: F-test significance heatmap")

def create_india_overlay_chart(results_df, master_df):
    """HT-5: India vs emerging peers overlay with current position"""
    
    # Define emerging markets (same as Task 2)
    emerging_countries = ['india', 'south korea']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot beauty consumption for emerging markets
    for country in emerging_countries:
        country_data = master_df[master_df['country'] == country]
        
        if 'BeautyPC' not in country_data.columns or country_data.empty:
            continue
            
        x = country_data['gdppcppp']
        y = country_data['BeautyPC']
        
        # Remove NaN values - allow y = 0
        mask = ~(np.isnan(x) | np.isnan(y)) & (x > 0)
        if mask.sum() < 3:
            continue
            
        x_clean, y_clean = x[mask], y[mask]
        
        # Plot country trajectory
        if country == 'india':
            ax.plot(x_clean, y_clean, 'o-', linewidth=3, markersize=6, 
                   label=f'{country.title()}', alpha=0.8)
            
            # Mark India's current position (latest year)
            current_x = x_clean.iloc[-1]
            current_y = y_clean.iloc[-1]
            ax.scatter(current_x, current_y, s=200, color='red', marker='*', 
                      edgecolor='black', linewidth=2, zorder=10, label='India Today')
            
        else:
            ax.plot(x_clean, y_clean, 'o-', linewidth=2, markersize=4, 
                   label=f'{country.title()}', alpha=0.7)
        
        # Add peer breakpoint lines
        peer_bp = results_df.query("country==@country & series=='Beauty'")['breakpoint']
        if not peer_bp.empty and not np.isnan(peer_bp.iloc[0]):
            if country == 'india':
                ax.axvline(peer_bp.iloc[0], color='red', linestyle='--', alpha=0.7, linewidth=2,
                          label=f'India Threshold: ${peer_bp.iloc[0]:,.0f}')
            else:
                ax.axvline(peer_bp.iloc[0], color='grey', linestyle=':', alpha=0.4, linewidth=1)
    
    ax.set_xlabel('GDP per capita PPP (2021 International $)', fontsize=12)
    ax.set_ylabel('Beauty Consumption per Capita (2015 USD)', fontsize=12)
    ax.set_title('India vs Emerging Market Peers: Beauty Consumption Trajectory\nIndia\'s Position Relative to Income Thresholds', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add interpretation helper
    ax.text(0.02, 0.95,
            '★ = India 2024\nKorea breached its\ninflection in 2004',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='grey', alpha=0.6))
    
    plt.tight_layout()
    script_dir = Path(__file__).resolve().parent.parent.parent
    figures_dir = script_dir / "figures"
    plt.savefig(figures_dir / 'HT-5_india_vs_peers_overlay.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created HT-5: India vs emerging peers overlay")

def generate_summary_report(results_df, elasticity_df):
    """Generate summary report of findings"""
    
    print("\n" + "="*60)
    print("TASK 3: HYPOTHESIS TESTING SUMMARY REPORT")
    print("="*60)
    
    if results_df.empty:
        print("No piecewise regression results available.")
        return
    
    # Overall statistics
    total_analyses = len(results_df)
    significant_results = len(results_df[results_df['p_value'] < 0.05])
    
    print(f"Total piecewise analyses: {total_analyses}")
    print(f"Statistically significant thresholds (p<0.05): {significant_results} ({significant_results/total_analyses*100:.1f}%)")
    
    # By category analysis
    print(f"\nBREAKPOINT ANALYSIS BY CATEGORY:")
    print("-" * 40)
    
    for series in results_df['series'].unique():
        series_data = results_df[results_df['series'] == series]
        sig_data = series_data[series_data['p_value'] < 0.05]
        
        print(f"\n{series}:")
        print(f"  Countries analyzed: {len(series_data)}")
        print(f"  Significant thresholds: {len(sig_data)}")
        
        if len(sig_data) > 0:
            median_threshold = sig_data['breakpoint'].median()
            avg_slope_change = sig_data['delta_b'].mean()
            print(f"  Median threshold: ${median_threshold:,.0f}")
            print(f"  Average slope acceleration: {avg_slope_change:.4f}")
    
    # India-specific analysis
    print(f"\nINDIA-SPECIFIC FINDINGS:")
    print("-" * 25)
    
    india_results = results_df[results_df['country'] == 'india']
    if not india_results.empty:
        for _, row in india_results.iterrows():
            significance = "SIGNIFICANT" if row['p_value'] < 0.05 else "not significant"
            print(f"{row['series']}: Threshold ${row['breakpoint']:,.0f} ({significance}, p={row['p_value']:.3f})")
            print(f"  Slope change: {row['delta_b']:.4f}")
    else:
        print("No India results available.")
    
    # Cross-category comparison
    print(f"\nCROSS-CATEGORY INSIGHTS:")
    print("-" * 26)
    
    if len(results_df) > 0:
        # Which category breaks out first (lowest threshold)?
        sig_results = results_df[results_df['p_value'] < 0.05]
        if not sig_results.empty:
            by_category = sig_results.groupby('series')['breakpoint'].median().sort_values()
            print("Categories by typical threshold (lowest first):")
            for series, threshold in by_category.items():
                print(f"  {series}: ${threshold:,.0f}")
        
        # Which category shows strongest acceleration?
        if not sig_results.empty:
            by_acceleration = sig_results.groupby('series')['delta_b'].mean().sort_values(ascending=False)
            print("\nCategories by acceleration strength:")
            for series, accel in by_acceleration.items():
                print(f"  {series}: {accel:.4f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    print("TASK 3: HYPOTHESIS TESTING")
    print("Testing: 'Beauty consumption accelerates disproportionately once GDP per capita crosses a threshold'")
    print("="*80)
    
    # Run main analysis
    results_df, elasticity_df, master_df = run_hypothesis_testing()
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results_df, elasticity_df, master_df)
    
    # Generate summary report
    generate_summary_report(results_df, elasticity_df)
    
    print(f"\nTask 3 completed successfully!")
    print(f"Generated files:")
    print(f"  - figures/T3_piecewise_results.csv")
    print(f"  - figures/T3_elasticity_results.csv") 
    print(f"  - figures/HT-1_piecewise_fits_small_multiples.png")
    print(f"  - figures/HT-2_slope_change_bar_chart.png")
    print(f"  - figures/HT-3_breakpoint_histogram.png")
    print(f"  - figures/HT-4_ftest_heatmap.png")
    print(f"  - figures/HT-5_india_vs_peers_overlay.png")