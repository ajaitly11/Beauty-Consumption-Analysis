import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set larger font sizes for better readability
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16
import scipy.stats
import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter, ScalarFormatter

# Consistent country color palette matching Tasks 1 and 2
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
    'usa': '#00008B'            # DarkBlue (unique color for USA)
}

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
    Perform piecewise linear regression in log-space with grid search for breakpoint
    Returns optimal breakpoint, slopes, and F-test results
    """
    # Clean data - only require x > 0 and y > 0 for log transformation
    mask = ~(np.isnan(x) | np.isnan(y)) & (x > 0) & (y > 0)
    if mask.sum() < 8:  # Need minimum observations for two segments
        return None
    
    x_clean, y_clean = x[mask], y[mask]
    n = len(x_clean)
    
    # Minimum observation guard
    if len(x_clean) < 10:
        return None
    
    # Transform to log space for fitting
    log_x = np.log(x_clean)
    log_y = np.log(y_clean)
    
    # Calculate R² for initial screening in log space
    try:
        X_single = sm.add_constant(log_x)
        model_single = sm.OLS(log_y, X_single).fit()
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
            # Fit segment 1 in log space: log(y) = a1 + b1*log(x)
            log_x1 = log_x[mask1]
            log_y1 = log_y[mask1]
            X1 = sm.add_constant(log_x1)
            model1 = sm.OLS(log_y1, X1).fit()
            sse1 = model1.ssr
            
            # Fit segment 2 in log space: log(y) = a2 + b2*log(x)
            log_x2 = log_x[mask2]
            log_y2 = log_y[mask2]
            X2 = sm.add_constant(log_x2)
            model2 = sm.OLS(log_y2, X2).fit()
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
    
    # Calculate F-test for best breakpoint (in log space)
    try:
        # Single line regression in log space (null model)
        X_single = sm.add_constant(log_x)
        model_single = sm.OLS(log_y, X_single).fit()
        sse_single = model_single.ssr
        
        # F-test: F = ((SSE_single - SSE_two)/1) / (SSE_two/(N-4))
        f_stat = ((sse_single - best_sse) / 1) / (best_sse / (n - 4))
        p_value = scipy.stats.f.sf(f_stat, 1, n - 4)
        
        return {
            'country': country_name,
            'series': series_name,
            'breakpoint': best_breakpoint,
            'b1': best_model1_params[1],  # log-space slope
            'b2': best_model2_params[1],  # log-space slope
            'intercept1': best_model1_params[0],  # log-space intercept
            'intercept2': best_model2_params[0],  # log-space intercept
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
    
    # Comment out other charts for faster testing
    # HT-2: Slope change bar chart  
    create_slope_change_chart(results_df)
    
    # HT-3: Breakpoint histogram
    create_breakpoint_histogram(results_df)
    
    # HT-4: F-test heatmap
    create_ftest_heatmap(results_df)
    
    # HT-5: India vs emerging peers overlay
    create_india_overlay_chart(results_df, master_df)

def generate_clean_ticks(x_data, y_data, country_name):
    """Generate clean tick positions and labels for log-scale axes, optimized for individual plots"""
    if len(x_data) == 0 or len(y_data) == 0:
        return [], [], [], []
    
    # X-axis ticks (GDP values) - more precise for individual plots
    x_min, x_max = np.min(x_data), np.max(x_data)
    x_range = x_max - x_min
    
    # Country-specific optimal tick selection
    if country_name.lower() == 'india':
        x_ticks = [2000, 3000, 4000, 5000, 6000, 8000]
        x_labels = ['2K', '3K', '4K', '5K', '6K', '8K']
    elif country_name.lower() == 'china':
        x_ticks = [2000, 5000, 8000, 10000, 15000, 20000]
        x_labels = ['2K', '5K', '8K', '10K', '15K', '20K']
    elif country_name.lower() == 'brazil':
        x_ticks = [8000, 10000, 12000, 15000, 18000, 20000]
        x_labels = ['8K', '10K', '12K', '15K', '18K', '20K']
    elif country_name.lower() == 'mexico':
        x_ticks = [8000, 10000, 12000, 15000, 18000, 20000, 25000]
        x_labels = ['8K', '10K', '12K', '15K', '18K', '20K', '25K']
    elif x_max <= 8000:  # Low income countries
        x_ticks = [1000, 2000, 3000, 4000, 5000, 6000, 8000]
        x_labels = ['1K', '2K', '3K', '4K', '5K', '6K', '8K']
    elif x_max <= 20000:  # Lower-middle income
        x_ticks = [5000, 8000, 10000, 15000, 20000]
        x_labels = ['5K', '8K', '10K', '15K', '20K']
    elif x_max <= 50000:  # Upper-middle income
        x_ticks = [20000, 30000, 40000, 50000]
        x_labels = ['20K', '30K', '40K', '50K']
    else:  # High income countries
        x_ticks = [30000, 40000, 50000, 60000]
        x_labels = ['30K', '40K', '50K', '60K']
    
    # Filter to actual data range and ensure we have 4-6 ticks
    valid_x_indices = [(i, tick) for i, tick in enumerate(x_ticks) if x_min*0.8 <= tick <= x_max*1.2]
    if valid_x_indices and len(valid_x_indices) >= 3:
        x_tick_positions = [tick for _, tick in valid_x_indices]
        x_tick_labels = [x_labels[i] for i, _ in valid_x_indices]
    else:
        # Fallback: create evenly spaced ticks
        x_tick_positions = np.logspace(np.log10(x_min), np.log10(x_max), 5)
        x_tick_labels = [f'{int(tick/1000)}K' if tick >= 1000 else f'{int(tick)}' for tick in x_tick_positions]
    
    # Y-axis ticks (Beauty consumption values)
    y_min, y_max = np.min(y_data), np.max(y_data)
    y_range = y_max - y_min
    
    if y_max <= 1:  # Very low consumption
        y_ticks = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        y_labels = ['0.05', '0.1', '0.2', '0.3', '0.5', '0.7', '1.0']
    elif y_max <= 5:  # Low consumption
        y_ticks = [0.1, 0.2, 0.5, 1, 2, 3, 5]
        y_labels = ['0.1', '0.2', '0.5', '1', '2', '3', '5']
    elif y_max <= 30:  # Medium consumption
        y_ticks = [1, 2, 5, 10, 15, 20, 30]
        y_labels = ['1', '2', '5', '10', '15', '20', '30']
    elif y_max <= 100:  # High consumption
        y_ticks = [10, 20, 30, 50, 70, 100]
        y_labels = ['10', '20', '30', '50', '70', '100']
    else:  # Very high consumption
        y_ticks = [50, 75, 100, 150, 200]
        y_labels = ['50', '75', '100', '150', '200']
    
    # Filter to actual data range
    valid_y_indices = [(i, tick) for i, tick in enumerate(y_ticks) if y_min*0.8 <= tick <= y_max*1.2]
    if valid_y_indices and len(valid_y_indices) >= 3:
        y_tick_positions = [tick for _, tick in valid_y_indices]
        y_tick_labels = [y_labels[i] for i, _ in valid_y_indices]
    else:
        # Fallback: create evenly spaced ticks
        y_tick_positions = np.logspace(np.log10(max(y_min, 0.01)), np.log10(y_max), 5)
        y_tick_labels = [f'{tick:.1f}' if tick < 10 else f'{int(tick)}' for tick in y_tick_positions]
    
    return x_tick_positions, x_tick_labels, y_tick_positions, y_tick_labels

def create_individual_plot(country, series, results_df, master_df, figures_dir):
    """Create a single individual plot for one country-series combination"""
    
    # Get country data
    country_data = master_df[master_df['country'] == country]
    series_col = {'Beauty': 'BeautyPC', 'Skincare': 'SkincarePC', 
                 'Mens Wear': 'MenPC', 'Womens Wear': 'WomenPC'}[series]
    
    if series_col not in country_data.columns:
        return False
    
    x = country_data['gdppcppp']
    y = country_data[series_col]
    
    # Remove NaN values - allow y = 0
    mask = ~(np.isnan(x) | np.isnan(y)) & (x > 0)
    if mask.sum() < 3:
        return False
        
    x_clean, y_clean = x[mask], y[mask]
    
    # Create individual figure
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    
    # Generate clean ticks for this specific plot
    x_tick_pos, x_tick_labels, y_tick_pos, y_tick_labels = generate_clean_ticks(x_clean, y_clean, country)
    
    # Plot scatter with consistent colors
    country_color = COUNTRY_COLORS.get(country, '#666666')
    ax.scatter(x_clean, y_clean, alpha=0.7, s=60, color=country_color, 
              edgecolors='white', linewidth=1)
    
    # Get piecewise result for this country-series
    pw_result = results_df[(results_df['country'] == country) & 
                          (results_df['series'] == series)]
    
    if not pw_result.empty:
        breakpoint = pw_result.iloc[0]['breakpoint']
        b1 = pw_result.iloc[0]['b1']
        b2 = pw_result.iloc[0]['b2']
        intercept1 = pw_result.iloc[0]['intercept1']
        intercept2 = pw_result.iloc[0]['intercept2']
        
        # Plot piecewise lines (back-transform from log space)
        x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
        
        # Segment 1: y = exp(intercept1) * x^b1
        x1_range = x_range[x_range <= breakpoint]
        if len(x1_range) > 0:
            y1_range = np.exp(intercept1) * (x1_range ** b1)
            ax.plot(x1_range, y1_range, 'k-', linewidth=3, alpha=0.8, label='Pre-threshold')
        
        # Segment 2: y = exp(intercept2) * x^b2
        x2_range = x_range[x_range > breakpoint]
        if len(x2_range) > 0:
            y2_range = np.exp(intercept2) * (x2_range ** b2)
            ax.plot(x2_range, y2_range, 'k--', linewidth=3, alpha=0.8, label='Post-threshold')
        
        # Vertical line at breakpoint
        ax.axvline(breakpoint, color='red', linestyle=':', alpha=0.8, linewidth=2,
                  label=f'Threshold: ${breakpoint:,.0f}')
        
        # Add p-value annotation
        p_val = pw_result.iloc[0]['p_value']
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        ax.text(0.05, 0.95, f'p={p_val:.3f}{significance}', 
               transform=ax.transAxes, fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Set titles and labels
    ax.set_title(f'{country.replace("_", " ").title()} - {series} Consumption\nPiecewise Regression Analysis', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('GDP Per Capita PPP', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{series} Per Capita', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=12)
    
    # Set log scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Force clean tick formatting by disabling matplotlib's automatic formatting
    from matplotlib.ticker import FixedLocator, FixedFormatter
    
    # Apply clean pre-calculated ticks with complete control
    if len(x_tick_pos) > 0 and len(x_tick_labels) > 0:
        ax.xaxis.set_major_locator(FixedLocator(x_tick_pos))
        ax.xaxis.set_major_formatter(FixedFormatter(x_tick_labels))
        ax.xaxis.set_minor_locator(FixedLocator([]))  # Disable minor ticks
    
    if len(y_tick_pos) > 0 and len(y_tick_labels) > 0:
        ax.yaxis.set_major_locator(FixedLocator(y_tick_pos))
        ax.yaxis.set_major_formatter(FixedFormatter(y_tick_labels))
        ax.yaxis.set_minor_locator(FixedLocator([]))  # Disable minor ticks
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    if not pw_result.empty:
        ax.legend(loc='lower right', fontsize=10)
    
    # Save individual plot
    filename = f'HT-1_{country.lower()}_{series.lower().replace(" ", "_")}.png'
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True

def create_small_multiples_chart(results_df, master_df):
    """HT-1: Create individual plots and combined series charts following README scope"""
    
    # Core countries from README: Japan, South Korea, China, US, India
    core_countries = ['japan', 'south korea', 'china', 'usa', 'india']
    
    # Filter to core countries only
    results_df = results_df[results_df['country'].isin(core_countries)]
    
    # All series for extended hypothesis testing
    series_list = ['Beauty', 'Skincare', 'Mens Wear', 'Womens Wear']
    
    script_dir = Path(__file__).resolve().parent.parent.parent
    figures_dir = script_dir / "figures"
    
    # Create subdirectory for individual plots
    individual_plots_dir = figures_dir / "HT-1_individual_plots"
    individual_plots_dir.mkdir(exist_ok=True)
    
    created_individual_plots = []
    
    # Skip individual plot creation for now since they're already created
    print(f"  Skipping individual plot creation (already exists in HT-1_individual_plots/)")
    
    # Create combined charts for each series (Option B)
    combined_charts_created = 0
    for series in series_list:
        success = create_series_comparison_chart(series, results_df, master_df, figures_dir, core_countries)
        if success:
            combined_charts_created += 1
            print(f"  Created combined chart: HT-1_{series.lower().replace(' ', '_')}_comparison.png")
    
    print(f"  Created {combined_charts_created} combined series comparison charts in main figures/")

def create_series_comparison_chart(series, results_df, master_df, figures_dir, core_countries):
    """Create a comparison chart for one series across all core countries (Option B)"""
    
    # Filter to this series and core countries
    series_results = results_df[(results_df['series'] == series) & 
                               (results_df['country'].isin(core_countries))]
    
    if series_results.empty:
        print(f"  No data for {series} series")
        return False
    
    # Create subplot layout: 1 row, 5 columns for core countries
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), dpi=300)
    
    plot_idx = 0
    created_subplots = []
    
    for country in core_countries:
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        
        # Get country data
        country_data = master_df[master_df['country'] == country]
        series_col = {'Beauty': 'BeautyPC', 'Skincare': 'SkincarePC', 
                     'Mens Wear': 'MenPC', 'Womens Wear': 'WomenPC'}[series]
        
        if series_col not in country_data.columns or country_data.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{country.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            plot_idx += 1
            continue
        
        x = country_data['gdppcppp']
        y = country_data[series_col]
        
        # Remove NaN values - allow y = 0
        mask = ~(np.isnan(x) | np.isnan(y)) & (x > 0)
        if mask.sum() < 3:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{country.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            plot_idx += 1
            continue
            
        x_clean, y_clean = x[mask], y[mask]
        
        # Generate clean ticks for this specific subplot
        x_tick_pos, x_tick_labels, y_tick_pos, y_tick_labels = generate_clean_ticks(x_clean, y_clean, country)
        
        # Plot scatter with consistent colors
        country_color = COUNTRY_COLORS.get(country, '#666666')
        ax.scatter(x_clean, y_clean, alpha=0.7, s=40, color=country_color, 
                  edgecolors='white', linewidth=0.5)
        
        # Get piecewise result for this country-series
        pw_result = series_results[series_results['country'] == country]
        
        if not pw_result.empty:
            breakpoint = pw_result.iloc[0]['breakpoint']
            b1 = pw_result.iloc[0]['b1']
            b2 = pw_result.iloc[0]['b2']
            intercept1 = pw_result.iloc[0]['intercept1']
            intercept2 = pw_result.iloc[0]['intercept2']
            
            # Plot piecewise lines (back-transform from log space)
            x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
            
            # Segment 1: y = exp(intercept1) * x^b1
            x1_range = x_range[x_range <= breakpoint]
            if len(x1_range) > 0:
                y1_range = np.exp(intercept1) * (x1_range ** b1)
                ax.plot(x1_range, y1_range, 'k-', linewidth=2, alpha=0.8)
            
            # Segment 2: y = exp(intercept2) * x^b2
            x2_range = x_range[x_range > breakpoint]
            if len(x2_range) > 0:
                y2_range = np.exp(intercept2) * (x2_range ** b2)
                ax.plot(x2_range, y2_range, 'k--', linewidth=2, alpha=0.8)
            
            # Vertical line at breakpoint
            ax.axvline(breakpoint, color='red', linestyle=':', alpha=0.7, linewidth=2)
            
            # Add p-value annotation (top-left)
            p_val = pw_result.iloc[0]['p_value']
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            ax.text(0.05, 0.95, f'p={p_val:.3f}{significance}', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Add threshold annotation (bottom-right)
            threshold_val = pw_result.iloc[0]['breakpoint']
            threshold_text = f'${threshold_val/1000:.0f}K' if threshold_val >= 1000 else f'${threshold_val:.0f}'
            ax.text(0.95, 0.05, f'Threshold:\n{threshold_text}', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.8))
        
        # Set titles and labels
        ax.set_title(f'{country.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('GDP Per Capita PPP', fontsize=12)
        ax.set_ylabel(f'{series} Per Capita', fontsize=12)
        ax.tick_params(labelsize=10)
        
        # Set log scales
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Apply the same clean formatting approach as individual plots
        from matplotlib.ticker import FixedLocator, FixedFormatter
        
        if len(x_tick_pos) > 0 and len(x_tick_labels) > 0:
            ax.xaxis.set_major_locator(FixedLocator(x_tick_pos))
            ax.xaxis.set_major_formatter(FixedFormatter(x_tick_labels))
            ax.xaxis.set_minor_locator(FixedLocator([]))  # Disable minor ticks
        
        if len(y_tick_pos) > 0 and len(y_tick_labels) > 0:
            ax.yaxis.set_major_locator(FixedLocator(y_tick_pos))
            ax.yaxis.set_major_formatter(FixedFormatter(y_tick_labels))
            ax.yaxis.set_minor_locator(FixedLocator([]))  # Disable minor ticks
        
        # Add light grid
        ax.grid(True, alpha=0.3)
        
        created_subplots.append(f"{country}-{series}")
        plot_idx += 1
    
    # Add overall title
    plt.suptitle(f'{series} Consumption vs GDP Per Capita: Cross-Country Comparison\nTesting Threshold Hypothesis Across Core Countries', 
                fontsize=16, fontweight='bold', y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save series comparison chart
    filename = f'HT-1_{series.lower().replace(" ", "_")}_comparison.png'
    plt.savefig(figures_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True

def create_slope_change_chart(results_df):
    """HT-2: Improved horizontal bar chart of slope changes (Δb) with separate panels per category"""
    
    if results_df.empty:
        print("  Skipping HT-2: No piecewise results available")
        return
    
    # Prepare data for plotting
    plot_data = results_df[['country', 'series', 'delta_b', 'p_value']].copy()
    plot_data['significant'] = plot_data['p_value'] < 0.05
    plot_data['abs_delta_b'] = plot_data['delta_b'].abs()
    
    # Get all series and create subplot layout
    series_list = ['Beauty', 'Skincare', 'Mens Wear', 'Womens Wear']
    available_series = [s for s in series_list if s in plot_data['series'].unique()]
    
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Colors for categories (consistent with other charts)
    series_colors = {
        'Beauty': '#1f77b4',      # Blue
        'Skincare': '#ff7f0e',    # Orange  
        'Mens Wear': '#d62728',   # Red
        'Womens Wear': '#e377c2'  # Pink
    }
    
    for idx, series in enumerate(available_series):
        if idx >= 4:  # Safety check
            break
            
        ax = axes[idx]
        series_data = plot_data[plot_data['series'] == series].copy()
        
        if series_data.empty:
            ax.text(0.5, 0.5, f'No data for {series}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{series}', fontsize=14, fontweight='bold')
            continue
        
        # Sort countries by magnitude of delta_b (absolute value) - largest changes first
        series_data = series_data.sort_values('abs_delta_b', ascending=True)  # ascending=True for horizontal bars
        
        countries = series_data['country'].values
        delta_b_values = series_data['delta_b'].values
        significance = series_data['significant'].values
        p_values = series_data['p_value'].values
        
        # Create horizontal bar chart
        y_positions = np.arange(len(countries))
        color = series_colors.get(series, '#666666')
        
        # Create bars with different alpha for significance
        bars = ax.barh(y_positions, delta_b_values, 
                      color=color, alpha=0.8, height=0.6)
        
        # Add significance markers (asterisks)
        for i, (bar, sig, p_val) in enumerate(zip(bars, significance, p_values)):
            if sig:
                width = bar.get_width()
                # Position asterisk just outside the bar
                if width >= 0:
                    x_pos = width + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
                    ha = 'left'
                else:
                    x_pos = width - 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
                    ha = 'right'
                
                ax.text(x_pos, bar.get_y() + bar.get_height()/2, '*', 
                       ha=ha, va='center', fontweight='bold', fontsize=16, color='red')
        
        # Formatting
        ax.set_yticks(y_positions)
        ax.set_yticklabels([c.replace('_', ' ').title() for c in countries], fontsize=14)
        ax.set_xlabel('Slope Change (Δb)', fontsize=16, fontweight='bold')
        ax.set_title(f'{series}', fontsize=18, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        # Adjust x-axis limits to accommodate asterisks
        xlim = ax.get_xlim()
        x_range = xlim[1] - xlim[0]
        ax.set_xlim(xlim[0] - 0.05 * x_range, xlim[1] + 0.05 * x_range)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, delta_b_values)):
            if abs(val) > 0.001:  # Only show labels for non-zero values
                x_label = bar.get_width() / 2
                ax.text(x_label, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                       ha='center', va='center', fontweight='bold', fontsize=12, color='white')
    
    # Hide unused subplots
    for idx in range(len(available_series), 4):
        axes[idx].set_visible(False)
    
    # Overall title and layout
    plt.suptitle('Slope Change at Income Threshold by Country and Category\n(* indicates p < 0.05, sorted by magnitude)', 
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    
    # Save the chart
    script_dir = Path(__file__).resolve().parent.parent.parent
    figures_dir = script_dir / "figures"
    plt.savefig(figures_dir / 'HT-2_slope_change_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created HT-2: Improved slope change bar chart with separate panels")

def create_breakpoint_histogram(results_df):
    """HT-3: Improved histogram of breakpoints by category with consistent formatting"""
    
    if results_df.empty:
        print("  Skipping HT-3: No piecewise results available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Define consistent bin edges (10k width) and axis ranges for all panels
    bin_edges = np.arange(0, 60000, 10000)  # 0, 10k, 20k, ..., 50k
    x_min, x_max = 0, 55000
    
    # Calculate overall max frequency for consistent y-axis scaling
    all_breakpoints = results_df['breakpoint'].dropna()
    if len(all_breakpoints) > 0:
        # Calculate max frequency across all series to set consistent y-axis
        max_freq = 0
        series_list = ['Beauty', 'Skincare', 'Mens Wear', 'Womens Wear']
        for series in series_list:
            series_data = results_df[results_df['series'] == series]
            if not series_data.empty:
                breakpoints = series_data['breakpoint'].dropna()
                if len(breakpoints) > 0:
                    hist_counts, _ = np.histogram(breakpoints, bins=bin_edges)
                    max_freq = max(max_freq, hist_counts.max())
        
        # Set y-axis to show integers only, with some headroom
        y_max = max(3, int(max_freq) + 1)
    else:
        y_max = 3
    
    series_list = ['Beauty', 'Skincare', 'Mens Wear', 'Womens Wear']
    
    for i, series in enumerate(series_list):
        if i >= 4:
            break
            
        ax = axes[i]
        series_data = results_df[results_df['series'] == series]
        
        if not series_data.empty:
            breakpoints = series_data['breakpoint'].dropna()
            
            if len(breakpoints) > 0:
                # Get country names for each breakpoint
                breakpoint_countries = series_data[['country', 'breakpoint']].dropna()
                
                # Create bins and get country assignments
                hist_counts, bin_edges_used = np.histogram(breakpoints, bins=bin_edges)
                
                # Create bars with spacing (rwidth parameter adds gaps)
                n, bins, patches = ax.hist(breakpoints, bins=bin_edges, alpha=0.7, 
                                         edgecolor='black', color='steelblue', rwidth=0.8)
                
                # Add country names to bars
                for i, (count, left_edge, right_edge) in enumerate(zip(hist_counts, bin_edges_used[:-1], bin_edges_used[1:])):
                    if count > 0:
                        # Find countries in this bin
                        countries_in_bin = breakpoint_countries[
                            (breakpoint_countries['breakpoint'] >= left_edge) & 
                            (breakpoint_countries['breakpoint'] < right_edge)
                        ]['country'].tolist()
                        
                        # Format country names (capitalize and remove underscores)
                        formatted_countries = [c.replace('_', ' ').title() for c in countries_in_bin]
                        
                        # Position text in center of bar
                        bin_center = (left_edge + right_edge) / 2
                        
                        # Add country names vertically stacked in the bar
                        if len(formatted_countries) == 1:
                            # Single country - center it
                            ax.text(bin_center, count/2, formatted_countries[0], 
                                   ha='center', va='center', fontsize=9, fontweight='bold',
                                   color='black', rotation=0)
                        elif len(formatted_countries) == 2:
                            # Two countries - stack them
                            ax.text(bin_center, count * 0.75, formatted_countries[0], 
                                   ha='center', va='center', fontsize=8, fontweight='bold',
                                   color='black', rotation=0)
                            ax.text(bin_center, count * 0.25, formatted_countries[1], 
                                   ha='center', va='center', fontsize=8, fontweight='bold',
                                   color='black', rotation=0)
                        else:
                            # More than 2 countries - use smaller font and stack
                            for j, country in enumerate(formatted_countries):
                                y_pos = count * (len(formatted_countries) - j) / (len(formatted_countries) + 1)
                                ax.text(bin_center, y_pos, country, 
                                       ha='center', va='center', fontsize=7, fontweight='bold',
                                       color='black', rotation=0)
                
                # Add median line
                median_val = breakpoints.median()
                ax.axvline(median_val, color='red', linestyle='--', linewidth=2)
                
                # Add median label inside panel, positioned near the line
                ax.text(median_val + 1500, y_max * 0.85, f'Median: ${median_val:,.0f}', 
                       fontsize=10, fontweight='bold', color='red',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                ax.set_xlabel('GDP Per Capita PPP', fontsize=11, fontweight='bold')
                ax.set_ylabel('Number of Countries', fontsize=11, fontweight='bold')
                ax.set_title(f'{series} Breakpoints\n({len(breakpoints)} countries)', 
                           fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Set consistent axes ranges and integer ticks
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(0, y_max)
                ax.set_yticks(range(0, y_max + 1))  # Integer y-axis ticks only
                
                # Format x-axis with clean ticks
                ax.set_xticks(np.arange(0, 60000, 10000))
                ax.set_xticklabels(['0', '10K', '20K', '30K', '40K', '50K'])
                
            else:
                ax.text(0.5, 0.5, 'No significant\nbreakpoints found', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{series} Breakpoints', fontsize=12, fontweight='bold')
                ax.set_xlabel('GDP Per Capita PPP', fontsize=11, fontweight='bold')
                ax.set_ylabel('Number of Countries', fontsize=11, fontweight='bold')
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(0, y_max)
                ax.set_yticks(range(0, y_max + 1))
                ax.set_xticks(np.arange(0, 60000, 10000))
                ax.set_xticklabels(['0', '10K', '20K', '30K', '40K', '50K'])
                ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{series} Breakpoints', fontsize=12, fontweight='bold')
            ax.set_xlabel('GDP Per Capita PPP (USD)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Number of Countries', fontsize=11, fontweight='bold')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, y_max)
            ax.set_yticks(range(0, y_max + 1))
            ax.set_xticks(np.arange(0, 60000, 10000))
            ax.set_xticklabels(['0', '10K', '20K', '30K', '40K', '50K'])
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots (shouldn't be needed with 4 series, but keep for safety)
    for i in range(len(series_list), 4):
        axes[i].set_visible(False)
    
    plt.suptitle('Distribution of Income Thresholds by Category\n(Breakpoints where consumption accelerates)', 
                fontsize=15, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    
    script_dir = Path(__file__).resolve().parent.parent.parent
    figures_dir = script_dir / "figures"
    plt.savefig(figures_dir / 'HT-3_breakpoint_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created HT-3: Improved breakpoint histogram with consistent formatting")

def create_ftest_heatmap(results_df):
    """HT-4: Dot plot of F-test p-values with diverging color palette"""
    
    if results_df.empty:
        print("  Skipping HT-4: No piecewise results available")
        return
    
    # Create pivot table for data organization
    heatmap_data = results_df.pivot(index='country', columns='series', values='p_value')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up diverging colormap centered at 0.05
    import matplotlib.colors as colors
    cmap = plt.cm.RdBu
    norm = colors.TwoSlopeNorm(vcenter=0.05, vmin=0, vmax=0.4)
    
    # Create dot plot
    x_positions = np.arange(len(heatmap_data.columns))
    y_positions = np.arange(len(heatmap_data.index))
    
    # Plot dots for each data point
    for i, country in enumerate(heatmap_data.index):
        for j, category in enumerate(heatmap_data.columns):
            p_val = heatmap_data.loc[country, category]
            
            if not pd.isna(p_val):
                # Size rectangles inversely proportional to p-value (smaller p = larger rectangle)
                # Base width and height, with scaling for significance
                base_width = 0.35
                base_height = 0.25
                scale_factor = 1 + (1 - min(p_val, 0.4) / 0.4) * 0.5  # 1.0 to 1.5x scaling
                rect_width = base_width * scale_factor
                rect_height = base_height * scale_factor
                
                # Color based on p-value using diverging palette
                color = cmap(norm(p_val))
                
                # Create rectangle centered at (j, i)
                from matplotlib.patches import Rectangle
                rect = Rectangle((j - rect_width/2, i - rect_height/2), 
                               rect_width, rect_height,
                               facecolor=color, edgecolor='white', 
                               linewidth=2, alpha=0.8)
                ax.add_patch(rect)
                
                # Add text annotation with larger font
                ax.text(j, i, f'{p_val:.3f}', ha='center', va='center', 
                       fontsize=12, fontweight='bold',
                       color='white' if p_val < 0.15 else 'black')
    
    # Customize the plot
    ax.set_xticks(x_positions)
    ax.set_yticks(y_positions)
    ax.set_xticklabels(heatmap_data.columns, fontsize=11)
    ax.set_yticklabels(heatmap_data.index, fontsize=11)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set limits with padding
    ax.set_xlim(-0.5, len(heatmap_data.columns) - 0.5)
    ax.set_ylim(-0.5, len(heatmap_data.index) - 0.5)
    
    # Labels and title
    ax.set_title('Statistical Significance of Income Thresholds\n(F-test p-values, red = significant, blue = not significant, white = missing data)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Country', fontsize=12, fontweight='bold')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('p-value', fontsize=11, fontweight='bold')
    
    # Add significance threshold line to colorbar
    cbar.ax.axhline(y=0.05, color='black', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    script_dir = Path(__file__).resolve().parent.parent.parent
    figures_dir = script_dir / "figures"
    plt.savefig(figures_dir / 'HT-4_ftest_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created HT-4: F-test significance dot plot")

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
        
        # Sort by x-axis values to avoid kinks in the line plot
        sort_idx = np.argsort(x_clean)
        x_clean, y_clean = x_clean.iloc[sort_idx], y_clean.iloc[sort_idx]
        
        # Plot country trajectory with consistent colors
        country_color = COUNTRY_COLORS.get(country, '#666666')
        
        if country == 'india':
            ax.plot(x_clean, y_clean, 'o-', linewidth=3, markersize=6, 
                   color=country_color, label=f'{country.title()}', alpha=0.8)
            
            # Mark India's current position (latest year)
            current_x = x_clean.iloc[-1]
            current_y = y_clean.iloc[-1]
            ax.scatter(current_x, current_y, s=200, color=country_color, marker='*', 
                      edgecolor='black', linewidth=2, zorder=10, label='India Today')
            
        else:
            ax.plot(x_clean, y_clean, 'o-', linewidth=2, markersize=4, 
                   color=country_color, label=f'{country.title()}', alpha=0.7)
        
        # Add peer breakpoint lines
        peer_bp = results_df.query("country==@country & series=='Beauty'")['breakpoint']
        if not peer_bp.empty and not np.isnan(peer_bp.iloc[0]):
            if country == 'india':
                ax.axvline(peer_bp.iloc[0], color='red', linestyle='--', alpha=0.7, linewidth=2,
                          label=f'India Threshold: ${peer_bp.iloc[0]:,.0f}')
            else:
                ax.axvline(peer_bp.iloc[0], color='grey', linestyle=':', alpha=0.4, linewidth=1)
    
    ax.set_xlabel('GDP Per Capita PPP', fontsize=12, fontweight='bold')
    ax.set_ylabel('Beauty Consumption Per Capita', fontsize=12, fontweight='bold')
    ax.set_title('India vs Emerging Market Peers: Beauty Consumption Trajectory\nIndia\'s Position Relative to Income Thresholds', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
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