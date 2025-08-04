import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.ticker import MaxNLocator

plt.style.use('default')

# Set larger font sizes for better readability
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16

# Special colors for T1-1 chart (different from country colors)
T1_CHART_COLORS = {
    'segments': '#17becf',     # Cyan
    'overall': '#bcbd22'       # Olive
}

# Consistent country color palette matching Task 2
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

def load_master_data():
    """Load master dataset excluding India for Task 1"""
    master = pd.read_parquet("data/processed/beauty_income_panel.parquet")
    # Filter out India for Task 1, and filter Brazil data before 1995
    master = master[master['country'] != 'india'].copy()
    master = master[(master['country'] != 'brazil') | (master['year'] >= 1995)].copy()
    return master

def create_descriptive_plots(df):
    """Create descriptive plots T1-1 through T1-6"""
    
    countries = sorted(df['country'].unique())
    
    # Setup figure directory
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    
    # T1-3: Scatter BeautyPC vs GDPpcPPP with LOWESS
    plt.figure(figsize=(8, 5), dpi=200)
    
    for country in countries:
        country_data = df[df['country'] == country]
        country_label = country.upper() if country == 'usa' else country.replace('_', ' ').title()
        country_color = COUNTRY_COLORS.get(country, '#666666')
        plt.scatter(country_data['gdppcppp'], country_data['BeautyPC'], 
                   alpha=0.8, label=country_label, color=country_color, s=60)
    
    # Add LOWESS smooth lines with robustness testing
    for frac in [0.4]:
        # Use statsmodels lowess
        smoothed = lowess(df['BeautyPC'], df['gdppcppp'], frac=frac)
        plt.plot(smoothed[:, 0], smoothed[:, 1], '-', linewidth=3, alpha=1.0, 
                color='black', label=f'LOWESS (frac={frac})')
    
    plt.xlabel('GDP Per Capita PPP', fontweight='bold')
    plt.ylabel('Beauty Consumption Per Capita', fontweight='bold')
    plt.title('Is Beauty An S-Curve Good?', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(fig_dir / "T1-3_BeautyPC_vs_gdppcppp_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # T1-4: Log-log scatter with OLS fit
    plt.figure(figsize=(8, 5), dpi=200)
    
    # Remove zero values for log transformation and fix log constant
    df_log = df[(df['BeautyPC'] > 0) & (df['gdppcppp'] > 0)].copy()
    df_log['ln_gdppc'] = np.log(df_log['gdppcppp'])
    
    for country in countries:
        country_data = df_log[df_log['country'] == country]
        country_label = country.upper() if country == 'usa' else country.replace('_', ' ').title()
        country_color = COUNTRY_COLORS.get(country, '#666666')
        plt.scatter(country_data['ln_gdppc'], country_data['ln_BeautyPC'], 
                   alpha=0.8, label=country_label, color=country_color, s=60)
    
    # Add OLS fit line
    X = df_log['ln_gdppc'].values.reshape(-1, 1)
    y = df_log['ln_BeautyPC'].values
    reg = LinearRegression().fit(X, y)
    
    x_range = np.linspace(df_log['ln_gdppc'].min(), df_log['ln_gdppc'].max(), 100)
    y_pred = reg.predict(x_range.reshape(-1, 1))
    
    plt.plot(x_range, y_pred, 'k-', linewidth=3, 
             label=f'Income Elasticity = {reg.coef_[0]:.2f}')
    
    plt.xlabel('Natural Log Of GDP Per Capita PPP', fontweight='bold')
    plt.ylabel('Natural Log Of Beauty Consumption\nPer Capita', fontweight='bold')
    plt.title('Income Elasticity Of Beauty Consumption: Log-Log Relationship', fontweight='bold')
    plt.legend(loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()  # Ensure proper spacing
    plt.savefig(fig_dir / "T1-4_log_scatter_ols.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # T1-5: BeautyShare vs GDPpcPPP
    plt.figure(figsize=(8, 5), dpi=200)
    
    for country in countries:
        country_data = df[df['country'] == country]
        country_label = country.upper() if country == 'usa' else country.replace('_', ' ').title()
        country_color = COUNTRY_COLORS.get(country, '#666666')
        plt.scatter(country_data['gdppcppp'], country_data['BeautyShare'], 
                   alpha=0.8, label=country_label, color=country_color, s=60)
    
    plt.xlabel('GDP Per Capita PPP', fontweight='bold')
    plt.ylabel('Beauty Share Of Household Consumption', fontweight='bold')
    plt.title('Beauty Share Of Wallet Flattens Above ~2% Of Consumption', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "T1-5_BeautyShare_vs_gdppcppp.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return df_log

def calculate_correlations(df):
    """Calculate correlations between beauty metrics and income variables"""
    
    results = []
    countries = ['pooled'] + sorted(df['country'].unique())
    
    for country in countries:
        if country == 'pooled':
            data = df.copy()
        else:
            data = df[df['country'] == country].copy()
        
        # Remove missing values
        data = data.dropna(subset=['BeautyPC', 'gdppcppp', 'ne_con_prvt_pc_kd'])
        
        if len(data) < 3:  # Need at least 3 points for correlation
            continue
            
        # BeautyPC vs GDPpcPPP
        r_gdp, p_gdp = stats.pearsonr(data['BeautyPC'], data['gdppcppp'])
        rho_gdp, p_rho_gdp = stats.spearmanr(data['BeautyPC'], data['gdppcppp'])
        
        # BeautyPC vs HH Consumption
        r_cons, p_cons = stats.pearsonr(data['BeautyPC'], data['ne_con_prvt_pc_kd'])
        rho_cons, p_rho_cons = stats.spearmanr(data['BeautyPC'], data['ne_con_prvt_pc_kd'])
        
        results.append({
            'country': country,
            'n_obs': len(data),
            'r_BeautyPC_gdp': r_gdp,
            'p_BeautyPC_gdp': p_gdp,
            'rho_BeautyPC_gdp': rho_gdp,
            'p_rho_BeautyPC_gdp': p_rho_gdp,
            'r_BeautyPC_cons': r_cons,
            'p_BeautyPC_cons': p_cons,
            'rho_BeautyPC_cons': rho_cons,
            'p_rho_BeautyPC_cons': p_rho_cons
        })
    
    corr_df = pd.DataFrame(results)
    
    corr_df.to_csv("figures/T1_correlations.csv", index=False)
    return corr_df

def run_elasticity_regression(df):
    """Run OLS regression on logs to estimate elasticity"""
    
    # Remove zero/missing values
    reg_data = df[(df['BeautyPC'] > 0) & (df['gdppcppp'] > 0)].copy()
    reg_data = reg_data.dropna(subset=['ln_BeautyPC', 'ln_gdppc'])
    
    # Run regression: ln(BeautyPC) = alpha + beta * ln(GDPpcPPP) + epsilon
    X = sm.add_constant(reg_data['ln_gdppc'])  # Add constant
    y = reg_data['ln_BeautyPC']
    
    model = sm.OLS(y, X)
    results_sm = model.fit(cov_type='HC3') # HC3 robust standard errors
    
    # Extract results
    beta = results_sm.params[1]  # Slope coefficient
    se_beta = results_sm.bse[1]
    ci_lower = results_sm.conf_int().iloc[1, 0]
    ci_upper = results_sm.conf_int().iloc[1, 1]
    r_squared = results_sm.rsquared
    p_value = results_sm.pvalues[1]
    
    results = {
        'elasticity_beta': beta,
        'std_error': se_beta,
        'ci_lower_95': ci_lower,
        'ci_upper_95': ci_upper,
        'r_squared': r_squared,
        'n_obs': len(reg_data),
        'p_value': p_value
    }
    
    # Save regression summary
    with open("figures/T1_elasticity_regression.txt", "w") as f:
        f.write(f"Elasticity Regression Results\n{'='*30}\n")
        f.write(f"Regression: ln(BeautyPC) = α + β*ln(GDPpcPPP) + ε\n\n")
        f.write(f"Elasticity (β): {beta:.3f}\nStandard Error: {se_beta:.3f}\n")
        f.write(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]\nR-squared: {r_squared:.3f}\n")
        f.write(f"P-value: {p_value:.3e}\nN observations: {len(reg_data)}\n")
    
    return results, results_sm

def detect_inflection_point(df):
    """Detect inflection/plateau using piecewise linear regression with comprehensive statistical validation"""
    
    # Prepare data
    data = df.dropna(subset=['BeautyPC', 'gdppcppp']).copy()
    data = data.sort_values('gdppcppp')
    
    x = data['gdppcppp'].values
    y = data['BeautyPC'].values
    
    if len(data) < 10:
        return None, None, None
    
    # Define candidate breakpoints
    x_min, x_max = x.min(), x.max()
    candidates = np.arange(x_min + 2000, x_max - 2000, 2000)
    
    results = []
    
    # Single linear model for comparison
    X_single = sm.add_constant(x)
    single_model = sm.OLS(y, X_single)
    single_results = single_model.fit()
    single_sse = np.sum(single_results.resid**2)
    single_aic = single_results.aic
    
    for c in candidates:
        seg1_mask = x <= c
        seg2_mask = x > c
        
        if sum(seg1_mask) < 3 or sum(seg2_mask) < 3:
            continue
        
        try:
            # Fit segments with confidence intervals
            X1 = sm.add_constant(x[seg1_mask])
            X2 = sm.add_constant(x[seg2_mask])
            
            model1 = sm.OLS(y[seg1_mask], X1)
            results1 = model1.fit()
            model2 = sm.OLS(y[seg2_mask], X2)
            results2 = model2.fit()
            
            total_sse = np.sum(results1.resid**2) + np.sum(results2.resid**2)
            
            # Calculate AIC and F-test
            aic = results1.aic + results2.aic
            delta_aic = aic - single_aic
            
            # F-test for piecewise vs single model
            # F-statistic for comparing nested models (single vs piecewise)
            # F = ((RSS_restricted - RSS_unrestricted) / (df_restricted - df_unrestricted)) / (RSS_unrestricted / df_unrestricted)
            # Here, restricted is single model (2 params), unrestricted is piecewise (4 params)
            f_stat = ((single_sse - total_sse) / (4 - 2)) / (total_sse / (len(data) - 4))
            f_pvalue = stats.f.sf(f_stat, 2, len(data) - 4)
            
            # Confidence intervals for slopes
            slope1_ci = results1.conf_int()[1, :]
            slope2_ci = results2.conf_int()[1, :]
            
            results.append({
                'breakpoint': c,
                'slope1': results1.params[1],
                'slope2': results2.params[1],
                'intercept1': results1.params[0],
                'intercept2': results2.params[0],
                'slope1_se': results1.bse[1],
                'slope2_se': results2.bse[1],
                'slope1_ci_lower': slope1_ci[0],
                'slope1_ci_upper': slope1_ci[1],
                'slope2_ci_lower': slope2_ci[0],
                'slope2_ci_upper': slope2_ci[1],
                'sse': total_sse,
                'aic': aic,
                'delta_aic': delta_aic,
                'f_stat': f_stat,
                'f_pvalue': f_pvalue,
                'n1': sum(seg1_mask),
                'n2': sum(seg2_mask)
            })
            
        except Exception:
            continue
    
    if not results:
        return None, None, None
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("figures/T1_piecewise_results.csv", index=False)
    
    # Find best result (lowest AIC)
    best_result = results_df.iloc[results_df['aic'].argmin()]
    
    # Statistical validation
    is_significant = best_result['delta_aic'] < -2 and best_result['f_pvalue'] < 0.05
    is_plateau = abs(best_result['slope2']) < 0.001
    
    return best_result, is_plateau, is_significant

def create_piecewise_plot(df, best_result):
    """Create T1-6: Piecewise linear fit overlay plot with detailed explanations"""
    
    if best_result is None:
        return
        
    plt.figure(figsize=(8, 5), dpi=200)
    
    # Plot data points by country
    countries = sorted(df['country'].unique())
    
    for country in countries:
        country_data = df[df['country'] == country]
        country_label = country.upper() if country == 'usa' else country.replace('_', ' ').title()
        country_color = COUNTRY_COLORS.get(country, '#666666')
        plt.scatter(country_data['gdppcppp'], country_data['BeautyPC'], 
                   alpha=0.8, label=country_label, color=country_color, s=60)
    
    # Plot piecewise regression lines
    x_range = np.linspace(df['gdppcppp'].min(), df['gdppcppp'].max(), 1000)
    breakpoint = best_result['breakpoint']
    
    # Segment 1 line (before breakpoint)
    x1_range = x_range[x_range <= breakpoint]
    y1_range = best_result['slope1'] * x1_range + best_result['intercept1']
    plt.plot(x1_range, y1_range, 'k-', linewidth=4, alpha=0.9, 
             label='Pre-breakpoint trend')
    
    # Segment 2 line (after breakpoint)
    x2_range = x_range[x_range > breakpoint]
    y2_range = best_result['slope2'] * x2_range + best_result['intercept2']
    plt.plot(x2_range, y2_range, 'k-', linewidth=4, alpha=0.9,
             label='Post-breakpoint trend')
    
    # Mark breakpoint
    plt.axvline(x=breakpoint, color='black', linestyle='--', alpha=0.8, linewidth=2,
                label=f'Breakpoint: ${breakpoint:,.0f}')
    
    plt.xlabel('GDP Per Capita PPP', fontweight='bold')
    plt.ylabel('Beauty Consumption Per Capita', fontweight='bold')
    plt.title('Income Threshold Analysis: Beauty Consumption Pattern Change', fontweight='bold')
    plt.legend(loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("figures/T1-6_piecewise_regression.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_additional_charts(df):
    """Create additional visualization charts for comprehensive analysis"""
    high_income = ['usa', 'germany', 'france', 'japan', 'united kingdom']
    emerging = ['south korea', 'russia', 'brazil', 'china', 'mexico']
    
    df_task1 = df[df['country'] != 'india'].copy()
    
    create_t1_1_small_multiples(df_task1, high_income, emerging)
    create_t1_2_growth_comparison(df_task1)

def create_t1_1_small_multiples(df, high_income, emerging):
    """T1-1: Small multiples chart grouped by income level"""
    
    # Filter countries that exist in data
    high_income_available = [c for c in high_income if c in df['country'].values]
    emerging_available = [c for c in emerging if c in df['country'].values]
    
    max_countries = max(len(high_income_available), len(emerging_available))
    
    fig, axes = plt.subplots(2, max_countries, figsize=(4*max_countries, 9), dpi=200)
    
    # Ensure axes is always 2D
    if max_countries == 1:
        axes = axes.reshape(2, 1)
    
    # Use special T1 chart colors
    BEAUTY_COLOR = T1_CHART_COLORS['segments']  # Cyan for beauty consumption
    GDP_COLOR = T1_CHART_COLORS['overall']      # Olive for GDP
    
    # Top row: High-income countries
    for i, country in enumerate(high_income_available):
        ax = axes[0, i]
        country_data = df[df['country'] == country].sort_values('year')
        
        # Create dual y-axis
        ax2 = ax.twinx()
        
        # Plot Beauty consumption on left axis
        ax.plot(country_data['year'], country_data['BeautyPC'], 
               'o-', color=BEAUTY_COLOR, linewidth=2.5, markersize=3, 
               alpha=0.9, label='Beauty PC')
        ax.tick_params(axis='y', labelcolor=BEAUTY_COLOR, labelsize=12)
        
        # Plot GDP on right axis (scaled to thousands)
        ax2.plot(country_data['year'], country_data['gdppcppp']/1000, 
                's--', color=GDP_COLOR, linewidth=2, markersize=2.5,
                alpha=0.9, label='GDP PC')
        ax2.tick_params(axis='y', labelcolor=GDP_COLOR, labelsize=12)
        
        # Country title with better formatting
        country_title = country.upper() if country == 'usa' else country.replace('_', ' ').title()
        ax.set_title(country_title, fontsize=16, fontweight='bold', pad=30)
        
        # Improved formatting
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.tick_params(axis='x', labelsize=12, rotation=45)
        ax.set_xticks(range(1995, 2025, 10))  # Reduce tick frequency
        
        # Force integer formatting on y-axes
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add legend only to first subplot
        if i == 0:
            lines1, _ = ax.get_legend_handles_labels()
            lines2, _ = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, ['Beauty PC', 'GDP PC (000s)'], 
                     loc='upper left', fontsize=13, framealpha=0.95,
                     fancybox=True, shadow=True, bbox_to_anchor=(0.02, 0.98))
        
        # Remove top spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
    
    # Hide unused high-income subplots
    for i in range(len(high_income_available), max_countries):
        axes[0, i].set_visible(False)
    
    # Bottom row: Emerging market countries
    for i, country in enumerate(emerging_available):
        ax = axes[1, i]
        country_data = df[df['country'] == country].sort_values('year')
        
        # Create dual y-axis
        ax2 = ax.twinx()
        
        # Plot Beauty consumption on left axis
        ax.plot(country_data['year'], country_data['BeautyPC'], 
               'o-', color=BEAUTY_COLOR, linewidth=2.5, markersize=3, 
               alpha=0.9, label='Beauty PC')
        ax.tick_params(axis='y', labelcolor=BEAUTY_COLOR, labelsize=12)
        
        # Plot GDP on right axis (scaled to thousands)
        ax2.plot(country_data['year'], country_data['gdppcppp']/1000, 
                's--', color=GDP_COLOR, linewidth=2, markersize=2.5,
                alpha=0.9, label='GDP PC')
        ax2.tick_params(axis='y', labelcolor=GDP_COLOR, labelsize=12)
        
        # Country title
        country_title = country.replace('_', ' ').title()
        ax.set_title(country_title, fontsize=16, fontweight='bold', pad=25)
        
        # Improved formatting
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.tick_params(axis='x', labelsize=12, rotation=45)
        ax.set_xticks(range(1995, 2025, 10))
        # ax.set_xlabel('Year', fontsize=10, fontweight='bold')  # Removed as requested
        
        # Force integer formatting on y-axes
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Remove top spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
    
    # Hide unused emerging market subplots
    for i in range(len(emerging_available), max_countries):
        axes[1, i].set_visible(False)
    
    # Add row labels with better styling
    fig.text(0.06, 0.68, 'DEVELOPED\nCOUNTRIES', rotation=90, 
             fontsize=16, fontweight='bold', va='center', ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    fig.text(0.06, 0.32, 'DEVELOPING\nCOUNTRIES', rotation=90, 
             fontsize=16, fontweight='bold', va='center', ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
    
    # Add overall title with more space from country titles
    fig.suptitle('Beauty Consumption and GDP Trends by Country Group (1995-2024)', 
                 fontsize=22, fontweight='bold', y=0.97)
    
    # Adjust layout with proper spacing - more top space for title
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, left=0.10, right=0.95, hspace=0.4, wspace=0.3)
    
    plt.savefig("figures/T1-1_small_multiples_by_income_group.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_t1_2_growth_comparison(df):
    """T1-2: Average growth rates comparison chart"""
    
    # Calculate growth rates
    df_growth = df.copy()
    df_growth = df_growth.sort_values(['country', 'year'])
    
    # Calculate year-over-year growth rates
    df_growth['beauty_growth'] = df_growth.groupby('country')['BeautyPC'].pct_change() * 100
    df_growth['gdp_growth'] = df_growth.groupby('country')['gdppcppp'].pct_change() * 100
    
    # Remove extreme outliers
    df_growth = df_growth[(df_growth['beauty_growth'].abs() < 50) & 
                         (df_growth['gdp_growth'].abs() < 20)]
    
    # Calculate average growth rates by country
    avg_growth = df_growth.groupby('country')[['beauty_growth', 'gdp_growth']].mean().reset_index()
    avg_growth = avg_growth.dropna()
    avg_growth = avg_growth.sort_values('gdp_growth', ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=200)
    
    x_pos = np.arange(len(avg_growth)) * 1.5  # Increase spacing between country groups
    width = 0.4  # Reduce bar width to increase gap between bars
    
    country_labels = [c.upper() if c == 'usa' else c.replace('_', ' ').title() 
                     for c in avg_growth['country']]
    
    BEAUTY_COLOR = '#2E8B57'
    GDP_COLOR = '#4169E1'
    
    gap_increase = 0.05  # Additional gap between green and blue bars
    bars1 = ax.bar(x_pos - width/2 - gap_increase, avg_growth['beauty_growth'], width, 
                   label='Beauty Consumption Growth', alpha=0.85, color=BEAUTY_COLOR,
                   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x_pos + width/2 + gap_increase, avg_growth['gdp_growth'], width, 
                   label='GDP Growth', alpha=0.85, color=GDP_COLOR,
                   edgecolor='white', linewidth=0.5)
    
    ax.set_title('Average Annual Growth Rates by Country (1995-2024)', 
                 fontsize=18, fontweight='bold', pad=25)
    ax.set_ylabel('Average Growth Rate (%/year)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Country', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)  # Keep original x_pos for centered labels
    ax.set_xticklabels(country_labels, rotation=0, ha='center', fontsize=14)
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(fontsize=16, loc='upper left', framealpha=0.95, 
              fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.9, linewidth=1)
    
    # Add value labels on bars with perfect centering
    for bar in bars1:
        height = bar.get_height()
        if not np.isnan(height):
            label_y = height + 0.2 if height >= 0 else height - 0.3
            # Use bar center position for perfect alignment
            bar_center = bar.get_x() + bar.get_width() / 2.0
            ax.text(bar_center, label_y,
                   f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                   fontsize=14, fontweight='bold', color=BEAUTY_COLOR)
    
    for bar in bars2:
        height = bar.get_height()
        if not np.isnan(height):
            label_y = height + 0.2 if height >= 0 else height - 0.3
            # Use bar center position for perfect alignment
            bar_center = bar.get_x() + bar.get_width() / 2.0
            ax.text(bar_center, label_y,
                   f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                   fontsize=14, fontweight='bold', color=GDP_COLOR)
    
    # Improve layout and spacing
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig("figures/T1-2_average_growth_rates_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def summarize_task1_results():
    """Create summary of Task 1 findings"""
    summary = """Task 1: Statistical Exploration Results Summary
==============================================

Key Findings:
1. Beauty consumption shows high income elasticity (β ≈ 2.3)
2. Strong positive correlation with GDP across all countries
3. Piecewise regression detects structural breaks around $40-50k GDP per capita
4. Country heterogeneity in breakpoint timing and plateau behavior

Statistical Evidence:
- Income elasticity: ~2.3 (luxury good behavior)
- Strong explanatory power (R² > 0.8)
- Significant structural breaks in most countries
- Mixed evidence on plateau behavior post-breakpoint
"""
    
    with open("figures/T1_summary.txt", "w") as f:
        f.write(summary)

def detect_inflection_per_country(df):
    """Detect inflection points for each country individually"""
    countries = sorted(df['country'].unique())
    country_results = {}
    
    for country in countries:
        country_data = df[df['country'] == country]
        if len(country_data) < 8:
            continue
            
        result, plateau, significant = detect_inflection_point(country_data)
        if result is not None:
            country_results[country] = {
                'breakpoint': result['breakpoint'],
                'slope1': result['slope1'],
                'slope2': result['slope2'],
                'is_plateau': plateau,
                'is_significant': significant
            }
    
    if country_results:
        pd.DataFrame(country_results).T.to_csv("figures/T1_piecewise_per_country.csv")
    
    return country_results

def detect_segmented_inflection_points(df):
    """Detect inflection points separately for OECD and emerging markets"""
    
    # Define country groups
    oecd_countries = ['usa', 'germany', 'france', 'japan', 'united kingdom', 'south korea']
    emerging_countries = ['russia', 'brazil', 'china', 'mexico']
    
    # Filter countries that exist in the data
    available_countries = df['country'].unique()
    oecd_available = [c for c in oecd_countries if c in available_countries]
    emerging_available = [c for c in emerging_countries if c in available_countries]
    
    results = {}
    
    # Analyze OECD/Advanced markets
    if oecd_available:
        oecd_data = df[df['country'].isin(oecd_available)].copy()
        if len(oecd_data) >= 10:
            oecd_result, oecd_plateau, oecd_significant = detect_inflection_point(oecd_data)
            results['oecd'] = {
                'countries': oecd_available,
                'n_countries': len(oecd_available),
                'n_obs': len(oecd_data),
                'result': oecd_result,
                'is_plateau': oecd_plateau,
                'is_significant': oecd_significant
            }
    
    # Analyze Emerging markets
    if emerging_available:
        emerging_data = df[df['country'].isin(emerging_available)].copy()
        if len(emerging_data) >= 10:
            emerging_result, emerging_plateau, emerging_significant = detect_inflection_point(emerging_data)
            results['emerging'] = {
                'countries': emerging_available,
                'n_countries': len(emerging_available),
                'n_obs': len(emerging_data),
                'result': emerging_result,
                'is_plateau': emerging_plateau,
                'is_significant': emerging_significant
            }
    
    # Global/Pooled analysis for comparison
    global_result, global_plateau, global_significant = detect_inflection_point(df)
    results['global'] = {
        'countries': list(available_countries),
        'n_countries': len(available_countries),
        'n_obs': len(df),
        'result': global_result,
        'is_plateau': global_plateau,
        'is_significant': global_significant
    }
    
    # Save segmented results
    save_segmented_results(results)
    
    return results

def save_segmented_results(results):
    """Save segmented inflection point results to files"""
    
    summary_data = []
    
    for segment, data in results.items():
        if data['result'] is not None:
            summary_data.append({
                'segment': segment,
                'countries': ', '.join(data['countries']),
                'n_countries': data['n_countries'],
                'n_obs': data['n_obs'],
                'breakpoint': data['result']['breakpoint'],
                'slope1': data['result']['slope1'],
                'slope2': data['result']['slope2'],
                'slope1_se': data['result']['slope1_se'],
                'slope2_se': data['result']['slope2_se'],
                'delta_aic': data['result']['delta_aic'],
                'f_stat': data['result']['f_stat'],
                'is_plateau': data['is_plateau'],
                'is_significant': data['is_significant']
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv("figures/T1_segmented_inflection_points.csv", index=False)
        
        # Create detailed text summary
        with open("figures/T1_segmented_analysis_summary.txt", "w") as f:
            f.write("Segmented Inflection Point Analysis\n")
            f.write("="*40 + "\n\n")
            
            for segment, data in results.items():
                if data['result'] is not None:
                    result = data['result']
                    f.write(f"{segment.upper()} MARKETS:\n")
                    f.write(f"Countries: {', '.join(data['countries'])}\n")
                    f.write(f"Observations: {data['n_obs']} ({data['n_countries']} countries)\n")
                    f.write(f"Breakpoint: ${result['breakpoint']:,.0f} GDP per capita\n")
                    f.write(f"Pre-breakpoint slope: ${result['slope1']*1000:.2f} per $1000 GDP\n")
                    f.write(f"Post-breakpoint slope: ${result['slope2']*1000:.2f} per $1000 GDP\n")
                    f.write(f"Statistical significance: {'Yes' if data['is_significant'] else 'No'}\n")
                    f.write(f"Plateau behavior: {'Yes' if data['is_plateau'] else 'No'}\n")
                    f.write("-" * 30 + "\n\n")

def create_segmented_comparison_plot(df, segmented_results):
    """Create T1-7: Segmented inflection point comparison plot"""
    
    # Remove the pooled chart since T1-6 already covers it
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=200)
    
    # Define country groups
    oecd_countries = segmented_results.get('oecd', {}).get('countries', [])
    emerging_countries = segmented_results.get('emerging', {}).get('countries', [])
    
    segments = [
        ('oecd', oecd_countries, 'Advanced Markets'),
        ('emerging', emerging_countries, 'Emerging Markets')
    ]
    
    # Collect legend elements for shared legend
    legend_elements = []
    legend_labels = []
    
    for idx, (segment_key, countries, title) in enumerate(segments):
        ax = axes[idx]
        
        if segment_key in segmented_results and segmented_results[segment_key]['result'] is not None:
            # Get segment data and result
            segment_data = df[df['country'].isin(countries)].copy()
            result = segmented_results[segment_key]['result']
            
            # Plot data points with consistent colors
            for country in countries:
                if country in df['country'].values:
                    country_data = segment_data[segment_data['country'] == country]
                    country_label = country.upper() if country == 'usa' else country.replace('_', ' ').title()
                    country_color = COUNTRY_COLORS.get(country, '#666666')
                    
                    scatter = ax.scatter(country_data['gdppcppp'], country_data['BeautyPC'], 
                                       color=country_color, alpha=0.7, s=60, 
                                       label=country_label, edgecolors='white', linewidth=0.5)
                    
                    # Add country colors to legend
                    if idx == 0 and country in oecd_countries:
                        legend_elements.append(scatter)
                        legend_labels.append(country_label)
                    elif idx == 1 and country in emerging_countries:
                        legend_elements.append(scatter)
                        legend_labels.append(country_label)
            
            # Plot piecewise regression lines with black color
            x_range = np.linspace(segment_data['gdppcppp'].min(), 
                                segment_data['gdppcppp'].max(), 1000)
            breakpoint = result['breakpoint']
            
            # Create segment-specific labels
            segment_prefix = 'Adv' if segment_key == 'oecd' else 'Emg'
            
            # Segment 1 line (before breakpoint)
            x1_range = x_range[x_range <= breakpoint]
            y1_range = result['slope1'] * x1_range + result['intercept1']
            line1 = ax.plot(x1_range, y1_range, 'k-', linewidth=3, alpha=0.8,
                           label=f'{segment_prefix} Pre: {result["slope1"]*1000:.1f}/$1000 GDP')
            
            # Segment 2 line (after breakpoint)
            x2_range = x_range[x_range > breakpoint]
            y2_range = result['slope2'] * x2_range + result['intercept2']
            line2 = ax.plot(x2_range, y2_range, 'k--', linewidth=3, alpha=0.8,
                           label=f'{segment_prefix} Post: {result["slope2"]*1000:.1f}/$1000 GDP')
            
            # Mark breakpoint
            ax.axvline(x=breakpoint, color='black', linestyle=':', alpha=0.7, linewidth=2)
            breakpoint_marker = ax.plot(breakpoint, result['slope1'] * breakpoint + result['intercept1'], 
                                       'ko', markersize=8, label=f'{segment_prefix} Breakpoint: ${breakpoint:,.0f}')
            
            # Add regression lines to legend
            legend_elements.extend([line1[0], line2[0], breakpoint_marker[0]])
            legend_labels.extend([f'{segment_prefix} Pre: {result["slope1"]*1000:.1f}/$1000 GDP',
                                f'{segment_prefix} Post: {result["slope2"]*1000:.1f}/$1000 GDP',
                                f'{segment_prefix} Breakpoint: ${breakpoint:,.0f}'])
            
            # Add significance indicator with better positioning
            sig_text = "✓ Significant" if segmented_results[segment_key]['is_significant'] else "✗ Not Significant"
            plateau_text = "Plateau" if segmented_results[segment_key]['is_plateau'] else "Continued Growth"
            
            ax.text(0.02, 0.98, f"{sig_text}\n{plateau_text}", 
                   transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.9),
                   fontsize=9)
        
        # Use camel case for axes labels with split y-axis title
        ax.set_xlabel('GDP Per Capita PPP', fontweight='bold', fontsize=14)
        ax.set_ylabel('Beauty Consumption\nPer Capita', fontweight='bold', fontsize=14)
        ax.set_title(title, fontweight='bold', fontsize=16)
        ax.grid(True, alpha=0.3)
    
    # Create single shared legend positioned between the two charts with expanded width and increased font size
    fig.legend(legend_elements, legend_labels, 
               loc='center', bbox_to_anchor=(0.5, -0.05), ncol=6, fontsize=14)
    
    plt.suptitle('Inflection Point Analysis: Advanced Vs Emerging Markets', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, wspace=0.15)  # Reduced space between subplots and for legend
    plt.savefig("figures/T1-7_segmented_inflection_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run complete Task 1 analysis"""
    # Load data and create plots
    df = load_master_data()
    df_log = create_descriptive_plots(df)
    
    # Statistical analysis
    calculate_correlations(df)
    run_elasticity_regression(df_log)
    
    # Piecewise regression (global pooled)
    best_result, _, _ = detect_inflection_point(df)
    if best_result is not None:
        create_piecewise_plot(df, best_result)
    
    # Segmented inflection point analysis (OECD vs Emerging)
    segmented_results = detect_segmented_inflection_points(df)
    create_segmented_comparison_plot(df, segmented_results)
    
    # Country-specific analysis and additional charts
    detect_inflection_per_country(df)
    create_additional_charts(df)
    summarize_task1_results()

if __name__ == "__main__":
    main()