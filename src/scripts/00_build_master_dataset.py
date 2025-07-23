import pandas as pd
import numpy as np
from pathlib import Path
import logging

def pearsonr(x, y):
    """Simple Pearson correlation coefficient calculation"""
    x, y = np.array(x), np.array(y)
    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    if len(x) < 2:
        return 0.0, 1.0
    
    # Calculate correlation
    r = np.corrcoef(x, y)[0, 1]
    
    # Simple p-value approximation for large n
    n = len(x)
    if n > 10:
        t_stat = r * np.sqrt((n-2)/(1-r**2)) if abs(r) < 0.999 else 0
        p_value = 0.05 if abs(t_stat) > 2 else 0.1  # Rough approximation
    else:
        p_value = 0.1
    
    return r, p_value

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_worldbank_data():
    """Load World Bank data: GDP per capita, consumption, population"""
    base_path = Path("data/raw/worldbank")
    
    # GDP per capita PPP
    gdp_path = base_path / "GDP per capita" / "API_NY.GDP.PCAP.PP.KD_DS2_en_csv_v2_20423.csv"
    gdp_df = pd.read_csv(gdp_path, skiprows=4)
    gdp_df = gdp_df.melt(id_vars=['Country Name', 'Country Code'], 
                         value_vars=[str(y) for y in range(1990, 2025)],
                         var_name='year', value_name='gdppcppp')
    gdp_df['year'] = gdp_df['year'].astype(int)
    gdp_df.columns = gdp_df.columns.str.lower().str.replace(' ', '_')
    
    # Household consumption per capita
    cons_path = base_path / "Consumption" / "API_NE.CON.PRVT.PC.KD_DS2_en_csv_v2_30892.csv"
    cons_df = pd.read_csv(cons_path, skiprows=4)
    cons_df = cons_df.melt(id_vars=['Country Name', 'Country Code'],
                          value_vars=[str(y) for y in range(1990, 2025)],
                          var_name='year', value_name='ne_con_prvt_pc_kd')
    cons_df['year'] = cons_df['year'].astype(int)
    cons_df.columns = cons_df.columns.str.lower().str.replace(' ', '_')
    
    # Population
    pop_path = base_path / "population" / "API_SP.POP.TOTL_DS2_en_csv_v2_38144.csv"
    pop_df = pd.read_csv(pop_path, skiprows=4)
    pop_df = pop_df.melt(id_vars=['Country Name', 'Country Code'],
                         value_vars=[str(y) for y in range(1990, 2025)],
                         var_name='year', value_name='population')
    pop_df['year'] = pop_df['year'].astype(int)
    pop_df.columns = pop_df.columns.str.lower().str.replace(' ', '_')
    
    # Merge all World Bank data
    wb_data = gdp_df.merge(cons_df[['country_name', 'year', 'ne_con_prvt_pc_kd']], 
                          on=['country_name', 'year'], how='outer')
    wb_data = wb_data.merge(pop_df[['country_name', 'year', 'population']], 
                           on=['country_name', 'year'], how='outer')
    
    # Standardize country names with mappings (expanded for more countries)
    country_mapping = {
        'united states': 'usa',
        'korea, rep.': 'south korea', 
        'china': 'china',
        'japan': 'japan',
        'india': 'india',
        'germany': 'germany',
        'united kingdom': 'united kingdom',
        'france': 'france',
        'brazil': 'brazil',
        'mexico': 'mexico',
        'russian federation': 'russia'
    }
    
    wb_data['country'] = wb_data['country_name'].str.lower().str.strip().map(country_mapping)
    wb_data = wb_data.dropna(subset=['country'])  # Remove unmapped countries
    
    # Debug info
    logger.info(f"World Bank data shape: {wb_data.shape}")
    logger.info(f"Countries in WB after mapping: {sorted(wb_data['country'].unique())}")
    
    return wb_data

def load_comtrade_data():
    """Load UN Comtrade data for HS 3303-3307"""
    import csv
    
    comtrade_path = Path("data/raw/comtrade")
    comtrade_files = list(comtrade_path.glob("*.csv"))
    
    all_rows = []
    for file in comtrade_files:
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)
    
    comtrade_df = pd.DataFrame(all_rows)
    
    # Convert numeric columns
    numeric_cols = ['refYear', 'primaryValue']
    for col in numeric_cols:
        if col in comtrade_df.columns:
            comtrade_df[col] = pd.to_numeric(comtrade_df[col], errors='coerce')
    
    # Create working columns with country name mapping (expanded)
    comtrade_mapping = {
        'usa': 'usa',
        'rep. of korea': 'south korea', 
        'china': 'china',
        'japan': 'japan',
        'india': 'india',
        'germany': 'germany',
        'united kingdom': 'united kingdom',
        'france': 'france',
        'brazil': 'brazil',
        'mexico': 'mexico',
        'russian federation': 'russia'
    }
    
    comtrade_df['country'] = (comtrade_df['reporterDesc'].str.lower().str.strip()
                             .map(comtrade_mapping))
    comtrade_df['year'] = comtrade_df['refYear']
    comtrade_df['tradeflow'] = comtrade_df['flowCode']  
    comtrade_df['tradevalue'] = comtrade_df['primaryValue'].fillna(0)
    
    # Filter to mapped countries only
    comtrade_df = comtrade_df.dropna(subset=['country'])
    
    # Debug: print some info
    logger.info(f"Comtrade data shape: {comtrade_df.shape}")
    logger.info(f"Countries in Comtrade after mapping: {sorted(comtrade_df['country'].unique())}")
    logger.info(f"Years range: {comtrade_df['year'].min()}-{comtrade_df['year'].max()}")
    
    return comtrade_df

def load_cpi_data():
    """Load and process CPI data from FRED"""
    cpi_df = pd.read_csv("data/raw/CPIAUCSL.csv")
    cpi_df['date'] = pd.to_datetime(cpi_df['observation_date'])
    cpi_df['year'] = cpi_df['date'].dt.year
    
    # Create annual CPI (average of 12 months)
    annual_cpi = cpi_df.groupby('year')['CPIAUCSL'].mean().reset_index()
    
    # Rebase to 2015 = 100
    cpi_2015 = annual_cpi[annual_cpi['year'] == 2015]['CPIAUCSL'].iloc[0]
    annual_cpi['cpi_2015base'] = (annual_cpi['CPIAUCSL'] / cpi_2015) * 100
    
    return annual_cpi[['year', 'cpi_2015base']]

def build_beauty_proxy(comtrade_df):
    """Build beauty proxy using three formulas and select best"""
    
    # First extract re-exports before flow mapping
    re_export_agg = (comtrade_df[comtrade_df['tradeflow'] == 'RX']
                     .groupby(['country', 'year'])['tradevalue']
                     .sum().reset_index()
                     .rename(columns={'tradevalue': 're_export'}))
    
    # Map flow types using flowcode - complete mapping per plan
    flow_mapping = {
        'M': 'inbound',    # Import
        'RM': 'inbound',   # Re-import
        'IG': 'inbound',   # Import of goods after outward processing
        'X': 'outbound',   # Export
        'RX': 'outbound',  # Re-export  
        'EG': 'outbound',  # Export of goods after inward processing
        'EO': 'outbound',  # Export of goods for outward processing
    }
    
    comtrade_df['flow_type'] = comtrade_df['tradeflow'].map(flow_mapping)
    comtrade_df['flow_type'] = comtrade_df['flow_type'].fillna('other')
    
    # Aggregate by country-year-flow_type
    trade_agg = (comtrade_df.groupby(['country', 'year', 'flow_type'])['tradevalue']
                 .sum().reset_index())
    trade_pivot = trade_agg.pivot_table(index=['country', 'year'], 
                                       columns='flow_type', 
                                       values='tradevalue', 
                                       fill_value=0).reset_index()
    
    # Merge re-export data
    trade_pivot = trade_pivot.merge(re_export_agg, on=['country', 'year'], how='left')
    trade_pivot['re_export'] = trade_pivot['re_export'].fillna(0)
    
    # Calculate three proxy candidates per plan specifications
    inbound = trade_pivot.get('inbound', 0)
    outbound = trade_pivot.get('outbound', 0)
    re_export = trade_pivot['re_export']
    
    trade_pivot['p1_net_inbound'] = inbound - re_export
    trade_pivot['p2_half_total'] = 0.5 * (inbound + outbound)
    trade_pivot['p3_balanced_net'] = inbound - re_export + 0.5 * outbound
    
    return trade_pivot

def select_best_proxy(trade_pivot, wb_data):
    """Select best beauty proxy based on correlation with GDP and consumption"""
    
    # Merge with WB data for ALL available countries (no hardcoded benchmark list)
    merged = trade_pivot.merge(wb_data[['country', 'year', 'gdppcppp', 'ne_con_prvt_pc_kd']], 
                              on=['country', 'year'], how='inner')
    merged = merged.dropna(subset=['gdppcppp', 'ne_con_prvt_pc_kd'])
    
    proxies = ['p1_net_inbound', 'p2_half_total', 'p3_balanced_net']
    correlations = {}
    detailed_correlations = {}
    
    print("\\nProxy Selection Analysis:")
    print("=" * 50)
    print(f"Using {len(merged['country'].unique())} countries: {sorted(merged['country'].unique())}")
    print(f"Total observations: {len(merged)}")
    
    for proxy in proxies:
        try:
            r_gdp, p_gdp = pearsonr(merged[proxy], merged['gdppcppp'])
            r_cons, p_cons = pearsonr(merged[proxy], merged['ne_con_prvt_pc_kd'])
            mean_corr = np.mean([abs(r_gdp), abs(r_cons)])
            
            correlations[proxy] = mean_corr
            detailed_correlations[proxy] = {
                'r_gdp': r_gdp,
                'p_gdp': p_gdp,
                'r_cons': r_cons,
                'p_cons': p_cons,
                'mean_abs_r': mean_corr
            }
            
            print(f"\\n{proxy}:")
            print(f"  vs GDP per capita:     r = {r_gdp:6.3f} (p = {p_gdp:.3f})")
            print(f"  vs HH consumption:     r = {r_cons:6.3f} (p = {p_cons:.3f})")
            print(f"  Mean |r|:              {mean_corr:6.3f}")
            
        except Exception as e:
            correlations[proxy] = 0
            print(f"\\n{proxy}: Error calculating correlations - {e}")
    
    best_proxy = max(correlations, key=correlations.get)
    print(f"\\n" + "=" * 50)
    print(f"SELECTED PROXY: {best_proxy}")
    print(f"Mean correlation: {correlations[best_proxy]:.3f}")
    print("=" * 50)
    
    trade_pivot['beauty_value_nominal_usd'] = trade_pivot[best_proxy]
    trade_pivot['proxy_used'] = best_proxy
    
    return trade_pivot, best_proxy

def handle_missing_data(wb_data):
    """Handle missing data with interpolation and flagging"""
    
    print("\\nMissing Data Analysis:")
    print("=" * 30)
    
    # Check missing data by country and variable
    key_vars = ['gdppcppp', 'ne_con_prvt_pc_kd', 'population']
    missing_summary = wb_data.groupby('country')[key_vars].apply(lambda x: x.isnull().sum())
    
    if missing_summary.sum().sum() > 0:
        print("Missing observations by country and variable:")
        print(missing_summary[missing_summary.sum(axis=1) > 0])
        
        # Add flag columns for interpolated values
        for var in key_vars:
            wb_data[f'{var}_interpolated'] = False
        
        # Linear interpolation by country
        wb_data = wb_data.sort_values(['country', 'year'])
        
        for country in wb_data['country'].unique():
            country_mask = wb_data['country'] == country
            country_data = wb_data.loc[country_mask].copy()
            
            for var in key_vars:
                # Only interpolate if there are sufficient surrounding values
                if country_data[var].isnull().sum() > 0:
                    # Flag which values will be interpolated
                    interpolate_mask = country_data[var].isnull()
                    
                    # Interpolate
                    interpolated_values = country_data[var].interpolate(method='linear')
                    
                    # Update main dataframe
                    wb_data.loc[country_mask, var] = interpolated_values
                    wb_data.loc[country_mask & interpolate_mask, f'{var}_interpolated'] = True
                    
                    if interpolate_mask.sum() > 0:
                        print(f"  {country}: Interpolated {interpolate_mask.sum()} values for {var}")
    else:
        print("No missing data detected in key World Bank variables.")
        # Still add flag columns for consistency
        for var in key_vars:
            wb_data[f'{var}_interpolated'] = False
    
    print("Missing data handling completed.\\n")
    return wb_data

def create_master_dataset():
    """Create the master dataset following steps 0.1-0.6"""
    
    logger.info("Loading World Bank data...")
    wb_data = load_worldbank_data()
    
    logger.info("Handling missing data...")
    wb_data = handle_missing_data(wb_data)
    
    logger.info("Loading Comtrade data...")
    comtrade_df = load_comtrade_data()
    
    logger.info("Loading CPI data...")
    cpi_data = load_cpi_data()
    
    logger.info("Building beauty proxy...")
    trade_pivot = build_beauty_proxy(comtrade_df)
    
    logger.info("Selecting best proxy...")
    trade_pivot, chosen_proxy = select_best_proxy(trade_pivot, wb_data)
    
    logger.info("Creating master panel...")
    # Merge all data
    master = wb_data.merge(trade_pivot[['country', 'year', 'beauty_value_nominal_usd']], 
                          on=['country', 'year'], how='outer')
    master = master.merge(cpi_data, on='year', how='left')
    
    # Deflate to 2015 USD first, then per capita
    master['beauty_value_const2015'] = (master['beauty_value_nominal_usd'] * 
                                       100 / master['cpi_2015base'])
    
    # Create derived metrics
    master['beautypc'] = master['beauty_value_const2015'] / master['population']
    master['beautyshare'] = master['beautypc'] / master['ne_con_prvt_pc_kd']
    
    # Log variables (avoid log(0))
    master['ln_gdppc'] = np.log(master['gdppcppp'].fillna(1))
    master['ln_beautypc'] = np.log(master['beautypc'].fillna(0) + 1)
    
    # YoY growth
    master = master.sort_values(['country', 'year'])
    master['beautypc_growth'] = master.groupby('country')['beautypc'].pct_change(fill_method=None)
    
    # 5-year CAGR (forward-looking)
    def calc_cagr(series, periods=5):
        return (series.shift(-periods) / series) ** (1/periods) - 1
    
    master['beautypc_cagr_5y'] = master.groupby('country')['beautypc'].transform(lambda x: calc_cagr(x))
    
    # Add metadata
    master['chosen_proxy'] = chosen_proxy
    
    # Check for missing data
    missing_counts = master.groupby('country')[['gdppcppp', 'ne_con_prvt_pc_kd', 'beautypc']].apply(lambda x: x.isnull().sum())
    if missing_counts.sum().sum() > 0:
        print("Missing data by country:")
        print(missing_counts)
    
    # Clean and filter
    master = master.dropna(subset=['beautypc', 'gdppcppp'])
    master = master[master['year'] >= 1995]  # Focus on recent decades
    
    # UAE already excluded from raw data
    
    # Save master dataset
    output_path = "data/processed/beauty_income_panel.parquet"
    master.to_parquet(output_path, index=False)
    logger.info(f"Master dataset saved to {output_path}")
    logger.info(f"Dataset shape: {master.shape}")
    logger.info(f"Countries: {sorted(master['country'].unique())}")
    
    return master

if __name__ == "__main__":
    master = create_master_dataset()
    print("\nMaster dataset summary:")
    print(master.groupby('country')[['year', 'beautypc', 'gdppcppp']].agg({
        'year': ['min', 'max'],
        'beautypc': ['count', 'mean'],
        'gdppcppp': 'mean'
    }).round(2))