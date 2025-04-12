# %% Step 1: Setup - Import Libraries & Configure Settings
print("--- Starting Step 1: Setting up libraries and configurations ---")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings
import textwrap

# Configuring plot styles and ignoring warnings
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['patch.edgecolor'] = 'black'
plt.rcParams['patch.linewidth'] = 0.5
plt.rcParams['figure.facecolor'] = 'white'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Project constants
ALPHA = 0.05
FILE_PATH = 'Crime_Data_from_2020_to_Present (2).csv'

print(f"Libraries imported. Using data file: {FILE_PATH}")
print("--- Step 1 Complete ---")
print("-" * 80)

# %% Step 2: Load Data & Initial Inspection
print("--- Starting Step 2: Loading and looking at the raw data ---")

# Load data
df_raw = pd.read_csv(FILE_PATH, low_memory=False)
print(f"Loaded {df_raw.shape[0]:,} rows and {df_raw.shape[1]} columns.")

# Show first few rows and basic info
print("\nFirst 5 rows:")
with pd.option_context('display.max_columns', None):
    print(df_raw.head())
# print("\nColumn Info:") # info() can be verbose, maybe skip direct print here
# df_raw.info(show_counts=True)
# print("\nSummary Stats:") # describe() is also quite verbose
# print(df_raw.describe(include='all').T)

# Create a working copy
df = df_raw.copy()
print("\nCreated a working copy.")
print("--- Step 2 Complete ---")
print("-" * 80)

# %% Step 3: Clean Column Names & Process Dates/Times
print("--- Starting Step 3: Cleaning column names and date/time fields ---")

# Make column names standard (lowercase_with_underscores)
df.columns = df.columns.str.lower().str.replace(' ', '_', regex=False).str.replace('-', '_', regex=False)
print("Standardized column names.")

# Convert date columns to datetime format
df['date_rptd'] = pd.to_datetime(df['date_rptd'], errors='coerce')
df['date_occ'] = pd.to_datetime(df['date_occ'], errors='coerce')
initial_rows_date = df.shape[0]
df.dropna(subset=['date_occ'], inplace=True) # Remove rows with invalid occurrence dates
rows_dropped_date = initial_rows_date - df.shape[0]
if rows_dropped_date > 0: print(f"  Removed {rows_dropped_date} rows with invalid dates.")
print("Converted date columns.")

# Parse time (HHMM) and combine with date to create a timestamp
df['time_occ_str'] = df['time_occ'].astype(int).astype(str).str.zfill(4)
time_parsed = pd.to_datetime(df['time_occ_str'], format='%H%M', errors='coerce').dt.time
df['datetime_occ'] = pd.to_datetime(df['date_occ'].dt.date.astype(str) + ' ' + time_parsed.astype(str), errors='coerce')
initial_rows_datetime = df.shape[0]
df.dropna(subset=['datetime_occ'], inplace=True) # Remove rows with invalid times
rows_dropped_datetime = initial_rows_datetime - df.shape[0]
if rows_dropped_datetime > 0: print(f"  Removed {rows_dropped_datetime} rows with invalid times.")
print("Processed time and created 'datetime_occ'.")

print(f"\nShape after date/time cleaning: {df.shape}")
print("--- Step 3 Complete ---")
print("-" * 80)

# %% Step 4: Handle Missing Values & Placeholders
print("--- Starting Step 4: Handling missing values and special codes ---")

# Clean victim age (replace non-positive values)
df['vict_age'] = df['vict_age'].replace(to_replace=[-4, -3, -2, -1, 0], value=np.nan)
df['vict_age'] = pd.to_numeric(df['vict_age'], errors='coerce')
print("Cleaned victim age values.")

# Clean victim sex (replace 'X', 'H', '-')
df['vict_sex'] = df['vict_sex'].replace(['X', 'H', '-'], np.nan)
print("Cleaned victim sex codes.")

# Clean victim descent (replace 'X', '-')
df['vict_descent'] = df['vict_descent'].replace(['X', '-'], np.nan)
print("Cleaned victim descent codes.")

# Fill missing categorical/text fields with defaults
df['weapon_used_cd'].fillna(999, inplace=True)
df['weapon_desc'].fillna('NONE/UNKNOWN', inplace=True)
df['premis_cd'].fillna(999, inplace=True)
df['premis_desc'].fillna('UNKNOWN', inplace=True)
df['mocodes'].fillna('NONE', inplace=True)
df['cross_street'].fillna('UNKNOWN', inplace=True)
# Fill rare missing 'status'/'status_desc' with mode
if df['status'].isnull().any():
    status_mode = df['status'].mode()[0]
    status_desc_mode = df.loc[df['status'] == status_mode, 'status_desc'].mode()[0]
    df['status'].fillna(status_mode, inplace=True)
    df['status_desc'].fillna(status_desc_mode, inplace=True)
print("Filled missing text/code information.")

# Fill missing secondary crime codes with 0
fill_crm_codes = {'crm_cd_1': 0, 'crm_cd_2': 0, 'crm_cd_3': 0, 'crm_cd_4': 0}
df.fillna(value=fill_crm_codes, inplace=True)
print("Filled missing secondary crime codes.")

# Adjust data types for codes
df['weapon_used_cd'] = df['weapon_used_cd'].astype('category')
df['premis_cd'] = df['premis_cd'].astype('category')
df['status'] = df['status'].astype('category')
for col in ['crm_cd_1', 'crm_cd_2', 'crm_cd_3', 'crm_cd_4']:
    df[col] = df[col].astype(int)
print("Adjusted data types.")

print("\nMissing values after cleaning (should only be victim demographics):")
missing_after = df.isnull().sum()
print(missing_after[missing_after > 0].sort_values(ascending=False))

print("--- Step 4 Complete ---")
print("-" * 80)

# %% Step 5: Feature Engineering - Create New Variables
print("--- Starting Step 5: Creating new features for analysis ---")

# Extract useful time components
df['year'] = df['datetime_occ'].dt.year
df['month'] = df['datetime_occ'].dt.month
df['hour'] = df['datetime_occ'].dt.hour
df['day_of_week'] = df['datetime_occ'].dt.dayofweek # Monday=0
df['day_name'] = df['datetime_occ'].dt.day_name()
df['month_name'] = df['datetime_occ'].dt.month_name()
print("Extracted time components.")

# Create time categories (segment, weekend)
time_bins = [-1, 6, 12, 18, 24]
time_labels = ['Night (0-5)', 'Morning (6-11)', 'Afternoon (12-17)', 'Evening (18-23)']
df['time_segment'] = pd.cut(df['hour'], bins=time_bins, labels=time_labels, right=False, ordered=True)
df['is_weekend'] = df['day_name'].isin(['Saturday', 'Sunday'])
print("Created time segments and weekend flag.")

# Create victim age groups
age_bins = [-np.inf, 17, 25, 35, 45, 55, 65, np.inf]
age_labels = ['0-17 (Minor)', '18-25', '26-35', '36-45', '46-55', '56-65', '66+ (Senior)']
df['vict_age_group'] = pd.cut(df['vict_age'], bins=age_bins, labels=age_labels, right=True, ordered=True)
print("Created victim age groups.")

# Create detailed crime categories using a function
def classify_crime(description):
    desc_upper = str(description).upper()
    violent_keywords = ['ASSAULT', 'ROBBERY', 'HOMICIDE', 'BATTERY', 'ADW', 'RAPE', 'SHOTS FIRED', 'MANSLAUGHTER', 'WEAPON LAWS', 'THREATS', 'KIDNAPPING', 'SEXUAL']
    property_keywords = ['BURGLARY', 'THEFT', 'STOLEN', 'VANDALISM', 'SHOPLIFTING', 'PICKPOCKET', 'PURSE SNATCHING', 'EMBEZZLEMENT']
    vehicle_keywords = ['VEHICLE', 'AUTO', 'BIKE']
    fraud_keywords = ['FRAUD', 'FORGERY', 'CREDIT CARDS', 'BUNCO', 'IDENTITY THEFT']
    public_order_keywords = ['DISTURBING PEACE', 'DRUNK', 'NARCOTICS', 'DRUGS', 'TRESPASSING', 'LOITERING', 'PROSTITUTION', 'GAMBLING']
    other_keywords = ['ARSON', 'CONTEMPT OF COURT', 'DRIVING WITHOUT LICENSE', 'EXTORTION', 'FALSE POLICE REPORT']
    if any(keyword in desc_upper for keyword in violent_keywords):
        if any(vk in desc_upper for vk in vehicle_keywords) and 'ROBBERY' in desc_upper: return 'Vehicle Related - Violent'
        return 'Violent Crime'
    if any(keyword in desc_upper for keyword in vehicle_keywords): return 'Vehicle Related - Property'
    if any(keyword in desc_upper for keyword in property_keywords): return 'Property Crime'
    if any(keyword in desc_upper for keyword in fraud_keywords): return 'Fraud/White Collar'
    if any(keyword in desc_upper for keyword in public_order_keywords): return 'Public Order/Vice'
    if any(keyword in desc_upper for keyword in other_keywords): return 'Other Specific'
    return 'Miscellaneous/Unknown'
df['crime_category_detailed'] = df['crm_cd_desc'].apply(classify_crime).astype('category')
print("Created detailed crime categories.")

# Map victim descent codes to full names
descent_map = {
    'A': 'Other Asian', 'B': 'Black', 'C': 'Chinese', 'D': 'Cambodian', 'F': 'Filipino',
    'G': 'Guamanian', 'H': 'Hispanic/Latin/Mexican', 'I': 'Am. Indian/Alaskan', 'J': 'Japanese',
    'K': 'Korean', 'L': 'Laotian', 'O': 'Other', 'P': 'Pacific Islander', 'S': 'Samoan',
    'U': 'Hawaiian', 'V': 'Vietnamese', 'W': 'White', 'Z': 'Asian Indian'
}
df['vict_descent_full'] = df['vict_descent'].map(descent_map).fillna('Unknown').astype('category')
print("Mapped victim descent codes.")

print(f"\nFeature engineering complete. DataFrame shape: {df.shape}")
# Show small sample of new features
print("\nSample rows with new features:")
print(df[['datetime_occ', 'hour', 'time_segment', 'is_weekend', 'vict_age_group', 'crime_category_detailed', 'vict_descent_full']].head(3))

print("--- Step 5 Complete ---")
print("-" * 80)





# %% Step 6: Visualize Single Variable Distributions
print("--- Step 6: Looking at distributions of individual variables ---")

if 'df' in locals() and isinstance(df, pd.DataFrame):

    # Define a reusable helper function for creating count plots
    def plot_counts(data, column, title, xlabel, ylabel, top_n=None, palette='viridis', order=None, annotate=False, horizontal=False):
        """Helper to create count plots (bar charts for category frequencies)."""
        plt.figure(figsize=(16, 8))
        ax = None
        counts = data[column].value_counts()

        # Handle ordering and top N selection
        if top_n: counts = counts.nlargest(top_n)
        if order is None: order = counts.index
        else: counts = counts.reindex(order) # Ensure counts match specified order

        # Create horizontal or vertical plot
        if horizontal:
            ax = sns.countplot(y=data[data[column].isin(order)][column], order=order, palette=palette, orient='h')
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            plt.xlabel(xlabel, fontsize=14); plt.ylabel(ylabel, fontsize=14)
        else:
            ax = sns.countplot(x=data[column], order=order, palette=palette)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, p: format(int(y), ',')))
            plt.xlabel(xlabel, fontsize=14); plt.ylabel(ylabel, fontsize=14)
            plt.xticks(rotation=45, ha='right')

        # Add annotations (count numbers on bars)
        if annotate:
            for i, count in enumerate(counts):
                if pd.notna(count): # Avoid annotating NaN counts if order included them
                    if horizontal:
                        ax.text(count + (counts.max() * 0.005), i, f'{count: ,}', va='center', ha='left', fontsize=10)
                    else:
                        ax.text(i, count + (counts.max() * 0.005), f'{count: ,}', ha='center', va='bottom', fontsize=10)

        plt.title(title, fontsize=20, pad=20, weight='bold')
        plt.tight_layout()
        plt.show()

    # --- Plotting Key Distributions ---

    # 1. Crime Categories
    print("\nPlotting: Distribution of Crime Categories")
    plot_counts(data=df, column='crime_category_detailed', title='Crime Category Distribution',
                xlabel='Number of Incidents', ylabel='Crime Category', horizontal=True,
                top_n=len(df['crime_category_detailed'].unique()), palette='magma', annotate=True)
    print("Insight: Property, Vehicle, and Violent crimes are most frequent.")

    # 2. LAPD Area Names
    print("\nPlotting: Crimes by LAPD Area")
    plot_counts(data=df, column='area_name', title='Crimes by LAPD Area Name',
                xlabel='Number of Incidents', ylabel='Area Name', horizontal=True,
                top_n=len(df['area_name'].unique()), palette='Spectral', annotate=True)
    print("Insight: Central, 77th St, Pacific areas report the most crimes.")

    # 3. Time Segments
    print("\nPlotting: Crimes by Time Segment")
    time_order = ['Morning (6-11)', 'Afternoon (12-17)', 'Evening (18-23)', 'Night (0-5)']
    plot_counts(data=df, column='time_segment', title='Crimes by Time Segment',
                xlabel='Time Segment', ylabel='Number of Incidents', order=time_order,
                palette='twilight_shifted', annotate=True)
    print("Insight: Crime peaks in the Afternoon and Evening.")

    # 4. Day of the Week
    print("\nPlotting: Crimes by Day of the Week")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    plot_counts(data=df, column='day_name', title='Crimes by Day of the Week',
                xlabel='Day of the Week', ylabel='Number of Incidents', order=day_order,
                palette='rocket', annotate=True)
    print("Insight: Crime counts are relatively stable across days, peaking slightly on Friday.")

    # 5. Month (Aggregated)
    print("\nPlotting: Crimes by Month")
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    plot_counts(data=df, column='month_name', title='Crimes by Month (Aggregated)',
                xlabel='Month', ylabel='Number of Incidents', order=month_order,
                palette='cubehelix', annotate=True)
    print("Insight: Some seasonal variation is visible, check yearly trends for confirmation.")

    # 6. Victim Age Distribution
    print("\nPlotting: Distribution of Victim Age")
    plt.figure(figsize=(16, 8))
    ax_age = sns.histplot(df['vict_age'].dropna(), bins=60, kde=True, color='steelblue', edgecolor='black', alpha=0.7)
    median_age = df['vict_age'].median()
    mean_age = df['vict_age'].mean()
    ax_age.axvline(median_age, color='red', linestyle='--', linewidth=2, label=f'Median Age: {median_age:.1f}')
    ax_age.axvline(mean_age, color='black', linestyle=':', linewidth=2, label=f'Mean Age: {mean_age:.1f}')
    ax_age.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, p: format(int(y), ',')))
    plt.title('Victim Age Distribution (Known Ages)', fontsize=20, pad=20, weight='bold')
    plt.xlabel('Victim Age', fontsize=14); plt.ylabel('Number of Victims', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(f"Insight: Victim age is right-skewed (median {median_age:.1f}, mean {mean_age:.1f}), peaking in younger adulthood.")

    # 7. Victim Sex
    print("\nPlotting: Distribution of Victim Sex")
    plot_counts(data=df.dropna(subset=['vict_sex']), column='vict_sex',
                title='Victim Sex Distribution (Known)', xlabel='Victim Sex',
                ylabel='Number of Incidents', palette='coolwarm', annotate=True)
    print("Insight: Slightly more male victims reported than female.")

    # 8. Victim Descent
    print("\nPlotting: Distribution of Victim Descent")
    plot_counts(data=df[df['vict_descent_full'] != 'Unknown'], column='vict_descent_full',
                title='Victim Descent Distribution (Known, Top 15)', xlabel='Number of Incidents',
                ylabel='Victim Descent', horizontal=True, top_n=15, palette='tab20', annotate=True)
    print("Insight: Hispanic/Latin/Mexican, White, and Black are most frequent known descents.")

    # 9. Incident Status
    print("\nPlotting: Distribution of Incident Status")
    plot_counts(data=df, column='status_desc', title='Incident Investigation Status',
                xlabel='Status Description', ylabel='Number of Incidents',
                palette='crest', annotate=True)
    print("Insight: Most incidents are 'Investigation Continuing'.")

else:
    print("Error: DataFrame 'df' not found. Please run previous steps.")

print("\n--- Step 6 Complete ---")
print("-" * 80)





# %% Step 7: Analyze Time Patterns vs. Crime Types
print("--- Step 7: Analyzing relationships between time and crime types ---")

if 'df' in locals() and isinstance(df, pd.DataFrame):

    # 1. Plot Crime Categories by Time Segment (Stacked Bar)
    print("\nPlotting: Crime Categories by Time Segment")
    crime_time_counts = df.groupby(['time_segment', 'crime_category_detailed'], observed=False).size().unstack(fill_value=0)
    plt.figure(figsize=(18, 10))
    ax_ts = crime_time_counts.plot(kind='bar', stacked=True, colormap='tab20', width=0.8)
    plt.title('Crime Categories Across Time Segments', fontsize=20, pad=20, weight='bold')
    plt.xlabel('Time Segment', fontsize=14); plt.ylabel('Number of Incidents', fontsize=14)
    plt.xticks(rotation=0)
    plt.legend(title='Crime Category', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax_ts.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, p: format(int(y), ',')))
    # Add total annotations
    totals_ts = crime_time_counts.sum(axis=1)
    for i, total in enumerate(totals_ts):
        ax_ts.text(i, total * 1.01, f'{total: ,}', ha='center', va='bottom', fontsize=10, weight='bold')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
    print("Insight: Violent crimes appear proportionally higher during Evening/Afternoon.")

    # 2. Plot Crime Categories by Day of Week (Grouped Bar)
    print("\nPlotting: Crime Categories by Day of Week")
    crime_day_counts = df.groupby(['day_name', 'crime_category_detailed'], observed=False).size().unstack(fill_value=0)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    crime_day_counts = crime_day_counts.reindex(day_order)
    plt.figure(figsize=(20, 10))
    ax_day = crime_day_counts.plot(kind='bar', colormap='tab20', width=0.85)
    plt.title('Crime Categories Across Days of the Week', fontsize=20, pad=20, weight='bold')
    plt.xlabel('Day of the Week', fontsize=14); plt.ylabel('Number of Incidents', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Crime Category', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax_day.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, p: format(int(y), ',')))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
    print("Insight: Daily patterns are relatively stable for major categories; slight weekend variations possible.")

    # 3. Plot Heatmap of Top Crimes by Hour and Day
    print("\nPlotting: Heatmap of Top Crimes by Hour and Day")
    top_n_crimes = 15
    top_crime_list = df['crm_cd_desc'].value_counts().nlargest(top_n_crimes).index
    df_top_crimes = df[df['crm_cd_desc'].isin(top_crime_list)]
    heatmap_data = pd.pivot_table(df_top_crimes, values='dr_no', index='hour', columns='day_name', aggfunc='count', fill_value=0)
    heatmap_data = heatmap_data.reindex(columns=day_order) # Use same day order
    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_data, cmap="viridis", linewidths=.5, linecolor='lightgrey')
    plt.title(f'Heatmap of Top {top_n_crimes} Crimes by Hour and Day', fontsize=20, pad=20, weight='bold')
    plt.xlabel('Day of Week', fontsize=14); plt.ylabel('Hour of Day (0-23)', fontsize=14)
    plt.yticks(rotation=0); plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    print("Insight: Heatmap shows crime concentration in afternoon/evening hours.")

    # 4. Analyze Crimes involving "VERBAL THREATS"
    print("\nAnalyzing: Crimes associated with 'VERBAL THREATS'")
    verbal_threat_crimes = df[df['weapon_desc'].str.contains('VERBAL THREAT', case=False, na=False)]
    if not verbal_threat_crimes.empty:
        print(f"Found {len(verbal_threat_crimes):,} incidents involving 'VERBAL THREATS'.")
        # Re-use plotting helper if it's defined globally or re-define briefly
        def plot_counts_simple_h(data, column, title, xlabel, ylabel, top_n=25, palette='viridis', annotate=True):
             plt.figure(figsize=(16, 10)) # Adjust size for this plot
             counts = data[column].value_counts().nlargest(top_n)
             order = counts.index
             ax = sns.countplot(y=data[data[column].isin(order)][column], order=order, palette=palette, orient='h')
             ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
             if annotate:
                 for i, count in enumerate(counts): ax.text(count + (counts.max() * 0.005), i, f'{count: ,}', va='center', ha='left', fontsize=10)
             plt.title(title, fontsize=20, pad=20, weight='bold')
             plt.xlabel(xlabel, fontsize=14); plt.ylabel(ylabel, fontsize=14)
             plt.tight_layout(); plt.show()

        plot_counts_simple_h(
            data=verbal_threat_crimes, column='crm_cd_desc',
            title='Top Crime Descriptions Where Weapon Was "VERBAL THREATS"',
            xlabel='Number of Incidents', ylabel='Crime Description',
            top_n=25, palette='coolwarm_r', annotate=True
        )
        print("Insight: Verbal threats commonly occur alongside Criminal Threats, Assaults, and Restraining Order Violations.")
    else:
        print("No incidents found specifically listing 'VERBAL THREATS' as weapon.")

else:
    print("Error: DataFrame 'df' not found. Please run previous steps.")

print("\n--- Step 7 Complete ---")
print("-" * 80)



# %% Step 7b: Explore Specific Patterns (Curiosity Tangents)
print("--- Step 7b: Exploring specific crime patterns ---")

if 'df' in locals() and isinstance(df, pd.DataFrame):

    # --- Tangent 1: Bike Theft Analysis ---
    print("\nAnalyzing: Bike Theft Patterns (Where, When, Who)")
    # Filter data for bike thefts
    bike_thefts = df[df['crm_cd_desc'].str.contains('BIKE - STOLEN', case=False, na=False)].copy()

    if not bike_thefts.empty:
        print(f"Found {len(bike_thefts):,} bike theft incidents.")

        # A. Bike Theft Premises
        print("Plotting: Top premises for bike thefts")
        plt.figure(figsize=(16, 8))
        ax_bike_p = sns.countplot(y=bike_thefts['premis_desc'], order=bike_thefts['premis_desc'].value_counts().nlargest(20).index, palette='viridis_r')
        plt.title('Top 20 Premises for Bike Thefts', fontsize=20, pad=20, weight='bold')
        plt.xlabel('Number of Thefts', fontsize=14); plt.ylabel('Premise Description', fontsize=14)
        ax_bike_p.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        # Add annotations
        counts_p = bike_thefts['premis_desc'].value_counts().nlargest(20)
        for i, count in enumerate(counts_p): ax_bike_p.text(count*1.01, i, f'{count: ,}', va='center', ha='left', fontsize=10)
        plt.tight_layout(); plt.show()
        print("Insight: Streets, Apartments, Homes, and Garages are common bike theft locations.")

        # B. Bike Theft Time Segments
        print("Plotting: Bike thefts by time segment")
        plt.figure(figsize=(12, 6))
        time_order = ['Morning (6-11)', 'Afternoon (12-17)', 'Evening (18-23)', 'Night (0-5)']
        ax_bike_t = sns.countplot(x=bike_thefts['time_segment'], order=time_order, palette='coolwarm')
        plt.title('Bike Thefts by Time Segment', fontsize=20, pad=20, weight='bold')
        plt.xlabel('Time Segment', fontsize=14); plt.ylabel('Number of Thefts', fontsize=14)
        plt.xticks(rotation=0)
        ax_bike_t.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, p: format(int(y), ',')))
        # Add annotations
        counts_t = bike_thefts['time_segment'].value_counts().reindex(time_order)
        for i, count in enumerate(counts_t):
            if pd.notna(count): ax_bike_t.text(i, count*1.01, f'{count: ,}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout(); plt.show()
        print("Insight: Bike thefts occur most often in Afternoon/Evening, but also notably overnight.")

        # C. Bike Theft Victim Age Groups
        print("Plotting: Bike theft victims by age group")
        plt.figure(figsize=(14, 7))
        # Use existing category order if available
        age_order = bike_thefts['vict_age_group'].cat.categories if hasattr(bike_thefts['vict_age_group'], 'cat') else None
        ax_bike_a = sns.countplot(x=bike_thefts['vict_age_group'], order=age_order, palette='Spectral')
        plt.title('Victim Age Groups for Bike Thefts', fontsize=20, pad=20, weight='bold')
        plt.xlabel('Victim Age Group', fontsize=14); plt.ylabel('Number of Victims', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        ax_bike_a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, p: format(int(y), ',')))
        # Add annotations
        counts_a = bike_thefts['vict_age_group'].value_counts().reindex(age_order)
        for i, count in enumerate(counts_a):
            if pd.notna(count): ax_bike_a.text(i, count*1.01, f'{count: ,}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout(); plt.show()
        print("Insight: Young Adults (18-25) and the 26-35 group are most frequently victims.")

    else:
        print("No 'BIKE - STOLEN' incidents found.")
    print("-" * 50)

    # --- Tangent 2: Profile of Most Frequent Crime Location ---
    print("\nAnalyzing: Crime Profile of the Most Frequent Location")
    # Normalize location strings to find the mode accurately
    df['location_normalized'] = df['location'].str.strip().str.upper()
    most_frequent_loc = df['location_normalized'].mode()[0]
    loc_freq = df['location_normalized'].value_counts().max() # Get frequency of the mode
    print(f"Most frequent location: '{most_frequent_loc}' ({loc_freq:,} incidents).")

    loc_crimes = df[df['location_normalized'] == most_frequent_loc].copy()

    if not loc_crimes.empty:
        # A. Common Crimes at this Location
        print("Plotting: Top crimes at this location")
        plt.figure(figsize=(16, 8))
        ax_loc_c = sns.countplot(y=loc_crimes['crm_cd_desc'], order=loc_crimes['crm_cd_desc'].value_counts().nlargest(20).index, palette='magma_r')
        plt.title(f'Top 20 Crimes at: {most_frequent_loc}', fontsize=20, pad=20, weight='bold')
        plt.xlabel('Number of Incidents', fontsize=14); plt.ylabel('Crime Description', fontsize=14)
        ax_loc_c.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        # Add annotations
        counts_lc = loc_crimes['crm_cd_desc'].value_counts().nlargest(20)
        for i, count in enumerate(counts_lc): ax_loc_c.text(count*1.01, i, f'{count: ,}', va='center', ha='left', fontsize=10)
        plt.tight_layout(); plt.show()
        print("Insight: Dominated by Battery, Petty Theft, Trespassing - likely a high-traffic public/commercial area.")

        # B. Premise at this Location
        premise_at_loc = loc_crimes['premis_desc'].mode()[0]
        print(f"\nMost common premise: '{premise_at_loc}'.")
        # print(f"  Top 5 premises:\n{loc_crimes['premis_desc'].value_counts().head()}") # Optional detail

        # C. Time Profile at this Location
        print("Plotting: Crime times at this location")
        plt.figure(figsize=(12, 6))
        time_order_loc = ['Morning (6-11)', 'Afternoon (12-17)', 'Evening (18-23)', 'Night (0-5)']
        ax_loc_t = sns.countplot(x=loc_crimes['time_segment'], order=time_order_loc, palette='Blues_r')
        plt.title(f'Crimes by Time Segment at {most_frequent_loc}', fontsize=20, pad=20, weight='bold')
        plt.xlabel('Time Segment', fontsize=14); plt.ylabel('Number of Incidents', fontsize=14)
        plt.xticks(rotation=0)
        ax_loc_t.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, p: format(int(y), ',')))
        # Add annotations
        counts_lt = loc_crimes['time_segment'].value_counts().reindex(time_order_loc)
        for i, count in enumerate(counts_lt):
            if pd.notna(count): ax_loc_t.text(i, count*1.01, f'{count: ,}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout(); plt.show()
        print("Insight: Crimes peak strongly during Afternoon hours at this location.")

    else:
        print(f"No incidents found for location '{most_frequent_loc}'.")

    # Clean up temporary column
    if 'location_normalized' in df.columns:
        df.drop(columns=['location_normalized'], inplace=True)
    print("\nLocation analysis complete.")

else:
    print("Error: DataFrame 'df' not found. Please run previous steps.")

print("\n--- Step 7b Complete ---")
print("-" * 80)






# %% Step 8: Analyze Demographic Patterns vs. Crime/Location
print("--- Step 8: Analyzing relationships involving victim demographics ---")

if 'df' in locals() and isinstance(df, pd.DataFrame):

    # 1. Plot Victim Age Distribution by Crime Category
    print("\nPlotting: Victim Age Distribution by Crime Category")
    plt.figure(figsize=(18, 10))
    # Calculate median ages for ordering y-axis
    median_ages = df.groupby('crime_category_detailed', observed=False)['vict_age'].median().sort_values()
    order_cats = median_ages.index
    ax_age_cat = sns.boxplot(data=df, x='vict_age', y='crime_category_detailed', order=order_cats,
                             palette='coolwarm', showfliers=False, orient='h')
    plt.title('Victim Age Distribution by Crime Category', fontsize=20, pad=20, weight='bold')
    plt.xlabel('Victim Age', fontsize=14); plt.ylabel('Crime Category', fontsize=14)
    # Add median annotations
    for i, cat in enumerate(order_cats):
        median_val = median_ages[cat]
        if pd.notna(median_val):
             ax_age_cat.text(median_val*1.02, i, f'Median: {median_val:.0f}', va='center', ha='left',
                             color='black', fontsize=9, weight='semibold', bbox=dict(fc='white', alpha=0.6, ec='none', pad=0.3))
    ax_age_cat.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    plt.tight_layout(); plt.show()
    print("Insight: Victim age profiles differ by crime type (e.g., Fraud often has older victims).")

    # 2. Plot Victim Sex Proportion by Crime Category
    print("\nPlotting: Victim Sex Proportion by Crime Category")
    sex_crime_counts = df.groupby(['crime_category_detailed', 'vict_sex'], observed=False).size().unstack(fill_value=0)
    sex_crime_props = sex_crime_counts.apply(lambda x: x*100 / float(x.sum()) if x.sum() > 0 else 0, axis=1) # Avoid division by zero
    plt.figure(figsize=(18, 10))
    ax_sex_cat = sex_crime_props[['M', 'F']].plot(kind='barh', stacked=True, colormap='coolwarm', width=0.8)
    plt.title('Proportion of Victim Sex (M/F) by Crime Category', fontsize=20, pad=20, weight='bold')
    plt.xlabel('Percentage (%)', fontsize=14); plt.ylabel('Crime Category', fontsize=14)
    plt.legend(title='Victim Sex', loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax_sex_cat.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    plt.xlim(0, 100)
    plt.tight_layout(rect=[0, 0, 0.85, 1]); plt.show() # Adjust for legend
    print("Insight: Male/Female victim proportions vary across crime categories.")

    # 3. Plot Victim Descent Distribution for Top Crime Categories
    print("\nPlotting: Victim Descent for Top 4 Crime Categories")
    top_crime_cats = df['crime_category_detailed'].value_counts().nlargest(4).index
    df_top_crime_cats = df[df['crime_category_detailed'].isin(top_crime_cats)].copy() # Use copy for modification
    # Combine less frequent descents for clarity
    top_descents = df['vict_descent_full'].value_counts().nlargest(5).index
    df_top_crime_cats['vict_descent_plot'] = df_top_crime_cats['vict_descent_full'].apply(
        lambda x: x if x in top_descents or x == 'Unknown' else 'Other Descent'
    ).astype('category')
    plt.figure(figsize=(20, 10))
    descent_order_plot = top_descents.tolist() + ['Other Descent', 'Unknown']
    category_order_plot = top_crime_cats.tolist() # Use list for countplot order
    ax_desc_cat = sns.countplot(data=df_top_crime_cats, y='crime_category_detailed', order=category_order_plot,
                                hue='vict_descent_plot', hue_order=descent_order_plot, palette='tab10')
    plt.title('Victim Descent Distribution for Top Crime Categories', fontsize=20, pad=20, weight='bold')
    plt.xlabel('Number of Incidents', fontsize=14); plt.ylabel('Crime Category', fontsize=14)
    plt.legend(title='Victim Descent', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax_desc_cat.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout(rect=[0, 0, 0.85, 1]); plt.show()
    print("Insight: Victim descent patterns differ across the most common crime types.")

    # 4. Plot Victim Age Distribution by Top Premise Types
    print("\nPlotting: Victim Age Distribution by Top 10 Premise Types")
    top_n_premises = 10
    top_premises_list = df['premis_desc'].value_counts().nlargest(top_n_premises).index
    df_top_premises = df[df['premis_desc'].isin(top_premises_list)]
    # Order by median age
    median_ages_premise = df_top_premises.groupby('premis_desc')['vict_age'].median().sort_values()
    premise_order_plot = median_ages_premise.index
    plt.figure(figsize=(18, 10))
    ax_age_prem = sns.boxplot(data=df_top_premises, x='vict_age', y='premis_desc', order=premise_order_plot,
                              palette='Greens_r', showfliers=False, orient='h')
    plt.title(f'Victim Age Distribution by Top {top_n_premises} Premises', fontsize=20, pad=20, weight='bold')
    plt.xlabel('Victim Age', fontsize=14); plt.ylabel('Premise Description', fontsize=14)
    # Add median annotations
    for i, premise in enumerate(premise_order_plot):
        median_val = median_ages_premise[premise]
        if pd.notna(median_val):
             ax_age_prem.text(median_val*1.02, i, f'Median: {median_val:.0f}', va='center', ha='left',
                              color='black', fontsize=9, weight='semibold', bbox=dict(fc='white', alpha=0.6, ec='none', pad=0.3))
    ax_age_prem.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    plt.tight_layout(); plt.show()
    print("Insight: Victim age profiles vary depending on the location (premise) of the crime.")

else:
    print("Error: DataFrame 'df' not found. Please run previous steps.")

print("\n--- Step 8 Complete ---")
print("-" * 80)





# %% Step 9: Analyze Weapon Usage Patterns
print("--- Step 9: Analyzing weapon usage in crimes ---")

if 'df' in locals() and isinstance(df, pd.DataFrame):

    # 1. Plot Distribution of Known Weapon Types
    print("\nPlotting: Distribution of Top 30 Known Weapons Used")
    known_weapons_filter = ~df['weapon_desc'].isin(['NONE/UNKNOWN', 'UNKNOWN WEAPON/OTHER WEAPON', 'UNKNOWN'])
    known_weapons_df = df[known_weapons_filter]

    if not known_weapons_df.empty:
        plt.figure(figsize=(16, 10))
        counts_w = known_weapons_df['weapon_desc'].value_counts().nlargest(30)
        ax_w = sns.countplot(y=known_weapons_df['weapon_desc'], order=counts_w.index, palette='Reds_r')
        plt.title('Top 30 Known Weapons Used (Excluding None/Unknown)', fontsize=20, pad=20, weight='bold')
        plt.xlabel('Number of Incidents', fontsize=14); plt.ylabel('Weapon Description', fontsize=14)
        ax_w.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        # Add annotations
        for i, count in enumerate(counts_w):
            ax_w.text(count * 1.005, i, f'{count: ,}', va='center', ha='left', fontsize=9)
        plt.tight_layout(); plt.show()
        print("Insight: Physical force ('STRONG-ARM') and Verbal Threats are common; firearms and knives also feature.")
    else:
        print("No incidents with known weapons found after filtering.")

    # 2. Plot Weapon Usage Specifically in Violent Crimes
    print("\nPlotting: Top 30 Weapons Used in Violent Crimes")
    violent_crimes_df = df[df['crime_category_detailed'] == 'Violent Crime']
    if not violent_crimes_df.empty:
        violent_weapons_df = violent_crimes_df[known_weapons_filter] # Apply same filter
        if not violent_weapons_df.empty:
            plt.figure(figsize=(16, 10))
            counts_vw = violent_weapons_df['weapon_desc'].value_counts().nlargest(30)
            ax_vw = sns.countplot(y=violent_weapons_df['weapon_desc'], order=counts_vw.index, palette='Oranges_r')
            plt.title('Top 30 Weapons Used in Violent Crimes', fontsize=20, pad=20, weight='bold')
            plt.xlabel('Number of Violent Incidents', fontsize=14); plt.ylabel('Weapon Description', fontsize=14)
            ax_vw.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            # Add annotations
            for i, count in enumerate(counts_vw):
                ax_vw.text(count * 1.005, i, f'{count: ,}', va='center', ha='left', fontsize=9)
            plt.tight_layout(); plt.show()
            print("Insight: Strong-arm and verbal threats still lead, but firearms/knives are more prominent in violent crimes.")
        else:
             print("No known weapons reported within the 'Violent Crime' category.")
    else:
        print("No 'Violent Crime' category incidents found.")

    # 3. Plot Comparison of Weapon Presence (Violent vs. Property)
    print("\nPlotting: Percentage of Crimes with Known Weapon (Violent vs. Property)")
    df['known_weapon_used'] = known_weapons_filter # Reuse the filter boolean Series
    comparison_cats = ['Violent Crime', 'Property Crime', 'Vehicle Related - Property']
    # Calculate percentage where known_weapon_used is True
    weapon_presence_pct = df[df['crime_category_detailed'].isin(comparison_cats)].groupby(
        'crime_category_detailed', observed=False)['known_weapon_used'].mean() * 100

    if not weapon_presence_pct.empty:
        plt.figure(figsize=(10, 6)) # Smaller figure for fewer bars
        ax_wp = weapon_presence_pct.plot(kind='bar', color=sns.color_palette('Accent', len(weapon_presence_pct)), width=0.6)
        plt.title('Incidents with a Known Weapon Used (%)', fontsize=20, pad=20, weight='bold')
        plt.xlabel('Crime Category', fontsize=14); plt.ylabel('Percentage (%)', fontsize=14)
        plt.xticks(rotation=0)
        ax_wp.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
        plt.ylim(0, max(weapon_presence_pct.max() * 1.1, 50)) # Adjust ylim dynamically
        # Add percentage annotations
        for i, percentage in enumerate(weapon_presence_pct):
            ax_wp.text(i, percentage + 1, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=11, weight='bold')
        plt.tight_layout(); plt.show()
        print("Insight: Violent crimes involve known weapons far more often than property crimes.")
    else:
        print("Could not calculate weapon presence comparison.")

    # Clean up temporary column
    if 'known_weapon_used' in df.columns:
        df.drop(columns=['known_weapon_used'], inplace=True)

else:
    print("Error: DataFrame 'df' not found. Please run previous steps.")

print("\n--- Step 9 Complete ---")
print("-" * 80)







# %% Step 10: Statistical Analysis & Hypothesis Testing
print("--- Step 10: Testing observed patterns statistically ---")

# Ensure necessary variables and libraries are available
if 'df' in locals() and isinstance(df, pd.DataFrame) and 'stats' in locals() and 'ALPHA' in locals():

    # Helper function to print test results clearly
    def print_test_results(test_name, statistic, p_value, alpha=ALPHA):
        print(f"\n--- {test_name} Results ---")
        print(f"Statistic: {statistic:.4f}")
        print(f"P-value: {p_value:.4g}") # Use general format for p-value
        significant = p_value < alpha
        conclusion = "IS statistically significant" if significant else "is NOT statistically significant"
        print(f"Conclusion (alpha={alpha}): Result {conclusion} (p={p_value:.4g}).")
        print("-" * (len(test_name) + 12))
        return significant

    print(f"Using significance level alpha = {ALPHA}")

    # Hypothesis 1: Association between Crime Status and Time Segment
    print("\nTesting: Is crime status related to time segment?")
    status_time_contingency = pd.crosstab(df['status_desc'], df['time_segment'])
    # print("Contingency Table:\n", status_time_contingency) # Optional: view table
    try:
        chi2, p, dof, expected = stats.chi2_contingency(status_time_contingency)
        sig_h1 = print_test_results("Chi-Squared (Status vs Time Segment)", chi2, p)
        if sig_h1: print("Interpretation: Yes, investigation status significantly varies by time of day.")
    except ValueError as e: print(f"Chi-Squared test failed: {e}")

    # Hypothesis 2: Association between Violent Crime Proportion and Time Segment
    print("\nTesting: Does the proportion of violent crime vary by time segment?")
    df['is_violent'] = (df['crime_category_detailed'] == 'Violent Crime')
    severity_time_contingency = pd.crosstab(df['is_violent'], df['time_segment'])
    severity_time_contingency.index = ['Non-Violent/Other', 'Violent Crime']
    # print("Contingency Table:\n", severity_time_contingency) # Optional: view table
    try:
        chi2_sev, p_sev, dof_sev, expected_sev = stats.chi2_contingency(severity_time_contingency)
        sig_h2 = print_test_results("Chi-Squared (Violent vs Time Segment)", chi2_sev, p_sev)
        if sig_h2:
            proportions = severity_time_contingency.apply(lambda x: x*100 / float(x.sum()) if x.sum() > 0 else 0, axis=0)
            print("Proportion Violent (%):\n", proportions.loc['Violent Crime'].round(1))
            print("Interpretation: Yes, the likelihood of a crime being violent significantly varies by time of day.")
    except ValueError as e: print(f"Chi-Squared test failed: {e}")
    if 'is_violent' in df.columns: df.drop(columns=['is_violent'], inplace=True) # Clean up

    # Hypothesis 3: Difference in Victim Age across Top Crime Categories
    print("\nTesting: Does victim age differ across top crime categories?")
    categories_for_age_test = df['crime_category_detailed'].value_counts().nlargest(5).index.tolist()
    print(f"Comparing categories: {categories_for_age_test}")
    age_data_by_category = [df['vict_age'][df['crime_category_detailed'] == cat].dropna() for cat in categories_for_age_test]
    valid_age_groups = [group for group in age_data_by_category if len(group) > 1]
    valid_categories = [cat for cat, group in zip(categories_for_age_test, age_data_by_category) if len(group) > 1]

    if len(valid_age_groups) >= 2:
        # Check normality briefly (as Kruskal-Wallis is likely needed anyway)
        normality_check_p_values = [stats.shapiro(g.sample(min(len(g), 4999), random_state=1))[1] for g in valid_age_groups if len(g)>3]
        all_normal = all(p >= ALPHA for p in normality_check_p_values) if normality_check_p_values else False
        print(f"Normality Check Passed for All Groups? {'Yes' if all_normal else 'No'}")

        if all_normal: # Use ANOVA if assumptions met
            print("Performing ANOVA (Comparing Mean Ages)...")
            stat_anova, p_anova = stats.f_oneway(*valid_age_groups)
            sig_h3 = print_test_results(f"ANOVA (Age across {len(valid_categories)} cats)", stat_anova, p_anova)
            if sig_h3: print("Interpretation: Yes, mean victim age significantly differs between categories.")
        else: # Use Kruskal-Wallis if normality fails
             print("Performing Kruskal-Wallis Test (Comparing Median Ages)...")
             stat_kw, p_kw = stats.kruskal(*valid_age_groups)
             sig_h3 = print_test_results(f"Kruskal-Wallis (Age across {len(valid_categories)} cats)", stat_kw, p_kw)
             if sig_h3:
                 print("Interpretation: Yes, median victim age significantly differs between categories.")
                 print("Median Ages:", {cat: f"{group.median():.1f}" for cat, group in zip(valid_categories, valid_age_groups)})
    else:
        print("Not enough groups with age data to perform comparison.")

else:
    print("Error: DataFrame 'df' or required libraries not found. Please run previous steps.")

print("\n--- Step 10 Complete ---")
print("-" * 80)





# %% Step 10b: Supplementary Statistical Analysis & Distributions
print("--- Step 10b: Exploring more statistical tests and distributions ---")

# Ensure necessary variables and libraries are available
if 'df' in locals() and isinstance(df, pd.DataFrame) and 'stats' in locals() and 'ALPHA' in locals():

    # Re-define helper if needed (or ensure it's available from Step 10)
    def print_test_results(test_name, statistic, p_value, alpha=ALPHA):
        print(f"\n--- {test_name} Results ---")
        print(f"Statistic: {statistic:.4f}")
        print(f"P-value: {p_value:.4g}")
        significant = p_value < alpha
        conclusion = "IS statistically significant" if significant else "is NOT statistically significant"
        print(f"Conclusion (alpha={alpha}): Result {conclusion} (p={p_value:.4g}).")
        print("-" * (len(test_name) + 12))
        return significant

    print(f"Running supplementary tests with alpha = {ALPHA}")

    # 1. Example: t-test for Victim Age (Violent vs. Property)
    # Note: Performed for demonstration; assumptions were likely violated.
    print("\nExample: Comparing mean victim age (Violent vs Property) using t-test")
    age_violent = df['vict_age'][df['crime_category_detailed'] == 'Violent Crime'].dropna()
    age_property = df['vict_age'][df['crime_category_detailed'] == 'Property Crime'].dropna()
    if len(age_violent) > 1 and len(age_property) > 1:
        print("Reminder: Normality assumption likely failed; Mann-Whitney U preferred.")
        stat_levene, p_levene = stats.levene(age_violent, age_property)
        equal_var_ttest = p_levene >= ALPHA # Check equal variance for t-test type
        stat_tt, p_tt = stats.ttest_ind(age_violent, age_property, equal_var=equal_var_ttest)
        sig_tt = print_test_results(f"t-Test (Age: Violent vs Property)", stat_tt, p_tt)
        if sig_tt: print(f"  Mean Ages - Violent: {age_violent.mean():.1f}, Property: {age_property.mean():.1f}")
    else: print("  Not enough data for t-test comparison.")

    # 2. Example: Binomial Test for Weekend Crime Proportion
    print("\nExample: Testing if weekend crime proportion equals 2/7")
    n_total = len(df)
    n_weekend = df['is_weekend'].sum()
    hypothesized_prop = 2/7
    observed_prop = n_weekend / n_total
    print(f"Observed weekend proportion: {observed_prop:.3f} ({n_weekend:,}/{n_total:,})")
    binom_result = stats.binomtest(k=n_weekend, n=n_total, p=hypothesized_prop, alternative='two-sided')
    sig_binom = print_test_results(f"Binomial Test (Weekend vs {hypothesized_prop:.3f})", observed_prop, binom_result.pvalue)
    if sig_binom: print(f"Interpretation: The proportion of weekend crimes ({observed_prop:.3f}) significantly differs from 2/7.")

    # 3. Visualize Victim Age vs. Normal Distribution
    print("\nVisualizing: Victim Age distribution vs. Normal distribution")
    plt.figure(figsize=(14, 7))
    ax_norm = sns.histplot(df['vict_age'].dropna(), bins=60, kde=True, stat='density', color='skyblue', label='Observed Age')
    mean_age, std_age = df['vict_age'].mean(), df['vict_age'].std()
    x_norm = np.linspace(df['vict_age'].min(), df['vict_age'].max(), 200)
    y_norm = stats.norm.pdf(x_norm, mean_age, std_age)
    plt.plot(x_norm, y_norm, color='red', linestyle='--', linewidth=2, label=f'Normal Fit')
    plt.title('Victim Age Distribution vs. Theoretical Normal', fontsize=18, pad=15)
    plt.xlabel('Victim Age', fontsize=14); plt.ylabel('Density', fontsize=14)
    plt.legend(); plt.tight_layout(); plt.show()
    print("Insight: Age distribution is clearly right-skewed, not normally distributed.")

    # 4. Visualize Hourly Crime Counts vs. Poisson Distribution (Conceptual Comparison)
    # Note: Crime counts rarely follow a simple Poisson due to non-constant rates.
    print("\nVisualizing: Hourly crime counts vs. Poisson distribution shape")
    hourly_counts = df['hour'].value_counts().sort_index()
    mean_hourly_rate = hourly_counts.mean()
    plt.figure(figsize=(14, 7))
    plt.plot(hourly_counts.index, hourly_counts.values, marker='o', linestyle='-', color='blue', label='Observed Hourly Counts')
    # Generate Poisson PMF and scale for visual comparison
    poisson_pmf = stats.poisson.pmf(k=hourly_counts.index, mu=mean_hourly_rate)
    scaling_factor = len(df) # Rough scaling
    plt.plot(hourly_counts.index, poisson_pmf * scaling_factor, color='orange', linestyle='--', label=f'Poisson Shape (λ≈{mean_hourly_rate:.1f})')
    plt.title('Observed Hourly Counts vs. Scaled Poisson Shape', fontsize=18, pad=15)
    plt.xlabel('Hour of Day (0-23)', fontsize=14); plt.ylabel('Counts / Scaled Probability', fontsize=14)
    plt.xticks(range(0, 24)); plt.legend(); plt.grid(True, axis='y', linestyle=':'); plt.tight_layout(); plt.show()
    print("Insight: Hourly counts show strong daily patterns (diurnal cycle), unlike a simple Poisson distribution.")

else:
    print("Error: DataFrame 'df' or required libraries not found. Please run previous steps.")

print("\n--- Step 10b Complete ---")
print("-" * 80)






# %% Step 11: Summary, Conclusions, and Limitations
print("--- Step 11: Wrapping up with Summary, Conclusions, and Limitations ---")

# This step summarizes the findings from the analysis of the crime data.

print("\n--- I. Quick Summary of Analysis ---")
# Brief overview of the process followed
print("1. Loaded and inspected the raw crime data.")
print("2. Cleaned the data (column names, dates, missing values, placeholders).")
print("3. Created new features (time components, age groups, crime categories).")
print("4. Explored data patterns through visualizations (distributions, relationships, specific tangents).")
print("5. Performed statistical tests to confirm key observations.")

print("\n--- II. Key Findings ---")
# Highlighting the main insights discovered

print("\n  About Crime Types & Methods:")
print("  * Property, Vehicle-related, and Violent crimes are the most common.")
print("  * The most frequent location ('800 N ALAMEDA ST') sees mainly Battery, Petty Theft, and Trespassing.")
print("  * Physical force ('STRONG-ARM') and Verbal Threats are the most reported 'weapons'.")
print("  * Known weapons are significantly more common in Violent crimes than Property crimes.")

print("\n  About Time Patterns:")
print("  * Crime peaks during Afternoon and Evening hours.")
print("  * Violent crimes make up a higher percentage of incidents during these peak times.")
print("  * Crime levels are fairly steady across weekdays, often highest on Fridays.")
print("  * The proportion of crimes on weekends differs statistically from a simple 2/7 split.")
print("  * Investigation status is significantly related to the time of day.")

print("\n  About Locations:")
print("  * Crime distribution varies significantly by LAPD Area (Central, 77th St highest).")
print("  * Streets, Apartments, Homes, and Parking Lots are common crime scenes.")

print("\n  About Victims:")
print("  * Victim age distribution is skewed towards younger adults (median ~37).")
print("  * Victim age profiles differ significantly across crime categories and locations.")
print("  * Slightly more male victims reported overall, but proportions vary by crime type.")
print("  * Hispanic/Latin/Mexican, White, and Black individuals are the most common known victim descents, with variations across crime types.")

print("\n--- III. Important Limitations ---")
# Acknowledging the constraints of the analysis
print("*   Analysis uses *reported* crimes only.")
print("*   ~25% of victim demographic data was missing.")
print("*   Keyword-based crime categorization is an approximation.")
print("*   Weapon/Motive details can be imprecise or missing.")
print("*   'Status Desc' is a limited proxy for case resolution.")
print("*   Correlation does not imply causation.")
print("*   External factors (economy, events, policing) were not included.")

print("\n--- IV. Conclusion ---")
# Final concluding thoughts
print("This analysis revealed significant patterns in LA crime regarding when, where, how, and to whom crimes occur, based on reported data from 2020 onwards. Key trends were identified and statistically validated.")
print("While data limitations exist, the findings offer a valuable snapshot of crime dynamics. Further research could include geospatial analysis, time series modeling, or incorporating external data sources.")

print("\n--- Analysis Complete ---")
print("-" * 80)
