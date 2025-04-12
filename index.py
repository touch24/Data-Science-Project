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
