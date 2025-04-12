# Steps 1-5: Setup, Load, Clean, Feature Engineering
print("--- Step 1: Setup ---")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings
import textwrap # Kept for potential use in plotting long labels later

# --- Core Libraries & Settings ---
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['figure.figsize'] = (16, 8) # Keep plots large for detail
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['patch.edgecolor'] = 'black' # Keep edges for clarity
plt.rcParams['patch.linewidth'] = 0.5
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Project Configuration ---
ALPHA = 0.05
FILE_PATH = 'Crime_Data_from_2020_to_Present (2).csv'

print(f"Libraries loaded. Style set.")
print(f"Significance Level (Alpha): {ALPHA}")
print(f"Data File: {FILE_PATH}")
print("Step 1 Complete.")
# --------------------------------------------------

print("\n--- Step 2: Load Data & Initial Look ---")

# Load the dataset
df_raw = pd.read_csv(FILE_PATH, low_memory=False)
print(f"Loaded data from: {FILE_PATH}")

# Basic Inspection
print(f"\nData Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

print("\nFirst 5 Rows:")
with pd.option_context('display.max_columns', None):
    print(df_raw.head())

print("\nColumn Types & Non-Null Counts:")
df_raw.info(show_counts=True)

print("\nSummary Statistics:")
print(df_raw.describe(include='all').T)

# Create a working copy
df = df_raw.copy()
print("\nCreated a working copy of the data.")
print("Step 2 Complete.")
# --------------------------------------------------

print("\n--- Step 3: Clean Column Names & Dates/Times ---")

if 'df' in locals() and isinstance(df, pd.DataFrame):

    # 1. Standardize column names (lowercase, underscores)
    print("Standardizing column names...")
    df.columns = df.columns.str.lower().str.replace(' ', '_', regex=False).str.replace('-', '_', regex=False)

    # 2. Convert date columns to datetime
    print("Converting date columns...")
    df['date_rptd'] = pd.to_datetime(df['date_rptd'], errors='coerce')
    df['date_occ'] = pd.to_datetime(df['date_occ'], errors='coerce')
    initial_rows = df.shape[0]
    df.dropna(subset=['date_occ'], inplace=True)
    rows_dropped = initial_rows - df.shape[0]
    if rows_dropped > 0:
        print(f"  Dropped {rows_dropped} rows with invalid occurrence dates.")

    # 3. Parse time (HHMM) and combine with date_occ
    print("Parsing time and creating combined 'datetime_occ'...")
    df['time_occ_str'] = df['time_occ'].astype(int).astype(str).str.zfill(4)
    time_parsed = pd.to_datetime(df['time_occ_str'], format='%H%M', errors='coerce').dt.time
    df['datetime_occ'] = pd.to_datetime(df['date_occ'].dt.date.astype(str) + ' ' + time_parsed.astype(str), errors='coerce')

    initial_rows_dt = df.shape[0]
    df.dropna(subset=['datetime_occ'], inplace=True)
    rows_dropped_dt = initial_rows_dt - df.shape[0]
    if rows_dropped_dt > 0:
         print(f"  Dropped {rows_dropped_dt} rows with invalid occurrence times.")

    print(f"\nCleaned DataFrame shape: {df.shape}")
    print("Date/Time columns processed.")

else:
    print("Error: DataFrame 'df' not found at the start of Step 3.")

print("Step 3 Complete.")
# --------------------------------------------------

print("\n--- Step 4: Handle Missing Data & Placeholders ---")

if 'df' in locals() and isinstance(df, pd.DataFrame):

    print("Initial missing value counts (Top 15):")
    missing_before = df.isnull().sum()
    print(missing_before[missing_before > 0].sort_values(ascending=False).head(15))

    # 1. Clean victim age (replace <= 0 with NaN)
    print("\nCleaning victim age (<=0 -> NaN)...")
    df['vict_age'] = df['vict_age'].replace(to_replace=[-4, -3, -2, -1, 0], value=np.nan)
    df['vict_age'] = pd.to_numeric(df['vict_age'], errors='coerce')

    # 2. Clean victim sex (replace placeholders with NaN)
    print("Cleaning victim sex ('X', 'H', '-' -> NaN)...")
    df['vict_sex'] = df['vict_sex'].replace(['X', 'H', '-'], np.nan)

    # 3. Clean victim descent (replace placeholders with NaN)
    print("Cleaning victim descent ('X', '-' -> NaN)...")
    df['vict_descent'] = df['vict_descent'].replace(['X', '-'], np.nan)

    # 4. Fill missing categorical info with defaults
    print("Filling missing text/code info with defaults...")
    df['weapon_used_cd'].fillna(999, inplace=True) # Using 999 for unknown codes
    df['weapon_desc'].fillna('NONE/UNKNOWN', inplace=True)
    df['premis_cd'].fillna(999, inplace=True)
    df['premis_desc'].fillna('UNKNOWN', inplace=True)
    df['mocodes'].fillna('NONE', inplace=True)
    df['cross_street'].fillna('UNKNOWN', inplace=True)

    # Fill rare missing status with mode if necessary
    if df['status'].isnull().any():
        status_mode = df['status'].mode()[0]
        # Find corresponding description for the mode status
        status_desc_mode = df.loc[df['status'] == status_mode, 'status_desc'].mode()[0]
        df['status'].fillna(status_mode, inplace=True)
        df['status_desc'].fillna(status_desc_mode, inplace=True)
        print("  Filled missing status/status_desc with mode.")

    # Fill missing secondary crime codes with 0
    fill_crm_codes = {'crm_cd_1': 0, 'crm_cd_2': 0, 'crm_cd_3': 0, 'crm_cd_4': 0}
    df.fillna(value=fill_crm_codes, inplace=True)

    # 5. Convert relevant columns to appropriate types
    print("Converting column types (codes to category, Crm Cd to int)...")
    df['weapon_used_cd'] = df['weapon_used_cd'].astype('category')
    df['premis_cd'] = df['premis_cd'].astype('category')
    df['status'] = df['status'].astype('category')
    for col in ['crm_cd_1', 'crm_cd_2', 'crm_cd_3', 'crm_cd_4']:
        df[col] = df[col].astype(int)

    print("\nMissing values AFTER handling (columns with >0 missing):")
    missing_after = df.isnull().sum()
    print(missing_after[missing_after > 0].sort_values(ascending=False))
    print("  (NaNs expected for vict_age, vict_sex, vict_descent)")

else:
    print("Error: DataFrame 'df' not found at the start of Step 4.")

print("Step 4 Complete.")
# --------------------------------------------------

print("\n--- Step 5: Create New Features ---")

if 'df' in locals() and isinstance(df, pd.DataFrame):

    # 1. Extract Time Features
    print("Extracting time features (year, month, hour, day name, etc.)...")
    df['year'] = df['datetime_occ'].dt.year
    df['month'] = df['datetime_occ'].dt.month
    df['hour'] = df['datetime_occ'].dt.hour
    df['day_of_week'] = df['datetime_occ'].dt.dayofweek # Monday=0
    df['day_name'] = df['datetime_occ'].dt.day_name()
    df['month_name'] = df['datetime_occ'].dt.month_name()

    # 2. Create Time Categories
    print("Creating time categories (segment, weekend)...")
    time_bins = [-1, 6, 12, 18, 24]
    time_labels = ['Night (0-5)', 'Morning (6-11)', 'Afternoon (12-17)', 'Evening (18-23)']
    df['time_segment'] = pd.cut(df['hour'], bins=time_bins, labels=time_labels, right=False, ordered=True)
    df['is_weekend'] = df['day_name'].isin(['Saturday', 'Sunday'])

    # 3. Create Age Groups
    print("Creating victim age groups...")
    age_bins = [-np.inf, 17, 25, 35, 45, 55, 65, np.inf]
    age_labels = ['0-17 (Minor)', '18-25', '26-35', '36-45', '46-55', '56-65', '66+ (Senior)']
    df['vict_age_group'] = pd.cut(df['vict_age'], bins=age_bins, labels=age_labels, right=True, ordered=True)

    # 4. Create Detailed Crime Categories
    print("Creating detailed crime categories...")
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
    print("Crime Category Counts:\n", df['crime_category_detailed'].value_counts()) # Show counts

    # 5. Map Victim Descent Codes
    print("Mapping victim descent codes...")
    descent_map = {
        'A': 'Other Asian', 'B': 'Black', 'C': 'Chinese', 'D': 'Cambodian', 'F': 'Filipino',
        'G': 'Guamanian', 'H': 'Hispanic/Latin/Mexican', 'I': 'Am. Indian/Alaskan', 'J': 'Japanese',
        'K': 'Korean', 'L': 'Laotian', 'O': 'Other', 'P': 'Pacific Islander', 'S': 'Samoan',
        'U': 'Hawaiian', 'V': 'Vietnamese', 'W': 'White', 'Z': 'Asian Indian' # Shortened Am Indian
    }
    df['vict_descent_full'] = df['vict_descent'].map(descent_map).fillna('Unknown').astype('category')

    print(f"\nDataFrame shape after adding features: {df.shape}")

else:
    print("Error: DataFrame 'df' not found at the start of Step 5.")

print("Step 5 Complete.\n" + "-"*80)

