# %% Refactored Step 1: Setup
print("--- Step 1: Setup ---")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings

# Basic Plot Style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 12

# Config
ALPHA = 0.05
FILE_PATH = 'Crime_Data_from_2020_to_Present (2).csv'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

print(f"Libraries loaded. Plotting style set.")
print(f"Significance Level (Alpha): {ALPHA}")
print(f"Data File: {FILE_PATH}")
print("Step 1 Complete.\n" + "-"*80)

# Initialize DataFrame variable
df = None





# %% Refactored Step 2: Load Data & Initial Look
print("--- Step 2: Load Data & Initial Look ---")

# Load the dataset
# Assuming FILE_PATH is defined in Step 1 and the file exists
df_raw = pd.read_csv(FILE_PATH, low_memory=False)
print(f"Loaded data from: {FILE_PATH}")

# Basic Inspection
print(f"\nData Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

print("\nFirst 5 Rows:")
print(df_raw.head())

print("\nData Types & Missing Value Info:")
df_raw.info() # Provides Dtypes and non-null counts

print("\nSummary Statistics:")
print(df_raw.describe(include='all').T) # Transposed for readability

# Create a working copy for modifications
df = df_raw.copy()
print("\nCreated a working copy of the data.")

print("Step 2 Complete.\n" + "-"*80)





# %% Refactored Step 3: Clean Column Names & Dates/Times
print("--- Step 3: Clean Column Names & Dates/Times ---")

# Ensure df exists from Step 2
if 'df' in locals() and isinstance(df, pd.DataFrame):

    # 1. Standardize column names
    print("Standardizing column names...")
    df.columns = df.columns.str.lower().str.replace(' ', '_', regex=False).str.replace('-', '_', regex=False)
    # print("New columns:", df.columns.tolist()) # Optional: uncomment to see new names

    # 2. Convert date columns
    print("Converting date columns...")
    df['date_rptd'] = pd.to_datetime(df['date_rptd'], errors='coerce')
    df['date_occ'] = pd.to_datetime(df['date_occ'], errors='coerce')
    # Drop rows if 'date_occ' couldn't be parsed (essential column)
    initial_rows = df.shape[0]
    df.dropna(subset=['date_occ'], inplace=True)
    if initial_rows > df.shape[0]:
        print(f"  Dropped {initial_rows - df.shape[0]} rows with invalid occurrence dates.")

    # 3. Parse time and combine with date
    print("Parsing time and creating 'datetime_occ'...")
    # Pad time, parse HHMM, combine with date_occ, coerce errors
    df['time_occ_str'] = df['time_occ'].astype(int).astype(str).str.zfill(4)
    time_parsed = pd.to_datetime(df['time_occ_str'], format='%H%M', errors='coerce').dt.time
    df['datetime_occ'] = pd.to_datetime(df['date_occ'].dt.date.astype(str) + ' ' + time_parsed.astype(str), errors='coerce')

    # Drop rows if combined datetime is invalid
    initial_rows_dt = df.shape[0]
    df.dropna(subset=['datetime_occ'], inplace=True)
    if initial_rows_dt > df.shape[0]:
         print(f"  Dropped {initial_rows_dt - df.shape[0]} rows with invalid occurrence times.")

    print(f"\nCleaned DataFrame shape: {df.shape}")
    print("Relevant column types after processing:")
    print(df[['date_rptd', 'date_occ', 'datetime_occ']].info())

else:
    print("Error: DataFrame 'df' not found. Please run previous steps.")

print("Step 3 Complete.\n" + "-"*80)
