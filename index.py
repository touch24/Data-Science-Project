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
