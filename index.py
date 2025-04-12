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
