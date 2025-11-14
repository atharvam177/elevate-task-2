"""
Complete EDA Pipeline with Advanced Missing Data Imputation
Comprehensive data exploration with professional visualizations
Using Titanic Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Enhanced visualization styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.1)
plt.rcParams['figure.facecolor'] = '#f8f9fa'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Modern color palette
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'accent': '#f093fb',
    'success': '#51cf66',
    'warning': '#ffd93d',
    'danger': '#ff6b6b',
    'info': '#6bcf7f',
    'gradient': ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a'],
    'survival': ['#ff6b6b', '#51cf66']
}

print("=" * 100)
print("🚀 COMPLETE EDA PIPELINE WITH ADVANCED MISSING DATA IMPUTATION")
print("=" * 100)

# ============================================================================
# PHASE 1: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

print("\n" + "=" * 100)
print("📊 PHASE 1: DATA LOADING AND INITIAL EXPLORATION")
print("=" * 100)

# Load dataset
import kagglehub
import os

print("\n📥 Loading Titanic dataset...")
path = kagglehub.dataset_download("yasserh/titanic-dataset")
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
csv_file = os.path.join(path, csv_files[0])
df_original = pd.read_csv(csv_file)

print(f"✓ Dataset loaded: {df_original.shape[0]} rows × {df_original.shape[1]} columns")

# Display basic info
print("\n📋 First 5 rows:")
print(df_original.head())

print("\n📋 Dataset Information:")
print(df_original.info())

print("\n📋 Statistical Summary:")
print(df_original.describe())

# ============================================================================
# PHASE 2: MISSING DATA ANALYSIS AND IMPUTATION
# ============================================================================

print("\n" + "=" * 100)
print("🔧 PHASE 2: MISSING DATA ANALYSIS AND IMPUTATION")
print("=" * 100)

# Analyze missing data
print("\n📋 Missing Data Summary:")
print("=" * 100)

missing_data = pd.DataFrame({
    'Column': df_original.columns,
    'Missing_Count': df_original.isnull().sum().values,
    'Missing_Percentage': (df_original.isnull().sum().values / len(df_original) * 100).round(2),
    'Data_Type': df_original.dtypes.values
})

missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_data) > 0:
    print(missing_data.to_string(index=False))
    print(f"\n📊 Total missing values: {df_original.isnull().sum().sum()}")
    print(f"📊 Percentage of dataset: {(df_original.isnull().sum().sum() / (len(df_original) * len(df_original.columns)) * 100):.2f}%")
else:
    print("✓ No missing values found!")

# Visualize missing data
print("\n📊 Creating missing data visualizations...")

fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor('#f8f9fa')

# 1. Missing data heatmap
ax1 = plt.subplot(2, 3, 1)
missing_mask = df_original.isnull()
sns.heatmap(missing_mask, cmap=['#51cf66', '#ff6b6b'], cbar=False, 
           yticklabels=False, ax=ax1, linewidths=0)
ax1.set_title('Missing Data Pattern\n(Red = Missing, Green = Present)', 
             fontsize=13, fontweight='bold', pad=15, color='#2c3e50')
ax1.set_xlabel('Features', fontsize=11, fontweight='bold')
ax1.set_ylabel('Samples', fontsize=11, fontweight='bold')

# 2. Missing data bar chart
ax2 = plt.subplot(2, 3, 2)
if len(missing_data) > 0:
    bars = ax2.barh(missing_data['Column'], missing_data['Missing_Percentage'],
                   color=COLORS['gradient'][:len(missing_data)], alpha=0.8, 
                   edgecolor='white', linewidth=2)
    
    for i, (bar, count, pct) in enumerate(zip(bars, missing_data['Missing_Count'], 
                                              missing_data['Missing_Percentage'])):
        width = bar.get_width()
        ax2.text(width + 1, i, f'{int(count)} ({pct:.1f}%)',
               va='center', ha='left', fontweight='bold', fontsize=10, color='#2c3e50')
    
    ax2.set_xlabel('Missing Percentage (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Missing Data by Feature', fontsize=13, fontweight='bold', 
                 pad=15, color='#2c3e50')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3, axis='x')

# 3. Missing data correlation
ax3 = plt.subplot(2, 3, 3)
missing_corr = df_original.isnull().corr()
sns.heatmap(missing_corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
           square=True, linewidths=2, cbar_kws={"shrink": 0.8}, ax=ax3,
           vmin=-1, vmax=1, annot_kws={'fontsize': 8, 'fontweight': 'bold'})
ax3.set_title('Missing Data Correlation', fontsize=13, fontweight='bold', 
             pad=15, color='#2c3e50')
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=9)
plt.setp(ax3.get_yticklabels(), rotation=0, fontsize=9)

# 4. Missing by category
ax4 = plt.subplot(2, 3, 4)
if 'Age' in missing_data['Column'].values and 'Pclass' in df_original.columns:
    age_missing = df_original.groupby('Pclass')['Age'].apply(lambda x: x.isnull().sum())
    age_total = df_original.groupby('Pclass')['Age'].count()
    age_missing_pct = (age_missing / (age_missing + age_total)) * 100
    
    bars = ax4.bar(age_missing_pct.index, age_missing_pct.values, 
                  color=COLORS['gradient'][:3], alpha=0.8, edgecolor='white', linewidth=2)
    
    for bar, pct in zip(bars, age_missing_pct.values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
               f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax4.set_xlabel('Passenger Class', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Missing Age (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Age Missingness by Class', fontsize=13, fontweight='bold', 
                 pad=15, color='#2c3e50')
    ax4.set_xticks([1, 2, 3])
    ax4.set_xticklabels(['1st Class', '2nd Class', '3rd Class'])
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(True, alpha=0.3, axis='y')

# 5. Mechanism guide
ax5 = plt.subplot(2, 3, 5)
mechanism_text = """MISSING DATA MECHANISMS:

MCAR: Missing Completely At Random
• Random, unrelated to any variables
• Any imputation method works

MAR: Missing At Random
• Related to observed variables
• Use related features for imputation

MNAR: Missing Not At Random
• Related to unobserved data
• Model missingness explicitly"""

ax5.text(0.05, 0.95, mechanism_text, transform=ax5.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='#fff3cd', 
                 edgecolor=COLORS['warning'], linewidth=2, alpha=0.9))
ax5.axis('off')

# 6. Strategy guide
ax6 = plt.subplot(2, 3, 6)
strategy_text = """IMPUTATION STRATEGY:

Missing < 5%: Simple (mean/median)
Missing 5-20%: KNN or MICE
Missing 20-50%: ML-based + indicators
Missing > 50%: Consider dropping

Age: Custom (Title + Class)
Embarked: Mode imputation
Cabin: Drop or create indicator
Fare: Median by Pclass"""

ax6.text(0.05, 0.95, strategy_text, transform=ax6.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='#d1ecf1', 
                 edgecolor=COLORS['info'], linewidth=2, alpha=0.9))
ax6.axis('off')

plt.suptitle('📊 Missing Data Analysis Dashboard', fontsize=18, 
            fontweight='bold', y=0.98, color='#2c3e50')
plt.tight_layout()
plt.show()

# Apply Custom Imputation (Recommended Strategy)
print("\n🔧 Applying Custom Domain-Specific Imputation...")
print("-" * 100)

df = df_original.copy()

# Age imputation based on Title and Class
if 'Age' in df.columns and 'Name' in df.columns:
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
    
    for title in df['Title'].unique():
        if pd.notna(title):
            for pclass in df['Pclass'].unique():
                mask = (df['Title'] == title) & (df['Pclass'] == pclass)
                median_age = df.loc[mask, 'Age'].median()
                
                if pd.notna(median_age):
                    df.loc[mask & df['Age'].isnull(), 'Age'] = median_age
    
    df['Age'].fillna(df['Age'].median(), inplace=True)
    print(f"✓ Age: Imputed using Title + Class (177 values filled)")

# Embarked: Mode
if 'Embarked' in df.columns:
    most_common = df['Embarked'].mode()[0]
    df['Embarked'].fillna(most_common, inplace=True)
    print(f"✓ Embarked: Filled with mode '{most_common}' (2 values filled)")

# Fare: Median by class
if 'Fare' in df.columns and 'Pclass' in df.columns:
    for pclass in df['Pclass'].unique():
        mask = df['Pclass'] == pclass
        median_fare = df.loc[mask, 'Fare'].median()
        df.loc[mask & df['Fare'].isnull(), 'Fare'] = median_fare
    print(f"✓ Fare: Imputed using Pclass median")

# Cabin: Create indicator
if 'Cabin' in df.columns:
    df['Cabin_Known'] = df['Cabin'].notna().astype(int)
    print(f"✓ Cabin: Created binary indicator (Cabin_Known)")

print(f"\n✓ Imputation complete! Remaining missing values: {df.isnull().sum().sum()}")

# Get column types after imputation
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols = [col for col in numerical_cols if 'id' not in col.lower() and col != 'Survived']
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# ============================================================================
# PHASE 3: ENHANCED DISTRIBUTION ANALYSIS
# ============================================================================

print("\n" + "=" * 100)
print("📊 PHASE 3: DISTRIBUTION ANALYSIS")
print("=" * 100)

fig = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor('#f8f9fa')

for idx, col in enumerate(numerical_cols[:6], 1):
    ax = plt.subplot(2, 3, idx)
    
    data = df[col].dropna()
    
    # Histogram with KDE
    n, bins, patches = ax.hist(data, bins=30, alpha=0.6, color=COLORS['gradient'][idx-1], 
                               edgecolor='white', linewidth=1.5, density=True)
    
    # Add KDE
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 200)
    ax.plot(x_range, kde(x_range), color=COLORS['gradient'][idx-1], 
           linewidth=3, label='KDE', alpha=0.8)
    
    # Statistics
    mean_val = data.mean()
    median_val = data.median()
    ax.axvline(mean_val, color='#ff6b6b', linestyle='--', linewidth=2.5, 
              label=f'Mean: {mean_val:.1f}', alpha=0.8)
    ax.axvline(median_val, color='#51cf66', linestyle='--', linewidth=2.5, 
              label=f'Median: {median_val:.1f}', alpha=0.8)
    
    ax.set_title(f'{col}', fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
    ax.set_xlabel('Value', fontsize=11, fontweight='600', color='#34495e')
    ax.set_ylabel('Density', fontsize=11, fontweight='600', color='#34495e')
    ax.legend(loc='best', frameon=True, shadow=True, fancybox=True, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Statistics box
    stats_text = f'Skew: {data.skew():.2f}\nStd: {data.std():.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
           facecolor=COLORS['gradient'][idx-1], alpha=0.15, edgecolor='none'),
           fontsize=9, fontweight='600', color='#2c3e50')

plt.suptitle('📊 Distribution Analysis with KDE Overlay', fontsize=18, 
            fontweight='bold', y=0.995, color='#2c3e50')
plt.tight_layout()
plt.show()

# ============================================================================
# PHASE 4: OUTLIER DETECTION
# ============================================================================

print("\n" + "=" * 100)
print("📦 PHASE 4: OUTLIER DETECTION")
print("=" * 100)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.patch.set_facecolor('#f8f9fa')
axes = axes.flatten()

for idx, col in enumerate(numerical_cols[:6]):
    ax = axes[idx]
    data = df[col].dropna()
    
    # Violin plot
    parts = ax.violinplot([data], positions=[0], showmeans=True, showmedians=True, widths=0.7)
    
    for pc in parts['bodies']:
        pc.set_facecolor(COLORS['gradient'][idx])
        pc.set_alpha(0.3)
        pc.set_edgecolor(COLORS['gradient'][idx])
        pc.set_linewidth(2)
    
    # Box plot overlay
    bp = ax.boxplot([data], positions=[0], widths=0.3, patch_artist=True,
                    boxprops=dict(facecolor=COLORS['gradient'][idx], alpha=0.7, 
                                 edgecolor='#2c3e50', linewidth=2),
                    whiskerprops=dict(color='#2c3e50', linewidth=2),
                    capprops=dict(color='#2c3e50', linewidth=2),
                    medianprops=dict(color='#e74c3c', linewidth=3),
                    flierprops=dict(marker='D', markerfacecolor='#e74c3c', 
                                   markersize=6, alpha=0.6, markeredgecolor='#c0392b'))
    
    # Calculate outliers
    Q1, Q3 = data.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
    
    ax.set_title(f'{col}\n{len(outliers)} outliers ({len(outliers)/len(data)*100:.1f}%)', 
                fontsize=13, fontweight='bold', pad=15, color='#2c3e50')
    ax.set_ylabel('Value', fontsize=11, fontweight='600', color='#34495e')
    ax.set_xticks([])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    
    # Quartile stats
    stats_text = f'Q1: {Q1:.1f}\nMedian: {data.median():.1f}\nQ3: {Q3:.1f}\nIQR: {IQR:.1f}'
    ax.text(0.6, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round,pad=0.6', 
           facecolor='white', alpha=0.9, edgecolor=COLORS['gradient'][idx], linewidth=2),
           fontsize=9, fontweight='600', color='#2c3e50', linespacing=1.5)

plt.suptitle('📦 Violin + Box Plot Analysis', fontsize=18, 
            fontweight='bold', y=0.995, color='#2c3e50')
plt.tight_layout()
plt.show()

# ============================================================================
# PHASE 5: CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 100)
print("🔗 PHASE 5: CORRELATION ANALYSIS")
print("=" * 100)

fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor('#f8f9fa')

correlation_matrix = df[numerical_cols].corr()

# Create mask for upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
           cmap='RdYlGn', center=0, square=True, linewidths=2, 
           cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
           annot_kws={'fontsize': 10, 'fontweight': 'bold'},
           vmin=-1, vmax=1, ax=ax)

ax.set_title('🔗 Correlation Matrix Heatmap', fontsize=16, 
            fontweight='bold', pad=20, color='#2c3e50')
plt.xticks(rotation=45, ha='right', fontsize=10, fontweight='600')
plt.yticks(rotation=0, fontsize=10, fontweight='600')

plt.tight_layout()
plt.show()

# Strong correlations
print("\n📊 Strong Correlations (|r| > 0.5):")
print("-" * 100)
corr_pairs = []
for i in range(len(correlation_matrix)):
    for j in range(i+1, len(correlation_matrix)):
        corr = correlation_matrix.iloc[i, j]
        if abs(corr) > 0.5:
            corr_pairs.append({
                'Feature 1': correlation_matrix.index[i],
                'Feature 2': correlation_matrix.columns[j],
                'Correlation': corr
            })

if corr_pairs:
    corr_df = pd.DataFrame(corr_pairs)
    corr_df = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=False).index)
    print(corr_df.to_string(index=False))
else:
    print("No strong correlations found (|r| > 0.5)")

# ============================================================================
# PHASE 6: CATEGORICAL ANALYSIS
# ============================================================================

print("\n" + "=" * 100)
print("📊 PHASE 6: CATEGORICAL FEATURE ANALYSIS")
print("=" * 100)

if len(categorical_cols) > 0:
    n_cats = min(len(categorical_cols), 4)
    
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('#f8f9fa')
    
    for idx, col in enumerate(categorical_cols[:n_cats], 1):
        ax = plt.subplot(2, 2, idx)
        
        value_counts = df[col].value_counts()
        colors_list = COLORS['gradient'][:len(value_counts)]
        
        # Horizontal bars
        bars = ax.barh(range(len(value_counts)), value_counts.values, 
                      color=colors_list, alpha=0.8, edgecolor='white', linewidth=2)
        
        # Labels
        for i, (bar, count) in enumerate(zip(bars, value_counts.values)):
            width = bar.get_width()
            percentage = (count / len(df)) * 100
            ax.text(width + max(value_counts.values)*0.01, i, 
                   f'{count} ({percentage:.1f}%)',
                   va='center', ha='left', fontweight='bold', 
                   fontsize=11, color='#2c3e50')
        
        ax.set_yticks(range(len(value_counts)))
        ax.set_yticklabels(value_counts.index, fontsize=11, fontweight='600')
        ax.set_xlabel('Count', fontsize=12, fontweight='bold', color='#34495e')
        ax.set_title(f'{col} Distribution\n({df[col].nunique()} unique values)', 
                    fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
        ax.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=0.5)
        ax.set_xlim(0, max(value_counts.values) * 1.15)
    
    plt.suptitle('📊 Categorical Features Analysis', fontsize=18, 
                fontweight='bold', y=0.995, color='#2c3e50')
    plt.tight_layout()
    plt.show()

# ============================================================================
# PHASE 7: TARGET ANALYSIS
# ============================================================================

print("\n" + "=" * 100)
print("🎯 PHASE 7: TARGET VARIABLE ANALYSIS")
print("=" * 100)

if 'Survived' in df.columns:
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor('#f8f9fa')
    
    # 1. Donut chart
    ax1 = plt.subplot(3, 3, 1)
    survival_counts = df['Survived'].value_counts()
    colors = [COLORS['survival'][0], COLORS['survival'][1]]
    
    wedges, texts, autotexts = ax1.pie(survival_counts, labels=['Died', 'Survived'],
                                       autopct='%1.1f%%', colors=colors, startangle=90,
                                       pctdistance=0.85, explode=(0.05, 0.05),
                                       wedgeprops=dict(width=0.5, edgecolor='white', linewidth=3),
                                       textprops={'fontsize': 12, 'fontweight': 'bold', 'color': 'white'})
    
    centre_circle = plt.Circle((0, 0), 0.70, fc='white', linewidth=0)
    ax1.add_artist(centre_circle)
    ax1.text(0, 0, f'{len(df)}\nPassengers', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='#2c3e50')
    ax1.set_title('Overall Survival', fontsize=13, fontweight='bold', pad=15, color='#2c3e50')
    
    # 2. Survival by Gender
    if 'Sex' in df.columns:
        ax2 = plt.subplot(3, 3, 2)
        survival_by_sex = pd.crosstab(df['Sex'], df['Survived'], normalize='index') * 100
        
        x = np.arange(len(survival_by_sex.index))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, survival_by_sex[0], width, label='Died', 
                       color=colors[0], alpha=0.8, edgecolor='white', linewidth=2)
        bars2 = ax2.bar(x + width/2, survival_by_sex[1], width, label='Survived',
                       color=colors[1], alpha=0.8, edgecolor='white', linewidth=2)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', 
                        fontweight='bold', fontsize=10, color='#2c3e50')
        
        ax2.set_ylabel('Survival Rate (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Survival by Gender', fontsize=13, fontweight='bold', pad=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels(survival_by_sex.index, fontsize=11, fontweight='600')
        ax2.legend(frameon=True, shadow=True, fancybox=True, fontsize=10)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Survival by Class
    if 'Pclass' in df.columns:
        ax3 = plt.subplot(3, 3, 3)
        survival_by_class = pd.crosstab(df['Pclass'], df['Survived'], normalize='index') * 100
        
        x = np.arange(len(survival_by_class.index))
        width = 0.6
        
        bars1 = ax3.bar(x, survival_by_class[0], width, label='Died',
                       color=colors[0], alpha=0.8, edgecolor='white', linewidth=2)
        bars2 = ax3.bar(x, survival_by_class[1], width, bottom=survival_by_class[0],
                       label='Survived', color=colors[1], alpha=0.8, 
                       edgecolor='white', linewidth=2)
        
        for i, (died, survived) in enumerate(zip(survival_by_class[0], survival_by_class[1])):
            ax3.text(i, died/2, f'{died:.1f}%', ha='center', va='center',
                    fontweight='bold', fontsize=11, color='white')
            ax3.text(i, died + survived/2, f'{survived:.1f}%', ha='center', va='center',
                    fontweight='bold', fontsize=11, color='white')
        
        ax3.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Survival by Class', fontsize=13, fontweight='bold', pad=15)
        ax3.set_xticks(x)
        ax3.set_xticklabels(['1st', '2nd', '3rd'], fontsize=11, fontweight='600')
        ax3.legend(frameon=True, shadow=True, fancybox=True, fontsize=10)
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Age distribution
    if 'Age' in df.columns:
        ax4 = plt.subplot(3, 3, 4)
        
        for survived in [0, 1]:
            ages = df[df['Survived'] == survived]['Age'].dropna()
            kde = stats.gaussian_kde(ages)
            x_range = np.linspace(0, 80, 200)
            y_vals = kde(x_range)
            
            ax4.plot(x_range, y_vals, linewidth=3, 
                    label='Died' if survived == 0 else 'Survived',
                    color=colors[survived], alpha=0.8)
            ax4.fill_between(x_range, y_vals, alpha=0.3, color=colors[survived])
        
        ax4.set_xlabel('Age', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax4.set_title('Age Distribution', fontsize=13, fontweight='bold', pad=15)
        ax4.legend(frameon=True, shadow=True, fancybox=True, fontsize=10)
        ax4.grid(True, alpha=0.3)
    
    # 5. Fare distribution
    if 'Fare' in df.columns:
        ax5 = plt.subplot(3, 3, 5)
        
        for survived in [0, 1]:
            fares = df[df['Survived'] == survived]['Fare'].dropna()
            fares = fares[fares > 0]
            
            ax5.hist(np.log10(fares + 1), bins=30, alpha=0.6,
                    label='Died' if survived == 0 else 'Survived',
                    color=colors[survived], edgecolor='white', linewidth=1.5)
        
        ax5.set_xlabel('Log10(Fare + 1)', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax5.set_title('Fare Distribution (Log)', fontsize=13, fontweight='bold', pad=15)
        ax5.legend(frameon=True, shadow=True, fancybox=True, fontsize=10)
        ax5.grid(True, alpha=0.3)
    
    # 6. Embarkation
    if 'Embarked' in df.columns:
        ax6 = plt.subplot(3, 3, 6)
        
        embarked_survival = df.groupby(['Embarked', 'Survived']).size().unstack(fill_value=0)
        embarked_pct = embarked_survival.div(embarked_survival.sum(axis=1), axis=0) * 100
        
        x = np.arange(len(embarked_pct.index))
        width = 0.35
        
        bars1 = ax6.bar(x - width/2, embarked_pct[0], width, label='Died',
                       color=colors[0], alpha=0.8, edgecolor='white', linewidth=2)
        bars2 = ax6.bar(x + width/2, embarked_pct[1], width, label='Survived',
                       color=colors[1], alpha=0.8, edgecolor='white', linewidth=2)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}%', ha='center', va='bottom',
                        fontweight='bold', fontsize=9)
        
        ax6.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax6.set_title('Survival by Port', fontsize=13, fontweight='bold', pad=15)
        ax6.set_xticks(x)
        ax6.set_xticklabels(['C', 'Q', 'S'], fontsize=10, fontweight='600')
        ax6.legend(frameon=True, shadow=True, fancybox=True, fontsize=10)
        ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Heatmap
    if 'Sex' in df.columns and 'Pclass' in df.columns:
        ax7 = plt.subplot(3, 3, 7)
        
        survival_pivot = df.pivot_table(values='Survived', index='Sex', 
                                       columns='Pclass', aggfunc='mean')
        
        sns.heatmap(survival_pivot, annot=True, fmt='.1%', cmap='RdYlGn',
                   ax=ax7, vmin=0, vmax=1, linewidths=3, linecolor='white',
                   cbar_kws={'label': 'Survival Rate', 'shrink': 0.8},
                   annot_kws={'fontsize': 12, 'fontweight': 'bold'})
        
        ax7.set_title('Survival Rate Heatmap\n(Gender × Class)', 
                     fontsize=13, fontweight='bold', pad=15)
        ax7.set_xlabel('Class', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Gender', fontsize=11, fontweight='bold')
        plt.setp(ax7.get_yticklabels(), rotation=0)
    
    # 8. Family size
    if 'SibSp' in df.columns and 'Parch' in df.columns:
        ax8 = plt.subplot(3, 3, 8)
        
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        family_survival = df.groupby('FamilySize')['Survived'].agg(['mean', 'count'])
        family_survival = family_survival[family_survival['count'] >= 5]
        
        bars = ax8.bar(family_survival.index, family_survival['mean'] * 100,
                      color=COLORS['gradient'], alpha=0.8, edgecolor='white', linewidth=2)
        
        for i, (idx, row) in enumerate(family_survival.iterrows()):
            ax8.text(idx, row['mean'] * 100 + 2, f"n={int(row['count'])}",
                    ha='center', va='bottom', fontsize=9, fontweight='600')
        
        ax8.set_xlabel('Family Size', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Survival Rate (%)', fontsize=11, fontweight='bold')
        ax8.set_title('Survival by Family Size', fontsize=13, fontweight='bold', pad=15)
        ax8.grid(True, alpha=0.3, axis='y')
        ax8.set_ylim(0, 100)
    
    # 9. Age groups
    if 'Age' in df.columns:
        ax9 = plt.subplot(3, 3, 9)
        
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                               labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        
        age_survival = df.groupby('AgeGroup')['Survived'].agg(['mean', 'count'])
        
        x = np.arange(len(age_survival.index))
        bars = ax9.bar(x, age_survival['mean'] * 100, color=COLORS['gradient'],
                      alpha=0.8, edgecolor='white', linewidth=2)
        
        for i, (idx, row) in enumerate(age_survival.iterrows()):
            ax9.text(i, row['mean'] * 100 + 2, 
                    f"{row['mean']*100:.1f}%\n(n={int(row['count'])})",
                    ha='center', va='bottom', fontsize=9, fontweight='600')
        
        ax9.set_ylabel('Survival Rate (%)', fontsize=11, fontweight='bold')
        ax9.set_title('Survival by Age Group', fontsize=13, fontweight='bold', pad=15)
        ax9.set_xticks(x)
        ax9.set_xticklabels(age_survival.index, fontsize=10, fontweight='600', rotation=15)
        ax9.grid(True, alpha=0.3, axis='y')
        ax9.set_ylim(0, 100)
    
    plt.suptitle('🎯 Comprehensive Survival Analysis Dashboard', fontsize=18, 
                fontweight='bold', y=0.995, color='#2c3e50')
    plt.tight_layout()
    plt.show()
    
    # Print key insights
    print("\n💡 Key Survival Insights:")
    print("-" * 100)
    survival_rate = df['Survived'].mean() * 100
    print(f"• Overall survival rate: {survival_rate:.1f}%")
    
    if 'Sex' in df.columns:
        female_survival = df[df['Sex'] == 'female']['Survived'].mean() * 100
        male_survival = df[df['Sex'] == 'male']['Survived'].mean() * 100
        print(f"• Female survival: {female_survival:.1f}%")
        print(f"• Male survival: {male_survival:.1f}%")
        print(f"  → Gender gap: {abs(female_survival - male_survival):.1f}%")
    
    if 'Pclass' in df.columns:
        print(f"\n• Survival by Class:")
        for pclass in sorted(df['Pclass'].unique()):
            rate = df[df['Pclass'] == pclass]['Survived'].mean() * 100
            print(f"  - Class {pclass}: {rate:.1f}%")

# ============================================================================
# PHASE 8: INTERACTIVE VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 100)
print("🎨 PHASE 8: INTERACTIVE VISUALIZATIONS")
print("=" * 100)

# 1. 3D Scatter
if len(numerical_cols) >= 3 and 'Survived' in df.columns:
    fig = px.scatter_3d(df, 
                        x=numerical_cols[0], 
                        y=numerical_cols[1], 
                        z=numerical_cols[2],
                        color='Survived',
                        color_discrete_map={0: '#ff6b6b', 1: '#51cf66'},
                        hover_data=df.columns[:6].tolist(),
                        title='🌐 3D Interactive Scatter Plot',
                        labels={'Survived': 'Survival Status'},
                        opacity=0.7)
    
    fig.update_layout(
        template='plotly_white',
        height=700,
        font=dict(family="Arial, sans-serif", size=12),
        title_font=dict(size=18, color='#2c3e50', family="Arial Black"),
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(230, 230,230, 0.5)", gridcolor="white"),
            yaxis=dict(backgroundcolor="rgba(230, 230,230, 0.5)", gridcolor="white"),
            zaxis=dict(backgroundcolor="rgba(230, 230,230, 0.5)", gridcolor="white")
        )
    )
    fig.show()
    print("✓ 3D scatter plot created")

# 2. Sunburst chart
if 'Survived' in df.columns and 'Pclass' in df.columns and 'Sex' in df.columns:
    sunburst_data = df.groupby(['Survived', 'Pclass', 'Sex']).size().reset_index(name='count')
    sunburst_data['Survived'] = sunburst_data['Survived'].map({0: 'Died', 1: 'Survived'})
    sunburst_data['Pclass'] = sunburst_data['Pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
    
    fig = px.sunburst(sunburst_data, 
                      path=['Survived', 'Pclass', 'Sex'], 
                      values='count',
                      color='Survived',
                      color_discrete_map={'Died': '#ff6b6b', 'Survived': '#51cf66'},
                      title='☀️ Hierarchical Survival Analysis (Sunburst)')
    
    fig.update_layout(
        template='plotly_white',
        height=700,
        font=dict(family="Arial, sans-serif", size=13),
        title_font=dict(size=18, color='#2c3e50', family="Arial Black")
    )
    fig.show()
    print("✓ Sunburst chart created")

# 3. Parallel coordinates
if len(numerical_cols) >= 3 and 'Survived' in df.columns:
    plot_df = df[numerical_cols[:4] + ['Survived']].dropna()
    
    fig = px.parallel_coordinates(
        plot_df,
        dimensions=numerical_cols[:4],
        color='Survived',
        color_continuous_scale=[(0, '#ff6b6b'), (1, '#51cf66')],
        title='📊 Parallel Coordinates - Feature Relationships'
    )
    
    fig.update_layout(
        template='plotly_white',
        height=600,
        font=dict(family="Arial, sans-serif", size=12),
        title_font=dict(size=18, color='#2c3e50', family="Arial Black")
    )
    fig.show()
    print("✓ Parallel coordinates plot created")

# 4. Enhanced scatter with marginals
if 'Age' in df.columns and 'Fare' in df.columns:
    fig = px.scatter(df, 
                     x='Age', 
                     y='Fare',
                     color='Survived' if 'Survived' in df.columns else 'Pclass',
                     size='Fare' if 'Survived' in df.columns else None,
                     hover_data=['Pclass', 'Sex'] if 'Sex' in df.columns else ['Pclass'],
                     title='💎 Interactive Scatter: Age vs Fare',
                     labels={'Survived': 'Survival Status'},
                     color_discrete_map={0: '#ff6b6b', 1: '#51cf66'},
                     marginal_x='violin',
                     marginal_y='violin')
    
    fig.update_layout(
        template='plotly_white',
        height=700,
        font=dict(family="Arial, sans-serif", size=12),
        title_font=dict(size=18, color='#2c3e50', family="Arial Black")
    )
    fig.show()
    print("✓ Enhanced scatter plot created")

# ============================================================================
# PHASE 9: KEY FINDINGS AND RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 100)
print("💡 PHASE 9: KEY FINDINGS AND RECOMMENDATIONS")
print("=" * 100)

print("\n🎯 DATA QUALITY SUMMARY:")
print("-" * 100)
print(f"• Total samples: {len(df)}")
print(f"• Total features: {len(df.columns)}")
print(f"• Numerical features: {len(numerical_cols)}")
print(f"• Categorical features: {len(categorical_cols)}")
print(f"• Missing values (after imputation): {df.isnull().sum().sum()}")
print(f"• Duplicate rows: {df.duplicated().sum()}")

print("\n📊 DISTRIBUTION INSIGHTS:")
print("-" * 100)
for col in numerical_cols[:5]:
    data = df[col].dropna()
    skew = data.skew()
    if abs(skew) < 0.5:
        interpretation = "Approximately symmetric"
    elif skew > 0.5:
        interpretation = "Right-skewed (positive)"
    else:
        interpretation = "Left-skewed (negative)"
    print(f"• {col:15s}: Skew = {skew:6.3f} → {interpretation}")

print("\n🔗 CORRELATION INSIGHTS:")
print("-" * 100)
if len(corr_pairs) > 0:
    print(f"• Found {len(corr_pairs)} strong correlation(s) (|r| > 0.5)")
    for pair in corr_pairs[:3]:
        print(f"  - {pair['Feature 1']} ↔ {pair['Feature 2']}: r = {pair['Correlation']:.3f}")
else:
    print("• No strong correlations detected (good for independence)")

print("\n🎨 RECOMMENDATIONS:")
print("=" * 100)

recommendations = [
    "1️⃣  Feature Engineering:",
    "    • Family size feature created (SibSp + Parch + 1)",
    "    • Age groups created for pattern detection",
    "    • Title extracted from names (already used for imputation)",
    "    • Cabin_Known indicator created",
    "",
    "2️⃣  Data Preprocessing:",
    "    ✓ Age imputed using Title + Pclass median",
    "    ✓ Embarked filled with mode",
    "    ✓ Fare imputed using Pclass median",
    "    ✓ Cabin converted to binary indicator",
    "",
    "3️⃣  Next Steps for Modeling:",
    "    • Encode categorical variables (Sex, Embarked, Title)",
    "    • Apply StandardScaler for numerical features",
    "    • Consider feature selection based on correlation",
    "    • Handle class imbalance if needed",
    "",
    "4️⃣  Model Selection:",
    "    • Start with Logistic Regression (baseline)",
    "    • Try Random Forest (non-linear relationships)",
    "    • Consider XGBoost (high performance)",
    "    • Use cross-validation for evaluation",
    "",
    "5️⃣  Feature Importance:",
    "    • Sex (highest predictor based on EDA)",
    "    • Pclass (strong correlation with survival)",
    "    • Age (especially children)",
    "    • Fare (proxy for socioeconomic status)",
    "    • Family size (moderate families survived better)"
]

for rec in recommendations:
    print(rec)

# ============================================================================
# PHASE 10: SAVE OUTPUTS
# ============================================================================

print("\n" + "=" * 100)
print("💾 PHASE 10: SAVING OUTPUTS")
print("=" * 100)

# Save cleaned and imputed dataset
df.to_csv('titanic_cleaned_imputed.csv', index=False)
print("✓ Saved: titanic_cleaned_imputed.csv")

# Save summary statistics
df[numerical_cols].describe().to_csv('summary_statistics.csv')
print("✓ Saved: summary_statistics.csv")

# Save correlation matrix
correlation_matrix.to_csv('correlation_matrix.csv')
print("✓ Saved: correlation_matrix.csv")

# Save EDA report
import json

eda_report = {
    'Dataset_Overview': {
        'Total_Rows': len(df),
        'Total_Columns': len(df.columns),
        'Numerical_Features': len(numerical_cols),
        'Categorical_Features': len(categorical_cols),
        'Missing_Values_After_Imputation': int(df.isnull().sum().sum()),
        'Duplicate_Rows': int(df.duplicated().sum())
    },
    'Imputation_Summary': {
        'Age': 'Custom (Title + Pclass median)',
        'Embarked': 'Mode imputation',
        'Fare': 'Median by Pclass',
        'Cabin': 'Converted to binary indicator (Cabin_Known)'
    },
    'Key_Insights': {
        'Overall_Survival_Rate': f"{df['Survived'].mean() * 100:.2f}%" if 'Survived' in df.columns else 'N/A',
        'Strong_Correlations': len(corr_pairs) if corr_pairs else 0,
        'Features_With_Outliers': len([col for col in numerical_cols 
                                      if len(df[col].dropna()[(df[col] < df[col].quantile(0.25) - 1.5*(df[col].quantile(0.75)-df[col].quantile(0.25))) | 
                                                               (df[col] > df[col].quantile(0.75) + 1.5*(df[col].quantile(0.75)-df[col].quantile(0.25)))]) > 0])
    },
    'Recommendations': recommendations
}

with open('complete_eda_report.json', 'w') as f:
    json.dump(eda_report, f, indent=4, default=str)

print("✓ Saved: complete_eda_report.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 100)
print("✨ COMPLETE EDA PIPELINE FINISHED!")
print("=" * 100)

print(f"""
🎉 COMPREHENSIVE ANALYSIS SUMMARY
{'='*100}

📊 Analysis Phases Completed:
   ✓ Phase 1: Data Loading & Initial Exploration
   ✓ Phase 2: Missing Data Analysis & Advanced Imputation
   ✓ Phase 3: Distribution Analysis with KDE
   ✓ Phase 4: Outlier Detection (Violin + Box Plots)
   ✓ Phase 5: Correlation Analysis
   ✓ Phase 6: Categorical Feature Analysis
   ✓ Phase 7: Target Variable Analysis (9-panel dashboard)
   ✓ Phase 8: Interactive Visualizations (Plotly)
   ✓ Phase 9: Key Findings & Recommendations
   ✓ Phase 10: Output Generation

🎨 Visualizations Generated:
   • Missing data analysis dashboard (6 panels)
   • Distribution plots with KDE overlays (6 features)
   • Violin + Box plot combinations (6 features)
   • Modern correlation heatmap
   • Categorical analysis (horizontal bars)
   • Comprehensive survival dashboard (9 panels)
   • 3D interactive scatter plot
   • Hierarchical sunburst chart
   • Parallel coordinates plot
   • Enhanced scatter with marginals

💾 Files Generated:
   📄 titanic_cleaned_imputed.csv - Ready for modeling
   📄 summary_statistics.csv - Statistical summary
   📄 correlation_matrix.csv - Correlation data
   📄 complete_eda_report.json - Comprehensive report

🚀 Ready for Machine Learning:
   • Data cleaned and imputed
   • Outliers identified
   • Features engineered
   • Patterns discovered
   • Recommendations provided

💡 Next Step: Build and train ML models!

{'='*100}
🎊 Thank you for using the Complete EDA Pipeline!
{'='*100}
""")

print("\n💡 Pro Tip: Use the cleaned dataset 'titanic_cleaned_imputed.csv' for model training!")
print("📚 Review 'complete_eda_report.json' for detailed insights and recommendations.\n")