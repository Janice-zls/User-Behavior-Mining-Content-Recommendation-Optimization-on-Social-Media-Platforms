import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- SCI Style Configuration ---
# Setting up plot style for publication-quality figures
plt.rcParams['font.family'] = 'sans-serif'
# Prioritize fonts that support Chinese characters on Windows
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False # Fix for minus sign
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = False # Cleaner look often preferred
plt.rcParams['lines.linewidth'] = 2

# Using a high-contrast palette
sns.set_palette("viridis")

def analyze_data():
    file_path = r"g:\B.比赛\2026正大杯\2.社交媒体平台用户行为特征挖掘与内容推荐优化研究——基于问卷调研与用户互动数据的实证分析\问卷星.csv"
    output_dir = os.path.dirname(file_path)
    
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig') # Assuming utf-8-sig from previous step
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print("\n--- Data Overview ---")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())

    # --- Data Preprocessing ---
    # Define Factor Groups (Based on column naming convention F1_Q1...F5_Q6)
    factor_groups = {
        'F1_使用习惯': [col for col in df.columns if col.startswith('F1_')],
        'F2_互动参与': [col for col in df.columns if col.startswith('F2_')],
        'F3_推荐质量': [col for col in df.columns if col.startswith('F3_')],
        'F4_隐私关注': [col for col in df.columns if col.startswith('F4_')],
        'F5_内容价值': [col for col in df.columns if col.startswith('F5_')]
    }
    
    # Calculate Mean Scores for each Factor
    for factor_name, cols in factor_groups.items():
        if cols:
            df[f'{factor_name}_Mean'] = df[cols].mean(axis=1)
            print(f"\nCalculated mean for {factor_name} using columns: {cols}")

    # Encode Categorical Targets for Correlation
    # C1_信息茧房感知: 总是感觉 > 经常 > 偶尔 > 很少 > 从不 (Assuming ordinal)
    c1_mapping = {
        '总是感觉': 5, '经常感觉': 4, '偶尔感觉': 3, '很少感觉': 2, '从不感觉': 1
    }
    # C2_平台总体满意度: 非常满意 > 满意 > 一般 > 不满意 > 非常不满意
    c2_mapping = {
        '非常满意': 5, '满意': 4, '一般': 3, '不满意': 2, '非常不满意': 1
    }
    
    df['C1_Score'] = df['C1_信息茧房感知'].map(c1_mapping)
    df['C2_Score'] = df['C2_平台总体满意度'].map(c2_mapping)

    # --- Descriptive Statistics ---
    print("\n--- Descriptive Statistics (Factors) ---")
    factor_means = [f'{k}_Mean' for k in factor_groups.keys()]
    print(df[factor_means].describe())

    # --- Correlation Analysis ---
    correlation_cols = factor_means + ['C1_Score', 'C2_Score']
    corr_matrix = df[correlation_cols].corr()
    
    print("\n--- Correlation Matrix ---")
    print(corr_matrix)

    # --- Visualizations ---
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Factors and Satisfaction', fontweight='bold')
    heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, bbox_inches='tight')
    print(f"Saved heatmap to {heatmap_path}")
    plt.close()

    # 2. Factor Distributions (Box Plot)
    plt.figure(figsize=(14, 8))
    # Melt dataframe for seaborn boxplot
    melted_df = df[factor_means].melt(var_name='Factor', value_name='Score')
    sns.boxplot(x='Factor', y='Score', data=melted_df, palette="viridis")
    plt.title('Distribution of Scores Across Factors', fontweight='bold')
    plt.xticks(rotation=45)
    boxplot_path = os.path.join(output_dir, 'factor_boxplot.png')
    plt.savefig(boxplot_path, bbox_inches='tight')
    print(f"Saved boxplot to {boxplot_path}")
    plt.close()
    
    # 3. Demographics: Age Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(y='A2_年龄段', data=df, order=df['A2_年龄段'].value_counts().index, palette="magma")
    plt.title('Age Distribution of Respondents', fontweight='bold')
    age_path = os.path.join(output_dir, 'age_distribution.png')
    plt.savefig(age_path, bbox_inches='tight')
    print(f"Saved age distribution to {age_path}")
    plt.close()

    # 4. Platform Usage
    plt.figure(figsize=(10, 6))
    sns.countplot(y='B1_最常用平台', data=df, order=df['B1_最常用平台'].value_counts().index, palette="plasma")
    plt.title('Most Used Social Media Platforms', fontweight='bold')
    platform_path = os.path.join(output_dir, 'platform_usage.png')
    plt.savefig(platform_path, bbox_inches='tight')
    print(f"Saved platform usage to {platform_path}")
    plt.close()

    print("\nAnalysis Complete.")

if __name__ == "__main__":
    analyze_data()
