import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from scipy.stats import pearsonr, ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Configuration
OUTPUT_DIR = r"g:\B.比赛\2026正大杯\2.社交媒体平台用户行为特征挖掘与内容推荐优化研究——基于问卷调研与用户互动数据的实证分析\thesis_tables"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_and_preprocess():
    file_path = r"g:\B.比赛\2026正大杯\2.社交媒体平台用户行为特征挖掘与内容推荐优化研究——基于问卷调研与用户互动数据的实证分析\问卷星.csv"
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(file_path, encoding='gbk')

    # Mapping
    c1_mapping = {'总是感觉': 5, '经常感觉': 4, '偶尔感觉': 3, '很少感觉': 2, '从不感觉': 1}
    c2_mapping = {'非常满意': 5, '满意': 4, '一般': 3, '不满意': 2, '非常不满意': 1}
    df['C1_Score'] = df['C1_信息茧房感知'].map(c1_mapping).fillna(3) 
    df['C2_Score'] = df['C2_平台总体满意度'].map(c2_mapping).fillna(3)

    # Factor Groups
    factor_groups = {
        'F1_Habits': [c for c in df.columns if c.startswith('F1_')],
        'F2_Interaction': [c for c in df.columns if c.startswith('F2_')],
        'F3_Recommendation': [c for c in df.columns if c.startswith('F3_')],
        'F4_Privacy': [c for c in df.columns if c.startswith('F4_')],
        'F5_Content': [c for c in df.columns if c.startswith('F5_')]
    }
    
    # Calculate Means
    for name, cols in factor_groups.items():
        df[f'{name}_Mean'] = df[cols].mean(axis=1)

    return df, factor_groups

def cronbach_alpha(df):
    # Cronbach's alpha calculation
    df_corr = df.corr()
    N = df.shape[1]
    rs = np.array([])
    for i, col1 in enumerate(df_corr.columns):
        for j, col2 in enumerate(df_corr.columns):
            if i > j:
                rs = np.append(rs, df_corr[col1][col2])
    mean_r = np.mean(rs)
    cronbach_alpha = (N * mean_r) / (1 + (N - 1) * mean_r)
    return cronbach_alpha

def generate_demographics_table(df):
    """Generates Table 3-1: Sample Basic Information Description"""
    demographic_cols = {
        'A1_性别': 'Gender',
        'A2_年龄段': 'Age Group',
        'A3_学历': 'Education',
        'A4_职业': 'Occupation',
        'A5_月收入': 'Income',
        'A6_所在城市级别': 'City Tier'
    }
    
    rows = []
    for col, name in demographic_cols.items():
        counts = df[col].value_counts()
        percentages = df[col].value_counts(normalize=True) * 100
        for cat in counts.index:
            rows.append({
                'Variable': name,
                'Category': cat,
                'Frequency (N)': counts[cat],
                'Percentage (%)': f"{percentages[cat]:.2f}%"
            })
    
    result_df = pd.DataFrame(rows)
    result_df.to_csv(os.path.join(OUTPUT_DIR, 'Table_3_1_Demographics.csv'), index=False, encoding='utf-8-sig')
    print("Generated Table 3-1: Demographics")

def generate_reliability_table(df, factor_groups):
    """Generates Table 3-2: Reliability and Validity Analysis (Simplified)"""
    rows = []
    for name, cols in factor_groups.items():
        subset = df[cols]
        alpha = cronbach_alpha(subset)
        # KMO is complex to implement from scratch without factor_analyzer, skipping or using placeholder
        # We will focus on Cronbach's Alpha and Mean/SD here
        mean_val = subset.mean().mean()
        std_val = subset.std().mean()
        
        rows.append({
            'Latent Variable': name,
            'Number of Items': len(cols),
            'Cronbach\'s Alpha': f"{alpha:.3f}",
            'Mean': f"{mean_val:.2f}",
            'SD': f"{std_val:.2f}"
        })
        
    # Add C1 and C2 if they were multi-item, but they are single items here based on previous code.
    # If they are single items, alpha is not applicable.
    
    result_df = pd.DataFrame(rows)
    result_df.to_csv(os.path.join(OUTPUT_DIR, 'Table_3_2_Reliability.csv'), index=False, encoding='utf-8-sig')
    print("Generated Table 3-2: Reliability")

def generate_descriptive_stats(df, factor_groups):
    """Generates Table 4-1: Descriptive Statistics for Latent Variables"""
    # Includes Mean, SD, Skewness, Kurtosis
    rows = []
    factor_means = [f'{k}_Mean' for k in factor_groups.keys()]
    factor_means.extend(['C1_Score', 'C2_Score'])
    
    mapping = {
        'F1_Habits_Mean': 'Platform Habits',
        'F2_Interaction_Mean': 'Interaction',
        'F3_Recommendation_Mean': 'Recommendation Quality',
        'F4_Privacy_Mean': 'Privacy Concern',
        'F5_Content_Mean': 'Content Value',
        'C1_Score': 'Information Cocoon',
        'C2_Score': 'Satisfaction'
    }

    for col in factor_means:
        desc = df[col].describe()
        skew = df[col].skew()
        kurt = df[col].kurt()
        
        rows.append({
            'Variable': mapping.get(col, col),
            'N': desc['count'],
            'Min': f"{desc['min']:.2f}",
            'Max': f"{desc['max']:.2f}",
            'Mean': f"{desc['mean']:.2f}",
            'SD': f"{desc['std']:.2f}",
            'Skewness': f"{skew:.3f}",
            'Kurtosis': f"{kurt:.3f}"
        })
        
    result_df = pd.DataFrame(rows)
    result_df.to_csv(os.path.join(OUTPUT_DIR, 'Table_4_1_Descriptive_Stats.csv'), index=False, encoding='utf-8-sig')
    print("Generated Table 4-1: Descriptive Statistics")

def generate_cluster_table(df, factor_groups):
    """Generates Table 4-3: K-Means Cluster Profiles"""
    factor_means_cols = [f'{k}_Mean' for k in factor_groups.keys()]
    X = df[factor_means_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Re-run clustering to ensure consistency
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Calculate means by cluster
    cluster_means = df.groupby('Cluster')[factor_means_cols].mean().T
    cluster_means.columns = ['Cluster 1 (Active)', 'Cluster 2 (Passive)', 'Cluster 3 (Privacy)']
    
    # Add count/percentage row
    counts = df['Cluster'].value_counts().sort_index()
    percentages = df['Cluster'].value_counts(normalize=True).sort_index() * 100
    
    # Append count info
    count_row = pd.Series({
        'Cluster 1 (Active)': f"{counts[0]} ({percentages[0]:.1f}%)",
        'Cluster 2 (Passive)': f"{counts[1]} ({percentages[1]:.1f}%)",
        'Cluster 3 (Privacy)': f"{counts[2]} ({percentages[2]:.1f}%)"
    }, name='Sample Size (N/%)')
    
    result_df = pd.concat([cluster_means, count_row.to_frame().T])
    result_df.index.name = 'Variable'
    result_df.reset_index(inplace=True)
    
    result_df.to_csv(os.path.join(OUTPUT_DIR, 'Table_4_3_Cluster_Profiles.csv'), index=False, encoding='utf-8-sig')
    print("Generated Table 4-3: Cluster Profiles")

def generate_correlation_matrix(df, factor_groups):
    """Generates Table 5-1: Pearson Correlation Matrix"""
    cols = [f'{k}_Mean' for k in factor_groups.keys()] + ['C1_Score', 'C2_Score']
    labels = ['Habits', 'Interaction', 'Recommendation', 'Privacy', 'Content', 'Cocoon', 'Satisfaction']
    
    corr_matrix = df[cols].corr()
    
    # Format with stars for significance
    # Need to calculate p-values
    p_values = pd.DataFrame(np.zeros_like(corr_matrix), columns=cols, index=cols)
    for i in cols:
        for j in cols:
            if i != j:
                _, p = pearsonr(df[i], df[j])
                p_values.loc[i, j] = p
            else:
                p_values.loc[i, j] = 0.0
                
    formatted_matrix = corr_matrix.copy().astype(str)
    for i in cols:
        for j in cols:
            val = corr_matrix.loc[i, j]
            p = p_values.loc[i, j]
            star = ""
            if p < 0.01: star = "**"
            elif p < 0.05: star = "*"
            
            if i == j:
                formatted_matrix.loc[i, j] = "1"
            else:
                formatted_matrix.loc[i, j] = f"{val:.3f}{star}"
                
    formatted_matrix.columns = labels
    formatted_matrix.index = labels
    
    formatted_matrix.to_csv(os.path.join(OUTPUT_DIR, 'Table_5_1_Correlation.csv'), index=True, encoding='utf-8-sig')
    print("Generated Table 5-1: Correlation Matrix")

def generate_regression_table(df, factor_groups):
    """Generates Table 5-2: Regression Analysis Results"""
    factor_means_cols = [f'{k}_Mean' for k in factor_groups.keys()]
    X = df[factor_means_cols]
    # Add Cocoon Perception as Independent Variable as per Hypothesis 6
    X['Information_Cocoon'] = df['C1_Score']
    
    y = df['C2_Score'] # Satisfaction
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    
    # Extract results
    summary_df = pd.DataFrame({
        'Variable': X.columns,
        'Beta (Coeff)': model.params,
        'Std. Error': model.bse,
        't-value': model.tvalues,
        'p-value': model.pvalues,
    })
    
    # VIF Calculation
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_data = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    summary_df['VIF'] = vif_data
    
    # Add significance stars to Beta
    summary_df['Sig.'] = summary_df['p-value'].apply(lambda p: '**' if p < 0.01 else ('*' if p < 0.05 else ''))
    
    # Formatting
    summary_df['Beta (Coeff)'] = summary_df.apply(lambda row: f"{row['Beta (Coeff)']:.3f}{row['Sig.']}", axis=1)
    summary_df['p-value'] = summary_df['p-value'].apply(lambda x: f"{x:.3e}" if x < 0.001 else f"{x:.3f}")
    summary_df['VIF'] = summary_df['VIF'].apply(lambda x: f"{x:.2f}")
    
    # Add R-squared info as a separate row or just print
    r2_row = {'Variable': 'R-squared', 'Beta (Coeff)': f"{model.rsquared:.3f}"}
    adj_r2_row = {'Variable': 'Adj. R-squared', 'Beta (Coeff)': f"{model.rsquared_adj:.3f}"}
    f_row = {'Variable': 'F-statistic', 'Beta (Coeff)': f"{model.fvalue:.3f}"}
    
    summary_df = pd.concat([summary_df, pd.DataFrame([r2_row, adj_r2_row, f_row])], ignore_index=True)
    
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'Table_5_2_Regression.csv'), index=False, encoding='utf-8-sig')
    print("Generated Table 5-2: Regression Analysis")
    
    # Generate Hypothesis Summary (Table 5-4)
    # H1: Habits -> Sat
    # H2: Interaction -> Sat
    # H3: Rec -> Sat
    # H4: Privacy -> Sat (-)
    # H5: Content -> Sat
    # H6: Cocoon -> Sat (-)
    
    hypotheses = [
        {'H': 'H1', 'Path': '使用习惯 -> 满意度', 'Var': 'F1_Habits_Mean', 'Expected': '+'},
        {'H': 'H2', 'Path': '互动参与 -> 满意度', 'Var': 'F2_Interaction_Mean', 'Expected': '+'},
        {'H': 'H3', 'Path': '推荐质量 -> 满意度', 'Var': 'F3_Recommendation_Mean', 'Expected': '+'},
        {'H': 'H4', 'Path': '隐私关注 -> 满意度', 'Var': 'F4_Privacy_Mean', 'Expected': '-'},
        {'H': 'H5', 'Path': '内容价值 -> 满意度', 'Var': 'F5_Content_Mean', 'Expected': '+'},
        {'H': 'H6', 'Path': '信息茧房 -> 满意度', 'Var': 'Information_Cocoon', 'Expected': '-'}
    ]
    
    h_rows = []
    for h in hypotheses:
        # Find coeff and p-value
        if h['Var'] in model.params:
            beta = model.params[h['Var']]
            p = model.pvalues[h['Var']]
            
            supported = "不成立"
            if p < 0.05:
                if (h['Expected'] == '+' and beta > 0) or (h['Expected'] == '-' and beta < 0):
                    supported = "成立"
            
            h_rows.append({
                '假设编号': h['H'],
                '假设路径': h['Path'],
                '回归系数': f"{beta:.3f}",
                'P值': f"{p:.3f}",
                '验证结果': supported
            })
            
    h_df = pd.DataFrame(h_rows)
    h_df.to_csv(os.path.join(OUTPUT_DIR, 'Table_5_4_Hypothesis_Test.csv'), index=False, encoding='utf-8-sig')
    print("Generated Table 5-4: Hypothesis Testing Summary")

def generate_item_descriptive_stats(df, factor_groups):
    """Generates Table 4-0: Item-Level Descriptive Statistics (Mean & SD)"""
    rows = []
    
    # Collect all items
    all_items = []
    for group_name, cols in factor_groups.items():
        for col in cols:
            all_items.append((col, group_name))
            
    for col, group in all_items:
        desc = df[col].describe()
        rows.append({
            'Factor': group,
            'Item Code': col,
            'N': desc['count'],
            'Mean': f"{desc['mean']:.2f}",
            'SD': f"{desc['std']:.2f}",
            'Skewness': f"{df[col].skew():.2f}",
            'Kurtosis': f"{df[col].kurt():.2f}"
        })
        
    result_df = pd.DataFrame(rows)
    result_df.to_csv(os.path.join(OUTPUT_DIR, 'Table_4_0_Item_Stats.csv'), index=False, encoding='utf-8-sig')
    print("Generated Table 4-0: Item-Level Statistics")

def generate_platform_usage_crosstab(df):
    """Generates Table 4-1-1: Platform Usage Frequency & Duration"""
    # Cross-tab of Platform vs Daily Usage Time
    # Assuming A7 is Platform and A8 is Daily Usage Time (based on context, might need verification if column names differ)
    # Let's check columns first. Based on previous code, we don't have explicit A7/A8 mapping in load_and_preprocess.
    # We will infer or use known columns if available. 
    # Since I don't have exact column mapping for A7/A8 in the snippet, I will check if they exist or skip.
    
    # Let's try to find them in df.columns
    platform_col = 'A7_主要使用的社交媒体平台' if 'A7_主要使用的社交媒体平台' in df.columns else None
    time_col = 'A8_日均使用时长' if 'A8_日均使用时长' in df.columns else None
    
    if platform_col and time_col:
        crosstab = pd.crosstab(df[platform_col], df[time_col], margins=True, margins_name='Total')
        crosstab.to_csv(os.path.join(OUTPUT_DIR, 'Table_4_1_1_Platform_Time_Crosstab.csv'), encoding='utf-8-sig')
        print("Generated Table 4-1-1: Platform Usage Crosstab")
    else:
        print("Skipping Table 4-1-1: Columns not found")

def generate_cluster_evaluation_metrics(df, factor_groups):
    """Generates Table 4-3-1: Cluster Evaluation Metrics (Elbow/Silhouette)"""
    from sklearn.metrics import silhouette_score
    
    factor_means_cols = [f'{k}_Mean' for k in factor_groups.keys()]
    X = df[factor_means_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    rows = []
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertia = kmeans.inertia_
        sil = silhouette_score(X_scaled, labels)
        
        rows.append({
            'Number of Clusters (k)': k,
            'Inertia (SSE)': f"{inertia:.2f}",
            'Silhouette Score': f"{sil:.3f}"
        })
        
    result_df = pd.DataFrame(rows)
    result_df.to_csv(os.path.join(OUTPUT_DIR, 'Table_4_3_1_Cluster_Evaluation.csv'), index=False, encoding='utf-8-sig')
    print("Generated Table 4-3-1: Cluster Evaluation Metrics")

def generate_group_differences_anova(df, factor_groups):
    """Generates Table 5-3: Demographic Differences Analysis (ANOVA/T-test) on Satisfaction"""
    from scipy.stats import f_oneway, ttest_ind
    
    target_var = 'C2_Score' # Satisfaction
    
    demographics = {
        'Gender': 'A1_性别',
        'Age': 'A2_年龄段',
        'Education': 'A3_学历',
        'CityTier': 'A6_所在城市级别'
    }
    
    rows = []
    
    for label, col in demographics.items():
        if col not in df.columns:
            continue
            
        groups = df.groupby(col)[target_var].apply(list)
        group_names = groups.index.tolist()
        group_data = [groups[g] for g in group_names]
        
        # Calculate overall mean for groups to show difference
        group_means = df.groupby(col)[target_var].mean()
        mean_str = " | ".join([f"{n}:{m:.2f}" for n, m in group_means.items()])
        
        if len(group_data) < 2:
            continue
            
        # Use ANOVA for >2 groups, T-test for 2 groups (or ANOVA, equivalent for 2)
        stat, p = f_oneway(*group_data)
        
        sig = ""
        if p < 0.01: sig = "**"
        elif p < 0.05: sig = "*"
        
        rows.append({
            'Demographic Variable': label,
            'Group Means': mean_str,
            'F-value': f"{stat:.3f}",
            'P-value': f"{p:.3f}",
            'Significance': sig
        })
        
    result_df = pd.DataFrame(rows)
    result_df.to_csv(os.path.join(OUTPUT_DIR, 'Table_5_3_Group_Differences_ANOVA.csv'), index=False, encoding='utf-8-sig')
    print("Generated Table 5-3: Group Differences (ANOVA)")

def main():
    print("Starting Table Generation...")
    df, factor_groups = load_and_preprocess()
    
    generate_demographics_table(df)
    generate_reliability_table(df, factor_groups)
    generate_descriptive_stats(df, factor_groups)
    generate_cluster_table(df, factor_groups)
    generate_correlation_matrix(df, factor_groups)
    generate_regression_table(df, factor_groups)
    
    # New Tables
    generate_item_descriptive_stats(df, factor_groups)
    generate_platform_usage_crosstab(df)
    generate_cluster_evaluation_metrics(df, factor_groups)
    generate_group_differences_anova(df, factor_groups)
    
    print(f"All tables saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
