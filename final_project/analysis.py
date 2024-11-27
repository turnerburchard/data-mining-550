# take in arrays of scores and perform ANOVAs, etc on them

import pandas as pd
from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# Example Data: Replace these with your actual silhouette scores
data = {
    'Algorithm': [
        'KMeans', 'KMeans', 'KMeans',
        'DBSCAN', 'DBSCAN', 'DBSCAN',
        'Hierarchical', 'Hierarchical', 'Hierarchical',
        'SOM', 'SOM', 'SOM'
    ],
    'SilhouetteScore': [
        0.65, 0.67, 0.64,  # KMeans scores
        0.55, 0.56, 0.54,  # DBSCAN scores
        0.62, 0.63, 0.61,  # Hierarchical scores
        0.70, 0.68, 0.69   # SOM scores
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Perform ANOVA
anova_result = f_oneway(
    df[df['Algorithm'] == 'KMeans']['SilhouetteScore'],
    df[df['Algorithm'] == 'DBSCAN']['SilhouetteScore'],
    df[df['Algorithm'] == 'Hierarchical']['SilhouetteScore'],
    df[df['Algorithm'] == 'SOM']['SilhouetteScore']
)

print(f"ANOVA F-statistic: {anova_result.statistic}, p-value: {anova_result.pvalue}")

# Check for significance
if anova_result.pvalue < 0.05:
    print("Significant differences found. Performing pairwise comparisons using Tukey's HSD...")

    # Perform Tukey's HSD test
    tukey_result = pairwise_tukeyhsd(
        endog=df['SilhouetteScore'],  # Dependent variable
        groups=df['Algorithm'],      # Independent variable
        alpha=0.05                   # Significance level
    )
    print(tukey_result)
else:
    print("No significant differences found between the algorithms.")

# Optionally, for non-parametric data
kruskal_result = kruskal(
    df[df['Algorithm'] == 'KMeans']['SilhouetteScore'],
    df[df['Algorithm'] == 'DBSCAN']['SilhouetteScore'],
    df[df['Algorithm'] == 'Hierarchical']['SilhouetteScore'],
    df[df['Algorithm'] == 'SOM']['SilhouetteScore']
)

print(f"Kruskal-Wallis H-statistic: {kruskal_result.statistic}, p-value: {kruskal_result.pvalue}")
if kruskal_result.pvalue < 0.05:
    print("Significant differences found in non-parametric test.")
