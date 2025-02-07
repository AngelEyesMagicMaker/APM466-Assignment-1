from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# Compute daily yield changes
yield_changes = bond_data.iloc[:, -10:].diff(axis=1).dropna(axis=1)  # Compute daily differences

# Perform PCA
pca = PCA(n_components=3)  # First 3 principal components
pca.fit(yield_changes.T)  # Transpose to align time-series data

# Extract PCA results
explained_variance = pca.explained_variance_ratio_
principal_components = pca.components_

# Store results in DataFrame
pca_results = pd.DataFrame({
    "Principal Component": [1, 2, 3],
    "Explained Variance": explained_variance
})

# Print the results
print(pca_results)

# Interpretation of PCA on Yield Changes
#
# PC1: Level Shift – Explains most variance, indicating parallel shifts in the
# yield curve where interest rates move together. Driven by monetary policy,
# inflation, or economic growth.
#
# PC2: Slope Change – Captures steepening or flattening of the curve. A positive
# shift means short-term rates fall while long-term rates rise (steepening);
# a negative shift means the opposite (flattening), often linked to rate hike
# expectations or recessions.
#
# PC3: Curvature Change – Represents bowing effects where medium-term rates move
# differently from short- and long-term rates. Positive shifts lower mid-term
# yields, while negative shifts raise them, often due to liquidity or risk
# sentiment changes.
#
# Key Takeaways: PC1 drives most yield movements, PC2 reflects short vs.
# long-term rate changes, and PC3 captures mid-curve behavior.
