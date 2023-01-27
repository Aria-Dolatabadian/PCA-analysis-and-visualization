from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from bioinfokit.analys import get_data
import numpy as np
import pandas as pd
df = pd.read_csv('PCA.csv')
print(df)
#Generate correlation matrix plot
df_st =  StandardScaler().fit_transform(df)
pd.DataFrame(df_st, columns=df.columns).head(2)
pca_out = PCA().fit(df_st)
pca_out.explained_variance_ratio_
np.cumsum(pca_out.explained_variance_ratio_)
loadings = pca_out.components_
num_pc = pca_out.n_features_
pc_list = ["PC" + str(i) for i in list(range(1, num_pc + 1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = df.columns.values
loadings_df = loadings_df.set_index('variable')
loadings_df
import seaborn as sns
import matplotlib.pyplot as plt
ax = sns.heatmap(loadings_df, annot=True, cmap='Spectral')
plt.show()
#Generate Scree plot
pca_out.explained_variance_
from bioinfokit.visuz import cluster
cluster.screeplot(obj=[pc_list, pca_out.explained_variance_ratio_])
#Generate 2D PCA loadings plot (2 PCs) plot and 3D PCA loadings plot (3 PCs) plot
cluster.pcaplot(x=loadings[0], y=loadings[1], labels=df.columns.values,
    var1=round(pca_out.explained_variance_ratio_[0]*100, 2),
    var2=round(pca_out.explained_variance_ratio_[1]*100, 2))
cluster.pcaplot(x=loadings[0], y=loadings[1], z=loadings[2],  labels=df.columns.values,
    var1=round(pca_out.explained_variance_ratio_[0]*100, 2), var2=round(pca_out.explained_variance_ratio_[1]*100, 2),
    var3=round(pca_out.explained_variance_ratio_[2]*100, 2))
#Generate 2D and 3D biplot
pca_scores = PCA().fit_transform(df_st)
cluster.biplot(cscore=pca_scores, loadings=loadings, labels=df.columns.values,
               var1=round(pca_out.explained_variance_ratio_[0] * 100, 2),
               var2=round(pca_out.explained_variance_ratio_[1] * 100, 2))
cluster.biplot(cscore=pca_scores, loadings=loadings, labels=df.columns.values,
               var1=round(pca_out.explained_variance_ratio_[0] * 100, 2),
               var2=round(pca_out.explained_variance_ratio_[1] * 100, 2),
               var3=round(pca_out.explained_variance_ratio_[2] * 100, 2))

#See WD to find the results
