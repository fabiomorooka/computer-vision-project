# encoding: utf-8
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Because they are images we will use MinMaxScaler
# Transform features by scaling each feature to a given range.
# Ref. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
def scale(x, name):
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    print(f'Scale per feature relative of the data {scaler.scale_}')
    print(f'Shape encoded {name} label : {x.shape}')

    return x

def plot(x, path, name=None):
    plt.figure(figsize=(20, 15))
    pca = PCA(n_components=x.shape[1]).fit(x)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.title(f'PCA analysis in the {name} dataset')
    plt.savefig(path)

# Linear dimensionality reduction using Singular Value Decomposition
# Ref.: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
def analyse(x, name, x_train=None):
    pca = PCA(n_components=50, svd_solver='full')
    pca.fit(x) if x_train is None else pca.fit(x_train)
    pca_x = pca.transform(x)
    if x_train is None: print(f'Sum variance ratio : {100*pca.explained_variance_ratio_.sum():.6f}%')
    print(f'Shape PCA {name} label : {pca_x.shape}')

    return pca_x