from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

##Function for mapping and plotting the Principal components on a Scree graph
def pca(X):
    
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled=scaler.transform(X)    

    pca = PCA(n_components=6, svd_solver = 'auto')
    Principal_components=pca.fit_transform(X_scaled)
    #print(Principal_components)

    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.show()