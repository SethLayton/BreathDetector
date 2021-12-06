import os
from matplotlib.pyplot import cla
from python_speech_features import mfcc
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier as knn
import sklearn
from sklearn.metrics import f1_score
import pickle
from FeatureExtractor import extract_features
from GridSearch import grid_search
from Plot import pca
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

##Extract features from the audio files based on the manual annotations
all_feats_x, all_feats_y = extract_features()


##Train/Test split       
X_train_n, X_test_n, Y_train_n, Y_test_n = sklearn.model_selection.train_test_split(all_feats_x, all_feats_y, test_size = 0.33, random_state = 42)
X_train_n = np.array(X_train_n)
X_test_n = np.array(X_test_n)
Y_train_n = np.array(Y_train_n)
Y_test_n = np.array(Y_test_n)


####Used for hyper-parameter tuning

# neighbors = [5, 6, 8]
# param_grid = {
#     "knn__n_neighbors": neighbors
# }
# grid_search(X_train_n, Y_train_n, param_grid=param_grid)


filename_sav = ".\\saved_objects\\finalized_model.sav"
##read model from file if stored
if not os.path.exists(filename_sav):
    ##Build the model
    pipe = Pipeline([('scaler', MinMaxScaler()), ('knn', knn(n_neighbors=5, weights='distance'))])
    ##Fit the model
    model_n = pipe.fit(X_train_n, Y_train_n)
    ##Write the model out to storage

    pickle.dump(model_n, open(filename_sav, 'wb'))
else:
    model_n= pickle.load(open(filename_sav, 'rb'))


##Get a training score
train_score_n = model_n.predict(X_train_n)
score_n = f1_score(Y_train_n,train_score_n, pos_label=1)
##Get a test score
test_score_n = model_n.predict(X_test_n)
score_n = f1_score(Y_test_n, test_score_n, pos_label=1)


print('training F1 Score: %.3f' % score_n)
print('test F1 Score: %.3f' % score_n)

##Call function to plot PCA on Scree graph to check feasibility
pca(all_feats_x)