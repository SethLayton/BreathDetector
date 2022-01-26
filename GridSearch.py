from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as knn
import pickle


def grid_search (x,y, param_grid):
    ##Set features and labels
    X_train_n = x
    Y_train_n = y

    ##create the pipeline
    pipe = Pipeline([('scaler', MinMaxScaler()), ('knn', knn())])
    #perform the Gridsearch with 5 fold cross validation
    search = GridSearchCV(pipe, param_grid, n_jobs=-1, scoring='f1', verbose=1)
    search.fit(X_train_n, Y_train_n)
    
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    pickle.dump(search.best_estimator_, '.\\saved_objects\\best_model.pkl', compress = 1)