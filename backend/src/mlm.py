#this should let gpu acceleration run
import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer
from tabulate import tabulate

def mcc_multioutput(y_true, y_pred):
    """
    Compute MCC for each output column and return the mean.
    Works with y_true, y_pred of shape (n_samples, n_outputs).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # If 1D, just use normal MCC
    if y_true.ndim == 1 or y_true.shape[1] == 1:
        return matthews_corrcoef(y_true.ravel(), y_pred.ravel())

    mccs = []
    for i in range(y_true.shape[1]):
        mccs.append(matthews_corrcoef(y_true[:, i], y_pred[:, i]))
    return float(np.mean(mccs))

#This functions uses GridSearch to fine tune the parameters for the models
def grid_search_tune(estimator, param_grid, X, y):
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=make_scorer(mcc_multioutput),
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X, y)
    return grid

#Step 3 - Define the dataset and use pandas to read the csv, only keeping the 5 parameters we want
def getDataset(filename):
    #defines the column names, and then defines the column names we want
    names = ["LightLevel", "Weather", "RoadCondition", "Longitude", "Latitude"]
    colsToUse = ["LightLevel", "Weather", "RoadCondition", "Longitude", "Latitude"]
    
    #creates the dataset
    dataset = pd.read_csv(filename, names=names, skiprows=1, usecols=colsToUse).dropna().reset_index(drop=True)
    return dataset

#Step 4 - Standardizing the data and removing text
def scaleData(dataset):
    #Text to Number Map
    lightLevelMap = {
        'Not reported':0, 'Other':0, 'Reported but invalid':0, 'Unknown':0,
        'Dawn':1, 'Dusk':1,
        'Daylight':2,
        'Dark - lighted roadway':3, 'Dark - roadway not lighted':3,
        'Dark - unknown roadway lighting':3
    }
    
    #Note that blowing sand is blowing sand or snow
    weatherMap = {'Blowing sand/snow':1,
                  'Blowing sand/snow/Blowing sand/snow':1,
                  'Blowing sand/snow/Clear':1,
                  'Blowing sand/snow/Cloudy':1,
                  'Blowing sand/snow/Fog/smog/smoke':1,
                  'Blowing sand/snow/Other':1,
                  'Blowing sand/snow/Rain':1,
                  'Blowing sand/snow/Severe crosswinds':2,
                  'Blowing sand/snow/Sleet/hail (freezing rain or drizzle)':1,
                  'Blowing sand/snow/Snow':1,
                  'Blowing sand/snow/Unknown':1,

                  'Clear':3,
                  'Clear/Blowing sand/snow':1,
                  'Clear/Clear':3,
                  'Clear/Cloudy':4,
                  'Clear/Fog/smog/smoke':5,
                  'Clear/Other':3,
                  'Clear/Rain':6,
                  'Clear/Reported but invalid':3,
                  'Clear/Severe crosswinds':7,
                  'Clear/Sleet/hail (freezing rain or drizzle)':8,
                  'Clear/Snow':9,
                  'Clear/Unknown':3,

                  'Cloudy':4,
                  'Cloudy/Blowing sand/snow':1,
                  'Cloudy/Clear':4,
                  'Cloudy/Cloudy':4,
                  'Cloudy/Fog/smog/smoke':5,
                  'Cloudy/Other':4,
                  'Cloudy/Rain':6,
                  'Cloudy/Reported but invalid':4,
                  'Cloudy/Severe crosswinds':7,
                  'Cloudy/Sleet/hail (freezing rain or drizzle)':8,
                  'Cloudy/Snow':9,
                  'Cloudy/Unknown':4,

                  'Fog/smog/smoke':5,
                  'Fog/smog/smoke/Clear':5,
                  'Fog/smog/smoke/Cloudy':5,
                  'Fog/smog/smoke/Fog/smog/smoke':5,
                  'Fog/smog/smoke/Other':5,
                  'Fog/smog/smoke/Rain':6,
                  'Fog/smog/smoke/Sleet/hail (freezing rain or drizzle)':8,
                  'Fog/smog/smoke/Snow':9,
                  'Fog/smog/smoke/Unknown':5,

                  'Not Reported':0,

                  'Other':0,
                  'Other/Blowing sand/snow':1,
                  'Other/Clear':3,
                  'Other/Cloudy':4,
                  'Other/Fog/smog/smoke':5,
                  'Other/Other':0,
                  'Other/Rain':6,
                  'Other/Severe crosswinds':7,
                  'Other/Sleet/hail (freezing rain or drizzle)':8,
                  'Other/Snow':9,
                  'Other/Unknown':0,

                  'Rain':6,
                  'Rain/Blowing sand/snow':1,
                  'Rain/Clear':6,
                  'Rain/Cloudy':6,
                  'Rain/Fog/smog/smoke':6,
                  'Rain/Other':6,
                  'Rain/Rain':6,
                  'Rain/Reported but invalid':6,
                  'Rain/Severe crosswinds':10,
                  'Rain/Sleet/hail (freezing rain or drizzle)':8,
                  'Rain/Snow':9,
                  'Rain/Unknown':6,

                  'Reported but invalid':0,
                  'Reported but invalid/Reported but invalid':0,

                  'Severe crosswinds':7,
                  'Severe crosswinds/Blowing sand/snow':2,
                  'Severe crosswinds/Clear':7,
                  'Severe crosswinds/Cloudy':7,
                  'Severe crosswinds/Other':7,
                  'Severe crosswinds/Rain':10,
                  'Severe crosswinds/Severe crosswinds':7,
                  'Severe crosswinds/Snow':11,
                  'Severe crosswinds/Unknown':7,

                  'Sleet/hail (freezing rain or drizzle)':8,
                  'Sleet/hail (freezing rain or drizzle)/Blowing sand/snow':8,
                  'Sleet/hail (freezing rain or drizzle)/Clear':8,
                  'Sleet/hail (freezing rain or drizzle)/Cloudy':8,
                  'Sleet/hail (freezing rain or drizzle)/Fog/smog/smoke':8,
                  'Sleet/hail (freezing rain or drizzle)/Other':8,
                  'Sleet/hail (freezing rain or drizzle)/Rain':8,
                  'Sleet/hail (freezing rain or drizzle)/Severe crosswinds':12,
                  'Sleet/hail (freezing rain or drizzle)/Sleet/hail (freezing rain or drizzle)':8,
                  'Sleet/hail (freezing rain or drizzle)/Snow':8,
                  'Sleet/hail (freezing rain or drizzle)/Unknown':8,

                  'Snow':9,
                  'Snow/Blowing sand/snow':9,
                  'Snow/Clear':9,
                  'Snow/Cloudy':9,
                  'Snow/Fog/smog/smoke':9,
                  'Snow/Other':9,
                  'Snow/Rain':9,
                  'Snow/Reported but invalid':9,
                  'Snow/Severe crosswinds':11,
                  'Snow/Sleet/hail (freezing rain or drizzle)':9,
                  'Snow/Snow':9,
                  'Snow/Unknown':9,

                  'Unknown':0,
                  'Unknown/Blowing sand/snow':1,
                  'Unknown/Clear':3,
                  'Unknown/Cloudy':4,
                  'Unknown/Other':0,
                  'Unknown/Rain':6,
                  'Unknown/Reported but invalid':0,
                  'Unknown/Sleet/hail (freezing rain or drizzle)':8,
                  'Unknown/Snow':9,
                  'Unknown/Unknown':0
    }
    
    roadConditionMap = {'Dry':1, 'Ice':2,
                        'Not reported':0, 'Other':0,
                        'Reported but invalid':0, 'Sand/mud/dirt/oil/gravel':3,
                        'Slush':4, 'Snow':5,
                        'Unknown':0, 'Water (standing - moving)':6,
                        'Wet':7
                        }

    #replacing the letters
    scaled_dataset = dataset.replace({
        'LightLevel': lightLevelMap,
        'Weather': weatherMap,
        'RoadCondition': roadConditionMap
    })

    return scaled_dataset
        
#Step 5 - Using RandomUnderSampler to create a balanced dataset of size 678
def balanceDataset(scaled_dataset):
    scaled_dataset = scaled_dataset[:10000]
    # Take the continuous longitude/latitude
    coords = scaled_dataset[["Longitude", "Latitude"]].to_numpy()

    # Discretize them into, say, 10 bins each (tweak n_bins as you like)
    discretizer = KBinsDiscretizer(
        n_bins=10,          # number of bins per dimension
        encode="ordinal",   # returns integers 0..n_bins-1
        strategy="quantile" # each bin has roughly same number of samples
    )

    y_binned = discretizer.fit_transform(coords).astype(int)

    # y now becomes integer class labels for each coordinate dimension
    dataset_results = pd.DataFrame(
        y_binned,
        columns=["Longitude_bin", "Latitude_bin"]
    )
    # X = all other columns
    dataset_balanced = scaled_dataset.drop(columns=["Longitude", "Latitude"])
    return dataset_balanced, dataset_results

def trainAndTest(X_train, X_test, y_train, y_test):

    # Multi-layer Neural Network setup
    mlp = MultiOutputClassifier(MLPClassifier(max_iter=2000, random_state=42))
    mlp_param_grid = {
        "estimator__hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "estimator__activation": ["identity", "logistic", "tanh", "relu"],
        "estimator__learning_rate": ["constant", "invscaling", "adaptive"]
    }

    # Support Vector Machine setup
    svc = MultiOutputClassifier(SVC())
    svc_param_grid = {
        "estimator__C": [0.1, 1, 10, 100],
        "estimator__kernel": ["linear", "poly", "rbf", "sigmoid"],
        "estimator__gamma": ["scale", "auto", 0.1, 0.01, 0.001, 0.0001]
    }

    # K-Nearest Neighbors setup
    knnc = MultiOutputClassifier(KNeighborsClassifier())
    knnc_param_grid = {
        "estimator__n_neighbors": [3, 5, 7, 11, 15],
        "estimator__p": [1, 2, 3],
        "estimator__algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
    }

    # Decision Tree setup
    dtc = MultiOutputClassifier(DecisionTreeClassifier(random_state=42))
    dtc_param_grid = {
        "estimator__criterion": ["gini", "entropy", "log_loss"],
        "estimator__max_depth": [None, 5, 10, 20, 40],
        "estimator__ccp_alpha": [0.0, 0.1, 0.01, 0.001]
    }

    # Logistic Regression setup
    log_reg = MultiOutputClassifier(LogisticRegression(max_iter=2000))
    log_reg_param_grid = [
        {
            "estimator__penalty": ["l1"],
            "estimator__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "estimator__solver": ["liblinear", "saga"]
        },
        {
            "estimator__penalty": ["l2"],
            "estimator__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "estimator__solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
        }
    ]

    #Training the models
    MLNN = grid_search_tune(mlp, mlp_param_grid, X_train, y_train)
    SVM = grid_search_tune(svc, svc_param_grid, X_train, y_train)
    KNN = grid_search_tune(knnc, knnc_param_grid, X_train, y_train)
    DT = grid_search_tune(dtc, dtc_param_grid, X_train, y_train)
    LR = grid_search_tune(log_reg, log_reg_param_grid, X_train, y_train)

    #Prints the summary table from training the models
    print("Step 6 - Training Data Summary:")
    print(
        tabulate(
            [
                ["Multi-layer Neural Network", f"({MLNN.best_params_})", MLNN.best_score_],
                ["Support Vector Machine",      f"({SVM.best_params_})",  SVM.best_score_],
                ["K-Nearest Neighbors",        f"({KNN.best_params_})",  KNN.best_score_],
                ["Decision Tree",              f"({DT.best_params_})",   DT.best_score_],
                ["Logistic Regression",        f"({LR.best_params_})",   LR.best_score_]
            ],
            headers=[
                "ML Trained Model",
                "Its Best Set of Parameter Values",
                "Its MCC-score on the 5-fold Cross Validation on Training Data (80%)"
            ]
        )
    )

    #Stores the scores when testing the trained models against the test data
    test_mcc_scores = {
        "Multi-layer Neural Network": mcc_multioutput(y_test, MLNN.best_estimator_.predict(X_test)),
        "Support Vector Machine":     mcc_multioutput(y_test, SVM.best_estimator_.predict(X_test)),
        "K-Nearest Neighbors":        mcc_multioutput(y_test, KNN.best_estimator_.predict(X_test)),
        "Decision Tree":              mcc_multioutput(y_test, DT.best_estimator_.predict(X_test)),
        "Logistic Regression":        mcc_multioutput(y_test, LR.best_estimator_.predict(X_test))
    }
    
    #Prints the results of testing the models on the test data
    print("Step 7 - Testing Data Summary:")
    print(
        tabulate(
            [
                ["Multi-layer Neural Network", f"({MLNN.best_params_})",
                 mcc_multioutput(y_test, MLNN.best_estimator_.predict(X_test))],
                ["Support Vector Machine",     f"({SVM.best_params_})",
                 mcc_multioutput(y_test, SVM.best_estimator_.predict(X_test))],
                ["K-Nearest Neighbors",        f"({KNN.best_params_})",
                 mcc_multioutput(y_test, KNN.best_estimator_.predict(X_test))],
                ["Decision Tree",              f"({DT.best_params_})",
                 mcc_multioutput(y_test, DT.best_estimator_.predict(X_test))],
                ["Logistic Regression",        f"({LR.best_params_})",
                 mcc_multioutput(y_test, LR.best_estimator_.predict(X_test))]
            ],
            headers=[
                "ML Trained Model",
                "Its Best Set of Parameter Values",
                "Its MCC-score on the 5-fold Cross Validation on Testing Data (20%)"
            ]
        )
    )

    return test_mcc_scores


#Defining main function
def main():
    filename = "oddYears(2005-2019).csv"
    original_dataset = getDataset(filename)
    scaled_dataset = scaleData(original_dataset)
    balanced_dataset, dataset_results = balanceDataset(scaled_dataset)

    print(dataset_results)

    #Step 6 - Splitting the training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        balanced_dataset, dataset_results, test_size=0.2, random_state=42
    )

    test_mcc_scores = trainAndTest(X_train, X_test, y_train, y_test)

    print()
    print(f"The Best Overall Model is: {max(test_mcc_scores, key=test_mcc_scores.get)}")

if __name__ == "__main__":
    main()
