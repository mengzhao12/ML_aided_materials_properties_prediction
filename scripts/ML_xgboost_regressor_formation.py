import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBRegressor

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error

# Load the data
data = pd.read_csv("data_complete_w_features.csv",index_col = 0)
data = data.dropna()

features = data.drop(['formula','structure_obj', 
                      'composition',
                      'structure',
                      'composition_oxid',
                      'formation_energy_ev_natom',
                      'bandgap_energy_ev'], axis = 1)

features = features.astype(float)

target1 = data['formation_energy_ev_natom']
target2 = data['bandgap_energy_ev']

random_state = 42

# split data for target1 formation energy
X_train1, X_test1, y_train1, y_test1 = train_test_split(features, 
                                                        target1, 
                                                        test_size=0.3, 
                                                        random_state=random_state)

# split data for target2 bandgap
X_train2, X_test2, y_train2, y_test2 = train_test_split(features, 
                                                        target2, 
                                                        test_size=0.3, 
                                                        random_state=random_state)

# Number of estimators
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# learn rate
learning_rate = [float(x) for x in np.arange(0.01, 1.0, step = 0.02)]
# gamma
gamma = [float(x) for x in np.arange(0, 1.0, step = 0.1)]
# colsample_bytree
colsample_bytree = [float(x) for x in np.arange(0.2, 1.0, step = 0.2)] 
# subsample
subsample = [float(x) for x in np.arange(0.2, 1.0, step = 0.2)] 

# Create the random grid

k_fold = 5
n_iter_search = 50

tuned_parameters_xgb = {'n_estimators': n_estimators, 
                        'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'gamma': gamma,
                        'colsample_bytree': colsample_bytree,
                        'subsample': subsample}

scaler= StandardScaler()
xgb_regr = XGBRegressor(objective="reg:linear", random_state=42)

randomsearch_xgb = RandomizedSearchCV(estimator = xgb_regr,
                                      param_distributions = tuned_parameters_xgb, 
                                      n_iter = n_iter_search,
                                      cv = k_fold,
                                      random_state = 42,
                                      verbose = 1,
                                      n_jobs= -1)

pipe_xgb = make_pipeline(scaler,randomsearch_xgb)

# List of pipelines for ease of iteration
pipelines = [pipe_xgb] 

# Dictionary of pipelines and classifier types for ease of reference
pipe_dict = {0: 'Xgboost regression'} 

# Fit the pipelines
for pipe in pipelines:
    pipe.fit(X_train1, y_train1) 
    y_test_pred = pipe.predict(X_test1)
    score = pipe.score(X_test1, y_test1) 
    print("the score for test data: {}".format(score))

filename = 'finalized_xgb_model_2_formation.sav'
joblib.dump(randomsearch_xgb.best_estimator_, open(filename, 'wb'))
print('model file is saved as {}'.format(filename))
print ("done")

# calculate mean absolute error
y_pred_test1 = pipelines[0].predict(X_test1)

mean_absolute_error = mean_absolute_error(y_test1, y_pred_test1)
print("MAE(mean absolute error) is {} eV/atom".format(round(mean_absolute_error,4)))

# number of testset
observations = len(y_test1)

pred_list = []
actual_list = []

pred_actual_diffs = []
most_negative1 = min(y_pred_test1)
most_negative2 = min(y_test1)

for pred, actual in zip(y_pred_test1,y_test1):
    pred_actual_diff = np.square(np.log(pred- most_negative2 + 1) - np.log(actual - most_negative2 + 1))
    pred_actual_diffs.append(pred_actual_diff)
    
evaluation = (1/observations)*np.sum(pred_actual_diffs)
rmsle = np.sqrt(evaluation)

print("RMSLE is {} eV/atom".format(round(rmsle,4)))

# plot the figure
fontsize = 12
plt.scatter(y_test1, y_pred_test1, marker='o', color = 'blue', alpha = 0.3)
straightline_x = [min(y_test1), max(y_test1)]
straightline_y = [min(y_test1), max(y_test1)]
plt.plot(straightline_x, straightline_y, 'k--', linewidth=2)
plt.xlabel("Actual Gibbs Formation Energy eV/atom", fontsize = fontsize)
plt.ylabel("Predicted Gibbs Formation Energy eV/atom", fontsize = fontsize)
label1 = "MAE: {} eV/atom".format(round(mean_absolute_error,2))
label2 = "ML algorithum:\n{}".format('Xgboost')
# plt.text(1.5, -4, label1, fontsize=fontsize)
# plt.text(1.5, -3, label2, fontsize=fontsize)
plt.savefig("./actual vs. predicted Gibbs formation energy_random_searchCV_xgbbost.jpeg",dpi = 400)


# # Load ML model from file
# xgb_model = joblib.load(filename)
# print("training model loaded")
# print(xgb_model)