import optuna
# from optuna.integration import OptunaSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import xgboost as xgb
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser
# from matplotlib import ticker
import numpy as np
from matplotlib import rcParams
base_folder=r'C:\\Users\\user\\git\\github\\py2401_xgb_pls\\'
file_name=base_folder+'regression_pls.csv'
model_name=base_folder+'xgb_pls.pickle'
PdfFile=base_folder+'pdf\\xgb_pls.pdf'
PdfFile_Fi=base_folder+'pdf\\xgb_Fi.pdf'
PdfFile_Pfi=base_folder+'pdf\\xgb_Pfi.pdf'
columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','Target']
target_column='Target'
df=pd.read_csv(file_name, encoding='utf-8', engine='python', usecols=columns)
features=[c for c in df.columns if c !=target_column]
train, test=train_test_split(df, test_size=0.2, random_state=115)
X_train=train[features]
y_train=train[target_column].values
X_test=test[features]
y_test=test[target_column].values
dtrain=xgb.DMatrix(X_train, label=y_train)
dtest=xgb.DMatrix(X_test, label=y_test)
def set_graph_params():
    rcParams['xtick.labelsize']=12
    rcParams['ytick.labelsize']=12
    rcParams['figure.figsize']=7,5
    sns.set_style('whitegrid')
def objective(trial):
    params = {
        'silent': 1,
        'max_depth': trial.suggest_int('max_depth', 6, 9),
        'min_child_weight': 1,
        'eta': trial.suggest_loguniform('eta', 0.01, 1.0),
        'tree_method': 'exact',
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'predictor': 'cpu_predictor'
    } 
    cv_results=xgb.cv(
        params,
        dtrain,
        num_boost_round=1000,
        seed=0,
        nfold=10,
        metrics={'rmse'},
        early_stopping_rounds=5
    )
    return cv_results['test-rmse-mean'].min()
study=optuna.create_study()
study.optimize(objective, n_trials=1000)
print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial=study.best_trial
print(' Value: {}'.format(trial.value))
print(' Params: ')
for key, value in trial.params.items():
    print('   {}: {}'.format(key, value))
params['max_depth']=trial.params['max_depth']
params['eta']=trial.params['eta']
model=xgb.train(params=params,
                dtrain=dtrain,
                num_boost_round=1000,
                early_stopping_rounds=5,
                evals=[(dtest, 'test')])
pred_train=model.predict(xgb.DMatrix(X_train))
pred_test=model.predict(xgb.DMatrix(X_test))
r2_train=r2_score(y_train, pred_train)
adjusted_r2_train=1-(1-r2_train)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
mse_train=mean_squared_error(y_train, pred_train)
rmse_train=mse_train**0.5
r2_test=r2_score(y_test, pred_test)
adjusted_r2_test=1-(1-r2_test)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
mse_test=mean_squared_error(y_test, pred_test)
rmse_test=mse_test**0.5
set_graph_params    
sns.set_color_codes()
plt.figure()
fig, ax=plt.subplots(figsize=(7,7))
plt.xlim([-12,2])
plt.ylim([-12,2])
sns.set(font='Arial')
plt.scatter(y_train, pred_train, alpha=0.5, color='blue', label='Train')
plt.scatter(y_test, pred_test, alpha=0.5, color='green', label='Test')
plt.plot(np.linspace(-12, 2, 14), np.linspace(-12, 2, 14), 'black')
plt.xlabel('True Target', fontsize=14)
plt.ylabel('Predicted True Target', fontsize=14)
plt.title(f'Train - Adjusted R2 Score: {adjusted_r2_train:.3f}, RMSE: {rmse_train:.3f}\nTest - Adjested R2 Score: {adjusted_r2_test:.3f}, RMSE: {rmse_test:.3f}', fontsize=13)
plt.legend(fontsize=14)
plt.savefig(PdfFile)
webbrowser.open_new(PdfFile)
best_estimator=optuna_search.best_estimator_
importance=best_estimator.feature_importances_
plot_Fi(importance, features, PdfFile_Fi)
plot_Pfi(X_test, y_test, features, PdfFile_Pfi)
webbrowser.open_new(PdfFile_Fi)
webbrowser.open_new(PdfFile_Pfi)