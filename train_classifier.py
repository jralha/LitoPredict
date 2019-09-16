#%%  Libraries
############################################################
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SVMSMOTE
import glob

np.set_printoptions(suppress=True)

os.chdir('LitoPredict')



#%% Function printing scores from a given prediction array
############################################################
def scores(pred,label):
    cmatrix = confusion_matrix(pred,label)
    tp = cmatrix[1][1]
    fp = cmatrix[0][1]
    tn = cmatrix[0][0]
    fn = cmatrix[1][0]
    recall0 = tp/(tp+fn)
    recall1 = tn/(tn+fp)
    precision0 = tp/(tp+fp)
    precision1 = tn/(tn+fn)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    print("Accuracy: "+str(accuracy))
    print("Recalls: "+str(recall0)+" "+str(recall1))
    print("Precisions: "+str(precision0)+" "+str(precision1))






#%% Merge all csvs into single dataframe
############################################################

folder0 = 'train_data'

dfs=[]
for file in glob.glob(folder0 + "/*.csv"):
    temp = pd.read_csv(file, header=[0],index_col=[0])
    if "RHOZ" in temp.columns:
        temp['RHOB']=temp['RHOZ'] #Can we merge RHOZ and RHOB?
        del temp['RHOZ']
    #Correcting typos
    temp['Formation'] = temp['Formation'].replace('Guratiba','Guaratiba')
    #Replacing null values with NaN
    temp = temp.replace("-9999",np.nan).replace(-9999,np.nan)
    temp = temp
    temp = temp.iloc[1:]
    #Remove top and bottom 50 samples from
    #each well to remove border interference
    temp = temp.iloc[50:]
    temp = temp.iloc[:-50]
    dfs.append(temp)

all_dfs = pd.concat(dfs,sort=False)

all_dfs=all_dfs.loc[:, ~all_dfs.columns.str.contains('^Unnamed')]


#%% Selects features
############################################################

#Removing values with PR >1, it's a ratio, should never be > 1.
data_pr = all_dfs.loc[all_dfs['PR'] <= 1]

#Separates labeled data from unlabeled data and backfills NaNs
#Backfill method chosen due to similar depths
classified = data_pr.loc[data_pr['Formation'] != 'Guaratiba'].dropna(axis=0)
label=classified['Formation'].map({'Barra Velha':1,'Itapema':0})

#Initial data DF
data0=classified.iloc[:,2:]#.drop(['PhiMac','PhiMic'],axis=1)
cols = data0.columns

#Convert numeric values to floats, ignores if non-numeric
for column in data0.columns:
    data0[column] = pd.to_numeric(data0[column],errors='ignore',downcast='float')

#Handling remaining NaN values by assigning mean value, probably doesn't do anything.
imp = SimpleImputer()
data_imp=pd.DataFrame(imp.fit_transform(data0))
data_imp.columns = cols


#%% Train-test split and pre-processing
############################################################
data_imp = data_imp[['MD','DTc','DTs','RHOB','ECGR']]
X_train, X_test, y_train, y_test = train_test_split(data_imp, label, test_size=0.2, random_state=123)

#Balancing dataset by oversampling minority label
x_col = X_train.columns
smote = SVMSMOTE(random_state=123,n_jobs=-1,sampling_strategy='minority')
X_train, y_train = smote.fit_resample(X_train,y_train)
X_train = pd.DataFrame(X_train)
X_train.columns = x_col

#Data is split with md, saving mds for future plotting
md_train = X_train['MD'].values
md_test = X_test['MD'].values

#Removes md from data
X_train = X_train.iloc[:,1:]
X_test = X_test.iloc[:,1:]

# Polynomial Features to generate additional features
polyfit = PolynomialFeatures().fit(X_train)
X_train = polyfit.transform(X_train)
X_test = polyfit.transform(X_test)



#%% XGBoost Model, hyperparameters were previously optimised with GridSearchCV.
############################################################

clf0=xgb.XGBClassifier(
    booster='gbtree',
    colsample_bylevel=0.5,
    colsample_bynode=0.5,
    colsample_bytree=1,
    learning_rate=0.1,
    max_depth=12,
    min_child_weight=0.5,
    n_estimators=1000,
    objective='binary:logistic',
    scale_pos_weight=1,
    subsample=1,
    min_split_loss=0
)

clf0.fit(X_train,y_train)
pred0 = clf0.predict(X_test)
prob_pred0 = clf0.predict_proba(X_test)
scores(pred0,y_test)



#%%Saving Model
############################################################
import pickle
out_file = open("model.pkl","wb")
pickle.dump(clf0,out_file)