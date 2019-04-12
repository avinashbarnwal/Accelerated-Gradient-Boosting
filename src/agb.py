import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

def get_perm(n_all,n_sub):
    perm = list(range(0, n_all))
    perm = random.sample(perm, n_sub)
    return(perm)

def agb(x,y,Nestrov=True,train_fraction=0.75,n_trees=500,shrinkage=0.01,
        distribution="gaussian",depth_tree=4,n_minobsinnode = 10,
        criterion = 'mse',nesterov=False,subsample=1, error_type = 'rmse'):

    n = x.shape[0]
    n_trees = n_trees +1
    n_train = int(round(n*train_fraction))
    n_valid = n-n_train

    x_train = x.iloc[:n_train]
    x_valid = x.iloc[n_train:]

    y_train = y.iloc[:n_train]
    y_valid = y.iloc[n_train:]

    err_train = np.zeros((n_trees-1,),dtype=np.float64) #rep(0,n.trees)
    err_valid = np.zeros((n_trees-1,),dtype=np.float64)


    if subsample < 1:
        n_sub  = int(round(n_train*subsample))
        scf    = get_perm(n_train,n_sub)
        sub_x_train = x_train.iloc[scf]
        sub_y_train = y_train.iloc[scf]
        fitted = pd.DataFrame(np.zeros((n_train,n_trees),dtype=np.float64))
        prev_valid = pd.DataFrame(np.zeros((n_valid,n_trees),dtype=np.float64))
        g_fitted   = fitted.copy()
        g_prev_valid = prev_valid.copy()
        fitted.iloc[:,0] = np.mean(y_train)
        prev_valid.iloc[:,0] = np.mean(y_valid)
        g_fitted.iloc[:,0] = fitted.iloc[:,0]
        g_prev_valid.iloc[:,0] = prev_valid.iloc[:,0]
        if error_type == 'rmse':
            err_train[0] = np.sqrt(np.mean((y_train-fitted.iloc[:,0])**2))
            err_valid[0] = np.sqrt(np.mean((y_valid-prev_valid.iloc[:,0])**2))
        elif error_type == 'mse':
            err_train[0] = np.mean((y_train-fitted.iloc[:,0])**2)
            err_valid[0] = np.mean((y_valid-prev_valid.iloc[:,0])**2)
        elif error_type == 'abs':
            err_train[0] = np.mean(np.abs(y_train-fitted.iloc[:,0]))
            err_valid[0] = np.mean(np.abs(y_valid-prev_valid.iloc[:,0]))
        data_boucle  = sub_x_train.copy()
        data_boucle.loc[:,"U"] = 0
    else :
        print(error_type)
        fitted       = pd.DataFrame(np.zeros((n_train,n_trees),dtype=np.float64))
        prev_valid   = pd.DataFrame(np.zeros((n_valid,n_trees),dtype=np.float64))
        g_fitted     = fitted.copy()
        g_prev_valid = prev_valid.copy()
        fitted.iloc[:,0]  = np.mean(y_train)
        prev_valid.iloc[:,0] = np.mean(y_valid)
        g_fitted.iloc[:,0] = fitted.iloc[:,0]
        g_prev_valid.iloc[:,0] = prev_valid.iloc[:,0]
        if error_type == 'rmse':
            err_train[0] = np.sqrt(np.mean((y_train-fitted.iloc[:,0])**2))
            err_valid[0] = np.sqrt(np.mean((y_valid-prev_valid.iloc[:,0])**2))
        elif error_type == 'mse':
            err_train[0] = np.mean((y_train-fitted.iloc[:,0])**2)
            err_valid[0] = np.mean((y_valid-prev_valid.iloc[:,0])**2)
        elif error_type == 'abs':
            err_train[0] = np.mean(np.abs(y_train-fitted.iloc[:,0]))
            err_valid[0] = np.mean(np.abs(y_valid-prev_valid.iloc[:,0]))

        data_boucle  = x_train.copy()
        data_boucle.loc[:,"U"] = 0

    tree_ctrl = {'max_depth': depth_tree, 'min_samples_split': n_minobsinnode
                ,'criterion':criterion, 'splitter':'best'}

    tree = DecisionTreeRegressor(**tree_ctrl)
    tree.fit(x_train,y_train)

    lamb  = np.zeros((n_trees,),dtype=np.float64)
    gamma = np.zeros((n_trees,),dtype=np.float64)
    gamma[0] = 1

    for i in range(1,n_trees):
        lamb[i] = 0.5*(1+math.sqrt(1+4*lamb[i-1]**2))
    if distribution=="gaussian":
        #n_trees-1
        for i in range(1,n_trees-1):
            if (nesterov==True and subsample==1):
                gamma[i] = (1-lamb[i])/lamb[i+1]
                U        = y_train-g_fitted.iloc[:,i-1]
                #data_boucle.loc[:,"U"] = U
                tree = DecisionTreeRegressor(**tree_ctrl)
                tree.fit(x_train,U)
                #data_boucle1 = data_boucle
                fitted.iloc[:,i]     = g_fitted.iloc[:,i-1]     + shrinkage*tree.predict(x_train)
                prev_valid.iloc[:,i] = g_prev_valid.iloc[:,i-1] + shrinkage*tree.predict(x_valid)
                g_fitted.iloc[:,i]   = (1-gamma[i-1])*fitted.iloc[:,i]+gamma[i-1]*fitted.iloc[:,i-1]
                g_prev_valid.iloc[:,i] = (1-gamma[i-1])*prev_valid.iloc[:,i]+gamma[i-1]*prev_valid.iloc[:,i-1]
                if error_type == 'rmse':
                    err_train[i] = np.sqrt(np.mean((y_train.values-fitted.iloc[:,i].values)**2))
                    err_valid[i] = np.sqrt(np.mean((y_valid.values-prev_valid.iloc[:,i].values)**2))
                elif error_type == 'mse':
                    err_train[i] = np.mean((y_train.values-fitted.iloc[:,i].values)**2)
                    err_valid[i] = np.mean((y_valid.values-prev_valid.iloc[:,i].values)**2)
                elif error_type == 'abs':
                    err_train[i] = np.mean(np.abs(y_train.values-fitted.iloc[:,i].values))
                    err_valid[i] = np.mean(np.abs(y_valid.values-prev_valid.iloc[:,i].values))

            elif(nesterov==False and subsample==1):
                U = y_train-fitted.iloc[:,i-1]
                data_boucle.loc[:,"U"] = U
                tree = DecisionTreeRegressor(**tree_ctrl)
                tree.fit(x_train,U)
                #data_boucle1 = data_boucle
                fitted.iloc[:,i] = fitted.iloc[:,i-1] + shrinkage*tree.predict(x_train)
                prev_valid.iloc[:,i] = prev_valid.iloc[:,i-1] + shrinkage*tree.predict(x_valid)
                if error_type == 'rmse':
                    err_train[i] = np.sqrt(np.mean((y_train.values-fitted.iloc[:,i].values)**2))
                    err_valid[i] = np.sqrt(np.mean((y_valid.values-prev_valid.iloc[:,i].values)**2))
                elif error_type == 'mse':
                    err_train[i] = np.mean((y_train.values-fitted.iloc[:,i].values)**2)
                    err_valid[i] = np.mean((y_valid.values-prev_valid.iloc[:,i].values)**2)
                elif error_type == 'abs':
                    err_train[i] = np.mean(np.abs(y_train.values-fitted.iloc[:,i].values))
                    err_valid[i] = np.mean(np.abs(y_valid.values-prev_valid.iloc[:,i].values))

            
            if err_valid[i]>=100:
                n_trees = i
                err_train = err_train[0:i]
                err_valid = err_valid[0:i]
                break

    result = {}
    result = {'error_train':err_train,'error_valid':err_valid}

    return result
if __name__ == "__main__":
    df_train = pd.read_csv('simulate_1.csv')
    y_train = df_train['Y']
    req_cols = [i for i in df_train.columns if i != 'Y']
    x_train = df_train[req_cols]
    agb_n_true_sub  = agb(x_train,y_train,nesterov=True,n_trees=500,subsample=0.5)
    fig = plt.figure()
    plt.plot(agb_n_true_sub['error_train'])
    plt.plot(agb_n_true_sub['error_valid'])
    plt.axvline(x=np.nanargmin(agb_n_true_sub['error_valid']))
    plt.show()
    fig.savefig("agb_n_true_sub.png")
    agb_n_false_sub = agb(x_train,y_train,nesterov=False,n_trees=500,subsample=0.5)
    fig = plt.figure()
    plt.plot(agb_n_false_sub['error_train'])
    plt.plot(agb_n_false_sub['error_valid'])
    plt.axvline(x=np.nanargmin(agb_n_false_sub['error_valid']))
    plt.show()
    fig.savefig("agb_n_false_sub.png")
    agb_n_true       = agb(x_train,y_train,nesterov=True,n_trees=500,subsample=1)
    agb_n_false      = agb(x_train,y_train,nesterov=False,n_trees=500,subsample=1)
    plt.plot(agb_n_false_sub['error_train'])
    plt.plot(agb_n_false_sub['error_valid'])
    plt.show()
    plt.plot(agb_n_false['error_train'])
    plt.plot(agb_n_false['error_valid'])
    plt.show()
    plt.plot(agb_n_true['error_train'])
    plt.plot(agb_n_true['error_valid'])
    plt.show()
