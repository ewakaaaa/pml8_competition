import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import numpy as np 

def ln_exp_obj(preds,dtrain):
    labels = dtrain.get_label()
    x = preds - labels
    exp_2x = np.exp(2*x)
    grad = (exp_2x -1 )/ (exp_2x + 1 )
    hess = (4 * exp_2x) / (exp_2x + 1 )**2 
    return grad, hess

def mae_log_eval(y_log_pred,dtran):
    y_log_true = dtrain.get_label()
    y_true = np.exp(y_log_true)
    y_pred = np.exp(y_log_pred)
    return 'mea', mean_absolute_error(y_true,y_pred)

# def train_and_predict(model, X, y):
#     model.fit(X, y)
#     y_pred = model.predict(X)
#     return mean_absolute_error(y, y_pred)

def run_cv(model, X, y, folds=4, target_log=False,cv_type=KFold):
    cv = cv_type(n_splits=folds)
    
    scores = []
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if target_log:
            y_train = np.log(y_train)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if target_log:
            y_pred = np.exp(y_pred)
            y_pred[y_pred < 0] = 0 #czasem może być wartość ujemna

        score = mean_absolute_error(y_train,y_pred)
        scores.append( score )
        
    return np.mean(scores), np.std(scores)


def plot_learning_curve(model, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), target_log=False):
    
    plt.figure(figsize=(12,8))
    #plt.title(title)
    if ylim is not None:plt.ylim(*ylim)

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    if target_log:
        y = np.log(y)
    
    def my_scorer(model, X, y):
        y_pred = model.predict(X)
        
        if target_log:
            y = np.exp(y)
            y_pred = np.exp(y_pred)
            y_pred[ y_pred<0 ] = 0
        
        return mean_absolute_error(y, y_pred)

        
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=my_scorer)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_feature_importances(model, feats, limit = 20):
    importances = model.feature_importances_

    indices = np.argsort(importances)[::-1][:limit]

    plt.figure(figsize=(20, 8))
    plt.title("Feature importances")

    print(range(limit))
    print(importances[indices])
    plt.bar(range(limit), importances[indices])
    plt.xticks(range(limit), [feats[i] for i in indices], rotation='vertical')
    plt.show()
  