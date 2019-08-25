# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from typing import List, Union, Any, Dict
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from statistics import mean, stdev
from itertools import product


def stat_score(data: List[float]) -> Union[float]:
    return mean(data), stdev(data)


def gridsearch_cv(X, y, models: Dict, params: Dict, n_fold=5, score_index=0):
    """
    input:
    - X: features for train and validation
    - y: label for train and validation
    - clf
    - params: 
    - n_fold: folding number, which corresponds to "n-fold" cross validation
    - score_index: declare the index of optimised score metric as followings
    """
    scorers = [
        accuracy_score,
        recall_score,
        precision_score,
        f1_score
    ]
    
    scores = list()
    kfold = KFold(n_splits=n_fold, shuffle=True)

    for name, model in models.items():
        # get hyperparameters combination
        model_params = {key.rsplit("__", 1)[-1]: val for key, val in params.items() if key.rsplit("__", 1)[0] == name}
        cols = list(model_params)
        param_comb = list(product(*list(model_params.values())))
        
        for param in param_comb:
            param_dict = dict(zip(cols, param))
            results = list()
            for time, (train_index, val_index) in enumerate(kfold.split(X)):
                # split train data for train and validation
                X_train_cv, X_val, y_train_cv, y_val = X[train_index,
                                                         :], X[val_index, :], y[train_index], y[val_index]

                model_instance = model(**param_dict)
                model_instance.fit(X_train_cv, y_train_cv)

                y_pred = model_instance.predict(X_val)

                result = [score(y_val, y_pred) for score in scorers]
                results.append(result)

            show_params = ", ".join(["{}: {}".format(key, val) for key, val in param_dict.items()])
            scores_1fold = [stat_score(data) for data in zip(*results)]
            
            scores.append((name, show_params, scores_1fold))
    return scores
#     return list(zip(*scores))


def get_best(result, score_index=0):
    """
    return best score record in result; model name and parameter
    
    arguments:
    sccore_index: index of scorers defined in gridsearch_cv function
    result =
      [('Extra Trees',
      'n_estimators: 25, random_state: 100',
      [(0.42000000000000004, 0.04472135954999579),
       (0.5857142857142857, 0.15278035454433886),
       (0.4807142857142857, 0.14527809753566698),
       (0.5033766233766234, 0.06540479453309993)]),
     ('Extra Trees',
      'n_estimators: 25, random_state: 110',
      [(0.34, 0.11401754250991379),
       (0.3833333333333333, 0.1322875655532295),
       (0.42857142857142855, 0.19849204762539366),
       (0.3745454545454545, 0.12304672229591729)]),
     ('Ridge', 'alpha: 0.01',
      [(0.4, 0.15811388300841897),
       (0.5161904761904762, 0.23342078050373977),
       (0.4661904761904762, 0.2347768757903889),
       (0.4587590187590188, 0.1963401236184742)]),
     ('Ridge', 'alpha: 0.1',
      [(0.44, 0.151657508881031),
       (0.579047619047619, 0.3845897776230593),
       (0.4531746031746032, 0.27826487904069636),
       (0.4797868797868797, 0.27212896419725746)])]
    """
    score_index = 1
    res = [res[-1][score_index][0] for res in result]
    return result[res.index(max(res))]


if __name__=='__main__':
    import numpy as np
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import RidgeClassifier
    # test  and how to use
    
    # set optimized models name and function
    models = {
        "Extra Trees": ExtraTreesClassifier,
        "Ridge": RidgeClassifier,
    }

    # set hyperparameters for model 
    # must add __ in the middle of model name and hyperparams name
    # model name must be defined in above and 
    # hyperparams name must be determined by sklearn
    params = {
        "Extra Trees__n_estimators": [i for i in range(25, 30, 5)],
        "Extra Trees__random_state": [i for i in range(100, 120, 10)],
        "Ridge__alpha": [10**i for i in range(-2, 0)]
    }

    # set optimized score index, 
    # accuracy=0
    # recall=1
    # precision=2
    # f1-score=3
    score_index = 0
    
    # sample data
    X = np.random.random_sample((50, 5))
    y = np.array([0, 1])[np.random.choice([False, True], size=50).astype('int')]
    assert X.shape == (50, 5)
    assert y.shape == (50, )
    
    # implement gridsearch
    result = gridsearch_cv(X, y, models, params, score_index=score_index)
    """
    result >>
      [('Extra Trees',
      'n_estimators: 25, random_state: 100',
      [(0.42000000000000004, 0.04472135954999579),
       (0.5857142857142857, 0.15278035454433886),
       (0.4807142857142857, 0.14527809753566698),
       (0.5033766233766234, 0.06540479453309993)]),
     ('Extra Trees',
      'n_estimators: 25, random_state: 110',
      [(0.34, 0.11401754250991379),
       (0.3833333333333333, 0.1322875655532295),
       (0.42857142857142855, 0.19849204762539366),
       (0.3745454545454545, 0.12304672229591729)]),
     ('Ridge', 'alpha: 0.01',
      [(0.4, 0.15811388300841897),
       (0.5161904761904762, 0.23342078050373977),
       (0.4661904761904762, 0.2347768757903889),
       (0.4587590187590188, 0.1963401236184742)]),
     ('Ridge', 'alpha: 0.1',
      [(0.44, 0.151657508881031),
       (0.579047619047619, 0.3845897776230593),
       (0.4531746031746032, 0.27826487904069636),
       (0.4797868797868797, 0.27212896419725746)])]
    """
    # get best model and hyperparameter
    best_model_and_params = get_best(result, score_index)
    """
    >>> 
    ('Extra Trees',
     'n_estimators: 25, random_state: 100',
     [(0.44, 0.18165902124584948),
      (0.26666666666666666, 0.09128709291752768),
      (0.34, 0.14747881203752625),
      (0.27714285714285714, 0.07515290535750078)])
    """


