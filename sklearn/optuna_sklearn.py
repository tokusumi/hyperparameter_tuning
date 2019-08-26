# # wrapper of optuna
# ## 1. set variables as followings
# ### 1. the definition of model name and object:
#
# ```
# trial_models = {
#     'Extra Trees': ExtraTreesClassifier,
#     'Ridge': RidgeClassifier,
#     'kneighbor': KNeighborsClassifier,
#     
# }
# ```
#
# ### 2. the definition of type and range of hyperparameters:
#
# ```
# trial_condition = {
#     'Extra Trees': {
#         'n_estimators': ('int', 1, 100),
#         'max_depth': ('dis', 1, 100, 5),
#         'random_state': 128
#     },
#     'Ridge': {
#         'alpha': ('log', 1e-2, 1e2)
#     },
#     'kneighbor': {
#         'n_neighbors': ('int', 1, 30),
#         'algorithm': ('cat', ('ball_tree', 'kd_tree')),
#     }
# }
# ```
#
# ### 3. the definition of evaluated score
#
# ```
# score_metric = accuracy_score
# ```
#
# ## 2. prepare train and test data
#
# ```
# from sklearn.model_selection import train_test_split
# X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)
# ```
#
# ## 3. define and implement optuna
#
# ```
# evaluate = EvaluateFunc(X_train, X_val, y_train, y_val, score_metric)
# objective = Objective(evaluate, trial_models, trial_condition)
#
# # Create a new study. Declare optimization direction; maximize or minimize(default).
# study = optuna.create_study(direction='maximize')  # Create a new study. 
# study.optimize(objective, n_trials=100)
# ```

def EvaluateFunc(X_train, X_val, y_train, y_val, score_metric):
    def _evaluate_func(model_obj):
        """
        evaluate model prediction.
        customize the followings if you want cross validation
        """
        model_obj.fit(X_train, y_train)
        y_pred = model_obj.predict(X_val)
        error = score_metric(y_val, y_pred)
        return error
    return _evaluate_func


def Objective(evaluate_func, trial_models, trial_condition):
    """
    Define an objective function to be minimized or maximized.
    type:
    - int: integer
    - uni: a uniform float sampling
    - log: a uniform float sampling on log scale
    - dis: a discretized uniform float sampling
    - cat: category; ('auto', 'mode1', 'mode2', )
    
    """
    model_names = list(trial_models)
    method_names = {
        'int': 'suggest_int',
        'uni': 'suggest_uniform',
        'log': 'suggest_loguniform',
        'dis': 'suggest_discrete_uniform',
        'cat': 'suggest_categorical',
    }
    model_params = {
        model_name: {key: (method_names.get(val[0]), ('{}_{}'.format(model_name, key), *val[1:])) if type(val) is tuple else val
            for key, val in trial_condition.get(model_name).items()}
                for model_name in model_names
    }
    
    def _objective(trial):

        # Invoke suggest methods of a Trial object to generate hyperparameters.
        model_name = trial.suggest_categorical('classifier', model_names)
        params = {
            key: getattr(trial, val[0])(*val[1]) if type(val) is tuple else val
                for key, val in model_params.get(model_name).items()
        }
        model_obj = trial_models.get(model_name)(**params)

        #  evaluation
        error = evaluate_func(model_obj)

        return error  # A objective value linked with the Trial object.
    return _objective
