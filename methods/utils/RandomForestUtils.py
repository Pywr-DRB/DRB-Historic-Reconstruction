from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error

import numpy as np

def TreeRFE(X, y, n_features_to_select, n_estimators=100, 
            tree = 'rf', 
                      min_samples_split=2, min_samples_leaf=1, 
                      max_depth=5,
                      cv_folds=5, scoring_metric='neg_mean_squared_error',
                      step=1, verbose=0, random_state=None):
    """
    Performs Recursive Feature Elimination (RFE) using a TreeClassifier.
    
    Parameters:
    - X : numpy array or pandas DataFrame
        Feature dataset.
    - y : numpy array or pandas Series
        Target variable.
    - n_features_to_select : int
        Number of features to select.
    - n_estimators : int, default=100
        The number of trees in the forest.
    - min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node.
    - min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
    - cv_folds : int, default=5
        Number of folds in cross-validation.
    - scoring_metric : string, default='accuracy'
        Metric used for scoring during cross-validation.
    - step : int or float, default=1
        If greater than or equal to 1, then 'step' corresponds to the (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), 'step' corresponds to the percentage (rounded down) of features to remove at each iteration.
    - verbose : int, default=0
        Controls verbosity of output.

    Returns:
    - list of selected features indices
    """
    # Initialize the base classifier
    if tree == 'rf':
        reg = RandomForestRegressor(n_estimators=n_estimators, 
                                 min_samples_split=min_samples_split, 
                                 min_samples_leaf=min_samples_leaf,
                                 bootstrap=True,
                                 random_state=random_state, n_jobs=-1)
    elif tree == 'gb':
        reg = GradientBoostingRegressor(n_estimators=n_estimators, 
                                        max_depth=max_depth,
                                 min_samples_split=min_samples_split, 
                                 min_samples_leaf=min_samples_leaf,
                                 random_state=random_state)

    # Initialize RFECV
    rfecv = RFECV(estimator=reg, step=step, cv=KFold(cv_folds),scoring=scoring_metric, 
                  min_features_to_select=n_features_to_select, verbose=verbose)
    
    # Fit RFECV
    rfecv.fit(X, y)
    
    # Print the optimal number of features
    print("Optimal number of features : %d" % rfecv.n_features_)
    
    # Get the ranking of the features
    ranking = rfecv.ranking_
    
    # Select features based on ranking
    selected_features = np.where(ranking == 1)[0]

    return selected_features.tolist()


def iterative_rfe(x_full, y, iterations, cv_folds, n_estimators, features_to_select,
                  tree='gb'):
    """
    Runs the TreeRFE function iteratively to select features in multiple steps.
    
    Parameters:
    - x_full : pandas DataFrame
        Feature dataset.
    - y : numpy array or pandas Series
        Target variable.
    - iterations : int
        Number of iterations to run.
    - cv_folds : list of int
        Number of folds in cross-validation for each iteration.
    - n_estimators : list of int
        Number of trees in the forest for each iteration.
    - features_to_select : list of int
        Number of features to select for each iteration.
    """
    feature_subset = x_full.columns
    for i in range(iterations):
        print(f'Iteration {i+1} of {iterations}')
        selected_indices = TreeRFE(x_full[feature_subset], y, 
                                   tree=tree,
                                           n_features_to_select=features_to_select[i], 
                                           n_estimators=n_estimators[i], 
                                           cv_folds=cv_folds[i])
        feature_subset = feature_subset[selected_indices]
        # Optionally, evaluate model performance here
    return feature_subset





def TreeCV(X, y, metrics, tree='gb', n_estimators=100, cv=5, random_state=None, max_depth=5):
    """
    Perform a cross-validated diagnostic of a Regressor model using multiple metrics.
    
    Parameters:
    - X : numpy array or pandas DataFrame
        Feature dataset.
    - y : numpy array or pandas Series
        Target variable.
    - metrics : list of str
        Metrics to evaluate the model. Supported metrics: 'mse', 'mae', 'r2'
    - n_estimators : int, default=100
        The number of trees in the forest.
    - cv : int, default=5
        The number of folds in k-fold cross-validation.
    - random_state : int or None, default=None
        Controls the randomness of the model.
        
    Returns:
    - dict : Results dictionary with keys as metrics and values containing the score arrays, mean, and std deviation.
    """
    # Define the model
    if tree == 'rf':
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    elif tree == 'gb':
        model = GradientBoostingRegressor(n_estimators=n_estimators,
                                          max_depth=max_depth, 
                                          random_state=random_state)

    # Setup scoring dictionary based on input metrics
    scoring = {}
    if 'mse' in metrics:
        scoring['mse'] = make_scorer(mean_squared_error, greater_is_better=False)
    if 'mae' in metrics:
        scoring['mae'] = make_scorer(mean_absolute_error, greater_is_better=False)
    if 'r2' in metrics:
        scoring['r2'] = make_scorer(r2_score)

    # Perform cross-validation
    results = cross_validate(model, X, y, scoring=scoring, cv=cv, 
                             return_train_score=False)

    # Prepare the output dictionary
    output = {}
    for metric in metrics:
        key = f"test_{metric}"
        if key in results:
            scores = results[key]
            output[metric] = {
                'scores': scores,
                'mean': np.mean(scores),
                'std': np.std(scores)
            }

    return output