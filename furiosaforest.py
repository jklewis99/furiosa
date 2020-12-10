'''
random forest regression
'''
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from utils.misc import plot_predictions, create_interactive_plot, create_df, generate_data

def forest(save_fig=False, compare=False):
    '''
    main function for testing Random Forest Regression models

    Keyword Args
    ===========
    `save_fig`:
        if True, save a matplotlib figure AND plotly interactive figure
        of predictions

    `compare`:
        if true, compare values with trailer data to the baseline
    '''
    x_train, x_test, y_train, y_test, _, _, dataset, test_indices = generate_data(
        "dbs/data_2010s.csv", scale_input=False)
    table_of_results = run_model(x_train, x_test, y_train, y_test, dataset, test_indices)
    print(table_of_results)
    if compare:
        # line is too long
        baseline_data = generate_data(
            "dbs/data_2010s.csv",
            drop_features=[
                'title', 'tmdb_id', 'year', 'view_count',
                'like_count', 'dislike_count', 'comment_count'],
            scale_input=False
            )
        x_train_b, x_test_b, y_train_b, y_test_b, _, _, _, _ = baseline_data
        table_of_results = run_model(x_train_b, x_test_b, y_train_b, y_test_b, save_fig=save_fig)
        print("Baseline\n", table_of_results)

def run_model(x_train, x_test, y_train, y_test, dataset=None, test_indices=None, bag_size=0.8, save_fig=False):
    '''
    logic of the model

    Parameters
    ==========
    `x_train`:
        numpy array with all feature values for each sample in training data

    `x_test`:
        numpy array with all feature values for each sample in testing data

    `y_train`:
        numpy array with correct continuous value (revenue) for each training sample

    `y_test`:
        numpy array with correct continuous value (revenue) for each testing sample

    `dataset`:
        DataFrame of data used to get test data if `save_fig` is True

    `test_indices`:
        test_indices of test data in `dataset` used if `save_fig` is True

    Keyword Args
    ==========
    `bag_size`:
        bag size to use for each tree in the forest. Default 0.8, meaning 80% of the number
        of training samples is used at each tree, with replacement. See the `max_samples`
        parameter in sklearn.ensemble.RandomForestRegressor documentation for more information.

    `save_fig`:
        if True, save a matplotlib figure AND plotly interactive figure
        of predictions

    Return
    ==========
    table_of_results
    '''

    # number of trees in the forest for comparison
    num_trees_list = np.arange(20, 96, 5)
    results_r2 = []
    #initialize "bests"
    best_preds = None
    best_r_squared = 0
    best_tree_count = 0

    for n_trees in num_trees_list:
        # TODO: Hyper-parameter Tuning
        # create instance of the RandomForestRegressor
        regressor_temp = RandomForestRegressor(
            n_estimators=n_trees,
            max_samples=bag_size,
            random_state=18)
        # fit the regressor to the training data
        regressor_temp.fit(x_train, y_train) # if scaled: .ravel())
        # use the regressor to test the model
        preds_forest_temp = regressor_temp.predict(x_test)
        # calculate metric
        r_squared = r2_score(y_test, preds_forest_temp)
        # update "bests", if this forest outperformed the current best
        if r_squared > best_r_squared:
            best_r_squared = r_squared
            best_preds = preds_forest_temp
            best_tree_count = n_trees
        results_r2.append(r_squared)
    
    if save_fig:
        test_data = create_df(best_preds, dataset, test_indices)
        plot_predictions(
            best_preds, y_test, best_r_squared,
            model=f"RandomForest-{best_tree_count}-trees-{bag_size}BS"
            )
        create_interactive_plot(
            test_data,
            model=f"RandomForest-{best_tree_count}-trees-{bag_size}BS"
            )
    # create the table of results
    results_r2 = np.array(results_r2)
    table_of_results = np.concatenate(
        (num_trees_list.reshape(len(num_trees_list), 1), results_r2.reshape(len(results_r2), 1)),
        axis=1)
    return table_of_results

if __name__ == "__main__":
    forest()
