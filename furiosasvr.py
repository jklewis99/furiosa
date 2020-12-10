'''
support vector machine for regression
'''
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from utils.misc import plot_predictions, inverse_transform, create_interactive_plot, create_df, generate_data

def svr(save_fig=False, compare=False):
    '''
    main function for testing Support Vector Regression models

    Keyword Args
    ===========
    `save_fig`:
        if True, save a matplotlib figure AND plotly interactive figure
        of predictions

    `compare`:
        if true, compare values with trailer data to the baseline
    '''
    x_train, x_test, y_train, y_test, _, scalar_y, dataset, test_indices = generate_data(
        "dbs/data_2010s.csv")

    # get parameters for comparing to the "baseline"
    if compare:
        # line is too long
        baseline_data = generate_data(
            "dbs/data_2010s.csv",
            drop_features=[
                'title', 'tmdb_id', 'year', 'view_count',
                'like_count', 'dislike_count', 'comment_count']
            )
        x_train_b, x_test_b, y_train_b, y_test_b, _, _, _, _ = baseline_data
    # define kernels that will be compared
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    #initialize lists that will be used to print results
    results_r2 = []
    baseline_results = []

    for kernel in kernels:
        # get preds and r_squared value per model
        preds, r_squared = test(kernel, x_train, x_test, y_train, y_test)
        # add it to the table
        results_r2.append(r_squared)
        if compare:
            # get preds and r_squared value per model for the baseline data
            preds, baseline_r2 = test(kernel, x_train_b, x_test_b, y_train_b, y_test_b)
            # add it to the table
            baseline_results.append(baseline_r2)
        if save_fig:
            # get the actual values of the revenue
            # using the the inverse transform of the data using scalar_y
            preds, actual = inverse_transform(preds, y_test, scalar_y)
            plot_predictions(preds, actual, r_squared, model=f"SVR-{kernel}")
            test_df = create_df(preds, dataset, test_indices)
            create_interactive_plot(test_df, model=f"SVR-{kernel}")
        # to compare without the outlier, uncomment the following line: 
        # remove_ghostbusters(df)

    # create a table of results
    kernels = np.array(kernels)
    results_r2 = np.array(results_r2)
    table_of_results = np.concatenate(
        (kernels.reshape(len(kernels), 1), results_r2.reshape(len(results_r2), 1)),
        axis=1)
    print(table_of_results)

    if compare:
        baseline_results = np.array(baseline_results)
        table_of_results = np.concatenate(
            (kernels.reshape(len(kernels), 1), baseline_results.reshape(len(baseline_results), 1)),
            axis=1)
        print(table_of_results)

def test(kernel, x_train, x_test, y_train, y_test):
    '''
    fit the SVR model with specified `kernel` to training data and test the SVR on the test data

    Parameters
    ==========
    `kernel`:
        string that specifies the kernel to be used
    `x_train`:
        numpy array with all feature values for each sample in training data
    `x_test`:
        numpy array with all feature values for each sample in testing data
    `y_train`:
        numpy array with correct continuous value (revenue) for each training sample
    `y_test`:
        numpy array with correct continuous value (revenue) for each testing sample

    Return
    ==========
    predictions, r_squared
    '''
    regressor_temp = SVR(kernel=kernel)
    regressor_temp.fit(x_train, y_train.flatten())
    preds = regressor_temp.predict(x_test)
    r_squared = r2_score(y_test, preds)
    return preds, r_squared

def remove_ghostbusters(df):
    '''
    Method to get the r-squaered value after 'Ghostbusters' is removed from the test data.
    Ghostbusters is a a significant outlier because of its high dislike count, which is
    excessively higher than the numbers seen in the training set.

    Parameters
    ==========
    `df`:
        DataFrame containing 'Ghostbusters'
    '''
    data = df.copy().loc[df['title'] != "Ghostbusters"][['revenue', 'predicted']].values
    revenue = data[:, 0]
    real_preds = data[:, 1]
    print(r2_score(revenue, real_preds))

if __name__ == "__main__":
    svr()
