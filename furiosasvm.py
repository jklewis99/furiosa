'''
support vector machine for regression
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from utils.misc import plot_predictions, inverse_transform, create_interactive_plot, create_df, generate_data

def main(compare=False):
    x_train, x_test, y_train, y_test, _, scalar_y, dataset, test_indices = generate_data(
        "dbs/data_2010s.csv")
    if compare:
        x_train_base, x_test_base, y_train_base, y_test_base, _, scalar_y_base, dataset_base, test_indices_base = generate_data(
            "dbs/data_2010s.csv", drop_features=['title', 'tmdb_id', 'year', 'view_count', 'like_count', 'dislike_count', 'comment_count'])
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    results_r2 = []
    results_r2_train = []
    baseline_results = []
    for kernel in kernels:
        preds, r_squared = test(kernel, x_train, x_test, y_train, y_test)
        results_r2.append(r_squared)
        if compare:
            preds, baseline_r2 = test(kernel, x_train_base, x_test_base, y_train_base, y_test_base)
            baseline_results.append(baseline_r2)
        preds, actual = inverse_transform(preds, y_test, scalar_y)
        # plot_predictions(preds, actual, r_squared, model=f"SVR-{kernel}")
        df = create_df(preds, dataset, test_indices)
        create_interactive_plot(df, model=f"SVR-{kernel}")
        # remove_ghostbusters(df)
        # preds = regressor_temp.predict(x_train)
        # results_r2_train.append(r2_score(y_train, preds))
    kernels = np.array(kernels)
    results_r2 = np.array(results_r2)
    table_of_results = np.concatenate((kernels.reshape(len(kernels), 1), results_r2.reshape(len(results_r2), 1)), axis=1)
    print(table_of_results)
    if compare:
        baseline_results = np.array(baseline_results)
        table_of_results = np.concatenate((kernels.reshape(len(kernels), 1), baseline_results.reshape(len(baseline_results), 1)), axis=1)
        print(table_of_results)
    # print(results_r2_train)

def test(kernel, x_train, x_test, y_train, y_test):
    '''
    test SVR with kernel
    '''
    regressor_temp = SVR(kernel=kernel)
    regressor_temp.fit(x_train, y_train.flatten())
    preds = regressor_temp.predict(x_test)
    r_squared = r2_score(y_test, preds)
    return preds, r_squared

def baseline(df):
    ['title', 'tmdb_id', 'year', 'view_count', 'like_count', 'dislike_count', 'comment_count']

def remove_ghostbusters(df):
    data = df.copy().loc[df['title'] != "Ghostbusters"][['revenue', 'predicted']].values
    revenue = data[:, 0]
    real_preds = data[:, 1]
    print(r2_score(revenue, real_preds))

if __name__ == "__main__":
    main()