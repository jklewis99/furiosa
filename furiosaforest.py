'''
module containing all functions related to a nerual network to predict revenue with
regression. Training, testing, and evaluating models are methods in this module.
'''

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from utils.misc import plot_predictions, create_interactive_plot, create_df
from furiosanet import generate_data, test

def main():
    x_train, x_test, y_train, y_test, _, scalar_y, dataset, test_indices = generate_data(
        "dbs/data_2010s.csv", scale_input=False)

    num_trees_list = np.arange(10, 100, 5)
    results_r2 = []
    best_preds = None
    best_r_squared = -1
    best_tree_count = 0
    for n_trees in num_trees_list:
        regressor_temp = RandomForestRegressor(n_estimators=n_trees)
        regressor_temp.fit(x_train, y_train)
        preds_forest_temp = regressor_temp.predict(x_test)
        r_squared = r2_score(y_test, preds_forest_temp)
        if r_squared > best_r_squared:
            best_r_squared = r_squared
            best_preds = preds_forest_temp
            best_tree_count = n_trees
        # plot_predictions(preds_forest_temp, y_test, r_squared, model=f"Random-Forest-{n_trees}")
        results_r2.append(r_squared)
    test_data = create_df(best_preds, dataset, test_indices)
    create_interactive_plot(test_data, model=f"RandomForest-{best_tree_count}")
    results_r2 = np.array(results_r2)
    table_of_results = np.concatenate((num_trees_list.reshape(len(num_trees_list), 1), results_r2.reshape(len(results_r2), 1)), axis=1)
    print(table_of_results)

if __name__ == "__main__":
    main()