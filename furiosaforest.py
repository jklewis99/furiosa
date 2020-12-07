'''
random forest regression
'''
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from utils.misc import plot_predictions, create_interactive_plot, create_df, generate_data
from sklearn import tree

def main(compare=False):
    x_train, x_test, y_train, y_test, _, _, dataset, test_indices = generate_data(
        "dbs/data_2010s.csv", scale_input=False)
    table_of_results = run_model(x_train, x_test, y_train, y_test, dataset, test_indices)
    print(table_of_results)
    if compare:
        x_train, x_test, y_train, y_test, _, _, dataset, test_indices = generate_data(
            "dbs/data_2010s.csv", drop_features=['title', 'tmdb_id', 'year', 'view_count', 'like_count', 'dislike_count', 'comment_count'],
            scale_input=False)
        table_of_results = run_model(x_train, x_test, y_train, y_test, dataset, test_indices)
        print("Baseline\n", table_of_results)

def run_model(x_train, x_test, y_train, y_test, dataset, test_indices):
    '''
    logic of the model
    '''
    num_trees_list = np.arange(20, 96, 5)
    results_r2 = []
    best_preds = None
    best_r_squared = 0
    best_tree_count = 0
    for n_trees in num_trees_list:
        # TODO: Hyper-parameter Tuning
        regressor_temp = RandomForestRegressor(n_estimators=n_trees, max_samples=0.8, random_state=18)
        regressor_temp.fit(x_train, y_train) # if scaled: .ravel())
        preds_forest_temp = regressor_temp.predict(x_test)
        r_squared = r2_score(y_test, preds_forest_temp)
        # plt.figure(figsize=(10,10))
        # tree.plot_tree(
        #         regressor_temp.estimators_[0],
        #         feature_names = dataset.columns[3:-1],
        #         filled=True,
        #         rounded=True,
        #         fontsize=5
        #     )
        # os.system('dot -Tpng tree.dot -o figures/tree.png')
        # plt.show()
        if r_squared > best_r_squared:
            best_r_squared = r_squared
            best_preds = preds_forest_temp
            best_tree_count = n_trees
        # plot_predictions(preds_forest_temp, y_test, r_squared, model=f"RandomForest-{n_trees}")
        results_r2.append(r_squared)
    test_data = create_df(best_preds, dataset, test_indices)
    # # test_data.to_csv("forest40-plotly.csv")
    # plot_predictions(best_preds, y_test, best_r_squared, model=f"RandomForest-{best_tree_count}-0.8BS")
    create_interactive_plot(test_data, model=f"RandomForest-{best_tree_count}-trees-0.8BS")
    results_r2 = np.array(results_r2)
    table_of_results = np.concatenate((num_trees_list.reshape(len(num_trees_list), 1), results_r2.reshape(len(results_r2), 1)), axis=1)
    return table_of_results

if __name__ == "__main__":
    main()