'''
linear regression
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from utils.misc import plot_predictions, inverse_transform, create_interactive_plot, create_df, generate_data

def main(compare=False):
    x_train, x_test, y_train, y_test, _, _, dataset, test_indices = generate_data(
        "dbs/data_2010s.csv", scale_input=False)
    if compare:
        x_train_base, x_test_base, y_train_base, y_test_base, _, _, _, _ = generate_data(
            "dbs/data_2010s.csv", drop_features=['title', 'tmdb_id', 'year', 'view_count', 'like_count', 'dislike_count', 'comment_count'],
            scale_input=False)
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    preds = regressor.predict(x_test)
    r_squared = r2_score(y_test, preds)
    print("Full: ", r_squared)
    if compare:
        regressor = LinearRegression()
        regressor.fit(x_train_base, y_train_base)
        preds = regressor.predict(x_test_base)
        r_squared = r2_score(y_test_base, preds)
        print("Baseline: ", r_squared)
    # plot_predictions(preds, y_test, r_squared, model="LinearRegression", show_fig=True)
    test_data = create_df(preds, dataset, test_indices)
    create_interactive_plot(test_data, model=f"Linear Regression")

if __name__ == "__main__":
    main()