'''
linear regression
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from utils.misc import plot_predictions, inverse_transform, create_interactive_plot, create_df
from furiosanet import generate_data

def main():
    x_train, x_test, y_train, y_test, _, _, dataset, test_indices = generate_data(
        "dbs/data_2010s.csv", scale_input=False)
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    preds = regressor.predict(x_test)
    r_squared = r2_score(y_test, preds)
    plot_predictions(preds, y_test, r_squared, model="LinearRegression", show_fig=True)
    # test_data = create_df(best_preds, dataset, test_indices)

if __name__ == "__main__":
    main()