'''
support vector machine for regression
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from utils.misc import plot_predictions, inverse_transform, create_interactive_plot, create_df
from furiosanet import generate_data

def main():
    x_train, x_test, y_train, y_test, _, scalar_y, dataset, test_indices = generate_data(
        "dbs/data_2010s.csv")
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    results_r2 = []
    results_r2_train = []
    for kernel in kernels:
        regressor_temp = SVR(kernel=kernel)
        regressor_temp.fit(x_train, y_train.flatten())
        preds = regressor_temp.predict(x_test)
        print(preds.shape, y_test.shape)
        r_squared = r2_score(y_test, preds)
        results_r2.append(r_squared)
        preds, actual = inverse_transform(preds, y_test, scalar_y)
        plot_predictions(preds, actual, r_squared, model=f"SVR-{kernel}")
        df = create_df(preds, dataset, test_indices)
        create_interactive_plot(df, model=f"SVR-{kernel}")
        # preds = regressor_temp.predict(x_train)
        # results_r2_train.append(r2_score(y_train, preds))
    kernels = np.array(kernels)
    results_r2 = np.array(results_r2)
    table_of_results = np.concatenate((kernels.reshape(len(kernels), 1), results_r2.reshape(len(results_r2), 1)), axis=1)
    print(table_of_results)
    print(results_r2_train)

if __name__ == "__main__":
    main()