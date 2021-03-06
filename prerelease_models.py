'''
testing on trailer only
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from naive_bayes import OverviewClassPrediction
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from utils.misc import plot_predictions, plot_history, create_interactive_plot, create_df, scale_data, stringify_model, inverse_transform
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.random import set_seed

set_seed(18)
def main():
    x_train, x_test, y_train, y_test, _, _, dataset, test_indices = get_trailer_data(
        "dbs/data_2010s.csv", scale_input=False)
    x_train_naive, x_test_naive = join_with_overview_bayes(x_train, x_test, dataset, test_indices)
    print("Forest Pre-release")
    forest(x_train, x_test, y_train, y_test)
    print("Forest Pre-release with Overview")
    forest(x_train_naive, x_test_naive, y_train, y_test)
    print("Linear Pre-release")
    linear(x_train, x_test, y_train, y_test)
    print("Linear Pre-release with Overview")
    linear(x_train_naive, x_test_naive, y_train, y_test)
    print("SVR Pre-release")
    support_vector(x_train, x_test, y_train, y_test)
    print("SVR Pre-release with Overview")
    support_vector(x_train_naive, x_test_naive, y_train, y_test)

def forest(x_train, x_test, y_train, y_test):
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
    # test_data = create_df(best_preds, dataset, test_indices)
    # plot_predictions(best_preds, y_test, best_r_squared, model=f"RandomForest-{best_tree_count}-0.8BS")
    # create_interactive_plot(test_data, model=f"RandomForest-{best_tree_count}-trees-0.8BS")
    results_r2 = np.array(results_r2)
    table_of_results = np.concatenate((num_trees_list.reshape(len(num_trees_list), 1), results_r2.reshape(len(results_r2), 1)), axis=1)
    print(table_of_results)

def linear(x_train, x_test, y_train, y_test):
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    preds = regressor.predict(x_test)
    r_squared = r2_score(y_test, preds)
    print(r_squared)
    # plot_predictions(preds, y_test, r_squared, model="LinearRegression", show_fig=True)

def support_vector(x_train, x_test, y_train, y_test):
    '''
    support vector try
    '''
    x_train, x_test, y_train, y_test, scalar_x, scalar_y =  scale_data(x_train, x_test, y_train, y_test)

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    results_r2 = []
    results_r2_train = []
    for kernel in kernels:
        regressor_temp = SVR(kernel=kernel)
        regressor_temp.fit(x_train, y_train.flatten())
        preds = regressor_temp.predict(x_test)
        r_squared = r2_score(y_test, preds)
        results_r2.append(r_squared)
        preds, actual = inverse_transform(preds, y_test, scalar_y)
        preds = regressor_temp.predict(x_train)
        results_r2_train.append(r2_score(y_train, preds))
    kernels = np.array(kernels)
    results_r2 = np.array(results_r2)
    table_of_results = np.concatenate((kernels.reshape(len(kernels), 1), results_r2.reshape(len(results_r2), 1)), axis=1)
    print(table_of_results)


def mlp(layers):
    # generate data with scaled input
    x_train, x_test, y_train, y_test, _, scalar_y, dataset, _ = get_trailer_data("dbs/data_2010s.csv", scale_input=True)
    loss_function = "mean_squared_error"
    # define the layers of the model, excluding the output layer
    layers = [x_train.shape[1]] + layers
    # call the function that will build the model
    model = build_model(layers, loss_function=loss_function)
    # define what weights we want to save and how we want to save them
    callback = ModelCheckpoint(
        filepath=f"weights/prereleased-{loss_function}-nn-{stringify_model(layers)}-1-weights.h5",
        verbose=1,
        save_best_only=True,
        monitor="val_" + loss_function,
        save_weights_only=True
        )

    # train the network
    results = model.fit(
        x_train, y_train,
        batch_size=50,
        epochs=50,
        validation_data=(x_test, y_test),
        callbacks=[callback]
        )

    # plot the history of the model based on the loss_function
    plot_history(results, layers, loss_function, norm="-prerelease", show_fig=True)

     # get the rescaled predictions
    predictions, actual_values = inverse_transform(model.predict(x_test), y_test, scalar_y)
    r_squared = r2_score(predictions, actual_values)
    print(r_squared)

def overview_mlp(layers):
    # generate data with scaled input
    x_train, x_test, y_train, y_test, _, scalar_y, dataset, test_indices = get_trailer_data("dbs/data_2010s.csv", scale_input=True)
    x_train, x_test = join_with_overview_bayes(x_train, x_test, dataset, test_indices)
    loss_function = "mean_squared_error"
    # define the layers of the model, excluding the output layer
    layers = [x_train.shape[1]] + layers
    # call the function that will build the model
    model = build_model(layers, loss_function=loss_function)
    # define what weights we want to save and how we want to save them
    callback = ModelCheckpoint(
        filepath=f"weights/prereleased-overview-{loss_function}-nn-{stringify_model(layers)}-1-weights.h5",
        verbose=1,
        save_best_only=True,
        monitor="val_" + loss_function,
        save_weights_only=True
        )

    # train the network
    results = model.fit(
        x_train, y_train,
        batch_size=50,
        epochs=50,
        validation_data=(x_test, y_test),
        callbacks=[callback]
        )

    # plot the history of the model based on the loss_function
    plot_history(results, layers, loss_function, norm="-prerelease-overview", show_fig=True)

     # get the rescaled predictions
    predictions, actual_values = inverse_transform(model.predict(x_test), y_test, scalar_y)
    r_squared = r2_score(predictions, actual_values)
    print(r_squared)

def neural_network(layers):
    '''
    build multilayer perceptron based on `layers`
    '''
    model = Sequential()
    model.add(Input(shape=(layers[0],)))
    for number_nodes in layers[1:]:
        model.add(Dense(number_nodes, kernel_initializer="normal", activation="relu"))
    model.add(Dense(1, activation="linear")) # one node for regression
    return model

def build_model(layers, loss_function="mean_squared_error"):
    '''
    top level function to build MLP and plot the model parameters
    '''
    model = neural_network(layers)
    model.compile(loss=loss_function, optimizer="adam", metrics=[loss_function])
    print(model.summary())
    return model

def test(weights_file):
    _, x_test, _, y_test, _, scalar_y, dataset, test_indices = get_trailer_data(
        "dbs/data_2010s.csv", scale_input=True)
    layers = [int(val) for val in weights_file.split("/")[-1].split("-") if val.isdigit()][:-1]
    model = build_model(layers)
    model.load_weights(weights_file)
    predictions, actual_values = inverse_transform(model.predict(x_test), y_test, scalar_y)
    r_squared = r2_score(predictions, actual_values)
    print(r_squared)
    # plot_predictions(
    #     predictions, actual_values, r_squared, layers=layers, best="best-", save_fig=False, show_fig=show_fig)
    return predictions, actual_values, dataset, test_indices

def overview_test(weights_file):
    _, x_test, _, y_test, _, scalar_y, dataset, test_indices = get_trailer_data(
        "dbs/data_2010s.csv", scale_input=True)
    layers = [int(val) for val in weights_file.split("/")[-1].split("-") if val.isdigit()][:-1]
    model = build_model(layers)
    model.load_weights(weights_file)
    predictions, actual_values = inverse_transform(model.predict(x_test), y_test, scalar_y)
    r_squared = r2_score(predictions, actual_values)
    print(r_squared)
    # plot_predictions(
    #     predictions, actual_values, r_squared, layers=layers, best="best-", save_fig=False, show_fig=show_fig)
    return predictions, actual_values, dataset, test_indices

def join_with_overview_bayes(x_train, x_test, dataset, test_indices):
    '''
    use naive bayues predictions based on specified `metric`
    '''
    overviews = pd.read_csv("dbs/movies_from_2010s.csv")[['tmdb_id', 'overview']]
    df = dataset.merge(overviews, on="tmdb_id")
    train = df.loc[~df.index.isin(test_indices)]
    test = df.iloc[test_indices]
    classifier = OverviewClassPrediction()
    classifier.fit(train.copy())
    # classifier.test_all_models(test.copy())
    # classifier.evaluate_models()
    train_preds = classifier.predict(train.copy(), model="Gaussian").reshape(len(x_train+1), 1)
    test_preds = classifier.predict(test.copy(), model="Gaussian").reshape(len(x_test+1), 1)
    x_train = np.concatenate((x_train, train_preds), axis=1)
    x_test = np.concatenate((x_test, test_preds), axis=1)

    return x_train, x_test

def get_trailer_data(csv, scale_input=False):
    dataset = pd.read_csv(csv)
    data = dataset.drop(columns=["tmdb_id", "year", "title", "vote_count", "vote_average", "popularity"]).astype('float')
    scalar_x = None
    scalar_y = None

    features = data.iloc[:, :-1].values
    output = data.iloc[:, -1].values
    indices = np.arange(len(output))
    x_train, x_test, y_train, y_test, _, test_indices = train_test_split(features, output, indices, test_size=20/100, random_state=18)

    if scale_input:
        x_train, x_test, y_train, y_test, scalar_x, scalar_y = scale_data(x_train, x_test, y_train, y_test)
    return x_train, x_test, y_train, y_test, scalar_x, scalar_y, dataset, test_indices

if __name__ == "__main__":
    # forest()
    # linear()
    # overview_mlp([10,20])
    # test("weights/prereleased-mean_squared_error-nn-24-10-20-1-weights.h5")
    main()