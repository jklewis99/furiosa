'''
module containing all functions related to a nerual network to predict revenue with
regression. Training, testing, and evaluating models are methods in this module.
'''

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.misc import stringify_model, plot_history, plot_predictions, inverse_transform, generate_data, create_df, create_interactive_plot
from sklearn.metrics import r2_score
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.random import set_seed

set_seed(18)

def neural_network(layers):
    '''
    build multilayer perceptron based on `layers`
    '''
    model = Sequential()
    model.add(Input(shape=(layers[0],)))
    for number_nodes in layers[1:]:
        model.add(Dense(number_nodes, activation="relu"))
        # model.add(Dense(number_nodes, kernel_initializer="normal", activation="relu"))
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

def get_layers_from_file(path):
    '''
    return the layers as a list of integers based on the `path`
    '''
    # all weights are formatted in "path/to/nn-{norm-}27-{H1}-{H2...}-1-weights.h5"
    # so we split by the path, then split by "-" and only take the layers, then remove the 
    # output layer
    layers = [int(val) for val in path.split("/")[-1].split("-") if val.isdigit()][:-1]
    return layers

def train(layers, loss_function, show_preds=False, scale_input=True):
    '''
    method for training FuriosaNet

    Parameters
    ==========
    `layers`:
        list of specified number of neurons per layer
    '''

    # generate data with scaled input
    x_train, x_test, y_train, y_test, _, scalar_y, dataset, _ = generate_data("dbs/data_2010s.csv", scale_input=scale_input)
    # x_train, x_test, y_train, y_test, _, scalar_y, dataset, _ = generate_data(
    #         "dbs/data_2010s.csv", drop_features=['title', 'tmdb_id', 'year', 'view_count', 'like_count', 'dislike_count', 'comment_count'])
    # define normalization string for specifying saved filed
    normalization = ""
    if scale_input:
        normalization = "-norm"
    # define the layers of the model, excluding the output layer
    layers = [x_train.shape[1]] + layers
    # call the function that will build the model
    model = build_model(layers, loss_function=loss_function)
    # define what weights we want to save and how we want to save them
    callback = ModelCheckpoint(
        filepath=f"weights/nn{normalization}-{loss_function}-{stringify_model(layers)}-1-weights.h5",
        verbose=1,
        save_best_only=True,
        monitor="val_" + loss_function,
        save_weights_only=True
        )

    # train the network
    results = model.fit(
        x_train, y_train,
        batch_size=50,
        epochs=100,
        validation_data=(x_test, y_test),
        callbacks=[callback]
        )

    # plot the history of the model based on the loss_function
    plot_history(results, layers, loss_function, norm=normalization)

    if show_preds:
        # get the rescaled predictions
        predictions, actual_values = inverse_transform(model.predict(x_test), y_test, scalar_y)
        # plot the predictions and get the r-squared value of the model
        r_squared = r2_score(predictions, actual_values)
        plot_predictions(predictions, actual_values, r_squared, layers=layers, norm=normalization)

def test(weights_file, layers, data_file="dbs/data_2010s.csv", create_fig=True, show_fig=True, scale_input=True):
    '''
    test a model with specified `layers` architecture using the `weights_file`

    Parameters
    ==========
    `weights_file`:
        path to the .h5 file containing the pretrained weights

    `layers`:
        list of integer values specifying the number of nodes at the each hidden layer
        except the final layer

    Keyword Args:
    ==========
    `show_fig`:
        default True; display graph of actual values vs predictions

    Returns
    ==========
    (predictions, actual_values, dataset, test_indices)
    '''
    _, x_test, _, y_test, _, scalar_y, dataset, test_indices = generate_data(
        data_file, scale_input=scale_input)
    # _, x_test, _, y_test, _, scalar_y, dataset, _ = generate_data(
    #         "dbs/data_2010s.csv", drop_features=['title', 'tmdb_id', 'year', 'view_count', 'like_count', 'dislike_count', 'comment_count'])
    model = build_model(layers)
    model.load_weights(weights_file)
    predictions, actual_values = inverse_transform(model.predict(x_test), y_test, scalar_y)
    r_squared = r2_score(predictions, actual_values)

    if create_fig:
        plot_predictions(
            predictions, actual_values, r_squared, layers=layers, best="best-", save_fig=True, show_fig=show_fig)
        df = create_df(predictions, dataset, test_indices)
        create_interactive_plot(df, model=layers)

    return predictions, actual_values, dataset, test_indices

def evaluate(metric, weights_folder, save_table=True, create_fig=False):
    '''
    method that will compare all models from a specified `weights_folder`
    based on `metric`

    Parameters
    ==========
    `metric`:
        metric to evaluate models

    `weights_folder`:
        path to fodler containing the weights of model which are meant to be modeled.
    '''
    metric_functions = {
        "r-squared": r2_score,
    }
    predictions = None
    _, x_test, _, y_test, _, scalar_y, dataset, _ = generate_data("dbs/data_2010s.csv", scale_input=True)

    models = dict()
    weights_files = [weights for weights in os.listdir(weights_folder)]
    for weights_file in weights_files:
        layers = get_layers_from_file(weights_file)
        model = build_model(layers)
        model.load_weights(os.path.join(weights_folder, weights_file))
        predictions, actual_values = inverse_transform(model.predict(x_test), y_test, scalar_y)
        r2 = metric_functions[metric](actual_values, predictions)
        if create_fig:
            # TODO: plot get some plottable points to be layers on one graph
            print()
        models[weights_file] = r2

    # TODO: show points plotted by each model maybe?
    if create_fig:
        print()
    if save_table:
        pd.DataFrame.from_dict(
            models, orient='index', columns=["r-squared"]).to_csv("model-evaluation.csv")
    else:
        print(f"{'Model':^40s}| {metric}")
        for k in models:
            print(f"{k:^40s}| {models[k]:0.3f}")
    return models

def main():
    parser = argparse.ArgumentParser(description='FuriosaNet model')
    parser.add_argument(
        'mode',
        choices=['train', 'test', 'evaluate'],
        help="Mode in which the model should be run"
        )
    parser.add_argument(
        '-weights',
        type=str,
        help="If testing, path of saved weights"
        )
    parser.add_argument(
        '-layers',
        type=str,
        default="50,100",
        help="If training, comma separated values defining the size of each hidden layer"
        )
    parser.add_argument(
        '-loss',
        type=str,
        default="mean_squared_error",
        help="loss function to monitor. Default: mean_squared_error"
        )
    parser.add_argument(
        '-weightsfolder',
        type=str,
        default="/weights",
        help="if evaluating, folder containing the weights that are to be compared"
        )
    parser.add_argument(
        '-by', '-evaluateby',
        choices=['r-squared'],
        default='r-squared',
        help="If evaluating, metric by which to compare models. Deafult: r-squared"
    )
    args = parser.parse_args()

    if args.mode == 'test':
        layers = get_layers_from_file(args.weights)
        test(args.weights, layers)
    elif args.mode == 'evaluate':
        evaluate(args.by, args.weightsfolder)
    else:
        train([int(val) for val in args.layers.split(",")], args.loss)

if __name__ == "__main__":
    main()
