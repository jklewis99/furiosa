import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.layers.core import Activation
from tensorflow.random import set_seed

set_seed(18)

def neural_network(layers=None):
    model = Sequential()
    model.add(Input(shape=(layers[0],)))
    for number_nodes in layers[1:]:
        model.add(Dense(number_nodes, kernel_initializer="normal", activation="relu"))
    model.add(Dense(1, activation="linear")) # one node for regression
    return model

def generate_data(csv, split=(80,20), output_index=-1, scale_input=True):
    '''
    read data from the specified `csv` file and normalize data, and
    split into training features, testing features, training labels,
    and testing labels

    Parameters
    ==========
    `csv`:
        string specifying the path to the file containing data

    Keyword Args
    ==========
    `split`:
        default (80, 20); tuple specifying the percentage of data for
        training and testing, respectively
    `output_index`:
        default -1; location in the dataset which is the value to predict
    `scale_input`:
        default True; boolean declaring whether to scale data

    Return
    ==========
    (x_train, x_test, y_train, y_test, scalar_x, scalar_y, dataset)
    numpy arrays: x_train, x_test, y_train, y_test;
    Scalar objects: scalar_x, scalar_y
    Pandas DataFrame: dataset
    '''
    dataset = pd.read_csv(csv)
    data = dataset.drop(columns=['title', 'tmdb_id', 'year']).astype('float')
    scalar_x = None
    scalar_y = None

    features = data.iloc[:, :output_index].values
    output = data.iloc[:, output_index].values

    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=split[1]/100, random_state=18)

    if scale_input:
        x_train, x_test, y_train, y_test, scalar_x, scalar_y = scale_data(x_train, x_test, y_train, y_test)
    return x_train, x_test, y_train, y_test, scalar_x, scalar_y, dataset

def scale_data(x_train, x_test, y_train, y_test):
    '''
    Uses Standard Scalar, which sets the mean to 0 and standard deviation to 1, to
    scale the inputs. Training values will be fit and transformed, and the test data
    will be transformed.

    Parameters
    ==========
    `x_train`:
        numpy array of features for all samples in training data
    `x_test`:
        numpy array of features for all outputs in testing data
    `y_train`:
        numpy array of outputs for all samples in training data
    `y_test`:
        numpy array of outputs for all samples in testing data

    Return
    ==========
    (x_train, x_test, y_train, y_test, scalar_x, scalar_y)
    tuple of the scaled x_train, transformed x_test, sacled y_train,
    transformed y_test, saclar used for features, scalar used for outputs
    '''
    scalar_x = StandardScaler() # need to scale our data (I think)
    scalar_y = StandardScaler() # need to scale our data (I think)

    x_train = scalar_x.fit_transform(x_train)
    y_train = scalar_y.fit_transform(y_train.reshape(-1, 1))
    x_test = scalar_x.transform(x_test)
    y_test = scalar_y.transform(y_test.reshape(-1, 1))

    return x_train, x_test, y_train, y_test, scalar_x, scalar_y

def plot_history(results, layers, loss, norm=""):
    '''
    plot the model's history during training
    '''
    plt.figure(figsize=(10, 8))
    plt.plot(np.array(results.history[loss])[15:])
    plt.plot(np.array(results.history["val_"+ loss])[15:])
    plt.title('MAE on training and testing data', fontsize=24)
    plt.ylabel('Mean Absolute Error', fontsize=18)
    plt.xlabel('Epoch', fontsize=18)
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(f"figures/history{norm}-{stringify_model(layers)}-1.png")
    plt.show()

def plot_predictions(predictions, actual, layers, norm="", best=""):
    '''
    plot the model's predictions
    '''
    plt.figure(figsize=(8, 8))
    plt.scatter(predictions, actual)
    plt.title('Actual vs Predicted', fontsize=24)
    plt.ylabel('Actual Value', fontsize=18)
    plt.xlabel('Predicted Value', fontsize=18)
    r_squared = r2_score(actual, predictions)
    plt.annotate(f"r^2 value = {r_squared:.3f}", (2, 0.9*np.max(actual)))
    plt.savefig(f"figures/{best}predictions{norm}-{stringify_model(layers)}-1.png")
    plt.show()

def build_model(layers, loss_function="mean_squared_error"):
    model = neural_network(layers)
    model.compile(loss=loss_function, optimizer="adam", metrics=[loss_function])
    print(model.summary())
    return model

def get_layers_from_file(path):
    '''
    return the layers as a list of integers based on the `path`
    '''
    # all weights are formatted in "path/to/nn-{norm-}27-{H1}-{H2...}-1-weights.h5"
    # so we split by the path, then split by "-" and only take the layers, then remove the output layer
    layers = [int(val) for val in path.split("/")[-1].split("-") if val.isdigit()][:-1]
    return layers

def stringify_model(layers):
    '''
    convert a list of the layers into a string for saving

    Parameters
    ==========
    `layers`:
        list of specified number of neurons per layer
    '''
    return "-".join(map(str, layers))

def train(layers, loss_function):
    '''
    method for training FuriosaNet

    Parameters
    ==========
    `layers`:
        list of specified number of neurons per layer
    '''

    # generate data with scaled input
    x_train, x_test, y_train, y_test, _, scalar_y, dataset = generate_data("dbs/data_2010s.csv", scale_input=True)
    # define normalization string for specifying saved filed
    normalization = "-norm"
    # define the layers of the model, excluding the output layer
    layers = [x_train.shape[1]] + layers
    # call the function that will build the model
    model = build_model(layers, loss_function=loss_function)
    # define what weights we want to save and how we want to save them
    callback = ModelCheckpoint(
        filepath=f"weights/nn{normalization}-{stringify_model(layers)}-1-weights.h5",
        verbose=1,
        save_best_only=True,
        monitor="val_" + loss_function,
        save_weights_only=True
        )
    
    # train the network
    results = model.fit(
        x_train, y_train,
        batch_size=50,
        epochs=500,
        validation_data=(x_test, y_test),
        callbacks=[callback]
        )

    # plot the history of the model based on the loss_function
    plot_history(results, layers, loss_function, norm=normalization)
    # get the rescaled predictions
    predictions = scalar_y.inverse_transform(np.array([val[0] for val in model.predict(x_test)]))
    # plot the predictions and get the r-squared value of the model
    plot_predictions(predictions, scalar_y.inverse_transform(y_test), layers, norm=normalization)

def test(weights_file, layers, loss_function):
    x_train, x_test, y_train, y_test, scalar_x, scalar_y, dataset = generate_data("dbs/data_2010s.csv", scale_input=True)
    model = build_model(layers)
    model.load_weights(weights_file)
    predictions = scalar_y.inverse_transform(np.array([val[0] for val in model.predict(x_test)]))
    plot_predictions(predictions, scalar_y.inverse_transform(y_test), layers, best="best-")

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
    _, x_test, _, y_test, _, scalar_y, dataset = generate_data("dbs/data_2010s.csv", scale_input=True)

    models = dict()
    weights_files = [weights for weights in os.listdir(weights_folder)]
    for weights_file in weights_files:
        layers = get_layers_from_file(weights_file)
        model = build_model(layers)
        model.load_weights(weights_file)
        predictions = scalar_y.inverse_transform(np.array([val[0] for val in model.predict(x_test)]))
        r2 = metric_functions[metric](y_test, predictions)
        if create_fig:
            # TODO: plot get some plottable points to be layers on one graph
            print()
        models[weights_file] = r2

    # TODO: show points plotted by each model maybe?
    if create_fig:
        print()
    if save_table:
        pd.DataFrame(models).to_csv("model-evaluation.csv", index=False)
    else:
        print(f"{'Model':^40s}| {metric}")
        for k in models:
            print(f"{k:^40s}| {models[k]:0.3f}")
    return models

def main():
    parser = argparse.ArgumentParser(description='FuriosaNet model')
    parser.add_argument(
        '-mode',
        choices=['train', 'test', 'evaluate'],
        default='train',
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
        default="mean_absolute_error",
        help="loss function to monitor. Default: mean_absolute_error"
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
        test(args.weights, layers, args.loss)
    elif args.mode == 'evaluate':
        evaluate(args.by, args.weightsfolder)
    else:
        train([int(val) for val in args.layers.split(",")], args.loss)

if __name__ == "__main__":
    main()