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
from tensorflow.keras.utils import to_categorical
from tensorflow.random import set_seed

set_seed(18)

class FuriosaNet():

    def __init__(self, num_classes):
        self.model = None
        self.num_classes = num_classes
        self.class_thresholds = self.create_class_thresholds(num_classes)
    
    def neural_network(self, layers):
        '''
        build multilayer perceptron based on `layers`
        '''
        self.model = Sequential()
        self.model.add(Input(shape=(layers[0],)))
        for number_nodes in layers[1:]:
            self.model.add(Dense(number_nodes, kernel_initializer="normal", activation="relu"))
        self.model.add(Dense(self.num_classes, activation="softmax")) # one node for each class
        return self.model

    def build_model(self, layers, loss_function="binary_crossentropy"):
        '''
        top level function to build MLP and plot the model parameters
        '''
        self.model = self.neural_network(layers)
        self.model.compile(loss=loss_function, optimizer="adam", metrics=[loss_function, "accuracy"])
        print(self.model.summary())
        return self.model

    def get_class(self, revenues, scalar_y):
        '''
        method to add a row to the data

        Parameters
        ==========
        `revenue`:
            numpy array of film revenue
        '''
        class_thresholds = scalar_y.transform(np.array(self.class_thresholds).reshape(-1, 1))
        labels = np.zeros_like(revenues)
        
        for i, revenue in enumerate(scalar_y.inverse_transform(revenues)):
            assigned = False
            for j, threshold in enumerate(self.class_thresholds):
                if revenue < threshold:
                    labels[i] = j
                    assigned = True
                    break
            if not assigned:
                labels[i] = len(self.class_thresholds)
        print(labels)
        return labels

    def create_class_thresholds(self, num_classes):
        '''
        create thresholds of length `num_classes` - 1
        '''
        basis_threshold = 500 / num_classes
        return [basis_threshold * 2**n * 10**6 for n in range(num_classes-1)]

    def load_model(self, weights_file, layers):
        self.model = self.build_model(layers)
        self.model.load_weights(weights_file)

    def train(self, layers, loss_function='binary_crossentropy', show_preds=False, scale_input=True):
        '''
        method for training FuriosaNet

        Parameters
        ==========
        `layers`:
            list of specified number of neurons per layer
        '''

        # generate data with scaled input
        # x_train, x_test, y_train, y_test, _, scalar_y, dataset, _ = generate_data("dbs/data_2010s.csv", scale_input=scale_input)
        x_train, x_test, y_train, y_test, _, scalar_y, dataset, _ = generate_data(
                "dbs/data_2010s.csv", drop_features=["tmdb_id", "year", "title", "vote_count", "vote_average", "popularity"])
        # updated revenues to classes
        y_train = to_categorical(self.get_class(y_train, scalar_y), self.num_classes)
        y_test = to_categorical(self.get_class(y_test, scalar_y), self.num_classes)
        
        # define normalization string for specifying saved filed
        normalization = ""
        if scale_input:
            normalization = "-norm"
        # define the layers of the model, excluding the output layer
        layers = [x_train.shape[1]] + layers
        # call the function that will build the model
        self.model = self.build_model(layers, loss_function)
        # define what weights we want to save and how we want to save them
        callback = ModelCheckpoint(
            filepath=f"weights/furiosanet{normalization}-{loss_function}-{stringify_model(layers)}-{self.num_classes}-weights.h5",
            verbose=1,
            save_best_only=True,
            monitor="val_accuracy", # + loss_function,
            save_weights_only=True
            )

        # train the network
        results = self.model.fit(
            x_train, y_train,
            batch_size=50,
            epochs=500,
            validation_data=(x_test, y_test),
            callbacks=[callback]
            )
        # plot the history of the model based on the loss_function
        plot_history(results, layers, loss_function, output_size=self.num_classes, norm=normalization)

        # if show_preds:
        #     # get the rescaled predictions
        #     predictions, actual_values = inverse_transform(self.model.predict(x_test), y_test, scalar_y)
        #     # plot the predictions and get the r-squared value of the model
        #     r_squared = r2_score(predictions, actual_values)
        #     plot_predictions(predictions, actual_values, r_squared, layers=layers, output_size=self.num_classes, norm=normalization)

    def test(self, data_file="dbs/data_2010s.csv", create_fig=True, show_fig=True, scale_input=True):
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
        # _, x_test, _, y_test, _, scalar_y, dataset, test_indices = generate_data(
        #     data_file, scale_input=scale_input)
        _, x_test, _, y_test, _, scalar_y, dataset, _ = generate_data(
                "dbs/data_2010s.csv", drop_features=["tmdb_id", "year", "title", "vote_count", "vote_average", "popularity"])
        # updated revenues to classes
        y_test = to_categorical(self.get_class(y_test, scalar_y), self.num_classes)

        predictions = np.argmax(self.model.predict(x_test), axis=1)
        actual = np.argmax(y_test, axis=1)

        accuracy = 100 * np.sum(actual == predictions) / len(actual)
        one_away_accuracy = 100 * self.one_away_count(predictions, actual) / len(actual)

        print(f"Accuracy: {accuracy:.0f}%")
        print(f"One Away Accuracy: {one_away_accuracy:.0f}%")
        # if create_fig:
        #     plot_predictions(
        #         predictions, actual_values, r_squared, layers=layers, best="bbest-", save_fig=True, show_fig=show_fig)
        #     df = create_df(predictions, dataset, test_indices)
        #     create_interactive_plot(df, model=layers)

        return predictions, actual #_values, dataset, test_indices

    def evaluate(self, metric, weights_folder, save_table=True, create_fig=False):
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
        # metric_functions = {
        #     "r-squared": r2_score,
        # }
        # predictions = None
        # _, x_test, _, y_test, _, scalar_y, dataset, _ = generate_data("dbs/data_2010s.csv", scale_input=True)

        # models = dict()
        # weights_files = [weights for weights in os.listdir(weights_folder)]
        # for weights_file in weights_files:
        #     layers = get_layers_from_file(weights_file)
        #     model = build_model(layers)
        #     model.load_weights(os.path.join(weights_folder, weights_file))
        #     predictions, actual_values = inverse_transform(model.predict(x_test), y_test, scalar_y)
        #     r2 = metric_functions[metric](actual_values, predictions)
        #     if create_fig:
        #         # TODO: plot get some plottable points to be layers on one graph
        #         print()
        #     models[weights_file] = r2

        # # TODO: show points plotted by each model maybe?
        # if create_fig:
        #     print()
        # if save_table:
        #     pd.DataFrame.from_dict(
        #         models, orient='index', columns=["r-squared"]).to_csv("model-evaluation.csv")
        # else:
        #     print(f"{'Model':^40s}| {metric}")
        #     for k in models:
        #         print(f"{k:^40s}| {models[k]:0.3f}")
        # return models
        pass

    def predict(self, data, human=True):
        '''
        predict on a set or example of data
        '''
        revenue_range = []
        preds = np.argmax(self.model.predict(data), axis=1)
        return preds
        # if human:
        #     for pred in preds:
        #         if pred > 0:
        #             p = f"{self.class_thresholds[pred-1]} < revenue "
        #             if pred < self.num_classes-1:
        #                 p = p + f"{self.class_thresholds[pred]}"
        #         else:
        #             p = f"revenue < {self.class_thresholds[pred]}"
        #         revenue_range.append(p)
        # return revenue_range

    
    @staticmethod
    def one_away_count(preds, actual):
        ''''''
        return len(preds[np.where(np.abs(actual-preds) <= 1)])

def get_layers_from_file(path):
    '''
    return the layers as a list of integers based on the `path`
    '''
    # all weights are formatted in "path/to/nn-{norm-}27-{H1}-{H2...}-1-weights.h5"
    # so we split by the path, then split by "-" and only take the layers, then remove the 
    # output layer
    layers = [int(val) for val in path.split("/")[-1].split("-") if val.isdigit()]
    return layers

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
        default="50,100,5",
        help="If training, comma separated values defining the size of each hidden layer"
        )
    parser.add_argument(
        '-loss',
        type=str,
        default="binary_crossentropy",
        help="loss function to monitor. Default: binary_crossentropy"
        )
    parser.add_argument(
        '-weightsfolder',
        type=str,
        default="/weights",
        help="if evaluating, folder containing the weights that are to be compared"
        )
    # parser.add_argument(
    #     '-by', '-evaluateby',
    #     choices=['r-squared'],
    #     default='r-squared',
    #     help="If evaluating, metric by which to compare models. Default: r-squared"
    # )
    args = parser.parse_args()

    if args.mode == 'test':
        hidden_layers = get_layers_from_file(args.weights)
        hidden_layers, output_size = hidden_layers[:-1], hidden_layers[-1]
        clf = FuriosaNet(output_size)
        clf.load_model(args.weights, hidden_layers)
        clf.test()
    elif args.mode == 'evaluate':
        pass
        # evaluate(args.by, args.weightsfolder)
    else:
        hidden_layers = [int(val) for val in args.layers.split(",")]
        hidden_layers, output_size = hidden_layers[:-1], hidden_layers[-1]
        clf = FuriosaNet(output_size)
        clf.train(hidden_layers, args.loss)

if __name__ == "__main__":
    main()
