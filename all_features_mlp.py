import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.layers.core import Activation
from tensorflow.random import set_seed

set_seed(18)
first_hidden = 50
second_hidden = 25
def neural_network(dimensions):
    model = Sequential()
    model.add(Input(shape=(dimensions,)))
    model.add(Dense(first_hidden, kernel_initializer="normal", activation="relu"))
    model.add(Dense(second_hidden, kernel_initializer="normal", activation="relu"))
    model.add(Dense(1, activation="linear")) # one node for regression
    return model

def generate_data(csv, split=(80,20), output_index=-1):
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

    Return
    ==========
    tuple of x_train, x_test, y_train, y_test
    '''
    dataset = pd.read_csv(csv)
    data = dataset.drop(columns=['title', 'tmdb_id', 'year']).astype('float')

    features = data.iloc[:, :output_index].values
    output = data.iloc[:, output_index].values

    # # need to scale our data (I think) scalar_x = StandardScaler() 
    # # need to scale our output (I think) scalar_y = StandardScaler() 
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=split[1]/100, random_state=18)
    return x_train, x_test, y_train, y_test, dataset

def plot_history(results):
    plt.figure(figsize=(10, 8))
    plt.plot(np.array(results.history['mean_absolute_error'])[15:])
    plt.plot(np.array(results.history['val_mean_absolute_error'])[15:])
    plt.title('MAE on training and testing data', fontsize=24)
    plt.ylabel('Mean Absolute Error', fontsize=18)
    plt.xlabel('Epoch', fontsize=18)
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(f"figures/history-{first_hidden}-{second_hidden}-1.png")
    plt.show()

def plot_predictions(predictions, actual):
    plt.figure(figsize=(10, 8))
    plt.scatter(predictions, actual)
    plt.title('Actual vs Predicted', fontsize=24)
    plt.ylabel('Actual Value', fontsize=18)
    plt.xlabel('Predicted Value', fontsize=18)
    r_squared = r2_score(actual, predictions)
    plt.annotate(f"r^2 value = {r_squared:.3f}", (2, 0.9*np.max(predictions)))
    plt.savefig(f"figures/predictions-{first_hidden}-{second_hidden}-1.png")
    plt.show()

def main():
    x_train, x_test, y_train, y_test, dataset = generate_data("dbs/data_2010s.csv")
    model = neural_network(x_train.shape[1])
    model.compile(loss="mean_absolute_error", optimizer="adam", metrics=['mean_absolute_error'])
    model.summary()
    callback = ModelCheckpoint(
        filepath=f"weights/nn-{first_hidden}-{second_hidden}-1-weights.h5",
        save_freq='epoch',
        verbose=1,
        save_weights_only=True
        )
    results = model.fit(
        x_train, y_train,
        batch_size=50,
        epochs=500,
        validation_data=(x_test, y_test),
        callbacks=[callback]
        )
    plot_history(results)
    predictions = np.array([val[0] for val in model.predict(x_test)])
    # print(y_test)
    plot_predictions(predictions, y_test)

if __name__ == "__main__":
    main()