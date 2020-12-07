import pandas as pd
import numpy as np
from furiosanet2 import FuriosaNet
from sklearn.model_selection import train_test_split
from utils.misc import scale_data

def main():
    data, labels, scalar_x, scalar_y, dataset = presentation_data()
    clf = FuriosaNet(5)
    clf.load_model("weights/furiosanet-norm-binary_crossentropy-24-50-100-5-weights.h5", [24,50,100])
    preds = clf.predict(data)
    actual = clf.get_class(labels, scalar_y)
    unique, counts = np.unique(actual, return_counts=True)
    print(dict(zip(unique, counts)))
    # print(preds)
    # print(actual)
    # for pred in preds:
    #     print(pred)
    # print(dataset[["title", "revenue"]])
    pass

def presentation_data():
    drop_features = ["tmdb_id", "year", "title", "vote_count", "vote_average", "popularity"]
    csv = "dbs/data_2010s.csv"
    dataset = pd.read_csv(csv)
    data = dataset.drop(columns=drop_features).astype('float')
    scalar_x = None
    scalar_y = None

    features = data.iloc[:, :-1].values
    output = data.iloc[:, -1].values
    indices = np.arange(len(output))
    x_train, x_test, y_train, y_test, _, test_indices = train_test_split(features, output, indices, test_size=20/100, random_state=18)

    x_train, x_test, y_train, y_test, scalar_x, scalar_y = scale_data(x_train, x_test, y_train, y_test)
    
    trials = np.random.choice(test_indices, size=len(y_test), replace=False)
    # print(trials)
    dataset = dataset.iloc[trials]
    x_test = features[trials]
    y_test = output[trials]
    return x_test, y_test, scalar_x, scalar_y, dataset

def generate_data(csv, drop_features=['title', 'tmdb_id', 'year'],
                  split=(80,20), output_index=-1, scale_input=True):
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
    `drop_features`:
        features to drop

    `split`:
        default (80, 20); tuple specifying the percentage of data for
        training and testing, respectively

    `output_index`:
        default -1; location in the dataset which is the value to predict

    `scale_input`:
        default True; boolean declaring whether to scale data

    Return
    ==========
    (x_train, x_test, y_train, y_test, scalar_x, scalar_y, dataset, test_indices)
    numpy arrays: x_train, x_test, y_train, y_test, test_indices;
    Scalar objects: scalar_x, scalar_y
    Pandas DataFrame: dataset
    '''


if __name__ == "__main__":
    main()