'''
miscellaneous functions used across the repository
'''
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FIGURES_PATH = os.path.join(os.path.dirname(__file__), "../figures")

def create_directory(path):
    '''
    create a directory recursively if path doesn't exist and return the path
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    return path

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
    dataset = pd.read_csv(csv)
    data = dataset.drop(columns=drop_features).astype('float')
    scalar_x = None
    scalar_y = None

    features = data.iloc[:, :output_index].values
    output = data.iloc[:, output_index].values
    indices = np.arange(len(output))
    x_train, x_test, y_train, y_test, _, test_indices = train_test_split(features, output, indices, test_size=split[1]/100, random_state=18)

    if scale_input:
        x_train, x_test, y_train, y_test, scalar_x, scalar_y = scale_data(x_train, x_test, y_train, y_test)
    return x_train, x_test, y_train, y_test, scalar_x, scalar_y, dataset, test_indices

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
    `(x_train, x_test, y_train, y_test, scalar_x, scalar_y)`:
    tuple of the scaled x_train, transformed x_test, scaled y_train,
    transformed y_test, scalar used for features, scalar used for outputs
    '''
    scalar_x = StandardScaler() # need to scale our data (I think)
    scalar_y = StandardScaler() # need to scale our data (I think)

    x_train = scalar_x.fit_transform(x_train)
    y_train = scalar_y.fit_transform(y_train.reshape(-1, 1))
    x_test = scalar_x.transform(x_test)
    y_test = scalar_y.transform(y_test.reshape(-1, 1))

    return x_train, x_test, y_train, y_test, scalar_x, scalar_y

def stringify_model(layers):
    '''
    convert a list of the layers into a string for saving

    Parameters
    ==========
    `layers`:
        list of specified number of neurons per layer
    '''
    return "-".join(map(str, layers))

def plot_history(results, layers, loss, output_size=1, norm="", save_fig=FIGURES_PATH, show_fig=False):
    '''
    plot the model's history during training

    Parameters
    ==========
    `results`:
        tensorflow keras history of model

    `layers`:
        list of model layers, excluding last layer, that defines the layer neurons

    `loss`:
        loss function used to train model

    Keyword Args
    ==========
    `norm`:
        default ""; string to define a description of the model's training data

    `save_fig`:
        default: "../figures"; string or False, path of location in which to save the figure

    `show_fig`:
        default: False; boolean determining whether to display the figure
    '''
    plt.figure(figsize=(10, 8))
    plt.plot(np.array(results.history[loss])) #[15:])
    plt.plot(np.array(results.history["val_"+ loss])) #[15:])
    plt.title('Loss on training and testing data', fontsize=24)
    plt.ylabel(loss, fontsize=18)
    plt.xlabel('Epoch', fontsize=18)
    plt.legend(['train', 'test'], loc='upper right')
    if save_fig:
        plt.savefig(f"{save_fig}/history{norm}-{loss}-{stringify_model(layers)}-{output_size}.png")
    if show_fig:
        plt.show()

def plot_predictions(predictions, actual, r_squared, model=None, layers=None, output_size=1,
                     norm="", best="", save_fig=FIGURES_PATH, show_fig=False):
    '''
    plot predicitions to a graph and save the image

    Parameters
    ==========
    `predictions`:
        output predictions of the model

    `actual`:
        target outputs for the model, real outputs for data

    `r_squared`:
        r_squared value for predictions and actual outputs

    Keyword Args
    ==========
    `model`:
        default: None; type of model that generated the data, used to define file name and
        title of plot, unless `layers` keyword arg is set

    `layers`:
        default: None; list of model layers, excluding last layer, that defines the layer neurons

    `norm`:
        default ""; string to define a description of the model's training data

    `best`:
        default: ""; string to define description of model, whether this is a graph of the "best"
        model for this architecture or just a plot of the last values of weights in training

    `save_fig`:
        default: "../figures"; string or False, path of location in which to save the figure

    `show_fig`:
        default: False; boolean determining whether to display the figure
    '''
    if layers:
        file_name = f"{best}predictions{norm}-{stringify_model(layers)}-1"
        model = "MLP " + stringify_model(layers) + f"-{output_size}"
    else:
        file_name = f"{model}-predictions"
    plt.figure(figsize=(8, 8))
    plt.scatter(predictions, actual)
    plt.title(f'{model}:\nActual vs Predicted', fontsize=18, wrap=True)
    plt.ylabel('Actual Value', fontsize=16)
    plt.xlabel('Predicted Value', fontsize=16)
    plt.annotate(f"r^2 value = {r_squared:.3f}", (1+np.min(predictions), 0.9*np.max(actual)))

    if save_fig:
        save_plot(plt, file_name, figure_type="matplotlib")
    if show_fig:
        plt.show()

def save_plot(figure, name, path=FIGURES_PATH, figure_type="plotly",
              interactive=True, file_type="png"):
    '''
    model to save figures from plotly in dynamic or static mode

    Parameters
    ==========
    `figure`:
        plotly figure or matplotlib figure

    `name`:
        name by which to save file

    Keyword Args
    ===========
    `path`:
        default: "../figures"; string path of location in which to save the figure

    `figure_type`:
        type of figure (matplotlib or plotly)

    `interactive`:
        default: True; defines whether to save file as HTML (only if figure type is 'plotly').
        Otherwise, this should be set to false.

    `file_type`:
        default: png; if `interactive` is False or the `figure_type` is "matplotlib", specifies
        the file type of the static image
    '''
    if figure_type=="plotly":
        if interactive:
            figure.write_html(f"{path}/{name}.html")
        else:
            figure.write_image(f"{path}/{name}.{file_type}")
    else:
        figure.savefig(f"{path}/{name}.{file_type}")

def create_interactive_plot(graph_data, model=None, layers=None, save_fig=FIGURES_PATH):
    '''
    create interactive graph to be saved to html file. *Requires either `model` or `layers` be set*

    Parameters
    ==========
    `graph_data`:
        pandas DataFrame object containing columns: title, predicted, and revenue

    Keyword Args:
    ==========
    `model`:
        default: None; type of model that generated the data, used to define file name and
        title of plot, unless `layers` keyword arg is set

    `layers`:
        default: None; list of model layers, excluding last layer, that defines the layer neurons

    `save_fig`:
        default: "../figures"; string or False, path of location in which to save the figure
    '''
    if layers:
        model = stringify_model(layers)
    fig = px.scatter(graph_data,
                    x="predicted",
                    y="revenue",
                    hover_name="title",
                    hover_data=["predicted__", "revenue____", "difference_", "percent_off"],
                    title=f"Actual Revenue and Predicted Revenue for {model}",
                    width=800,
                    height=800)
    fig.update_traces(marker=dict(size=12,
                                color='skyblue',
                                line=dict(width=1,
                                            color='black')))
    fig.update_layout(hovermode="closest",
                    hoverlabel=dict(
                            bgcolor="skyblue",
                            font_size=16,
                            font_family="Consolas"))

    save_plot(fig, f"{model}-predictions", path=os.path.join(FIGURES_PATH, "interactive"), interactive=True)

def inverse_transform(predictions, actual, scalar):
    '''
    transform the values to original scale

    Return
    ==========
    tuple of rescaled `predictions`, rescaled `actual`
    '''
    if len(predictions.shape) > 1:
        rescaled_predictions = scalar.inverse_transform(np.array([val[0] for val in predictions]))
    else:
        rescaled_predictions = scalar.inverse_transform(predictions)
    rescaled_actual = scalar.inverse_transform(actual).flatten()
    return rescaled_predictions, rescaled_actual

def create_df(predictions, dataset, test_indices):
    '''
    using the test_indices, set a predicted column to the predicted value
    by the model

    Parameters
    ==========
    `predictions`:
        output predictions of the model

    `dataset`:
        Pandas DataFrame from which training and test data was gathered

    `test_indices`:
        indices from the dataframe that specify the test data samples

    Return
    ==========
    Pandas DataFrame with updated column "predicted"
    '''
    test_data = dataset.copy().iloc[test_indices, :]
    test_data["predicted"] = predictions.T.astype("int64")
    test_data["difference_"] = test_data.apply(get_difference, axis=1)
    test_data["percent_off"] = test_data.apply(get_percent_off, axis=1)
    test_data["predicted__"] = test_data["predicted"].apply(convert_millions)
    test_data["revenue____"] = test_data["revenue"].apply(convert_millions)
    test_data["difference_"] = test_data["difference_"].apply(convert_abs_millions)
    return test_data

def get_difference(series_object):
    '''
    '''
    pred = series_object["predicted"]
    revenue = series_object["revenue"]
    return revenue - pred

def convert_abs_millions(val):
    mil = abs(val /(10**6))
    if mil >=  1000:
        val_str = f"${mil/1000:.3f} Billion"
    elif mil > 1:
        val_str = f"${mil:.0f} Million"
    else:
        val_str = f"${mil*1000:.0f} Thousand"
    return f"{val_str:>20}"

def convert_millions(val):
    mil = val /(10**6)
    if mil >=  1000:
        val_str = f"${mil/1000:.3f} Billion"
    elif mil >= 1:
        val_str = f"${mil:.0f} Million"
    elif mil >= 0:
        val_str = f"${mil*1000:.0f} Thousand"
    elif mil > -1:
        val_str = f"-${mil*-1000:.0f} Thousand"
    elif mil > -1000:
        val_str = f"-${-1*mil:.0f} Million"
    else:
        val_str = f"${mil/-1000:.3f} Billion"
    return f"{val_str:>20}"

def get_percent_off(series_object):
    '''
    '''
    diff = series_object["difference_"]
    revenue = series_object["revenue"]
    return f"{100*(diff / revenue):19.2f}%"
