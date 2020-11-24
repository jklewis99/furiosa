'''
function to test pretrained models on the test set and show the graph
of actual values and predictions
'''
import pandas as pd
from furiosanet import test, get_layers_from_file

def main():
    '''
    test models saved in the csv
    '''
    models = pd.read_csv("model-evaluation.csv", index_col=0).index.tolist()
    for model in models:
        weights_file = "weights/automated/" + model
        layers = get_layers_from_file(weights_file)
        test(weights_file, layers, show_fig=False)

if __name__ == "__main__":
    main()
        
