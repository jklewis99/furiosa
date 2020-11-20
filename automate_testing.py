import pandas as pd
import os
import numpy as np
import random
from furiosanet import test, get_layers_from_file

def main():
    
    models = pd.read_csv("model-evaluation.csv", index_col=0).index.tolist()
    # print(models)
    # for t in range(0, 200):
    #     layers = np.random.choice(range_params, random.choice(hidden_layers)).tolist()
    #     train(layers, "mean_squared_error", scale_input=False)z
    for model in models:
        weights_file = "weights/automated/" + model
        layers = get_layers_from_file(weights_file)
        test(weights_file, layers, "mean_squared_error")

if __name__ == "__main__":
    main()
        
