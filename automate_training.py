'''
module to train models with random numbr of layers and random number of nodes
per layer
'''
import random
import numpy as np
from furiosanet import train

def main():
    '''
    randomly create a model and train it
    '''
    num_trails = 100
    hidden_layers = np.arange(2, 6, 1)
    range_params = np.arange(20, 5000, 10)

    for _ in range(num_trails):
        layers = np.random.choice(range_params, random.choice(hidden_layers)).tolist()
        train(layers, "mean_squared_error", scale_input=True)

if __name__ == "__main__":
    main()
