import os
import numpy as np
import random
from furiosanet import train

def main():
    num_trails = 800
    hidden_layers = np.arange(2, 6, 1)
    range_params = np.arange(20, 5000, 10)

    # for t in range(0, 200):
    #     layers = np.random.choice(range_params, random.choice(hidden_layers)).tolist()
    #     train(layers, "mean_squared_error", scale_input=False)
    for t in range(0, num_trails):
        layers = np.random.choice(range_params, random.choice(hidden_layers)).tolist()
        train(layers, "mean_squared_error", scale_input=True)

if __name__ == "__main__":
    main()
        
