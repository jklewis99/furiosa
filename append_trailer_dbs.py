'''
Append/Concatenate all of the separate trailers csvs into one csv
'''

import pandas as pd
import argparse
import os

def main():
    '''
    locates all csv files in the specified input folder and concatenates
    them together
    '''
    parser = argparse.ArgumentParser(description='Concatenate trailer databases')
    parser.add_argument('input', type=str, help="Folder containing \
        trailers_2010s_{index}.csv files")
    parser.add_argument('output', type=str, help="Location to save \
        trailers_2010s.csv")
    args = parser.parse_args()

    trailers_csvs = [f for f in os.listdir(args.input) if f[:8] == "trailers"]
    trailers_2010s = pd.DataFrame()
    for csv in trailers_csvs:
        trailers_2010s = trailers_2010s.append(pd.read_csv(os.path.join(args.input, csv)))
    trailers_2010s.to_csv(os.path.join(args.output, "trailers_2010s.csv"), index=False)

if __name__ == "__main__":
    main()