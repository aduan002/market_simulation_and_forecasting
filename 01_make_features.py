import argparse
import pandas as pd


from features.features import VolatilityFeatures


def main(file_path: str, output_path:str): 
    data = pd.read_csv(file_path)

    # For changing to minutes instead of second level 
    data['t'] = data['t'] // 60
    data = data.groupby('t')['price'].last().reset_index()

    volatility_features = VolatilityFeatures()

    features_data = volatility_features.make_features(data, horizon=60)
    features_data.to_csv(output_path, index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
         prog = "make_features.py",
         description = "Creates features."
    )
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    

    args = vars(parser.parse_args()) 

    file_path = args["input"] 
    output_path = args["output"]
   

    main(file_path, output_path)

