import argparse
import yaml
import json
import pandas as pd
import logging
import os
import pickle

from datasets.cross_validation import get_dataset_split

from datasets.preprocess import Preprocess

from models.model import create_model, generate_grid

from metrics.regression_metrics import RegressionMetric

from utils.runs import create_run, dict_to_str
from utils.logger import create_global_logger, create_logger
import pprint

# TODO: Make logging prettier...
logger = create_global_logger(name = __name__, level = logging.DEBUG)

def main(config:dict, hyperparameters:dict):
    config["RESULTS"]["SAVE"] = False
    
    data_cfg = config["DATA"] 
    data_cfg["SEED"] = config["SEED"] 

    prep_config = config["PREPROCESSING"]
    prep_config["SEED"] = config["SEED"]

    data = pd.read_csv(data_cfg["DATA_PATH"])
    data = data[data_cfg["FEATURES"] + [data_cfg["TARGET"]]]

    data_features = data[data_cfg["FEATURES"]]
    data_target = data[[data_cfg["TARGET"]]] 


    train_indices, test_indices = get_dataset_split(data_cfg, data_features, data_target)

    feat_prep = Preprocess(prep_config["FEATURES"])
    target_prep = Preprocess(prep_config["TARGET"])  


    result_cfg = config["RESULTS"]
    folder_path = None
    run_logger = None
    folder_path = create_run(result_cfg)
    run_logger = create_logger(os.path.join(folder_path, "log.txt"), name = __name__)

    metric = RegressionMetric() 

    run_logger.info("\n"+pprint.pformat(config))

    for algorithm_name in hyperparameters:
        # NOTE: Not using sklearn gridsearch to allow for custom pytorch gridsearch
        for hp_cross_product in generate_grid(hyperparameters[algorithm_name]):
            model = create_model(algorithm_name, hp_cross_product, seed=config["SEED"])

            logger_msg = "\n"+"#"*50+"\n"+algorithm_name+"\n"+pprint.pformat(hp_cross_product)

            metric.reset() # Before starting on a new cross validation, reset the metric values.
            for fold_idx, (train_idx, test_idx) in enumerate(zip(train_indices, test_indices)):
                model.reset()
                X_train = data_features.iloc[train_idx]
                X_test = data_features.iloc[test_idx]

                y_train = data_target.iloc[train_idx]
                y_test = data_target.iloc[test_idx]
  
                X_train, X_test = feat_prep.transform_dataset(X_train, X_test, y_train)
                y_train = target_prep.fit_transform(y_train)           

                model.train(X_train, y_train)
                y_hat = target_prep.inverse_transform(model(X_test))
                
                metric.update(y_test, y_hat)

                if result_cfg["VERBOSE"] >= 1:
                    results = metric(y_test, y_hat)
                    logger_msg += "\n\n"+"-"*50+"\nFOLD {0}\n".format(fold_idx+1)+pprint.pformat(results)+"\n"+"-"*50

                if result_cfg["SAVE"]:
                    fold_folder = "fold_" + str(fold_idx + 1)

                    file_name = algorithm_name + "_" + dict_to_str(hp_cross_product) + ".pkl"
                    with open(os.path.join(folder_path, fold_folder, file_name), "wb") as file:
                        pickle.dump(model, file)
    
                    feature_prep_path = os.path.join(folder_path, fold_folder, "feat_preprocess" + ".pkl")
                    target_prep_path = os.path.join(folder_path, fold_folder, "target_prep" + ".pkl")
                    # Only save the preprocessing once for each fold, since subsequent repetitions contain the same data.
                    if not os.path.exists(feature_prep_path):
                        with open(feature_prep_path, "wb") as file:
                            pickle.dump(feat_prep, file)
                    if not os.path.exists(target_prep_path):
                        with open(target_prep_path, "wb") as file:
                            pickle.dump(target_prep, file)
                    
            logger_msg += "\n\nMEAN\n"+pprint.pformat(metric.mean())+"\n\nSTDEV\n"+pprint.pformat(metric.stdev())+"\n"
            logger_msg += "#"*50
            run_logger.info(logger_msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
         prog = "train.py",
         description = "Creates regression models on the given data."
    )
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-p", "--hyperparameters", required=True)

    args = vars(parser.parse_args()) 

    config_path = args["config"]  
    hyperparameter_path = args["hyperparameters"]
    with open(config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)

    with open(hyperparameter_path, "r") as file:
        hyperparameters = json.load(file)

    main(config, hyperparameters)