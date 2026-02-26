import os

def create_run(config:dict, num_folds:int = 10):
    folder_name = "run_0"
    if not os.path.exists(config["OUTPUT_DIR"]):
        os.makedirs(config["OUTPUT_DIR"])
    else:
        i = 0
        folder_exists = True
        while folder_exists:
            folder_name = "run_{0}".format(i)
            if not os.path.exists(os.path.join(config["OUTPUT_DIR"], folder_name)):
                folder_exists = False
            else:
                i += 1
    os.makedirs(os.path.join(config["OUTPUT_DIR"], folder_name))
    for i in range(1, num_folds + 1):
        os.mkdir(os.path.join(config["OUTPUT_DIR"], folder_name, "fold_"+str(i)))
    return os.path.join(os.path.join(config["OUTPUT_DIR"], folder_name))

def dict_to_str(hyperpameters:dict):
    name = []
    for key, value in hyperpameters.items():
        name.append(key + "-" + str(value))
    return "_".join(name)
