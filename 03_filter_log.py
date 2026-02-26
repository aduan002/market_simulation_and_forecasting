import re
import ast
import pandas as pd
import argparse 

def read_multiline_dict(lines, start_index):
    dict_lines = []
    brace_count = 0
    i = start_index

    while i < len(lines):
        line = lines[i]
        brace_count += line.count("{")
        brace_count -= line.count("}")

        dict_lines.append(line)

        if brace_count == 0:
            break

        i += 1

    dict_str = "".join(dict_lines)
    parsed_dict = ast.literal_eval(dict_str)

    return parsed_dict, i + 1


def parse_log_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    results = {}
    i = 0
    is_log_line = False

    while i < len(lines):
        line = lines[i].strip()

        if "INFO" in line:
            is_log_line = True

        if is_log_line and line == "##################################################":
            i += 1
            try: 
                model_name = lines[i].strip()
            except IndexError as e:
                continue

            is_log_line = False
            # Move to hyperparameter dict
            i += 1
            while not lines[i].strip().startswith("{"):
                i += 1

            hyperparams, i = read_multiline_dict(lines, i)

            folds = []
            mean_metrics = None

            while i < len(lines) and "##################################################" not in lines[i]:
                current_line = lines[i].strip()

                # Parse fold
                if current_line.startswith("FOLD"):
                    fold_number = int(current_line.split()[1])
                    i += 1

                    while not lines[i].strip().startswith("{"):
                        i += 1

                    fold_metrics, i = read_multiline_dict(lines, i)
                    fold_metrics["fold"] = fold_number
                    folds.append(fold_metrics)
                    continue

                # Parse MEAN
                if current_line == "MEAN":
                    i += 1
                    while not lines[i].strip().startswith("{"):
                        i += 1

                    mean_metrics, i = read_multiline_dict(lines, i)
                    continue

                i += 1

            if model_name not in results:
                results[model_name] = []

            results[model_name].append({
                "hyperparams": hyperparams,
                "folds": folds,
                "mean_metrics": mean_metrics
            })

        else:
            i += 1

    return results


def extract_best_per_algorithm(results_dict):
    best_rows = []

    for model_name, configs in results_dict.items():
        best_config = min(
            configs,
            key=lambda x: x["mean_metrics"]["mean_absolute_error"]
        )

        for fold_data in best_config["folds"]:
            best_rows.append({
                "algorithm": model_name,
                "fold": fold_data["fold"],
                "mean_absolute_error": fold_data["mean_absolute_error"],
                "mean_squared_error": fold_data["mean_squared_error"],
                "r2": fold_data["r2"],
                "max_error": fold_data["max_error"]
            })

    return pd.DataFrame(best_rows)


def main(file_path:str, output_path:str):
    parsed_results = parse_log_file(file_path)
    best_df = extract_best_per_algorithm(parsed_results)
    best_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
         prog = "filter_log.py",
         description = "Filters the log file for best folds"
    )
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    

    args = vars(parser.parse_args()) 

    file_path = args["input"]
    output_path = args["output"]

    main(file_path, output_path)

