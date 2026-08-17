import numpy as np
from tqdm import tqdm
import os, argparse
from glob import glob
from pathlib import Path
import torch
import pandas as pd

from util import get_list_distances_from_preds

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--preds_dir", type=str, help="directory with predictions of a VPR model")
    parser.add_argument("--inliers_dir", nargs="+", type=str, help="directory with image matching results")
    parser.add_argument("--num-preds", type=int, default=100, help="number of predictions to re-rank")
    parser.add_argument(
        "--positive-dist-threshold",
        type=int,
        default=25,
        help="distance (in meters) for a prediction to be considered a positive",
    )
    parser.add_argument(
        "--recall-values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 100],
        help="values for recall (e.g. recall@1, recall@5)",
    )

    return parser.parse_args()

def main(args):
    preds_folder = args.preds_dir
    inliers_folders = args.inliers_dir
    num_preds = args.num_preds
    threshold = args.positive_dist_threshold
    recall_values = args.recall_values

    for inliers_folder in inliers_folders:
        inliers_folder_name = inliers_folder # Save the name of the folder before converting it to path
        inliers_folder = Path(inliers_folder)
        txt_files = glob(os.path.join(preds_folder, "*.txt"))   # Each file in the preds_folder contains a query and the corresponding predictions made by the VPR method
        txt_files.sort(key=lambda x: int(Path(x).stem))

        total_queries = len(txt_files)

        folder = Path(args.preds_dir)
        print(f"folder name: {folder}")
        # Check if path exists AND is a directory first
        if folder.exists() and folder.is_dir():
            # any() checks if there is at least 1 item inside
            if any(folder.iterdir()):
                print("Folder has something in it!")
            else:
                print("Folder is completely empty.")
        else:
            print("Folder does not exist or is not a directory.")
        recalls = np.zeros(len(recall_values))

        for txt_file_query in tqdm(txt_files):  # For each query
            geo_dists = torch.tensor(get_list_distances_from_preds(txt_file_query))[:num_preds]     # Extraction of a vector of up to num_preds distances between query and predicted locations
            torch_file_query = inliers_folder.joinpath(Path(txt_file_query).name.replace('txt', 'torch'))
            query_results = torch.load(torch_file_query, weights_only=False)    # Load the image matching results for this query. These were saved when we called the image matcher
            query_db_inliers = torch.zeros(num_preds, dtype=torch.int32)
            num_preds = min(len(query_results), num_preds, len(geo_dists))

            for i in range(num_preds):
              result = query_results[i]
              if isinstance(result, dict):
                query_db_inliers[i] = result['num_inliers']       # Save the first num_preds inlier results for the predictions of this query
              else:
                query_db_inliers[i] = int(result)
            query_db_inliers, indices = torch.sort(query_db_inliers, descending=True)   # Reorder the inliers and save both the new order and the old indices reordered as well
            
            # The following is done to have all the vectors with the same length
            query_db_inliers = query_db_inliers[:num_preds]
            indices = indices[:num_preds]
            
            geo_dists = geo_dists[indices]      # Reorder the distances following this new order in decreasing number of inliers
            
            for i, n in enumerate(recall_values):
                if torch.any(geo_dists[:n] <= threshold):
                    recalls[i:] += 1
                    break

        recalls = recalls / total_queries * 100
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(recall_values, recalls)])

        print(recalls_str)

        matcher_name = inliers_folder_name.split("_")[-1]

        # Prepare results dictionary
        results_dict = {"Matcher": [matcher_name]} 
        for val, rec in zip(recall_values, recalls):
            results_dict[f"R@{val}"] = [round(float(rec), 2)]

        # Create DataFrame
        df = pd.DataFrame(results_dict)

        # Folder for rerankings
        destination_folder = Path("/content/drive/MyDrive/VPR_preds/logs/Rerankings")
        
        # Create target directory if it doesn't exist yet
        destination_folder.mkdir(parents=True, exist_ok=True)

        db_name = ""
        if "sf_xs" in preds_folder:
            db_name = "sf_xs"
        if "tokyo_xs" in preds_folder:
            db_name = "tokyo_xs"
        if "svox_queries_night" in preds_folder:
            db_name = "svox_queries_night"
        if "svox_queries_sun" in preds_folder:
            db_name = "svox_queries_sun"

        # Define output path in the destination folder directory
        excel_path = Path(destination_folder) / f"{db_name}_{matcher_name}_recalls.xlsx"

        # Save to Excel
        df.to_excel(excel_path, index=False)
        print(f"Saved results to {excel_path}")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)