"""
1. Pass images through VPR method and retrieve top-1 candidates
2. Pass the candidates through image matching method and compute inliers
3. Pass the inliers through the logistic regression
4. If regression classifies inliers as hard, rerank else do not rerank

"""
import numpy as np
from tqdm import tqdm
import os, argparse
from glob import glob
from pathlib import Path
import torch
import pandas as pd
import sys

sys.path.append("/content/Visual-Place-Recognition-Project")
from util import get_list_distances_from_preds

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--preds_dir", nargs="+", type=str, help="directory with predictions of a VPR model")
    parser.add_argument("--inliers_dir", nargs="+", type=str, help="directory with image matching results")
    parser.add_argument("--num-preds", type=int, default=100, help="number of predictions to re-rank")
    parser.add_argument(
        "--positive-dist-threshold",
        type=int,
        default=25,
        help="distance (in meters) for a prediction to be considered a positive",
    )

    return parser.parse_args()

def main(args):
    preds_folders = args.preds_dir
    inliers_folders = args.inliers_dir
    max_num_preds = args.num_preds
    threshold = args.positive_dist_threshold   # Threshold of 25 meters
    
    for preds_folder, inliers_folder in list(zip(preds_folders, inliers_folders)):
        log_reg_dataset = []
        inliers_folder_name = inliers_folder        # Save the name for later
        inliers_folder = Path(inliers_folder)

        txt_files = glob(os.path.join(preds_folder, "*.txt"))   # Each file in the preds_folder contains a query and the corresponding predictions made by the VPR method
        txt_files.sort(key=lambda x: int(Path(x).stem))     # Because glob may ruin the existing order, we sort the list here again

        folder = Path(preds_folder)
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
        label = -1     # y label for the classifier

        for txt_file_query in tqdm(txt_files):  # For each query
            # Extraction of up to num_preds vector of predicted images in order of increasing distances between query and predicted locations
            # This works because naturally the VPR method orders the predictions in that order
            geo_dists = torch.tensor(get_list_distances_from_preds(txt_file_query))[:max_num_preds]
            torch_file_query = inliers_folder.joinpath(Path(txt_file_query).name.replace('txt', 'torch'))

            if not torch_file_query.exists():
                continue

            query_results = torch.load(torch_file_query, weights_only=False)    # Load the image matching results for this query. These were saved when we called the image matcher
            
            num_preds = min(len(query_results), max_num_preds, len(geo_dists))      # At the end of the day, the num_preds to be delt with has to be consistent everywhere so we pick the smallest and work with it
            query_db_inliers = torch.zeros(num_preds, dtype=torch.int32)  

            for i in range(num_preds):      # Each prediction has data attached to it in the form of a dictionary. We are only interested in the inliers
                result = query_results[i]
                if isinstance(result, dict):
                    query_db_inliers[i] = result['num_inliers']       # Save the number of inliers for this predictions
                else:
                    query_db_inliers[i] = int(result)       # Sometimes the matcher can't match and adds a 0.0 in the dictionary for the predictions and it caused errors

            query_db_inliers, indices = torch.sort(query_db_inliers, descending=True)   # Reorder the inliers and save both the new order and the old indices reordered as well
            
            geo_dists = geo_dists[indices]      # Reorder the distances following this new order in decreasing number of inliers using the indices of the new order

            # We take the inlier computed for the top-1 predictor, the label and the corresponding query
            # Nothing about the predictio folder is saved as all the info needed is going to be in the txt_file_query
            if geo_dists[0] <= threshold:      # If any of the distances is below the threshold of 25, then it is counted as a success
                label = 1       # Mark the query as "easy"
            else:     # Meaning no good prediction was made
                label = 0       # Mark the query as "hard"

        log_reg_dataset.append(
            [query_db_inliers[0].item(), label, txt_file_query]
        )
        # Create DataFrame
        dataframe = pd.DataFrame(
            log_reg_dataset, columns=["inliers", "labels", "query_file"]
            ) # Save this data to a file for later use

        # Define output path in the destination folder directory
        matcher_name = inliers_folder_name.split("_")[-1]
        
        parent_folder = Path(preds_folder).parent
        csv_path = parent_folder / f"{matcher_name}.csv"

        # Save to Excel
        dataframe.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)