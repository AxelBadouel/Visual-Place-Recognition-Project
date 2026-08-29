import torch
import pandas as pd
import numpy as np
from logistic_classifier import AdaptiveClassifier
import os, argparse
from pathlib import Path
import sys

sys.path.append("/content/Visual-Place-Recognition-Project")
from util import get_list_distances_from_preds

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_inliers_df_cvs", nargs="+", type=str, help="file with the dataframe of the test inliers")
    parser.add_argument("--inliers_dir", type=str, help="Directory containing the saved .torch inlier files")
    parser.add_argument("--num-preds", type=int, default=100, help="Number of predictions to consider for the reranking")
    parser.add_argument("--classifier_weights", type=str, help="Weights to be of the trained classifier")
    parser.add_argument("--preds_dir", nargs="+", type=str, help="Directories where the predictions of the different methods have been saved")
    parser.add_argument("--out_dir", type=str, help="Directory were the checkpoint will be saved")
    parser.add_argument("--device", type=str, help="Device to use should be 'cpu' or 'cuda'")

    return parser.parse_args()

def rerank(txt_file_query, inliers_folder, num_preds=100):
    threshold = 25
    recall = 0
    # Extraction of up to num_preds vector of predicted images in order of increasing distances between query and predicted locations
    # This works because naturally the VPR method orders the predictions in that order
    geo_dists = torch.tensor(get_list_distances_from_preds(txt_file_query))[:num_preds]

    txt_path = Path(txt_file_query)
    torch_file_query = Path(inliers_folder) / txt_path.with_suffix(".torch").name

    if not torch_file_query.exists():
        return 0
    
    query_results = torch.load(torch_file_query, weights_only=False)    # Load the image matching results for this query. These were saved when we called the image matcher
        
    num_preds = min(len(query_results), num_preds, len(geo_dists))      # At the end of the day, the num_preds to be delt with has to be consistent everywhere so we pick the smallest and work with it
    query_db_inliers = torch.zeros(num_preds, dtype=torch.int32)  

    for i in range(num_preds):      # Each prediction has data attached to it in the form of a dictionary. We are only interested in the inliers
        result = query_results[i]
        if isinstance(result, dict):
            query_db_inliers[i] = result['num_inliers']       # Save the number of inliers for this predictions
        else:
            query_db_inliers[i] = int(result)       # Sometimes the matcher can't match and adds a 0.0 in the dictionary for the predictions and it caused errors

    query_db_inliers, indices = torch.sort(query_db_inliers, descending=True)   # Reorder the inliers and save both the new order and the old indices reordered as well

    geo_dists = geo_dists[indices]      # Reorder the distances following this new order in decreasing number of inliers using the indices of the new order
    
    if torch.any(geo_dists[0] <= threshold):   # If any of the distances is below the threshold of 25, then it is counted as a success
        recall = 1
    
    return recall

def main(args):
    device = args.device
    checkpoint = torch.load(args.classifier_weights)

    model = AdaptiveClassifier()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    train_mean = checkpoint["train_mean"]
    train_std = checkpoint["train_std"]
    probability_threshold = checkpoint["threshold"]

    test_csv_path = pd.read_csv(
        args.test_inliers_df_cvs[0]
        if isinstance(args.test_inliers_df_cvs, list)
        else args.test_inliers_df_cvs
    )
    test_dataframe = pd.read_csv(test_csv_path)

    test_inliers = test_dataframe["inliers"].values
    query_files = test_dataframe["query_file"].values

    x_test = torch.tensor(test_inliers, dtype=torch.float32).unsqueeze(1)
    x_test_scaled = (x_test - train_mean) / train_std

    model.to(device)
    x_test_scaled.to(device)
    with torch.no_grad():
        probs = model(x_test_scaled)

    # True = Low probability of success -> TRIGGER RE-RANKING
    is_hard_query = (probs < probability_threshold).squeeze().tolist()

    total_queries = len(query_files)
    successful_retrievals = 0
    total_reranking = 0

    for i in range(total_queries):
        txt_file = query_files[i]

        if is_hard_query[i]:
            total_reranked += 1
            #Rerank query using matchted inliers
            recall = rerank(
                txt_file, args.inliers_dir, num_preds=args.num_preds
            )
        else:
            # Keep pure VPR Top-1 result
            raw_dist = get_list_distances_from_preds(txt_file)
            recall = 1 if raw_dist[0] <= 25 else 0

        successful_retrievals += recall

        print(f"Total Test Queries: {total_queries}")
        print(f"Queries Re-ranked: {total_reranked} ({(total_reranked/total_queries)*100:.2f}%)")
        print(f"Final Recall@1 Accuracy: {successful_retrievals / total_queries:.4f}")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)