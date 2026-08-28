
import os
import sys
import argparse
import torch
from glob import glob
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
import time
from util import read_file_preds
import numpy as np

sys.path.append(str(Path(__file__).parent.joinpath("image-matching-models")))

from matching import get_matcher, available_models
from matching.utils import get_default_device

sys.path.append(str(Path(__file__).parent.joinpath("VPR-methods-evaluation")))  # Added in order to have access to the VPR-methods-evaluation\test_dataset modules

from test_dataset import TestDataset

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--method", type=str, help="method used before the image matching phase", choices=[
            "netvlad",
            "cosplace",
            "mixvpr",
            "megaloc"
        ])
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="set to true if you have no labels and just want to "
        "do standard image retrieval given two folders of queries and DB",
    )
    
    parser.add_argument(
        "--positive_dist_threshold",
        type=int,
        default=25,
        help="distance (in meters) for a prediction to be considered a positive",
    )
    parser.add_argument("--database-folder", nargs="+", type=str, help="Path/to/database")
    parser.add_argument("--queries-folder", type=str, help="Path/to/queries")
    parser.add_argument("--preds-dir", nargs="+", type=str, help="directory with predictions of a VPR model")  # Added nargs="+" so as to receive a list of input
    parser.add_argument("--out_dir", type=str, default=None, help="output directory of image matching results")
    # Choose matcher
    parser.add_argument(
        "--matcher",
        type=str,
        default="sift-lg",
        choices=available_models,
        help="choose your matcher",
    )
    parser.add_argument("--device", type=str, default=get_default_device(), choices=["cpu", "cuda"])
    parser.add_argument("--im_size", type=int, default=512, help="resize img to im_size x im_size")     # Image size for image matching test section is already fixed at 512
    parser.add_argument("--num_preds", type=int, default=20, help="number of predictions to match")  # Added nargs="+" so as to receive a list of input
    parser.add_argument("--start-query", type=int, default=-1, help="query to start from")
    parser.add_argument("--num-queries", type=int, default=-1, help="number of queries")
    parser.add_argument("--recall_values", nargs="+", type=int, default=100, help="list of recall values to match")  # Added nargs="+" so as to receive a list of input

    return parser.parse_args()

def main(args):
    device = args.device
    matcher_name = args.matcher
    img_size = args.im_size
    recall_nums = args.recall_values
    num_predictions = args.num_preds
    matcher = get_matcher(matcher_name, device=device)
    preds_folders = args.preds_dir
    start_query = args.start_query
    num_queries = args.num_queries

    # Added 2 for loops to compute recall. For each folder of predictions, compute the recalls: R@1, R@5, R@10
    for preds_folder in preds_folders:
        output_folder = Path(preds_folder + f"_{matcher_name}") if args.out_dir is None else Path(args.out_dir)
        
        output_folder.mkdir(exist_ok=True)
        
        txt_files = glob(os.path.join(preds_folder, "*.txt")) # Extracts the files ending in .txt and puts them in txt_files as a list in the same order it exists in preds_folder
        txt_files.sort(key=lambda x: int(Path(x).stem))


        start_query = start_query if start_query >= 0 else 0
        num_queries = num_queries if num_queries >= 0 else len(txt_files)

        for txt_file in tqdm(txt_files[start_query : start_query + num_queries]):
            q_num = Path(txt_file).stem

            out_file = output_folder.joinpath(f"{q_num}.torch")
            if out_file.exists():
                continue
            results = []
            q_path, pred_paths = read_file_preds(txt_file)
            
            img0 = matcher.load_image(q_path, resize=img_size)

            for pred_path in pred_paths[:num_predictions]:
                
                img1 = matcher.load_image(pred_path, resize=img_size)  # Load the image for the matcher
                result = matcher(deepcopy(img0), img1)  # Returns a dictionary of various objects. But we only need the inliers
                
                # The next line where in the original code from the professor I leave it as I don't know what we need to save results for yet
                result["all_desc0"] = result["all_desc1"] = None
                results.append(result)


            torch.save(results, out_file) 

if __name__ == "__main__":
    args = parse_arguments()
    main(args)