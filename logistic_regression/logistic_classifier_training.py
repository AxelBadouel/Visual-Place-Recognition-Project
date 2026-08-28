import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import os, argparse
from logistic_classifier import AdaptiveClassifier

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_inliers_df_cvs", nargs="+", type=str, help="file with the dataframe of the training inliers")
    parser.add_argument("--val_inliers_df_cvs", nargs="+", type=str, help="file with the dataframe of the validation inliers")
    parser.add_argument("--out_dir", type=str, help="Directory were the checkpoint will be saved")

    return parser.parse_args()

def main(args):
    train_data_csv = pd.read_csv(
        args.train_inliers_df_cvs
        if isinstance(args.train_inliers_df_cvs, list)
        else args.train_inliers_df_cvs
    )
    val_data_csv = pd.read_csv(
        args.val_inliers_df_cvs[0]
        if isinstance(args.val_inliers_df_cvs, list)
        else args.val_inliers_df_cvs
    )

    train_dataframe = pd.read_csv(train_data_csv)
    train_inliers = train_dataframe["inliers"].values
    train_labels = train_dataframe["label"].values

    val_dataframe = pd.read_csv(val_data_csv)
    val_inliers = val_dataframe["inliers"].values
    val_labels = val_dataframe["label"].values

    x_train = torch.tensor(train_inliers, dtype=torch.float32).unsqueeze(1)
    x_train_scaled = (x_train - x_train.mean()) / (x_train.std() + 1e-8)
    y_train = torch(train_labels, dtype=torch.float32).unsqueeze(1)

    x_val = torch.tensor(val_inliers, dtype=torch.float32).unsqueeze(1)
    x_val_scaled = (x_val - x_val.mean()) / (x_val.std() + 1e-8)
    y_val = torch(val_labels, dtype=torch.float32).unsqueeze(1)

    model = AdaptiveClassifier()
    criterion = nn.BCELoss()    # Binary Cross-Entropy
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training Loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(x_train_scaled)
        loss = criterion(outputs, y_train)

        # Backward pass & update
        loss.backward()
        optimizer.step()

    # Validation & Decision Cutoff Selection
    model.eval()
    with torch.no_grad():
        val_probs = model(x_val_scaled)

    # Tune the probablity decision boundary on validation set. We will pick what threshold is better based on these outputs.
    # (e.g., if P(correct) < probability_threshold, mark query as HARD and perform reranking)
    target_rerank_ratio = 0.3
    best_threshold = 0.5
    closest_diff = float('inf')

    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # Predict 1(Easy) vs 0 (Hard, re-rank)
        val_preds = (val_probs >= threshold).float()
        acc = (val_preds == y_val).float().mean().item()
        rerank_ratio = (val_preds == 0).float().mean().item() # Compute the fraction that has to be reranked
        print(f"Threshold {threshold:.1f} | Acc: {acc:.4f} | % Re-ranked (Hard): {rerank_ratio * 100:.1f}%")

    # Track which threshold gets closest to your target budget
    diff = abs(rerank_ratio - target_rerank_ratio)
    if diff < closest_diff:
        closest_diff = diff
        best_threshold = threshold

    print(f"\nSelected Optimal Threshold: {best_threshold}")

    checkpoint = {
        "state_dict": model.state_dict(),
        "train_mean": x_train.mean().item(),
        "train_std": x_train.std().item(),
        "threshold": best_threshold
    }

    os.makedirs(os.path.dirname(args.out_dir), exist_ok=True)
    torch.save(checkpoint, args.out_dir)
    print(f"Model saved to {args.out_dir}")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)