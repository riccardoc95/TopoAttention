import os
import numpy as np
import pandas as pd
from skimage import io
from sklearn.metrics import jaccard_score, accuracy_score

from astropy.io import fits
from PIL import Image
import pandas as pd


def load_image_as_array(filename):
    image = Image.open(filename)
    return np.array(image) / 255.

def dice_coefficient(gt, pred):
    """
    Compute the Dice Coefficient between two binary masks.
    """
    intersection = np.logical_and(gt, pred).sum()
    total = gt.sum() + pred.sum()
    dice = 2 * intersection / total if total > 0 else 1.0
    return dice

def calculate_segmentation_metrics(gt_path, pred_path):
    """
    Calculate Dice, IoU, and Pixel Accuracy for binary segmentation masks.

    Parameters:
        gt_path (str): Folder path containing ground truth masks.
        pred_path (str): Folder path containing predicted masks.

    Returns:
        metrics (list of dict): List containing metrics for each mask pair.
    """
    metrics = []

    # List all files in the directories
    gt_masks = sorted(os.listdir(gt_path))
    pred_masks = sorted(os.listdir(pred_path))

    for gt_file, pred_file in zip(gt_masks, pred_masks):
        # Load masks
        print(os.path.join(gt_path, gt_file), os.path.join(pred_path, pred_file))
        gt_mask = load_image_as_array(os.path.join(gt_path, gt_file))
        pred_mask = load_image_as_array(os.path.join(pred_path, pred_file))

        # Ensure both masks have the same dimensions
        if gt_mask.shape != pred_mask.shape:
            raise ValueError(f"Mask dimensions do not match: {gt_file} and {pred_file}")

        # Threshold masks to binary (0 or 1)
        gt_mask = (gt_mask > 0.5).astype(np.uint8)
        pred_mask = (pred_mask > 0.5).astype(np.uint8)

        # Calculate metrics
        dice = dice_coefficient(gt_mask, pred_mask)
        try:
            iou = jaccard_score(gt_mask.flatten(), pred_mask.flatten(), average="binary")
        except:
            iou = 0
        pixel_acc = accuracy_score(gt_mask.flatten(), pred_mask.flatten())

        metrics.append({
            "Image": gt_file,
            "Dice": dice,
            "IoU": iou,
            "Pixel Accuracy": pixel_acc
        })

    return metrics


# Example usage:
if __name__ == "__main__":
    # Replace these paths with your actual folders
    ground_truth_folder = "galaxysegment/target"
    for type in ["topo", "no_topo"]:
        predicted_folder = f"galaxysegment/restored/{type}/"

        results = calculate_segmentation_metrics(ground_truth_folder, predicted_folder)
        pd.DataFrame(results).to_csv(f"galaxysegment/metrics/metrics_{type}.csv", index=False)
        pd.DataFrame(results).describe().to_csv(f"galaxysegment/metrics/metrics_{type}_summary.csv", index=False)

    # Print results
    #for res in results:
    #    print(res)
