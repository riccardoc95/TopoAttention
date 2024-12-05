import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage import io
from sklearn.metrics import mean_squared_error as sklearn_mse

from astropy.io import fits
from PIL import Image
import pandas as pd


def load_image_as_array(filename):
    image = Image.open(filename)
    return np.array(image) / 255.


def calculate_metrics(gt_path, rec_path):
    """
    Calculate PSNR, SSIM, MSE, and NRMSE for ground truth and reconstructed images.

    Parameters:
        gt_path (str): Folder path containing ground truth images.
        rec_path (str): Folder path containing reconstructed images.

    Returns:
        metrics (list of dict): List containing metrics for each image pair.
    """
    metrics = []

    # List all files in the directories
    gt_images = sorted(os.listdir(gt_path))
    rec_images = sorted(os.listdir(rec_path))

    for gt_file, rec_file in zip(gt_images, rec_images):
        # Load images
        print(os.path.join(gt_path, gt_file), os.path.join(rec_path, rec_file))
        gt_image = load_image_as_array(os.path.join(gt_path, gt_file))
        rec_image = load_image_as_array(os.path.join(rec_path, rec_file))

        # Ensure both images have the same dimensions
        if gt_image.shape != rec_image.shape:
            raise ValueError(f"Image dimensions do not match: {gt_file} and {rec_file}")

        # Calculate metrics
        psnr_value = psnr(gt_image, rec_image, data_range=gt_image.max() - gt_image.min())
        ssim_value = ssim(gt_image, rec_image, data_range=gt_image.max() - gt_image.min(), multichannel=False)
        mse_value = mse(gt_image, rec_image)
        nrmse_value = np.sqrt(mse_value) / (gt_image.max() - gt_image.min())

        metrics.append({
            "Image": gt_file,
            "PSNR": psnr_value,
            "SSIM": ssim_value,
            "MSE": mse_value,
            "NRMSE": nrmse_value
        })

    return metrics


def remove_outliers(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns  # Select numeric columns

    df_filtered = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)  # First quartile
        Q3 = df[col].quantile(0.75)  # Third quartile
        IQR = Q3 - Q1  # Interquartile range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filter out outliers
        df_filtered = df_filtered[(df_filtered[col] >= lower_bound) & (df_filtered[col] <= upper_bound)]
    return df_filtered


# Example usage:
if __name__ == "__main__":
    # Replace these paths with your actual folders
    ground_truth_folder = "mnist/target"
    for type in ["topo", "no_topo"]:
        for noise in ["0_1", "0_2", "0_5"]:
            reconstructed_folder = f"mnist/restored/{type}/{noise}"

            results = calculate_metrics(ground_truth_folder, reconstructed_folder)
            noise_save = noise.replace("_",".")
            pd.DataFrame(results).to_csv(f"mnist/metrics/metrics_{type}_{noise_save}.csv", index=False)
            pd.DataFrame(results).describe().to_csv(f"mnist/metrics/metrics_{type}_{noise_save}_summary.csv", index=False)
            remove_outliers(pd.DataFrame(results)).describe().to_csv(f"mnist/metrics/metrics_{type}_{noise_save}_summary_no_out.csv",
                                                    index=False)

    # Print results
    #for res in results:
    #    print(res)
