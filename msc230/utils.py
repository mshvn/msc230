import fnmatch
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im

# from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def get_metrics(root_path, results_path, png_path) -> None:
    """
    Prints basic metrics (SSIM, PSNR).

    Args:
        root_path (str) : absolute path to root folder of the project
        results_path (str) : relative path to [results] folder
        png_path (str) : relative path to [png] folder

    Returns:
        Nothing

    """

    print("METRICS")

    # get list of files, filter PNG only
    results_list = sorted(os.listdir(os.path.join(root_path, results_path)))
    results_list = fnmatch.filter(results_list, "*.png")

    print(*results_list, sep="\n")
    print(f"Going to print SSIM and PSNR metrics for {len(results_list)} file(s) in [{results_path}] folder.")

    user_input = input("Do you want to continue? (y/n): ")
    if user_input.lower() not in ["yes", "y", "yep"]:
        return

    for result in results_list:

        print("\nMetrics for:", result)
        number = result.split("_")[1]

        ground_big = cv2.imread(
            os.path.join(root_path, png_path, f"ground_{number}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        upscaled_bic = cv2.imread(
            os.path.join(root_path, png_path, f"bicubic_{number}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        upscaled_nn = cv2.imread(
            os.path.join(root_path, results_path, f"compr_{number}_out.png"),
            cv2.IMREAD_GRAYSCALE,
        )

        print("SSIM BIC:", ssim(ground_big, upscaled_bic))
        print("SSIM NN: ", ssim(ground_big, upscaled_nn))

        print("PSNR BIC:", psnr(ground_big, upscaled_bic))
        print("PSNR NN: ", psnr(ground_big, upscaled_nn))

        # print(mse(ground_big, upscaled_bic))
        # print(mse(ground_big, upscaled_nn))

    return


def plot_polar(root_path, results_path, polar_path) -> None:
    """
    Converts images from cartesian to polar.
    Saves images to [polar] folder.

    Args:
        root_path (str) : absolute path to root folder of the project
        results_path (str) : relative path to [results] folder
        polar_path (str) : relative path to [polar] folder

    Returns:
        Nothing

    """

    print("PLOT POLAR")

    # get list of files, filter PNG only
    results_list = sorted(os.listdir(os.path.join(root_path, results_path)))
    results_list = fnmatch.filter(results_list, "*.png")

    print(*results_list, sep="\n")
    print(f"Going to process {len(results_list)} PNG files in [{results_path}] folder.")

    user_input = input("Do you want to continue? (y/n): ")
    if user_input.lower() not in ["yes", "y", "yep"]:
        return

    else:
        for result in results_list:
            image = np.array(im.open(os.path.join(root_path, results_path, result)))

            picture = np.zeros(([2000, 2000]), dtype=np.int16)
            theta_steps = np.arange(0, 361, 0.5)[:-2]

            for step in tqdm(range(len(image)), ncols=90, desc=f"Converting [{result}] "):
                # print('THETA STEPS:', theta_steps[step])
                # print(df.iloc[i])
                r = 0
                for val in image[step]:

                    # print(f'Value {val} in radius {r}, angle {angle_steps[step]}')
                    r = r + 1
                    x = (r * np.cos(np.deg2rad(theta_steps[step])) + 999)
                    y = (r * np.sin(np.deg2rad(theta_steps[step])) + 999)
                    # print('X Y:', x, y)
                    picture[round(x)][round(y)] = val

            plt.figure(figsize=(20, 20))
            plt.imshow(picture)
            plt.savefig(os.path.join(root_path, polar_path, "polar_figure_" + result))

            out_imgage = im.fromarray(np.uint8(picture), "L")
            out_imgage.save(os.path.join(root_path, polar_path, "polar_image_" + result))

        print("Done.")
        return
