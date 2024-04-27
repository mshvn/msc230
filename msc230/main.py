# import numpy as np
import os
# import pickle

# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt

# from tqdm import tqdm
# import fnmatch

# from PIL import Image as im
# import cv2

# from skimage.metrics import mean_squared_error as mse
# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import peak_signal_noise_ratio as psnr

from utils import get_metrics, plot_polar
from process import process_matlab_txt, processed_to_png, inverse_transform_to_txt

def main():
    # add ascii fancy header
    print('\
    ████████████████████████████████████\n\
    █▄─▀█▀─▄█─▄▄▄▄█─▄▄▄─█▀▄▄▀█▄▄▄░█─▄▄─█\n\
    ██─█▄█─██▄▄▄▄─█─███▀██▀▄███▄▄░█─██─█\n\
    ▀▄▄▄▀▄▄▄▀▄▄▄▄▄▀▄▄▄▄▄▀▄▄▄▄▀▄▄▄▄▀▄▄▄▄▀')

    root_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    print('\nRoot folder path:', root_path)

    # / added in end (?):
    raw_path = 'data/raw/'
    processed_path = 'data/processed/'
    png_path = 'data/png/'
    scalers_path = 'data/scalers/'
    models_path = 'models/'
    results_path = 'results/'
    output_path = 'output/'
    polar_path = 'polar/'


    # process_matlab_txt(root_path, raw_path, processed_path)
    # processed_to_png(root_path, processed_path, scalers_path, png_path)
    # inverse_transform_to_txt(root_path, png_path, results_path, output_path, scalers_path)
    # get_metrics(root_path, results_path, png_path)
    plot_polar(root_path, results_path, polar_path)

    return


if __name__ == "__main__":
    main()