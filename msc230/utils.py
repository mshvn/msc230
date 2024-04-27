import numpy as np
import os
# import pickle

# from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from tqdm import tqdm
import fnmatch

from PIL import Image as im
import cv2

# from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr



def get_metrics(root_path, results_path, png_path):
    '''
    Basic metrics
    '''

    print('METRICS!')

    # argPARSE filename/number here ??

    results_list = sorted(os.listdir(os.path.join(root_path, results_path)))
    results_list = fnmatch.filter(results_list, '*.png')

    print(*results_list, sep='\n')
    print('Metrics for:', results_list[0])
    number = results_list[0].split('_')[1]

    ground_big = cv2.imread(os.path.join(root_path, png_path, f'ground_{number}.png'), cv2.IMREAD_GRAYSCALE)
    upscaled_bic = cv2.imread(os.path.join(root_path, png_path, f'bicubic_{number}.png'), cv2.IMREAD_GRAYSCALE)
    upscaled_nn = cv2.imread(os.path.join(root_path, results_path, f'compr_{number}_out.png'), cv2.IMREAD_GRAYSCALE)

    print('SSIM BIC:', ssim(ground_big, upscaled_bic))
    print('SSIM NN: ', ssim(ground_big, upscaled_nn))

    print('PSNR BIC:', psnr(ground_big, upscaled_bic))
    print('PSNR NN: ', psnr(ground_big, upscaled_nn))

    # print(mse(ground_big, upscaled_bic))
    # print(mse(ground_big, upscaled_nn))

    return


def plot_polar(root_path, results_path, polar_path):
    '''
    Save to polar image
    '''

    print('POLAR PLOT!')

    # get list of files, filter only PNG
    results_list = sorted(os.listdir(os.path.join(root_path, results_path)))
    results_list = fnmatch.filter(results_list, '*.png')

    print(*results_list, sep='\n')
    print(f'Going to process {len(results_list)} PNG files in [{results_path}] folder.')


    user_input = input('Do you want to continue? (y/n): ')
    if user_input.lower() not in ['yes', 'y', 'yep']:
        return
        
    else:
        for result in results_list:
            image = np.array(im.open(os.path.join(root_path, results_path, result)))
            # print(image.shape)

            picture = np.zeros(([2000, 2000]), dtype=np.int16)
            theta_steps = np.arange(0, 361, 0.5)[:-2]

            # print(picture.shape)
            # print(len(theta_steps), theta_steps[-5:])


            for step in tqdm(range(len(image)), ncols=90, desc=f'Converting [{result}] '):
                # print('THETA STEPS:', theta_steps[step])
                # print(df.iloc[i])
                r = 0
                for val in image[step]:

                    # print(f'Value {val} in radius {r}, angle {angle_steps[step]}')
                    r = r + 1

                    x = r * np.cos(np.deg2rad(theta_steps[step])) + 999#len(r_steps)-3#1999
                    y = r * np.sin(np.deg2rad(theta_steps[step])) + 999#len(r_steps)-3#1999

                    # print('X Y:', x, y)
                    picture[round(x)][round(y)] = val

                    # if theta_steps[step] % 30 == 0:
                    #     print('*', end='') 
            
            plt.figure(figsize=(20,20))
            plt.imshow(picture)
            plt.savefig(os.path.join(root_path, polar_path, 'polar_figure_' + result))

            out_imgage = im.fromarray(np.uint8(picture), 'L')
            out_imgage.save(os.path.join(root_path, polar_path, 'polar_image_' + result))

        print('Done.')
        return