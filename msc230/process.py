import fnmatch
import os
import pickle

import numpy as np
from PIL import Image as im
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def process_matlab_txt(root_path, raw_path, processed_path) -> None:
    """
    Process (and reshape) Matlab matrices, from txt files to npy format.
    Saves npy files to [processed] folder.

    Args:
        root_path (str) : absolute path to root folder of the project
        raw_path (str) : relative path to [raw] folder
        processed_path (str) : relative path to [processed] folder

    Returns:
        Nothing

    """

    raw_files_list = os.listdir(os.path.join(root_path, raw_path))

    print(f"\nFiles to process ({len(raw_files_list)} files):\n")
    print(*raw_files_list, sep="\n")

    raw_files_list_enumerated = sorted(list(enumerate(raw_files_list)))

    print(f"\nSaving processed files to: {processed_path}*.npy\n")
    for i, j in raw_files_list_enumerated:

        df = np.loadtxt(os.path.join(root_path, raw_path, f"{j}"), skiprows=3)

        # TODO exeptions, when shape is wrong

        if df[:, 1:1444:2].shape[0] <= 721:
            outfile = df[:, 1:1444:2]
            print(i + 1, j, outfile.shape)

            with open((os.path.join(root_path, processed_path) + "/%03d.npy" % i), "wb") as f:
                np.save(f, outfile)

        else:
            outfile = df[::2, 1:1444:2]
            print(i + 1, j, outfile.shape)

            with open((os.path.join(root_path, processed_path) + "/%03d.npy" % i), "wb") as v:
                np.save(v, outfile)

    return


def processed_to_png(root_path, processed_path, scalers_path, png_path) -> None:
    """
    Converts processed NPY files to PNG images.
    MinMaxScaler to [0;255], saves scalers to [scalers] folder.

    Args:
        root_path (str) : absolute path to root folder of the project
        processed_path (str) : relative path to [processed] folder
        scalers_path (str) : relative path to [scalers] folder
        png_path (str) : relative path to [png_path] folder

    Returns:
        Nothing

    """

    npy_files_list = sorted(os.listdir(os.path.join(root_path, processed_path)))
    print(*npy_files_list, sep='\n')
    print(f'Total {len(npy_files_list)} files.')

    global_min = np.load(os.path.join(root_path, processed_path, npy_files_list[0])).min()
    global_max = np.load(os.path.join(root_path, processed_path, npy_files_list[0])).max()

    for i in npy_files_list:
        local_min = np.load(os.path.join(root_path, processed_path, i)).min()
        local_max = np.load(os.path.join(root_path, processed_path, i)).max()

        if local_min < global_min:
            global_min = local_min

        if local_max > global_max:
            global_max = local_max

    print("Global MIN, MAX:", global_min, global_max)

    # MINMAX Transform with global min, max
    scaler_0255_init = MinMaxScaler(feature_range=(0, 255), clip=False)
    scaler_0255_compr = MinMaxScaler(feature_range=(0, 255), clip=False)

    # open INITIAL for scaler
    init_mm = np.load(os.path.join(root_path, processed_path, npy_files_list[0]))
    init_mm = init_mm[:-1, :-2]  # attention ! cutting to 720x720 !

    init_mm[0, 0] = global_min
    init_mm[4, 0] = global_max

    # cut to COMPRESSED shape for compr scaler
    compr_mm = init_mm[::4, ::4]  # compressing/slicing to 180x180

    # flatten/reshape, fit
    scaler_0255_init.fit(init_mm.reshape(-1, 1))
    scaler_0255_compr.fit(compr_mm.reshape(-1, 1))

    pickle.dump(scaler_0255_init, open(os.path.join(root_path, scalers_path, "scaler_0255_init.sav"), "wb"))
    pickle.dump(scaler_0255_compr, open(os.path.join(root_path, scalers_path, "scaler_0255_compr.sav"), "wb"))

    # SAVE to PNG
    for i, j in tqdm(enumerate(npy_files_list), desc="Converting"):

        # INITIAL
        init_pic = np.load(os.path.join(root_path, processed_path, j))
        init_pic = init_pic[:-1, :-2]  # cutting to 720x720 !

        # GROUND
        out_ground_pic_np = scaler_0255_init.transform(init_pic.reshape(-1, 1))  # reshape to vector
        out_ground_pic_np = out_ground_pic_np.reshape((720, 720))  # reshape to matrix

        out_ground_pic = im.fromarray(np.uint8(out_ground_pic_np), "L")
        out_ground_pic.save(os.path.join(root_path, png_path) + "ground_%03d.png" % i)

        # COMPRESSED
        out_compr_pic_np = out_ground_pic_np[::4, ::4]  # slicing to 180x180
        out_compr_pic = im.fromarray(np.uint8(out_compr_pic_np), "L")
        out_compr_pic.save(os.path.join(root_path, png_path) + "compr_%03d.png" % i)

        # BICUBIC
        out_resized_pic = out_compr_pic.resize((720, 720), im.BICUBIC)
        out_resized_pic.save(os.path.join(root_path, png_path) + "bicubic_%03d.png" % i)

    return


def inverse_transform_to_txt(root_path, png_path, results_path, output_path, scalers_path) -> None:
    """
    Perfoms the inverse transformation, saves to the [output] folder.
    Output format is Matlab txt.

    Args:
        root_path (str) : absolute path to root folder of the project
        png_path (str) : relative path to [png_path] folder
        results_path (str) : relative path to [results] folder
        output_path (str) : relative path to [output] folder
        scalers_path (str) : relative path to [scalers] folder

    Returns:
        Nothing

    """

    print("INVERSE TRANSFORM")

    scaler_0255_init = pickle.load(open(os.path.join(root_path, scalers_path, "scaler_0255_init.sav"), "rb"))
    # scaler_0255_compr = pickle.load(open('/home/mike/MLDS/Dataset_export/scalers/scaler_0255_compr.sav', 'rb'))

    results_list = os.listdir(os.path.join(root_path, results_path))
    results_list = sorted(fnmatch.filter(results_list, "*.png"))
    print("Total files in [results] folder:", len(results_list))

    print("Working...")

    for file_name in results_list:
        number = file_name.split("_")[1]

        img_ground_big = np.array(im.open(os.path.join(root_path, png_path, f"ground_{number}.png")))
        # img_ground_small = np.array(im.open(f'dataset_png/compr_0{NUM}.png'))
        img_upscaled_bic = np.array(im.open(os.path.join(root_path, png_path, f"bicubic_{number}.png")))
        img_upscaled_nn = np.array(im.open(os.path.join(root_path, results_path, f"compr_{number}_out.png")))

        print(f"{file_name} -> {output_path}***.txt")

        # inverse transform, export to TXT
        np.savetxt(os.path.join(root_path, output_path, f"ground_big_{number}.txt"),
                   scaler_0255_init.inverse_transform(img_ground_big), delimiter=",")
        
        # np.savetxt(f'maps_output/ground_small_00{i}.txt', 
        #              scaler_0255_compr.inverse_transform(img_ground_small), delimiter=',')

        np.savetxt(os.path.join(root_path, output_path, f"upscaled_bicubic_{number}.txt"),
                   scaler_0255_init.inverse_transform(img_upscaled_bic), delimiter=",")
        
        np.savetxt(os.path.join(root_path, output_path, f"upscaled_nn_{number}.txt"),
                   scaler_0255_init.inverse_transform(img_upscaled_nn), delimiter=",")

    print("Done.")

    return
