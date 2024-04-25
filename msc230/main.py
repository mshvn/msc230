import numpy as np
import os

def main():
    # add ascii fancy header
    print('\
    ████████████████████████████████████\n\
    █▄─▀█▀─▄█─▄▄▄▄█─▄▄▄─█▀▄▄▀█▄▄▄░█─▄▄─█\n\
    ██─█▄█─██▄▄▄▄─█─███▀██▀▄███▄▄░█─██─█\n\
    ▀▄▄▄▀▄▄▄▀▄▄▄▄▄▀▄▄▄▄▄▀▄▄▄▄▀▄▄▄▄▀▄▄▄▄▀')

    root_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    print('\nRoot path:', root_path)

    raw_path = 'data/raw/'
    processed_path = 'data/processed'
    png_path = 'data/png'

    # process_matlab_txt(root_path, raw_path, processed_path)

    processed_to_png(root_path, processed_path, png_path)

    return


def process_matlab_txt(root_path, raw_path, processed_path):
    # find best way to use join and OS path...
    # raw_path = 'data/raw/'
    # processed_path = 'data/processed'
    raw_files_list = os.listdir(os.path.join(root_path, raw_path))

    print(f'\nFiles to process ({len(raw_files_list)} files):\n')
    print(*raw_files_list, sep='\n')

    raw_files_list_enumerated = sorted(list(enumerate(raw_files_list)))

    print(f'\nSaving processed files to: {processed_path}/*.npy\n')
    for i, j in raw_files_list_enumerated:

        df = np.loadtxt(os.path.join(root_path, raw_path, f'{j}'), skiprows=3)

        if df[:, 1:1444:2].shape[0] <= 721:
            # display(df[:, 1:1444:2].shape)
            outfile = df[:, 1:1444:2]
            print(i+1, j, outfile.shape)

            with open((os.path.join(root_path, processed_path) + '/%03d.npy'%i), 'wb') as f:
                np.save(f, outfile)

        else:
            # display(df[::2, 1:1444:2].shape)
            outfile = df[::2, 1:1444:2]
            print(i+1, j, outfile.shape)

            with open((os.path.join(root_path, processed_path) +'/%03d.npy'%i), 'wb') as v:
                np.save(v, outfile)
    return


def processed_to_png(root_path, processed_path, png_path):
    # print(root_path, png_path)

    npy_files_list = sorted(os.listdir(os.path.join(root_path, processed_path)))
    print(npy_files_list)


    global_min = np.load(os.path.join(root_path, processed_path, npy_files_list[0])).min()
    global_max = np.load(os.path.join(root_path, processed_path, npy_files_list[0])).max()
    
    # print(global_min, global_max)

    for i in npy_files_list:
        local_min = np.load(os.path.join(root_path, processed_path, i)).min()
        local_max = np.load(os.path.join(root_path, processed_path, i)).max()

        if local_min < global_min:
            global_min = local_min

        if local_max > global_max:
            global_max = local_max

    print(global_min, global_max)

    return



if __name__ == "__main__":
    main()