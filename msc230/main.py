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

    process_matlab_txt(root_path, raw_path, processed_path)

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





if __name__ == "__main__":
    main()