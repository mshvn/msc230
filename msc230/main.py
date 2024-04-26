import numpy as np
import os
import pickle

from sklearn.preprocessing import MinMaxScaler
from PIL import Image as im


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

    # get_metrics
    # save_to_(polarplot?)


    # process_matlab_txt(root_path, raw_path, processed_path)

    # processed_to_png(root_path, processed_path, scalers_path, png_path)

    inverse_transform_to_txt(root_path, png_path, results_path, output_path, scalers_path)

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
        
        #TODO exeptions, when shape is wrong
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

            with open((os.path.join(root_path, processed_path) + '/%03d.npy'%i), 'wb') as v:
                np.save(v, outfile)
    return


def processed_to_png(root_path, processed_path, scalers_path, png_path):
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

    print('Global MIN, MAX:', global_min, global_max)


    # MINMAX Transform with global min, max
    scaler_0255_init = MinMaxScaler(feature_range = (0, 255), clip=False)
    scaler_0255_compr = MinMaxScaler(feature_range = (0, 255), clip=False)

    # open INITIAL for scaler
    init_mm = np.load(os.path.join(root_path, processed_path, npy_files_list[0]))
    init_mm = init_mm[:-1,:-2] # attention ! cutting to 720x720 !
    print(init_mm.shape)
    print('init pic before:', init_mm.min(), init_mm.max())

    init_mm[0,0] = (global_min)# - 1)
    init_mm[4,0] = (global_max)# + 1)
    print('init pic after:', init_mm.min(), init_mm.max())


    # cut to COMPRESSED for scaler
    compr_mm = init_mm[::4, ::4] # compressing/slicing to 180x180
    print(compr_mm.shape)
    print('compr pic after:', compr_mm.min(), compr_mm.max())

    # flatten/reshape, fit
    scaler_0255_init.fit(init_mm.reshape(-1, 1))
    scaler_0255_compr.fit(compr_mm.reshape(-1, 1))

    pickle.dump(scaler_0255_init, open(os.path.join(root_path, scalers_path, 'scaler_0255_init.sav'), 'wb'))
    pickle.dump(scaler_0255_compr, open(os.path.join(root_path, scalers_path, 'scaler_0255_compr.sav'), 'wb'))


    # SAVE to PNG version 2 (global min max)


    for i, j in enumerate(npy_files_list):

        print(i, j)
        
        #INITIAL
        init_pic = np.load(os.path.join(root_path, processed_path, j))
        init_pic = init_pic[:-1,:-2] # attention ! cutting to 720x720 !
        # scaler_0255_init.fit(init_pic)

        print('shape 0, MIN MAX:', init_pic.shape, init_pic.min(), init_pic.max())
        out_ground_pic_np = scaler_0255_init.transform(init_pic.reshape(-1,1)) # reshape to vector
        out_ground_pic_np = out_ground_pic_np.reshape((720,720)) # reshape to matrix 

        print('shape 1, MIN MAX:', out_ground_pic_np.shape, out_ground_pic_np.min(), out_ground_pic_np.max())
        # out_ground_pic_np = np.repeat(out_ground_pic_np[None], 3, axis=0)
        # print('shape 2, MIN MAX:', out_ground_pic_np.shape, out_ground_pic_np.min(), out_ground_pic_np.max())
        print()
        out_ground_pic = im.fromarray(np.uint8(out_ground_pic_np), 'L')
        # out_gound_pic.save('dataset_png/ground_%03d.png'%i)
        out_ground_pic.save(os.path.join(root_path, png_path) + 'ground_%03d.png'%i)
        
        
        #COMPRESSED
        out_compr_pic_np = out_ground_pic_np[::4, ::4] # compressing/slicing to 180x180
        # scaler_0255_compr.fit(compr_pic)
        print('shape 1 compr, MIN MAX:', out_compr_pic_np.shape, out_compr_pic_np.min(), out_compr_pic_np.max())
        
        # out_compr_pic_np = scaler_0255_compr.transform(compr_pic.reshape(-1,1))
        # out_compr_pic_np = out_compr_pic_np.reshape((180, 180))

        # out_compr_pic_np = np.repeat(out_compr_pic_np[None], 3, axis=0)
        out_compr_pic = im.fromarray(np.uint8(out_compr_pic_np), 'L')
        # out_compr_pic.save('dataset_png/compr_%03d.png'%i)
        out_compr_pic.save(os.path.join(root_path, png_path) + 'compr_%03d.png'%i)

        
        #BICUBIC
        # out_resized_pic = skim_resize(out_compr_pic, (dim1,dim2)) # mode -- different versions of resize
        # out_resized_pic_np = scaler_0255_init.transform(resized_pic)
        # out_resized_pic_np = np.repeat(out_resized_pic_np[None], 3, axis=0)
        # out_resized_pic = im.fromarray(np.uint8(np.ascontiguousarray(out_resized_pic_np.transpose(1,2,0))), 'RGB')
        out_resized_pic = out_compr_pic.resize((720,720), im.BICUBIC)
        # out_resized_pic.save('dataset_png/bicubic_%03d.png'%i)
        out_resized_pic.save(os.path.join(root_path, png_path) + 'bicubic_%03d.png'%i)

        # break


    return


def inverse_transform_to_txt(root_path, png_path, results_path, output_path, scalers_path):
    print('INVERSE TRANSFORM!')
    print(*os.listdir(os.path.join(root_path, results_path)), sep='\n')
    print('Total files in [results] folder:', len(os.listdir(os.path.join(root_path, results_path))))
    
    scaler_0255_init = pickle.load(open(os.path.join(root_path, scalers_path, 'scaler_0255_init.sav'), 'rb'))
    # scaler_0255_compr = pickle.load(open('/home/mike/MLDS/Dataset_export/scalers/scaler_0255_compr.sav', 'rb'))
    # print((scaler_0255_init))

    results_list = os.listdir(os.path.join(root_path, results_path))

    print('Working...')

    for file_name in results_list:
        number = file_name.split('_')[1]

        img_ground_big = np.array(im.open(os.path.join(root_path, png_path, f'ground_{number}.png')))#[:,:,0]
        # img_ground_small = np.array(im.open(f'dataset_png/compr_0{NUM}.png'))#[:,:,0]
        img_upscaled_bic = np.array(im.open(os.path.join(root_path, png_path, f'bicubic_{number}.png')))#[:,:,0]
        img_upscaled_nn = np.array(im.open(os.path.join(root_path, results_path, f'compr_{number}_out.png')))#[:,:,0] # here was -- convert L !


        print(f'{file_name} -> {output_path}***.txt')

        # inverse transform and export to TXT

        np.savetxt(os.path.join(root_path, output_path, f'ground_big_{number}.txt'), scaler_0255_init.inverse_transform(img_ground_big), delimiter=',')
        # np.savetxt(f'maps_output/ground_small_00{i}.txt', scaler_0255_compr.inverse_transform(img_ground_small), delimiter=',')
        np.savetxt(os.path.join(root_path, output_path, f'upscaled_bicubic_{number}.txt'), scaler_0255_init.inverse_transform(img_upscaled_bic), delimiter=',')
        np.savetxt(os.path.join(root_path, output_path, f'upscaled_nn_{number}.txt'), scaler_0255_init.inverse_transform(img_upscaled_nn), delimiter=',')

        # break

    print('Done.')

    return


if __name__ == "__main__":
    main()