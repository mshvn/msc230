![banner](static/banner_text.png)

![HSE-MLDS22](https://img.shields.io/badge/HSE-MLDS22-blue)


## Description:

Computer Vision tools for MSC230 Super-Resolution project.

A set of tools to help you work with magnetic field maps, 
convert and preprocess them.

Start with:
1. Clone the repository
2. Create new Conda environment with Poetry
3. Run 'poetry install'

## Basic usage:

1. The magnetic field map files (Matlab matrices, txt format) 
    must be located in the [data/raw] directory.
2. Use "python msc230/main.py [action]" to start the program.
3. Valid actions:

    - process_matlab / pm : 
        Process (and reshape) Matlab matrices.

    - to_png / tp : 
        Convert processed NPY files to PNG images.

    - inverse_transform / it : 
        Perfom the inverse transformation.

    - metrics / ms : 
        Print basic metrics.

    - plotpolar / pp : 
        Convert images from cartesian to polar.

    - forward / ff : 
        Run "process_matlab", then run "to_png"