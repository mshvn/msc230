import argparse
import os

from process import inverse_transform_to_txt, process_matlab_txt, processed_to_png
from utils import get_metrics, plot_polar


def main(action) -> None:
    """
    Computer Vision tools for MSC230 Super-Resolution project.

    Basic usage:
    * * *

    """
    # add ascii fancy header
    print(
        "\
    ████████████████████████████████████\n\
    █▄─▀█▀─▄█─▄▄▄▄█─▄▄▄─█▀▄▄▀█▄▄▄░█─▄▄─█\n\
    ██─█▄█─██▄▄▄▄─█─███▀██▀▄███▄▄░█─██─█\n\
    ▀▄▄▄▀▄▄▄▀▄▄▄▄▄▀▄▄▄▄▄▀▄▄▄▄▀▄▄▄▄▀▄▄▄▄▀"
    )

    root_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    print("\nRoot folder path:", root_path)

    print("Selected action:", action)

    # models_path = 'models/'
    # / added in end (?):
    raw_path = "data/raw/"
    processed_path = "data/processed/"
    png_path = "data/png/"
    scalers_path = "data/scalers/"
    results_path = "results/"
    output_path = "output/"
    polar_path = "polar/"

    action = action.lower()

    if action in ["process_matlab", "process_mat", "process", "pm"]:
        process_matlab_txt(root_path, raw_path, processed_path)
    elif action in ["to_png", "png", "top", "tp"]:
        processed_to_png(root_path, processed_path, scalers_path, png_path)
    elif action in ["inverse_transform", "inv_trans", "invtrans", "invt", "it"]:
        inverse_transform_to_txt(
            root_path, png_path, results_path, output_path, scalers_path
        )
    elif action in ["metrics", "metric", "met", "ms"]:
        get_metrics(root_path, results_path, png_path)
    elif action in ["plotpolar", "plot", "polar", "pp"]:
        plot_polar(root_path, results_path, polar_path)
    elif action in ["full", "fullforward", "forward", "ff"]:
        process_matlab_txt(root_path, raw_path, processed_path)
        processed_to_png(root_path, processed_path, scalers_path, png_path)
    else:
        print(f"Wrong action provided ({action}).")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python msc230/main.py",
        description="Actions parser for MSC230 tools.\
                                        Please enter action.",
    )
    parser.add_argument("action", help="valid actions: pm, tp, it, ms, pp, ff")
    args = parser.parse_args()
    main(args.action)
