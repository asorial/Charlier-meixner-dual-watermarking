

from helpers import progress_bar, utils
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    # Directory
    base = "src/static"
    base = os.path.join(os.getcwd(), base)
    dir_covers = os.path.join(base, "results/ber")

    matrix_noise_for_all_methods = []

    methods_directory = os.listdir(dir_covers)

    methods_directory = sorted(
        methods_directory, key=lambda x: int(x.split("_")[1])
    )[:-1]

    # Initial call to print 0% progress
    progress_bar.printProgressBar(0, len(methods_directory), prefix='Progress:',
        suffix='Complete', length=50)
    for index, method_name in enumerate(methods_directory):

        dir_method_name = os.path.join(dir_covers, method_name)
        noises_directory = os.listdir(dir_method_name)
        noises_directory = sorted(
            noises_directory, key=lambda x: int(x.split("_")[0])
        )

        matrix_noise_for_each_methods = []

        for noise_name in noises_directory:
            dir_noise_name = os.path.join(dir_method_name, noise_name)
            third_directory = os.listdir(dir_noise_name)
            third_directory = sorted(third_directory, key=lambda x: int(x))
            matrix_noise = []
            for name_third_folder in third_directory:
                dir_third_folder = os.path.join(dir_noise_name, name_third_folder)
                signal_name = os.listdir(dir_third_folder)
                dir_signal = os.path.join(
                        dir_third_folder,
                        signal_name[0]
                )
                matrix_noise.append(
                        sorted(list(np.loadtxt(dir_signal)))
                    )
            matrix_noise_for_each_methods.append(
                    utils.mean_signal(matrix_noise)
                )
        matrix_noise_for_all_methods.append(
                matrix_noise_for_each_methods
        )
        progress_bar.printProgressBar(index+1, len(noises_directory),
                    prefix='Progress:', suffix='Complete', length=50)

    x = []

    x.append([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2])

    x.append(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        )

    x.append(
            [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02]
        )

    x.append([-20, -15, -10, -5, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])

    x.append([1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])

    x.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

    x.append(
            [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
        )

    x.append([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])

    x.append(
            [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
        )

    x.append([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])

    x.append(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    )

    x.append(
        [-75, -60, -45, -30, -15, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225]
    )
    x.append([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])

    noises = ['Gaussian noise', 'Minimum filter', 'Gaussian Laplace noise', 'Percentile filter', 'Uniform filter', 'Fourier ellipsoid filter', 'Fourier gaussian', 'Fourier uniform filter', 'Salt & pepper noise', 'Cropping noise', 'Scaling noise', 'Rotation noise', 'JPEG compression with QF']
    # JPEG compression with Quality Factor

    noises_name = ['0_gaussian', '1_minimum_filter', '2_gaussian_laplace', '3_percentile_filter', '4_uniform_filter', '5_fourier_ellipsoid_filter', '6_fourier_gaussian', '7_fourier_uniform_filter', '8_salt_pepper_noise', '9_cropping_noise', '11_scaling_noise', '12_rotation_noise', '13_jpeg_noise']

    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    ml = [
        "red", "green", "blue", "orange",
        "purple", "cyan", "magenta", "gold",
        "brown", "olive", "teal", "navy"
    ]

    Label = [
        'CS I', 'CS II', 'MS I', 'MS II', 'KS',
        'Fares et al.', 'Begum et al.', 'Zear et al.',
        'Sharma et al.', 'Rajput et al.', 'Alomoush et al.'
    ]

    for index, noise in enumerate(noises):
        fig, ax = plt.subplots(figsize=(12, 10))
        signal_noises = [
                meth_res[index] for meth_res in matrix_noise_for_all_methods
            ]
        for item, signal in enumerate(signal_noises):
            ax.plot(x[index], sorted(signal), '--o', ms=4, lw=1, alpha=0.5, mfc=ml[item], label=Label[item])
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        ax.grid()
        ax.set_xlabel(noise)
        ax.set_ylabel("The average BER")

        # Save the figures
        dirp = "src/static/plots/ber"
        if not os.path.exists(dirp): os.makedirs(dirp)
        direps = os.path.join(dirp, noises_name[index] + ".png")
        fig.savefig(direps, bbox_inches='tight')

if __name__ == '__main__':
    main()
