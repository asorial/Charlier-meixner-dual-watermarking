from helpers import progress_bar, utils
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    # Directory setup
    base = "src/static"
    base = os.path.join(os.getcwd(), base)
    dir_covers = os.path.join(base, "results/ber")

    matrix_noise_for_all_methods = []
    methods_directory = sorted(
        os.listdir(dir_covers), key=lambda x: int(x.split("_")[1])
    )

    # Initial call to print 0% progress
    progress_bar.printProgressBar(0, len(methods_directory), prefix='Progress:',
                                 suffix='Complete', length=50)

    for index, method_name in enumerate(methods_directory):
        dir_method_name = os.path.join(dir_covers, method_name)
        noises_directory = sorted(
            os.listdir(dir_method_name), key=lambda x: int(x.split("_")[0])
        )

        matrix_noise_for_each_methods = []
        for noise_name in noises_directory:
            dir_noise_name = os.path.join(dir_method_name, noise_name)
            third_directory = sorted(os.listdir(dir_noise_name), key=lambda x: int(x))
            matrix_noise = []
            for name_third_folder in third_directory:
                dir_third_folder = os.path.join(dir_noise_name, name_third_folder)
                signal_name = os.listdir(dir_third_folder)
                dir_signal = os.path.join(dir_third_folder, signal_name[0])
                matrix_noise.append(sorted(list(np.loadtxt(dir_signal))))
            matrix_noise_for_each_methods.append(utils.mean_signal(matrix_noise))
        matrix_noise_for_all_methods.append(matrix_noise_for_each_methods)
        progress_bar.printProgressBar(index + 1, len(methods_directory),
                                     prefix='Progress:', suffix='Complete', length=50)

    # --- Configuración de datos y estilos de visualización mejorada ---

    # Eje X para cada tipo de ruido
    x = [
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02],
        [-20, -15, -10, -5, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
        [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2],
        [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        [-75, -60, -45, -30, -15, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225],
        [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    ]

    # Nombres y etiquetas de ruido
    noises = [
        'Gaussian noise', 'Minimum filter', 'Gaussian Laplace noise', 'Percentile filter', 'Uniform filter',
        'Fourier ellipsoid filter', 'Fourier gaussian', 'Fourier uniform filter', 'Salt & pepper noise',
        'Cropping noise', 'Scaling noise', 'Rotation noise', 'JPEG compression with QF'
    ]
    noises_name = [
        '0_gaussian', '1_minimum_filter', '2_gaussian_laplace', '3_percentile_filter', '4_uniform_filter',
        '5_fourier_ellipsoid_filter', '6_fourier_gaussian', '7_fourier_uniform_filter', '8_salt_pepper_noise',
        '9_cropping_noise', '11_scaling_noise', '12_rotation_noise', '13_jpeg_noise'
    ]

    # Colores, estilos de línea y marcadores
    colors = [
        "red", "green", "blue", "orange", "purple", "cyan", "magenta", "gold", "brown", "olive", "teal", "navy", "black"
    ]
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']

    labels = [
        'CS_I', 'CS_II', 'MS_I', 'MS_II', 'KS', 'Fares et al.', 'Begum et al.',
        'Zear et al.', 'Sharma et al.', 'Rajput et al.', 'Alomoush et al.',
        'Loke et al.',
    ]

    # --- Generación de gráficos ---
    for index, noise in enumerate(noises):
        fig, ax = plt.subplots(figsize=(12, 8)) # Aumenta el tamaño del gráfico
        signal_noises = [meth_res[index] for meth_res in matrix_noise_for_all_methods]

        for item, signal in enumerate(signal_noises):
            if item < len(labels):
                ax.plot(
                    x[index],
                    sorted(signal),
                    color=colors[item % len(colors)],
                    linestyle=linestyles[item % len(linestyles)],
                    marker=markers[item % len(markers)],
                    ms=6, lw=2, alpha=0.8,
                    label=labels[item]
                )

        # Configuración del gráfico
        ax.set_title(f'Rendimiento BER bajo {noise}', fontsize=16) # Título descriptivo
        ax.set_xlabel(f'Parámetros de {noise}', fontsize=12)
        ax.set_ylabel("Tasa de error de bits (BER)", fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5) # Cuadrícula mejorada
        ax.set_yscale('log') # Escala logarítmica para ver mejor las diferencias pequeñas
        ax.legend(
            title="Métodos",
            bbox_to_anchor=(1.05, 1.0),
            loc='upper left',
            fontsize=10
        )
        plt.tight_layout() # Asegura que las etiquetas no se corten

        # Save the figures
        dirp = "src/static/plots/ber"
        if not os.path.exists(dirp):
            os.makedirs(dirp)
        direps = os.path.join(dirp, noises_name[index] + ".eps")
        fig.savefig(direps, format='eps', bbox_inches='tight')

if __name__ == '__main__':
    main()