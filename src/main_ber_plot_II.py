from helpers import progress_bar, utils
import matplotlib.pyplot as plt
from matplotlib import cycler
import numpy as np
import os

def main():
    # --- Ajustes globales para una mejor legibilidad ---
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
    })

    # Directorios
    base = os.path.join(os.getcwd(), "src/static")
    dir_covers = os.path.join(base, "results/ber")

    matrix_noise_for_all_methods = []

    methods_directory = sorted(os.listdir(dir_covers), key=lambda x: int(x.split("_")[1]))

    # Barra de progreso (por método)
    progress_bar.printProgressBar(0, len(methods_directory), prefix="Progress:", suffix="Complete", length=50)

    for idx_method, method_name in enumerate(methods_directory):
        dir_method_name = os.path.join(dir_covers, method_name)
        noises_directory = sorted(os.listdir(dir_method_name), key=lambda x: int(x.split("_")[0]))

        matrix_noise_for_each_methods = []

        for noise_name in noises_directory:
            dir_noise_name = os.path.join(dir_method_name, noise_name)
            third_directory = sorted(os.listdir(dir_noise_name), key=lambda x: int(x))
            matrix_noise = []
            for name_third_folder in third_directory:
                dir_third_folder = os.path.join(dir_noise_name, name_third_folder)
                signal_name = os.listdir(dir_third_folder)
                dir_signal = os.path.join(dir_third_folder, signal_name[0])
                # Aseguramos que sea 1D y ordenado
                vals = np.loadtxt(dir_signal)
                vals = np.ravel(vals)
                matrix_noise.append(sorted(list(vals)))
            # Promedio del conjunto (mantiene la longitud del eje X)
            matrix_noise_for_each_methods.append(utils.mean_signal(matrix_noise))

        matrix_noise_for_all_methods.append(matrix_noise_for_each_methods)
        # Progreso correcto respecto a métodos
        progress_bar.printProgressBar(idx_method + 1, len(methods_directory), prefix="Progress:", suffix="Complete", length=50)

    # Ejes X por prueba
    x = []
    x.append([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2])
    x.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    x.append([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02])
    x.append([-20, -15, -10, -5, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])
    x.append([1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    x.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    x.append([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2])
    x.append([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
    x.append([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2])
    x.append([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
    x.append([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
    x.append([-75, -60, -45, -30, -15, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225])
    x.append([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])

    noises = [
        'Gaussian noise', 'Minimum filter', 'Gaussian Laplace noise', 'Percentile filter',
        'Uniform filter', 'Fourier ellipsoid filter', 'Fourier gaussian', 'Fourier uniform filter',
        'Salt & pepper noise', 'Cropping noise', 'Scaling noise', 'Rotation noise', 'JPEG compression with QF'
    ]
    noises_name = [
        '0_gaussian', '1_minimum_filter', '2_gaussian_laplace', '3_percentile_filter',
        '4_uniform_filter', '5_fourier_ellipsoid_filter', '6_fourier_gaussian', '7_fourier_uniform_filter',
        '8_salt_pepper_noise', '9_cropping_noise', '11_scaling_noise', '12_rotation_noise', '13_jpeg_noise'
    ]

    # Colores/estilos/markers consistentes por método (12 métodos)
    Labels = [
        'CS_I', 'CS_II', 'MS_I', 'MS_II', 'KS',
        'Fares et al.', 'Begum et al.', 'Zear et al.',
        'Sharma et al.', 'Rajput et al.', 'Alomoush et al.',
        'Loke et al.',
    ]

    # Ciclos de estilo
    colors = plt.get_cmap("tab10").colors  # 10 colores
    # Extiende a 12 reutilizando de forma segura (manteniendo consistencia)
    colors = list(colors) + list(colors[:2])

    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-.", ":"]
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", ">", "<", "8"]

    method_style = {
        label: {"color": colors[i], "linestyle": linestyles[i], "marker": markers[i]}
        for i, label in enumerate(Labels)
    }

    # Directorio de salida
    out_dir = "src/static/plots/ber"
    os.makedirs(out_dir, exist_ok=True)

    # --- Graficar una figura por tipo de ruido ---
    for idx_noise, noise in enumerate(noises):
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.set_prop_cycle(cycler(color=[method_style[l]["color"] for l in Labels]))

        # Señales de todos los métodos para este ruido
        signal_noises = [meth_res[idx_noise] for meth_res in matrix_noise_for_all_methods]

        # Asegura y-range estándar de BER
        ax.set_ylim(0, 1)
        ax.set_xlabel(noise)
        ax.set_ylabel("Average BER")
        ax.set_title(f"BER vs {noise}")

        # Minor ticks y grid menor
        ax.minorticks_on()
        ax.grid(which="minor", linewidth=0.3, alpha=0.35)

        # Plot por método (orden y estilos fijos por método)
        for idx_method, signal in enumerate(signal_noises):
            label = Labels[idx_method]
            style = method_style[label]
            xi = np.asarray(x[idx_noise], dtype=float)
            yi = np.asarray(sorted(signal), dtype=float)

            # Seguridad: si las longitudes no coinciden, recorta a la mínima
            n = min(len(xi), len(yi))
            xi, yi = xi[:n], yi[:n]

            ax.plot(
                xi, yi,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4.5,
                linewidth=1.8,
                alpha=0.95,
                label=label
            )

        # Leyenda fuera, dos columnas
        leg = ax.legend(
            bbox_to_anchor=(1.02, 1.0), loc="upper left",
            borderaxespad=0., ncol=2, frameon=True
        )
        leg.get_frame().set_alpha(0.9)

        fig.tight_layout()

        # Guardar en múltiples formatos para informes/publicación
        base_name = os.path.join(out_dir, f"{noises_name[idx_noise]}")
        fig.savefig(base_name + ".png", bbox_inches="tight", facecolor="white")
        fig.savefig(base_name + ".svg", bbox_inches="tight", facecolor="white")
        fig.savefig(base_name + ".eps", bbox_inches="tight", facecolor="white")
        plt.close(fig)

if __name__ == "__main__":
    main()
