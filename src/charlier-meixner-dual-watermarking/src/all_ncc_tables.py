
from helpers import progress_bar, utils
import numpy as np
import os
from datetime import datetime

def format_float(v, ndigits=4):
    """Format a number with fixed decimals."""
    try:
        return f"{float(v):.{ndigits}f}"
    except Exception:
        return str(v)

def make_safe_label(s):
    """Create a LaTeX-safe label string."""
    return (
        s.lower()
        .replace(" ", "_")
        .replace("&", "and")
        .replace("%", "pct")
        .replace("-", "_")
    )

def write_latex_table(dir_out, noise_title, noise_key, x_vals, signals_by_method, method_labels):
    """
    signals_by_method: list of lists, each with 20 BER values for one method
    x_vals: list of 20 x-axis values for that noise
    method_labels: column headers (methods)
    """
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    # Match plotting logic: sort each method's series before tabulating
    sorted_signals = [sorted(sig) for sig in signals_by_method]

    # Column headers
    cols = ["Parameter"] + method_labels
    col_spec = "l" + "c" * len(method_labels)

    # Rows: for each x, take the i-th value from each (sorted) method series
    lines = []
    for i in range(len(x_vals)):
        row = [format_float(x_vals[i], 4)]
        for m in range(len(sorted_signals)):
            row.append(format_float(sorted_signals[m][i], 4))
        lines.append(" & ".join(row) + r" \\")

    caption = f"Average NCC for {noise_title}"
    label = f"tab:ber_{make_safe_label(noise_key)}"
    today = datetime.now().strftime("%Y-%m-%d")

    tex = []
    tex.append(r"\begin{table}[ht]")
    tex.append(r"\centering")
    tex.append(r"\small")
    tex.append(r"\setlength{\tabcolsep}{4pt}")  # <- 4pt as requested
    tex.append(fr"\caption{{{caption}}}")
    tex.append(fr"\label{{{label}}}")
    tex.append(r"\resizebox{\linewidth}{!}{%")  # <- wrap the tabular in resizebox
    tex.append(fr"\begin{{tabular}}{{{col_spec}}}")
    tex.append(r"\toprule")
    tex.append(" & ".join(cols) + r" \\")
    tex.append(r"\midrule")
    tex.extend(lines)
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"}")  # <- close \resizebox
    tex.append(fr"% Automatically generated on {today}")
    tex.append(r"\end{table}")
    tex_str = "\n".join(tex)

    out_path = os.path.join(dir_out, f"{noise_key}.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex_str)


def main():
    # Input directories
    base = "src/static"
    base = os.path.join(os.getcwd(), base)
    dir_covers = os.path.join(base, "results/ncc")

    matrix_noise_for_all_methods = []

    methods_directory = os.listdir(dir_covers)
    methods_directory = sorted(methods_directory, key=lambda x: int(x.split("_")[1]))[:-1]

    # Overall progress (by method)
    progress_bar.printProgressBar(0, len(methods_directory), prefix='Progress:', suffix='Complete', length=50)

    for index, method_name in enumerate(methods_directory):
        dir_method_name = os.path.join(dir_covers, method_name)
        noises_directory = os.listdir(dir_method_name)
        noises_directory = sorted(noises_directory, key=lambda x: int(x.split("_")[0]))

        matrix_noise_for_each_methods = []

        for noise_name in noises_directory:
            dir_noise_name = os.path.join(dir_method_name, noise_name)
            third_directory = os.listdir(dir_noise_name)
            third_directory = sorted(third_directory, key=lambda x: int(x))
            matrix_noise = []
            for name_third_folder in third_directory:
                dir_third_folder = os.path.join(dir_noise_name, name_third_folder)
                signal_name = os.listdir(dir_third_folder)
                dir_signal = os.path.join(dir_third_folder, signal_name[0])
                matrix_noise.append(sorted(list(np.loadtxt(dir_signal))))
            matrix_noise_for_each_methods.append(utils.mean_signal(matrix_noise))

        matrix_noise_for_all_methods.append(matrix_noise_for_each_methods)

        # Update progress bar by method
        progress_bar.printProgressBar(index + 1, len(methods_directory), prefix='Progress:', suffix='Complete', length=50)

    # Predefined X axes
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

    x.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

    x.append(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    )

    x.append(
        [-75, -60, -45, -30, -15, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225]
    )

    x.append([100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5])

    noises = ['Gaussian noise', 'Minimum filter', 'Gaussian Laplace noise', 'Percentile filter', 'Uniform filter', 'Fourier ellipsoid filter', 'Fourier gaussian', 'Fourier uniform filter', 'Salt & pepper noise', 'Cropping noise', 'Median filter noise', 'Scaling noise', 'Rotation noise', 'JPEG compression with QF']
    # JPEG compression with Quality Factor

    noises_name = ['0_gaussian', '1_minimum_filter', '2_gaussian_laplace', '3_percentile_filter', '4_uniform_filter', '5_fourier_ellipsoid_filter', '6_fourier_gaussian', '7_fourier_uniform_filter', '8_salt_pepper_noise', '9_cropping_noise', '10_median_filter_noise', '11_scaling_noise', '12_rotation_noise', '13_jpeg_noise']

    method_labels = [
        'CS I', 'CS II', 'MS I', 'MS II', 'KS',
        'Fares et al.', 'Begum et al.', 'Zear et al.',
        'Sharma et al.', 'Rajput et al.', 'Alomoush et al.'
    ]

    # Output directory for LaTeX tables
    out_dir = "src/static/tables/ncc"

    # For each noise, gather the corresponding series per method and write a table
    for idx_noise, noise_title in enumerate(noises):
        noise_key = noises_name[idx_noise]
        # signals_by_method: list with len(methods) elements; each is the 20-point series for that noise
        signals_by_method = [meth_res[idx_noise] for meth_res in matrix_noise_for_all_methods]
        x_vals = x[idx_noise]
        write_latex_table(out_dir, noise_title, noise_key, x_vals, signals_by_method, method_labels)

    # Optional: master file that \input{} all tables
    master_path = os.path.join(out_dir, "all_ncc_tables.tex")
    with open(master_path, "w", encoding="utf-8") as f:
        f.write("% Include this file in your main document with \\input{}\n")
        for key in noises_name:
            f.write(fr"\input{{{key}.tex}}" + "\n")

if __name__ == '__main__':
    main()


# from helpers import progress_bar, utils
# import numpy as np
# import os
# from datetime import datetime

# def format_float(v, ndigits=4):
#     try:
#         return f"{float(v):.{ndigits}f}"
#     except Exception:
#         return str(v)

# def make_safe_label(s):
#     # etiqueta LaTeX segura para \label
#     return s.lower().replace(" ", "_").replace("&", "and").replace("%", "pct").replace("-", "_")

# def write_latex_table(dir_out, noise_title, noise_key, x_vals, signals_by_method, method_labels):
#     """
#     signals_by_method: lista de listas, cada una con 20 valores (BER) para un método
#     x_vals: lista de 20 valores del eje X para ese ruido
#     method_labels: nombres de columnas (métodos)
#     """
#     if not os.path.exists(dir_out):
#         os.makedirs(dir_out)

#     # Asegurar que cada columna BER esté ordenada según hacías en los plots
#     # (tu plot usaba sorted(signal) por método). Replicamos esa lógica aquí.
#     sorted_signals = [sorted(sig) for sig in signals_by_method]

#     # Construir encabezado de columnas
#     cols = ["Parámetro"] + method_labels
#     col_spec = "l" + "c" * len(method_labels)

#     # Construir filas: una fila por cada valor en x
#     # Para la columna i, tomar el i-ésimo valor de cada método (ya ordenado).
#     lines = []
#     for i in range(len(x_vals)):
#         row = [format_float(x_vals[i], 4)]
#         for m in range(len(sorted_signals)):
#             row.append(format_float(sorted_signals[m][i], 4))
#         lines.append(" & ".join(row) + r" \\")

#     caption = f"Promedio de BER para {noise_title}."
#     label = f"tab:ber_{make_safe_label(noise_key)}"
#     today = datetime.now().strftime("%Y-%m-%d")

#     tex = []
#     tex.append(r"\begin{table}[ht]")
#     tex.append(r"\centering")
#     tex.append(r"\small")
#     tex.append(r"\setlength{\tabcolsep}{6pt}")
#     tex.append(fr"\caption{{{caption}}}")
#     tex.append(fr"\label{{{label}}}")
#     tex.append(fr"\begin{{tabular}}{{{col_spec}}}")
#     tex.append(r"\toprule")
#     tex.append(" & ".join(cols) + r" \\")
#     tex.append(r"\midrule")
#     tex.extend(lines)
#     tex.append(r"\bottomrule")
#     tex.append(r"\end{tabular}")
#     tex.append(fr"% Generado automáticamente el {today}")
#     tex.append(r"\end{table}")
#     tex_str = "\n".join(tex)

#     out_path = os.path.join(dir_out, f"{noise_key}.tex")
#     with open(out_path, "w", encoding="utf-8") as f:
#         f.write(tex_str)

# def main():
#     # Directorios de entrada
#     base = "src/static"
#     base = os.path.join(os.getcwd(), base)
#     dir_covers = os.path.join(base, "results/ber")

#     matrix_noise_for_all_methods = []

#     methods_directory = os.listdir(dir_covers)
#     methods_directory = sorted(methods_directory, key=lambda x: int(x.split("_")[1]))[:-1]

#     # Progreso global (por método)
#     progress_bar.printProgressBar(0, len(methods_directory), prefix='Progreso:', suffix='Completado', length=50)

#     for index, method_name in enumerate(methods_directory):
#         dir_method_name = os.path.join(dir_covers, method_name)
#         noises_directory = os.listdir(dir_method_name)
#         noises_directory = sorted(noises_directory, key=lambda x: int(x.split("_")[0]))

#         matrix_noise_for_each_methods = []

#         for noise_name in noises_directory:
#             dir_noise_name = os.path.join(dir_method_name, noise_name)
#             third_directory = os.listdir(dir_noise_name)
#             third_directory = sorted(third_directory, key=lambda x: int(x))
#             matrix_noise = []
#             for name_third_folder in third_directory:
#                 dir_third_folder = os.path.join(dir_noise_name, name_third_folder)
#                 signal_name = os.listdir(dir_third_folder)
#                 dir_signal = os.path.join(dir_third_folder, signal_name[0])
#                 matrix_noise.append(sorted(list(np.loadtxt(dir_signal))))
#             matrix_noise_for_each_methods.append(utils.mean_signal(matrix_noise))

#         matrix_noise_for_all_methods.append(matrix_noise_for_each_methods)

#         # actualizar barra de progreso por método
#         progress_bar.printProgressBar(index + 1, len(methods_directory), prefix='Progreso:', suffix='Completado', length=50)

#     # Ejes X predefinidos
#     x = []
#     x.append([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2])
#     x.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
#     x.append([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02])
#     x.append([-20, -15, -10, -5, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])
#     x.append([1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
#     x.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
#     x.append([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2])
#     x.append([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
#     x.append([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2])
#     x.append([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
#     x.append([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
#     x.append([-75, -60, -45, -30, -15, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225])
#     x.append([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])

#     noises = ['Gaussian noise', 'Minimum filter', 'Gaussian Laplace noise', 'Percentile filter', 'Uniform filter', 'Fourier ellipsoid filter', 'Fourier gaussian', 'Fourier uniform filter', 'Salt & pepper noise', 'Cropping noise', 'Scaling noise', 'Rotation noise', 'JPEG compression with QF']

#     noises_name = ['0_gaussian', '1_minimum_filter', '2_gaussian_laplace', '3_percentile_filter', '4_uniform_filter', '5_fourier_ellipsoid_filter', '6_fourier_gaussian', '7_fourier_uniform_filter', '8_salt_pepper_noise', '9_cropping_noise', '11_scaling_noise', '12_rotation_noise', '13_jpeg_noise']

#     method_labels = [
#         'CS I', 'CS II', 'MS I', 'MS II', 'KS',
#         'Fares et al.', 'Begum et al.', 'Zear et al.',
#         'Sharma et al.', 'Rajput et al.', 'Alomoush et al.'
#     ]

#     # Directorio de salida para tablas LaTeX
#     out_dir = "src/static/tables/ber"

#     # Para cada ruido, recopilar la columna correspondiente de cada método y escribir tabla
#     for idx_noise, noise_title in enumerate(noises):
#         noise_key = noises_name[idx_noise]
#         # signals_by_method: una lista con len(methods) elementos; cada elemento es la serie de 20 puntos para ese ruido
#         signals_by_method = [meth_res[idx_noise] for meth_res in matrix_noise_for_all_methods]
#         x_vals = x[idx_noise]
#         write_latex_table(out_dir, noise_title, noise_key, x_vals, signals_by_method, method_labels)

#     # (Opcional) archivo maestro que \input todas las tablas
#     master_path = os.path.join(out_dir, "all_ber_tables.tex")
#     with open(master_path, "w", encoding="utf-8") as f:
#         f.write("% Compila este archivo en tu documento principal con \\input{}\n")
#         for key in noises_name:
#             f.write(fr"\input{{{key}.tex}}" + "\n")

# if __name__ == '__main__':
#     main()
