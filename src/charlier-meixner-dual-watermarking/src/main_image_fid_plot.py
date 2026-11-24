

from helpers import progress_bar, utils
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os


def main():
    # Directory
    base = "src/static"
    base = os.path.join(os.getcwd(), base)
    dir_covers = os.path.join(base, "results/image_fid")

    signals_for_all_methods = []

    methods_directory = os.listdir(dir_covers)

    methods_directory = sorted(
        methods_directory, key=lambda x: int(x.split('.')[0])
    )

    for method_name in methods_directory:

        dir_method_name = os.path.join(dir_covers, method_name)

        signals_for_all_methods.append(
            sorted(list(np.loadtxt(dir_method_name)))
        )

    x = list(range(1, 25))

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

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for item, signal in enumerate(signals_for_all_methods):
        ax.plot(x, sorted(signal), '--o', ms=4, lw=1, alpha=0.5, mfc=ml[item], label=Label[item])
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.grid()
    ax.set_xlabel("Images")
    ax.set_ylabel("The average IF")

    # Save the figures
    dirp = "src/static/plots"
    if not os.path.exists(dirp): os.makedirs(dirp)
    direps = os.path.join(dirp, "image_fid.eps")
    fig.savefig(direps, bbox_inches='tight')

if __name__ == '__main__':
    main()
