import glob
import os
import sys

from DynamicWord2Vec.visualization.transform_for_viz import transform
from DynamicWord2Vec.visualization.tsne_of_results import plot_trajectories


def get_last_iter_file(files):
    last_iter = -1
    last_iter_file = None
    for file_path in files:
        iter_n = int(file_path.split('_iter')[-1].split('.')[0])
        if iter_n > last_iter:
            last_iter = iter_n
            last_iter_file = file_path
    return last_iter_file

def run_trajectories(exper_dir, word, word_step, font_size):
    result_dir = os.path.join(exper_dir, 'results')
    output_file = os.path.join(result_dir, 'embs_for_viz.mat')

    if not os.path.isfile(output_file):
        result_files = glob.glob(os.path.join(result_dir, '*U*.p'))
        # needed due to method 'sorted' does not work as you may expect
        result_file = get_last_iter_file(result_files)
        transform(result_file, output_file)

    plot_trajectories(exper_dir, output_file, word, word_step, font_size)


if __name__ == '__main__':
    run_trajectories(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
