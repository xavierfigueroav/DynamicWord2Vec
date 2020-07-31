import glob
import os
import pickle
import sys

from scipy.io import savemat


def get_points(directory_path):
    files_path = glob.glob(os.path.join(directory_path, 'wordPairPMI_*.csv'))
    get_frame_number = lambda path: int(path.split('_')[-1].split('.')[0])
    frames = list(map(get_frame_number, files_path))
    return sorted(frames)

def transform(result, output, embs_dir):
    with open(result, 'rb') as pfile:
        data = pickle.load(pfile)
        frames = get_points(embs_dir)
        embs = {}
        for i, emb in enumerate(data):
            embs[f'U_{frames[i]}'] = emb
        savemat(output, embs, oned_as='row')


if __name__ == '__main__':
    result = sys.argv[1]
    output = sys.argv[2]
    embs = sys.argv[3]
    transform(result, output, embs)
