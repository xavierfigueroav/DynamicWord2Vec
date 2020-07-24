import pickle
import sys

from scipy.io import savemat


def transform(result, output):
    with open(result, 'rb') as pfile:
        data = pickle.load(pfile)
        embs = {}
        for i, emb in enumerate(data):
            embs['U_' + str(i)] = emb
        savemat(output, embs, oned_as='row')


if __name__ == '__main__':
    result = sys.argv[1]
    output = sys.argv[2]
    transform(result, output)
