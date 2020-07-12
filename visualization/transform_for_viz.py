import pickle
import sys

from scipy.io import savemat

result = 'results/L10T50G100A1ngU_iter4.p'
output = 'results/embs.mat'

with open(result, 'rb') as pfile:
    data = pickle.load(pfile)
    embs = {}
    for i, emb in enumerate(data):
        embs['U_' + str(i)] = emb
    savemat(output, embs, oned_as='row')
