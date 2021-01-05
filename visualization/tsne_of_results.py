# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09  2017
"""
import glob
import os

import numpy as np
import pandas as pd
import scipy.io as sio


def get_points(directory_path):
    files_path = glob.glob(os.path.join(directory_path, 'wordPairPMI_*.csv'))
    get_frame_number = lambda path: int(path.split('_')[-1].split('.')[0])
    frames = list(map(get_frame_number, files_path))
    return sorted(frames)

def plot_trajectories(exper_dir, events, embeddings, word, word_step, font_size):
    embs_dir = os.path.join(exper_dir, 'embs')
    tsne_output = os.path.join(exper_dir, 'visualization')
    vocabulary = os.path.join(embs_dir, 'wordIDHash.csv')
    wordlist = []
    fid = open(vocabulary, 'r')
    for line in fid:
        word_id, _word = line.strip().split(',')
        wordlist.append(_word)
    fid.close()

    word2Id = {}
    for k in range(len(wordlist)):
        word2Id[wordlist[k]] = k

    times = get_points(embs_dir) # total number of time points (20/range(27) for ngram/nyt)

    emb_all = sio.loadmat(embeddings)

    emb = emb_all[f'U_{times[-1]}']
    nn = emb.shape[1]
                
    X = []
    list_of_words = []
    isword = []
    words_by_period = {}
    for year in times:
        emb = emb_all[f'U_{year}']
        embnrm = np.reshape(np.sqrt(np.sum(emb**2,1)),(emb.shape[0],1))
        emb_normalized = np.divide(emb, np.tile(embnrm, (1,emb.shape[1])))           
        print(emb_normalized.shape)
        v = emb_normalized[word2Id[word],:]

        d =np.dot(emb_normalized,v)
        
        idx = np.argsort(d)[::-1]
        newwords = [(wordlist[k], year) for k in list(idx[:nn])]
        print(newwords)
        list_of_words.extend(newwords)
        words_by_period[year] = list(map(lambda word: word[0], newwords))
        for k in range(nn):
            isword.append(k==0)
        X.append(emb[idx[:nn],:])
        
    X = np.vstack(X)
    print(X.shape)

    import umap
    model = umap.UMAP(n_neighbors=10, min_dist=0.75, metric='cosine', random_state=1)
    Z = model.fit_transform(X)

    import matplotlib.pyplot as plt
    import pickle

    plt.clf()
    traj = []
    target_indexes = []
    not_target_indexes = []
    sum_of_coor = dict()
    for k in range(len(list_of_words)):
        k_word = list_of_words[k][0] # e.g.: guayaquil
        period = list_of_words[k][1] # e.g.: 0 if first week, 1 if second week, etc.
        if isword[k] :
            target_indexes.append(k)
            marker = 's'
            color = 'red' if period in events else 'dodgerblue'
            traj.append(Z[k,:])
            plt.plot(Z[k,0], Z[k,1], marker, color=color, markersize=7)

            # plot only a few labels for clarity
            if period % word_step == 0 or period == times[-1]:
                plt.text(Z[k, 0], Z[k, 1], f'{k_word}::{period}', fontsize=font_size)
            else:
                plt.text(Z[k, 0], Z[k, 1], f'{period}', fontsize=font_size)
        else:
            not_target_indexes.append(k)
            sum_of_coor[k_word] = sum_of_coor.get(k_word, np.zeros(2))
            sum_of_coor[k_word] += Z[k]

    distances = []
    for i in target_indexes:
        differences = Z[not_target_indexes] - Z[i]
        distances.extend(np.linalg.norm(differences, axis=1))
    dist_threshold = np.quantile(distances, 0.95)
    print('==', dist_threshold)

    def plot_word(word_index, k_word, list_of_words):
        period = list_of_words[word_index][1] # e.g.: 0 if first week, 1 if second week, etc.
        plt.plot(Z[word_index, 0], Z[word_index, 1], 'o', color='mediumseagreen')
        plt.text(Z[word_index, 0], Z[word_index, 1], f'{k_word}::{period}', fontsize=font_size)
                        
    plot_indexes = set()
    plot_words = dict()
    for i in target_indexes:
        differences = Z[not_target_indexes] - Z[i]
        distances = np.linalg.norm(differences, axis=1)
        closest = sorted(zip(distances, not_target_indexes))
        top_threshold = 20
        for distance, word_index in closest[:top_threshold]:
            if distance < dist_threshold and not word_index in plot_indexes:
                k_word = list_of_words[word_index][0] # e.g.: guayaquil
                if plot_words.get(k_word) is None:
                    plot_word(word_index, k_word, list_of_words)
                    plot_indexes.add(word_index)
                    plot_words[k_word] = np.array([Z[word_index]])
                else:
                    differences = plot_words[k_word] - Z[word_index]
                    distances = np.linalg.norm(differences, axis=1)
                    if distances[distances < 1].shape[0] == 0:
                        plot_word(word_index, k_word, list_of_words)
                        plot_indexes.add(word_index)
                        plot_words[k_word] = np.append(plot_words[k_word], [Z[word_index]], axis=0)

    traj = np.vstack(traj)
    plt.plot(traj[:,0], traj[:,1], linewidth=2)
    plt.show()

    target_word_dir = os.path.join(tsne_output, word)
    if not os.path.isdir(target_word_dir):
        os.makedirs(target_word_dir)
    sio.savemat(os.path.join(target_word_dir, 'embs.mat'), {'emb':Z})
    pickle.dump(
        {'words':list_of_words,'isword':isword}, 
        open(os.path.join(target_word_dir, 'wordlist.pkl'),'wb')
    )
    for period, context_words in words_by_period.items():
        lines = []
        for context_word in context_words:
            lines.append(f'{word2Id[context_word]},{context_word}\n')
        with open(os.path.join(target_word_dir, f'closer2{word}_week_{period}.csv'), 'w') as file:
            file.writelines(lines)


    allwords = ['art','damn','gay','hell','maid','muslim']

    import matplotlib.pyplot as plt
    import pickle
    Z = sio.loadmat(os.path.join(target_word_dir, 'embs.mat'))['emb']
    data = pickle.load(
        open(os.path.join(target_word_dir, 'wordlist.pkl'),'rb')
    )
    list_of_words, isword = data['words'],data['isword']
    plt.clf()
    traj = []


    Zp = Z*1.
    Zp[:,0] = Zp[:,0]*2.
    all_dist = np.zeros((Z.shape[0],Z.shape[0]))
    for k in range(Z.shape[0]):
        all_dist[:,k] =np.sum( (Zp - np.tile(Zp[k,:],(Z.shape[0],1)))**2.,axis=1)

    dist_to_centerpoints = all_dist[:,isword]
    dist_to_centerpoints = np.min(dist_to_centerpoints,axis=1)

    dist_to_other = all_dist + np.eye(Z.shape[0])*1000.
    idx_dist_to_other = np.argsort(dist_to_other,axis=1)
    dist_to_other = np.sort(dist_to_other,axis=1)

    plt.clf()
    for k in range(len(list_of_words)-1,-1,-1):
        
        if isword[k] :
            #if list_of_words[k][1] % 3 != 0 and list_of_words[k][1] < 199 : continue
            marker = 'bo'
            traj.append(Z[k,:])
            plt.plot(Z[k,0], Z[k,1],marker)
        else: 
            if dist_to_centerpoints[k] > 200: continue
            skip =False
            for i in range(Z.shape[0]):
                if dist_to_other[k,i] < 150 and idx_dist_to_other[k,i] > k: 
                    skip = True
                    break
                if dist_to_other[k,i] >= 150: break
            
            if skip: continue
            if Z[k,0] > 8: continue
            plt.plot(Z[k,0], Z[k,1])
        
        
        plt.text(Z[k,0]-2, Z[k,1]+np.random.randn()*2,' %s-%d' % (list_of_words[k][0], list_of_words[k][1]*10))

    plt.axis('off')
    traj = np.vstack(traj)
    plt.plot(traj[:,0],traj[:,1])
    plt.show()
