#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:10:42 2016

"""

# main script for time CD 
# trainfile has lines of the form
# tok1,tok2,pmi

import glob
import os
import sys
import numpy as np
import pickle as pickle


def print_params(r,lam,tau,gam,emph,ITERS):
    
    print('rank = {}'.format(r))
    print('frob  regularizer = {}'.format(lam))
    print('time  regularizer = {}'.format(tau))
    print('symmetry regularizer = {}'.format(gam))
    print('emphasize param   = {}'.format(emph))
    print('total iterations = {}'.format(ITERS))
    
if __name__ == '__main__':
    import util_timeCD as util

    nw = 20936 # number of words in vocab (11068100/20936 for ngram/nyt)
    T = list(range(1990, 2016)) # total number of time points (20/range(27) for ngram/nyt)

    trainhead = 'data/wordPairPMI_' # location of training data
    savehead = 'results/'

    ITERS = 5 # total passes over the data
    lam = 10 #frob regularizer
    gam = 100 # forcing regularizer
    tau = 50  # smoothing regularizer
    r   = 50  # rank
    b = nw # batch size
    emph = 1 # emphasize the nonzero

    foo = sys.argv
    for i in range(1,len(foo)):
        if foo[i]=='-r':    r = int(float(foo[i+1]))        
        if foo[i]=='-iters': ITERS = int(float(foo[i+1]))            
        if foo[i]=='-lam':    lam = float(foo[i+1])
        if foo[i]=='-tau':    tau = float(foo[i+1])
        if foo[i]=='-gam':    gam = float(foo[i+1])
        if foo[i]=='-b':    b = int(float(foo[i+1]))
        if foo[i]=='-emph': emph = float(foo[i+1])
        if foo[i]=='-check': erchk=foo[i+1]

def get_vocabulary_size(vocabulary_directory):
    with open(vocabulary_directory, 'r') as voc_file:
        return len(voc_file.readlines())

def get_points(directory_path):
    files_path = glob.glob(os.path.join(directory_path, 'wordPairPMI_*.csv'))
    get_frame_number = lambda path: int(path.split('_')[-1].split('.')[0])
    frames = list(map(get_frame_number, files_path))
    return list(range(min(frames), max(frames) + 1))

def run_dw2v(exper_dir, iters, lam, tau, gam, emph, r):

    import train_model.util_timeCD as util

    data_dir = os.path.join(exper_dir, 'embs')
    trainhead = os.path.join(data_dir, 'wordPairPMI_')
    voc_path = os.path.join(data_dir, 'wordIDHash.csv')
    embmat = os.path.join(data_dir, 'emb_static.mat')
    b = nw = get_vocabulary_size(voc_path)
    ITERS = iters
    T = get_points(data_dir)
    savehead = os.path.join(exper_dir, 'results')
    savefile = 'L{}T{}G{}A{}'.format(lam, tau, gam, emph)
    savefile = os.path.join(savehead, savefile)

    print(f"b = {b}, nw = {nw}, T = {T}")
    print(iters, lam, tau, gam, emph, r)
    
    print('starting training with following parameters')
    print_params(r,lam,tau,gam,emph,ITERS)
    print('there are a total of {} words, and {} time points'.format(nw,T))
    
    print('X*X*X*X*X*X*X*X*X')
    print('initializing')
    
    #Ulist,Vlist = util.initvars(nw,T,r, trainhead)
    Ulist,Vlist = util.import_static_init(T, embmat)
    print(Ulist)
    print(Vlist)

    print('getting batch indices')
    if b < nw:
        b_ind = util.getbatches(nw,b)
    else:
        b_ind = [list(range(nw))]
    
    import time
    start_time = time.time()
    # sequential updates
    for iteration in range(ITERS):  
        print_params(r,lam,tau,gam,emph,ITERS)
        try:
            Ulist = pickle.load(open( "%sngU_iter%d.p" % (savefile,iteration), "rb" ) )
            Vlist = pickle.load(open( "%sngV_iter%d.p" % (savefile, iteration), "rb" ) )
            print('iteration %d loaded succesfully' % iteration)
            continue
        except(IOError):
            pass
        loss = 0
        # shuffle times
        if iteration == 0: times = T
        else: times = np.random.permutation(T)
        
        for t in range(len(times)):   # select a time
            print('iteration %d, time %d' % (iteration, t))
            f = trainhead + str(t) + '.csv'
            print(f)
            
            """
            try:
                Ulist = pickle.load( open( "%sngU_iter%d_time%d_tmp.p" % (savefile,iteration,t), "rb" ) )
                Vlist = pickle.load( open( "%sngV_iter%d_time%d_tmp.p" % (savefile, iteration,t), "rb" ) )
                times = pickle.load( open( "%sngtimes_iter%d_time%d_tmp.p" % (savefile, iteration,t), "rb" ) )
                print 'iteration %d time %d loaded succesfully' % (iteration, t)
                continue
            except(IOError):
                pass
            """
            
            pmi = util.getmat(f,nw,False)
            for j in range(len(b_ind)): # select a mini batch
                print('%d out of %d' % (j,len(b_ind)))
                ind = b_ind[j]
                ## UPDATE V
                # get data
                pmi_seg = pmi[:,ind].todense()
                
                if t==0:
                    vp = np.zeros((len(ind),r))
                    up = np.zeros((len(ind),r))
                    iflag = True
                else:
                    vp = Vlist[t-1][ind,:]
                    up = Ulist[t-1][ind,:]
                    iflag = False

                if t==len(T)-1:
                    vn = np.zeros((len(ind),r))
                    un = np.zeros((len(ind),r))
                    iflag = True
                else:
                    vn = Vlist[t+1][ind,:]
                    un = Ulist[t+1][ind,:]
                    iflag = False
                Vlist[t][ind,:] = util.update(Ulist[t],emph*pmi_seg,vp,vn,lam,tau,gam,ind,iflag)
                Ulist[t][ind,:] = util.update(Vlist[t],emph*pmi_seg,up,un,lam,tau,gam,ind,iflag)
            
                
            #pickle.dump(Ulist, open( "%sngU_iter%d_time%d_tmp.p" % (savefile,iteration,t), "wb" ) , pickle.HIGHEST_PROTOCOL)
            #pickle.dump(Vlist, open( "%sngV_iter%d_time%d_tmp.p" % (savefile, iteration,t), "wb" ) , pickle.HIGHEST_PROTOCOL)
            #pickle.dump(times, open( "%sngtimes_iter%d_time%d_tmp.p" % (savefile, iteration,t), "wb" ) , pickle.HIGHEST_PROTOCOL)
       
                
            ####  INNER BATCH LOOP END
                
        # save
        print('time elapsed = ', time.time()-start_time)
       

        pickle.dump(Ulist, open( "%sngU_iter%d.p" % (savefile,iteration), "wb" ) , pickle.HIGHEST_PROTOCOL)
        pickle.dump(Vlist, open( "%sngV_iter%d.p" % (savefile, iteration), "wb" ) , pickle.HIGHEST_PROTOCOL)
