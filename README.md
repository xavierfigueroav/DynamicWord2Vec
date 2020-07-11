# DynamicWord2Vec
Paper title:
Dynamic Word Embeddings for Evolving Semantic Discovery. 

Paper links:
https://dl.acm.org/citation.cfm?id=3159703
https://arxiv.org/abs/1703.00607

Files:

/embeddings
 - embeddings in loadable MATLAB files. 0 corresponds to 1990, 1 to 1991, ..., 19 to 2009.
 To save space, each year's embedding is saved separately. When used in visualization code, first merge to 1 embedding file.
 
/train_model
 - contains code used for training our embeddings
 - data file download: https://www.dropbox.com/s/nifi5nj1oj0fu2i/data.zip?dl=0
 
    /train_model/train_time_CD_smallnyt.py
     - main training script

    /train_model/util_timeCD.py
     - containing helper functions

/other_embeddings
 - contains code for training baseline embeddings
 - data file download: https://www.dropbox.com/s/tzkaoagzxuxtwqs/data.zip?dl=0
 
   /other_embeddings/staticw2v.py
    - static word2vec (Mikolov et al 2013)
    
   /other_embeddings/aw2v.py
    - aligned word2vec (Hamilton, Leskovec, Jufarsky 2016)
    
   /other_embeddings/tw2v.py
    - transformed word2vec (Kulkarni, Al-Rfou, Perozzi, Skiena 2015)
    
/visualization
 - scripts for visualizations in paper
 
   /visualization/norm_plots.py
    - changepoint detection figures
    
   /visualization/tsne_of_results.py
    - trajectory figures
    
/distorted_smallNYT
 - code for robust experiment
 - data file download: https://www.dropbox.com/s/6q5jhhmxdmc8n1e/data.zip?dl=0
 
/misc
 - contains general statistics and word hash file

## Set up the environment

1. Clone this repository: `git clone https://github.com/xavierfigueroav/DynamicWord2Vec.git`.

2. Change to the project directory: `cd DynamicWord2Vec`.

3. Create a Python 2 virtual environment in it and activate it. The steps to do it may change if you are in Windows, Linux or Mac. Google it :).

4. Install the dependencies: `pip install -r requirements.txt`


## Steps to reproduce

1. Create two new directories if absent: 'data' and 'results'.

2. 'data' must contain 3 types of files:

 - A single file named 'emb_static.mat'. This file contains the word embedding vectors learned from the whole corpus.
 - A single file named 'wordIDHash.csv'. This file contains every word in the corpus vocabulary as comma separeted pairs: id,word; where 'id' is a unique identifier for the 'word'. This file has no header.
 - N files with the name format 'wordPairPMI_N.csv'. There must be as many files as time frames you have, for example: if you are anually analysing documents from 2000 to 2003, you must have the files 'wordPairPMI_2000.csv', 'wordPairPMI_2001.csv', 'wordPairPMI_2002.csv', 'wordPairPMI_2003.csv'. Each file are lists of comma separated triples: word1,word2,ppmi; where word1 and word2 are word ids from the whole corpus vocabulary, and ppmi is the Positive Pointwise Mutual Information of word1 and word2 in the frame N. This file has the header 'word,context,pmi'.

3. Edit the file 'train_model/train_time_CD_smallnyt.py'. Set the variable `nw` to the number of words you have in the whole corpus, i.e., the size of your whole vocabulary. Set the variable `T` to a range Python object, from the beginning of your time frames to the end of them + 1, for example: if you are anually analysing documents from 2000 to 2003, `T = range(2000, 2004)`.

4. Run `python train_model/train_time_CD_smallnyt.py`. This may take several minutes. It will generate 10 files if you did not change the value of `ITERS` in 'train_model/train_time_CD_smallnyt.py'. The useful files are those whose names end with 'ngU_iter#.p', not those which end with 'ngV_iter#.p'. You may be only interested in the 'ngU_iter#.p' file with higher iter number, this is the file that contains the final aligned embeddings for each time frame.

5. 'visualization/tsne_of_results.py' is the file that generates the paper trajectory visualization. You need to transform the result obtained in the previous step for this script to work well. Change the variable `result` in 'visualization/transform_for_viz.py' as needed and the run `python visualization/transform_for_viz.py`. This will generate the file 'results/embs.mat' that will be used by 'visualization/tsne_of_results.py'.

6. Edit the file 'visualization/tsne_of_results.py'. Set the variable `times` to the same range as `T` in the step 3. Set the variable `emb` to `emb_all['U_%d' % times.index(X)]`; where X is the number of your last time frame, for example: if you are anually analysing documents from 2000 to 2003, `emb = emb_all['U_%d' % times.index(2003)]`. Set the variable `word` to the target word of which you want to analyse the trajectory.

7. Run `python visualization/tsne_of_results.py`.

Note: This process can be simplified a lot to make it painless. Coming soon.
