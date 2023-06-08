import numpy as np
from tqdm import tqdm, trange
from scipy.spatial.distance import cdist

def compute_AP(ranking):
    ap = 0
    num_positive = np.sum(ranking)
    rec_step = 1./num_positive
    cur_num_positive = 0
    for i in range(ranking.shape[0]):
        label = ranking[i]
        if label == 1:
            cur_num_positive += 1
            prec = cur_num_positive / (i+1)
            ap += prec*rec_step
            if cur_num_positive == num_positive:
                break
    return ap

def compute_mAP(ranking_matrix):
    mAP = []
    for i in trange(ranking_matrix.shape[0], leave = False):
        mAP.append(compute_AP(ranking_matrix[i]))
        # print("{:.1f}".format(mAP[-1]*100.),ranking_matrix[i])
    # print(["{:.1f}".format(ap*100.) for ap in mAP])
    return np.mean(mAP)

def ranking_mAP(latent_pair, label):
    """
    Compute the mean Average Precision (mAP) score,
    which jointly considers the ranking information and precision.
    A widely-used performance evaluation criterion in the
    research on cross-modal retrieval.
    """
    print("Computing ranking mAP")
    label_matrix = label.reshape(-1,1) == label.reshape(1,-1)
    label_matrix_ranking = np.zeros_like(label_matrix,dtype=np.int32) # (N, N)
    
    dists = cdist(latent_pair[0], latent_pair[1], metric="cosine") # (N, N)
    for i in range(label_matrix_ranking.shape[0]):
        label_matrix_ranking[i] = label_matrix[i][np.argsort(dists[i])]
    mAP = compute_mAP(label_matrix_ranking)
    return mAP