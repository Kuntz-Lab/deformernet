#!/usr/bin/env python3
import numpy as np
import pickle
from sklearn.decomposition import PCA

if __name__ == "__main__":
    with open("/home/baothach/shape_servo_data/extended_FPFH_vector_135.txt", 'rb') as f:
        FPFH_135_vectors = pickle.load(f)
    
    # for vector in FPFH_135_vectors:
        # print(np.array(vector))
    
    # print(np.array(FPFH_135_vectors).shape)

    X = np.array(FPFH_135_vectors)
    pca = PCA(n_components=30)
    pca.fit(X)

    Y = X[10].reshape(1, -1)
    # Y = np.random.rand(1,135)
    # print(Y)
    test_pca = pca.transform(Y)
    print(test_pca)
    # print(test_pca.shape)
