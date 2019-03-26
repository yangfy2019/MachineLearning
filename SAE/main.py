import numpy as np
import scipy.io as sio
import scipy
from scipy import linalg
from scipy import sparse

def SAE(X,S,lam):
    #For the error:Memory Error
    #We use csc_matrix to take place of ndarray
    S = scipy.transpose(S)
    S = sparse.csc_matrix(S,dtype='float16')
    A = np.dot(S,S.T)
    B = lam * np.dot(X,X.T)
    C = (1+lam) * np.dot(S,X.T)

    A = sparse.csc_matrix(A).toarray()
    B = sparse.csc_matrix(B).toarray()
    C = sparse.csc_matrix(C).toarray()

    W = linalg.solve_sylvester(A, B, C)     #Solve the Sylvester equation
    return W

if __name__=='__main__':
    print('########################################Loading Data###########################################')
    train_path = 'data/train/AWA_Trains_subset.mat'
    test_path = 'data/test/AWA_Tests_subset.mat'
    print('---------load train data------------')
    train_mat = sio.loadmat(train_path)
    train_data = {
        'TrainAttMat':train_mat['TrainAttMat'],
        'TrainLabel':train_mat['TrainLabel'],
        'Trains':train_mat['Trains']
    }
    print('TrainAttMat: ',train_data['TrainAttMat'].shape)
    print('TrainLabel: ',train_data['TrainLabel'].shape)
    print('Trains: ',train_data['Trains'].shape)
    print('---------load test data------------')
    test_mat = sio.loadmat(test_path)
    test_data = {
        'TestAttMat': test_mat['TestAttMat'],
        'TestLabel': test_mat['TestLabel'],
        'Tests': test_mat['Tests']
    }
    print('TestAttMat: ', test_data['TestAttMat'].shape)
    print('TestLabel: ', test_data['TestLabel'].shape)
    print('Tests: ', test_data['Tests'].shape)

    print('########################################Training Part###########################################')
    W = SAE(train_data['Trains'],train_data['TrainAttMat'],lam=0.1)
    Encoding_W = sparse.csc_matrix(W,dtype='float16')
    Decoding_W = sparse.csc_matrix(W.T, dtype='float16')
    print('Encoding Matrix Shape: ',Encoding_W.shape)
    print('Decoding Matrix Shape: ', Decoding_W.shape)

    print('########################################Testing Part###########################################')
    print('---------Testing with Encoding projection matrix---------')
    test_matrix = test_data['Tests']
    test_att = test_data['TestAttMat']
    Prediction_Att = np.dot(Encoding_W,test_matrix)
    Prediction_Att = sparse.csc_matrix(Prediction_Att).toarray()
    Prediction_Att[Prediction_Att>0]  = 1
    Prediction_Att[Prediction_Att<=0] = 0
    error = abs(Prediction_Att.T - test_data['TestAttMat'])
    (row,col) = error.shape
    accuracy = (row - error.sum(axis=0))/row
    print(accuracy)

