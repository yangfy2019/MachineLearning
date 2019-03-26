import numpy as np
import scipy.io as sio
from scipy import sparse
import scipy

def TrainHyperAttPredictors(Trains,TrainAttMat,method,params):
    #compute the Euclidean distance of each pair of samples.
    distance = dist2(Trains.T,Trains.T)

    #Raw Weights of hyperedge, each attribute is corresponding to a hyperedge
    attW = GetRawWeightByAffinity(distance,TrainAttMat)

    #the vertex-edge incident matrix is exactly the attribute label matrix
    H = TrainAttMat

    #Compute the Laplacian matrix of the attribute hypergraph
    mu = params['mu']
    lamb = params['lambda']
    gamma = params['gamma']
    L_att = ConstructHyperLaplacian(attW,TrainAttMat,mu)

    #Choose the Laplacian matrix based on the given method
    L = L_att

    if method=='HAP':
        #The basic model: Hypergraph-regularized Attribute Predictor: HAP
        #B = argmin {Trace(B^TXL_HX^TB)+\lambda ||X^TB-Y||^2+\gamma ||B||^2}
        #do nothing,
        None
    elif method=='CSHAP_G':
        #The Class Specific HAP with a penalty Graph: CSHAP_G
        #B = argmin {Trace(B^TXL_WX^TB)+\lambda ||X^TB-Y||^2+\gamma||B||^2}
        #where L_W = L_H + \eta L_* is the combined Laplacian matrix and L_* here
        #is a graph Lapalacian encodes class information.

        #load the class labels
        TrainLabel = params['labels']
        #set the eta
        eta = params['eta']
        L_lab = ConstructGraphLaplacian(distance,TrainLabel,mu)
        L  = L + eta*L_lab
    elif method=='CSHAP_H':
        #The Class Specific HAP with a penalty Graph: CSHAP_G
        #B = argmin {Trace(B^TXL_WX^TB)+\lambda ||X^TB-Y||^2+\gamma||B||^2}
        #where L_W = L_H + \eta L_* is the combined Laplacian matrix and L_* here
        #is a hypergraph Lapalacian encodes class information.

        # load the class labels
        TrainLabel = params['labels']
        # set the eta
        eta = params['eta']

        LabelMat = getLabelMatrix(TrainLabel)
        LabelMat[LabelMat==-1]  = 0
        #raw weights of hyperedges of the graph which encodes the class information
        LabelW = GetRawWeightByAffinity(distance, LabelMat)
        L_lab = ConstructHyperLaplacian(LabelW, LabelMat, mu)
        L = L + eta * L_lab
    else:
        print('Invalid algorithm, use HAP instead!')

    P = LearnGraphPredictors(L,Trains,H,lamb,gamma)
    return P

def getLabelMatrix(labels):
    libs = scipy.unique(labels)
    libs = libs[libs!=-1]
    (temp,lnum) = labels.shape
    cnum = len(libs)
    Lmat = scipy.zeros((lnum,cnum))
    for i in range(cnum):
        pind = scipy.where(labels==libs[i])
        nind = scipy.where(labels!=libs[i])
        Lmat[pind[1],i] = 1
        Lmat[nind[1],i] = -1
    return Lmat

def LearnGraphPredictors(L,X,H,lamb,gamma):
    #Learning the Graph (or Hypergraph)-based attribute predictors
    print('------------------------------------------------------------------------')
    print('Learning the Graph (or Hypergraph)-based attribute predictors')
    #Generate the Label matrix using edge-vertex incident matrix
    L = sparse.csc_matrix(L)
    X = sparse.csc_matrix(X)
    Y = 2.0*H-1.0

    Y = sparse.csc_matrix(Y)
    TempMat = scipy.dot(scipy.dot(X,L),X.T) + lamb * scipy.dot(X,X.T)   #pass
    local = scipy.mean(scipy.diag(TempMat.toarray()))       #pass
    #Learning the predictors by solving the issue as Least Square issue
    (row,col) = TempMat.shape
    I = (TempMat + local*gamma*scipy.eye(row,col)).I    #pass
    I = sparse.csc_matrix(I)
    P = scipy.dot(I,lamb*scipy.dot(X,Y))
    return P

def ConstructGraphLaplacian(W,labels,u):
    #Constructing the graph Laplacian matrix
    print('------------------------------------------------------------------------')
    print('Constructing the graph Laplacian matrix')
    meanW = scipy.mean(W)
    W = W/meanW
    W = scipy.exp(-W/u)
    Temp = scipy.zeros(W.shape)
    Label = scipy.unique(labels)
    for i in Label:
        inds = scipy.where(labels[0]==i)
        for m in inds[0]:
            for n in inds[0]:
                Temp[m,n] = W[m,n]
    W = Temp
    D = W.sum(axis=0)
    L = scipy.diag(D) - W
    D_h = scipy.sqrt(D)
    Dd = 1/D_h
    Lap = scipy.dot(scipy.dot(scipy.diag(Dd),L),Dd)
    return Lap

def ConstructHyperLaplacian(W,H,mu):
    #Constructing the Hypergraph Laplacian matrix
    print('------------------------------------------------------------------------')
    print('Constructing the Hypergraph Laplacian matrix')
    inds = scipy.where(W!=scipy.inf)
    W = W[inds[0]]
    H = H[:,inds[0]]
    #normalize the raw weights and kernelize them to present the final weights
    temp = scipy.mean(W)*mu
    W = scipy.exp(-W/(scipy.mean(W)*mu))
    # the diagonal weight matrix
    W = scipy.diag(W)
    #the diagonal vertex degree matrix
    Ve = scipy.diag(H.sum(axis=0))

    #zhou's method for constructing the normalized Laplacian of a hypergraph
    Ve = scipy.diag(1/scipy.diag(Ve))

    Dv = ((scipy.dot(H,W)).T).sum(axis=0)
    NewW = scipy.dot(scipy.dot(scipy.dot(H,W),Ve),H.T)
    Dv_I = 1/Dv
    Dv_I = scipy.diag(Dv_I)
    Lap = scipy.dot(scipy.dot(scipy.sqrt(Dv_I),(scipy.diag(Dv)-NewW)),scipy.sqrt(Dv_I))
    return Lap


def GetRawWeightByAffinity(AMatrix,AttMat):
    #Generating the raw weights of hyperedges via using the affinity matrix
    print('------------------------------------------------------------------------')
    print('Generating the raw weights of hyperedges via using the affinity matrix')
    (row,col) = AttMat.shape
    W = scipy.zeros(col)
    for i in range(col):
        ind = scipy.where(AttMat[:,i]!=0)
        length = len(ind[0])
        if length!=1 and length!=0:
            sum = 0
            for m in ind[0]:
                for n in ind[0]:
                    sum += AMatrix[m,n]
            W[i] = sum/(length*(length-1))
        else:
            W[i] = scipy.inf
    return W

def dist2(x,c):
    #Calculate squared distance between two sets of points
    print('------------------------------------------------------------------------')
    print('Calculate squared distance between two sets of points')
    (rowx,colx) = x.shape
    (rowc,colc) = c.shape
    if colc != colx:
        print('Data dimension does not match dimension of centres')
        return

    x2T = sparse.csc_matrix(np.multiply(x.toarray(),x.toarray())).T
    c2T = sparse.csc_matrix(np.multiply(c.toarray(),c.toarray())).T
    a2 = (scipy.dot(scipy.ones((rowc,1)),x2T.sum(axis=0))).T
    b2 = scipy.dot(scipy.ones((rowx,1)),c2T.sum(axis=0))
    ab2 = 2*scipy.dot(x,c.T)

    distance = a2 + b2 -ab2
    return distance

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
    print('.')
    print('.')
    #paramaters setting
    params = {
        'mu':1,
        'lambda':1000,
        'gamma':10,
        'eta':1,
        'labels':train_data['TrainLabel']
    }
    print('########################################Training HAP###########################################')
    P = TrainHyperAttPredictors(train_data['Trains'], train_data['TrainAttMat'], 'HAP', params)
    print('########################################Training CSHAP_G###########################################')
    PG = TrainHyperAttPredictors(train_data['Trains'], train_data['TrainAttMat'], 'CSHAP_G', params)
    print('########################################Training CSHAP_H###########################################')
    PH = TrainHyperAttPredictors(train_data['Trains'], train_data['TrainAttMat'], 'CSHAP_H', params)

    print('.')
    print('.')
    print('########################################Testing Part###########################################')

    print('------------------------------------------------------------------------')
    print('Calculate Test Attributes')
    #getting the attribute predictions InputAtt where Tests is a test sample
    #matrix, Y = X'* B, Y = attribute predictions, X = testing samples, B =
    #the learned predictors.
    Tests = test_data['Tests']
    Labels = sparse.csc_matrix(test_data['TestAttMat']).toarray()
    (sampleNum,labelNum) = Labels.shape

    #get every attribute prediction accuracy on AWA
    print('------------------------------------------------------------------------')
    print('---------------------------HAP--------------------------------')
    InputAtt = scipy.dot(Tests.T,P)
    InputAtt = sparse.csc_matrix(InputAtt).toarray()
    InputAtt[InputAtt>0.0] = 1
    InputAtt[InputAtt<=0.0] = 0
    HAP_accuracy = (sampleNum - (np.abs(InputAtt-Labels)).sum(axis=0))/sampleNum
    print('average attribute prediction accuracy of HAP: ',np.mean(HAP_accuracy))
    print('---------------------------CSHAP_G--------------------------------')
    InputAttG = scipy.dot(Tests.T, PG)
    InputAttG = sparse.csc_matrix(InputAttG).toarray()
    InputAttG[InputAttG>0.0] = 1
    InputAttG[InputAttG<=0.0] = 0
    CSHAP_G_accuracy = (sampleNum - (np.abs(InputAttG-Labels)).sum(axis=0))/sampleNum
    print('average attribute prediction accuracy of CSHAP_G: ', np.mean(CSHAP_G_accuracy))
    print('---------------------------CSHAP_H--------------------------------')
    InputAttH = scipy.dot(Tests.T, PH)
    InputAttH = sparse.csc_matrix(InputAttH).toarray()
    InputAttH[InputAttH>0.0] = 1
    InputAttH[InputAttH<=0.0] = 0
    CSHAP_H_accuracy = (sampleNum - (np.abs(InputAttH-Labels)).sum(axis=0))/sampleNum
    print('average attribute prediction accuracy of CSHAP_H: ', np.mean(CSHAP_H_accuracy))



