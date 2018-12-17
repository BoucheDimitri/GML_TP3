import scipy.io as sio

path = "/home/dimitribouche/Bureau/MVA/S1/GML/TP3/code_material_python"
from helper import *


def iterative_hfs(niter=20):
    # load the data   
    # a skeleton function to perform HFS, needs to be completed
    #  Input
    #  niter:
    #      number of iterations to use for the iterative propagation

    #  Output
    #  labels:
    #      class assignments for each (n) nodes
    #  accuracy
 
    mat = sio.loadmat(path + "/data/data_iterative_hfs_graph.mat")
    W, Y, Y_masked = mat["W"], mat["Y"], mat["Y_masked"]

    # Reshape Y_masked to a 1d array
    Y_masked = Y_masked[:, 0]

    # Change to positive/negative labels
    Y_masked[Y_masked == 1] = -1
    Y_masked[Y_masked == 2] = 1
    Y[Y == 1] = -1
    Y[Y == 2] = 1

    # Compute random walk transition matrix
    # We are summing each column so we access W columnwise, so it is fine
    D = np.array(W.sum(axis=0))[0, :]
    Dinv = scipy.sparse.csc_matrix(np.diag(1 / D))
    # In this dot, we only access W columnwise (also this is a dot product between two csc matrix and such product
    # Optimize in scipy.sparse.
    P = Dinv.dot(W)

    # Get unabelled indexes
    unlabelled = np.argwhere(Y_masked == 0)


    #####################################
    # Compute the initializion vector f #
    #####################################
    F = Y_masked.copy()

    #####################################
    #####################################

    accuracies = []
    #################################################################
    # compute the hfs solution, using iterated averaging            #
    # remember that column-wise slicing is cheap, row-wise          #
    # expensive and that W is already undirected                    #
    #################################################################
    for it in range(0, niter):
        for u in unlabelled:
            F[u] = np.array(F.dot(P[:, u].todense()))[0, 0]
        # Record accuracies to monitor convergence
        labels = np.sign(F)
        accuracies.append((labels == Y.reshape(-1)).mean())
        print(it)

    ################################################
    # Assign the label in {1,...,c}                #
    ################################################
    labels = np.sign(F)
    accuracy = (labels == Y.reshape(-1)).mean()
    
    ################################################
    ################################################
    return labels, accuracies
    

        
