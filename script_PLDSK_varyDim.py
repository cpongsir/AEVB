from utils import *

T = sio.loadmat('TT.mat')['TT']
numTrial = T.size
yDim = 192
# xDim = 20
vDim = 2
learning_rate = 1e-3
batch_size = 20
numFolds = 10
n_epochs = 200

gen = 'PLDSKinematics'
rec = 'SmoothingLDSTimeSeries'
win_size = 1
use_kinematics = True

xDims = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

for xDim in xDims :

    for fold in range(numFolds):

        print '\nFOLD ', fold, '\n'

        name = 'PLDSK_xDim='+str(xDim)+'_fold='+str(fold)

        T_train, T_test = split_data(T, numFolds, fold)

        sgvb, sgvb_best = sgvb_init(gen, rec, win_size, xDim, yDim, vDim)

        run(sgvb, learning_rate, use_kinematics, T_train, T_test, n_epochs, name)
