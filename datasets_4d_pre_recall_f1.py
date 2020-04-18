import numpy as np
from numpy import load
from numpy import save
import sklearn
import random 
import pdb
from sklearn.metrics import *
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from imblearn.combine import *
from imblearn.ensemble import *
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit
import torch

def evalSamplingp(ytest,ypred):
    return precision_score(ytest,ypred)
def evalSamplingr(ytest,ypred):
    return recall_score(ytest,ypred)
def evalSamplingf(ytest,ypred):
    return f1_score(ytest,ypred)


prediction_y = torch.load("datasets_4d_y_prediction.pt")#, map_location='cpu')#, allow_pickle = True)

# prediction_y = torch.load('datasets_4d_y_prediction_y.pt')
y_test_datasets_5d_resampled = np.load("datasets_4d_y_test_resampled.npy", allow_pickle = True)

# y_test_datasets_5d_resampled = load('../datasets_4d_y_test_resampled.npy')

print(len(prediction_y))#.size())
print(np.shape(y_test_datasets_5d_resampled))



data = []
data2 = []
data3 = []


matrixp =  np.empty((900*14,21))
matrixr =  np.empty((900*14,21))
matrixf1 =  np.empty((900*14,21))


c  = 0
for i in range(900):
    
    row = ["Dataset" + str(i+1)]
    rowdash = ["Dataset" + str(i+1)]
    row3 = ["Dataset" + str(i+1)]
    
    for j in range(21):
        ytest = y_test_datasets_5d_resampled[(i)*21+j]
        c = c + 1
        
        
        
        
        for k in range(14):
            ypred = (prediction_y[(i)*21*14 + k + 14*(j)].data).cpu().numpy()

   
            try:
                precision = evalSamplingp(ytest, ypred)
                matrixp[(i)*14 + k ][j] = round(precision,3)
                #print("precision" + str(t))
            except:
                t = -10.0


            try:
                recall = evalSamplingr(ytest, ypred)
                matrixr[(i)*14 + k ][j] = round(recall,3)
            except:
                t = -10.0


            try:

                f1 = evalSamplingf(ytest, ypred)
                matrixf1[(i)*14 + k ][j] = round(f1,3)
            except:

                t = -10.0


        

    
    

np.savetxt('datasets_nn_4d_precision.csv', matrixp.tolist() ,delimiter=',',fmt='%f') 
np.savetxt('datasets_nn_4d_recall.csv', matrixr.tolist() ,delimiter=',',fmt='%f') 
np.savetxt('datasets_nn_4d_f1.csv', matrixf1.tolist() ,delimiter=',',fmt='%f') 

