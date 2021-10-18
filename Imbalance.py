# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Imblearn library
#
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN 



def makeOverSamplesSMOTE(X, y):
    '''
    Application of SMOTE for handling imbalanced data
    
    X: Independent Variable in DataFrame\
    y: dependent Variable in Pandas DataFrame format
    '''
    
    
    sm   = SMOTE()
    X, y = sm.fit_sample(X, y)
    
    return X, y



def makeOverSamplesADASYN(X, y):
    '''
    Application of ADASYN for handling imbalanced data
    
    X: Independent Variable in DataFrame\
    y: dependent Variable in Pandas DataFrame format
    '''
    
    
    
    sm   = ADASYN()
    X, y = sm.fit_sample(X, y)
    
    return(X,y)

