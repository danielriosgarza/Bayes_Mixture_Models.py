from __future__ import division
import numpy as np
import math
import random

from Cholesky import Cholesky
from Gauss_Wishart_probability_model import Gauss_Wishart_model
from Gaussian_component import Gaussian_component
from sklearn.cluster import KMeans as km

class Gaussian_clusters:
    def __init(self, k=1, Xm=None, alpha=None, nk=None):
        self.k=k
        self.Xm=Xm
        self.alpha=alpha
    
    
    def __nk(self, nk=None):
        assert len(nk)==len(self.Xm)
        if nk is None:
            self.nk = km(n_clusters=self.k).fit_predict(self.Xm)
    
        
