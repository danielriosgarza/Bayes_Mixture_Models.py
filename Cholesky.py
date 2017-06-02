from __future__ import division
import numpy as np
import math
import scipy.linalg as linalg
trm = linalg.get_blas_funcs('trmm') #multiply triangular matrices. If lower use lower=1.

try:
    import pychud
    pychud_im =True
except ImportError:
    pychud_im = False
#https://github.com/danielriosgarza/pychud
#sudo apt-get install gfortran
#python setup.py install



class Cholesky:
    '''Methods for the Cholesky decomposition of positive matrices.
    Must be instantiated with a square matrix, called A.
    This matrix can be a full positive definite matrix or a Cholesky upper/lower decomposition or some approximation.
    'method' defines the kind of input matrix. Provide one of the terms: 'full' (default), 'upper', 'lower', 
    or 'approximate'. Potential linear algebra errors imply an approximate matrix, the Cholesky object tries
    to figure this out and if verbose is on, it prints a warning.
    
        '''
    
    def __init__(self, A, method='full', verbose=True):
        self.method = self.__method(method)
        self.d = len(A)
        self.verbose=verbose
        self.lower = None
        self.upper = None
        self.matrix = self.__mat(A, method)
        self.inv=None
        self.lower_inv = None
        
        
    
    def __method(self, method='full'):
        assert method=='full' or method=='lower' or method=='upper' or method=='approximate',\
        "\n\n\nError: supported methods are 'full', 'lower', 'upper', and  'approximate'"
        self.method=method
        return self.method
    
    def __mat(self, A, method='full'):
        assert A.shape==(self.d, self.d), "\n\n\nError: A must be a square matrix\n\n\n"
        if method=='full':
            if np.allclose(A, A.T):
                try:
                    self.lower = linalg.cholesky(A, lower=1, check_finite=0, overwrite_a=0)
                    self.upper = self.lower.T
                    self.matrix = A
                except np.linalg.LinAlgError:
                    self.matrix = A
                    self.method = 'approximate'
                    self.lower = self.lower_semidefinite()
                    self.upper = self.lower.T
            else:
                self.matrix=A
                self.method = 'approximate'
                self.lower = self.lower_semidefinite()
                self.upper = self.lower.T

        elif method =='lower':
            assert np.all(A[np.triu_indices(self.d, 1)]==0), \
            "\n\n\nError: if method is 'lower', a lower triangular matrix is required\n\n\n"
            if np.all(np.diag(A)>0):
                self.matrix=trm(alpha=1, a=A, b=A.T,lower=1)
                self.lower = A
                self.upper = self.lower.T
            else:
                self.matrix=trm(alpha=1, a=A, b=A.T,lower=1)
                self.lower = self.lower_semidefinite()
                self.upper = self.lower.T
                self.method='approximate'
                
        elif method =='upper':
            assert np.all(A[np.tril_indices(self.d, -1)]==0), \
            "\n\n\nError: if method is 'upper', an upper triangular matrix is required\n\n\n"
            if np.all(np.diag(A)>0):
                self.matrix=trm(alpha=1, a=A.T, b=A,lower=1)
                self.upper = A
                self.lower = self.upper.T
            else:
                self.matrix=trm(alpha=1, a=A.T, b=A,lower=1)
                self.lower = self.lower_semidefinite()
                self.upper = self.lower.T
                self.method='approximate'
                    
        return self.matrix
                    
        
            
            
    
    def lower_semidefinite(self):
        '''returns the approximation of the cholesky decomposition for positive semi-definite matrices
        and aproximately positive definite matrices. Notice that this is an approximation to a matrix that is singular, 
        so the inverse will not result in the identity A*inv(A)=I'''
        if self.verbose:
            print '\x1b[5;31;46m'+'Warning: A is not positive definite. Applied approximate method that could be wrong.'+ '\x1b[0m'
        a,b = linalg.eigh(0.5*(self.matrix+self.matrix.T))
        a[a<0]=0
        a = np.diag(a)
        B = b.dot(a).dot(b.T)
        return linalg.cholesky(B+np.eye(self.d)*0.00000000001, lower=1, check_finite=0,overwrite_a=0)


    
    def  __chol_of_the_inv(self):
        '''return the Cholesky decompostion of the inverse of A'''
        v1 = linalg.solve_triangular(self.lower, np.eye(self.d), lower=1,trans=0, overwrite_b=0,check_finite=0) 
        
        if self.method=='approximate':
            try:
                self.inv = np.linalg.inv(self.matrix)
            except np.linalg.LinAlgError:
                self.inv = linalg.solve_triangular(self.upper, v1, lower=0,trans=0, overwrite_b=0,check_finite=0)
                if self.verbose:
                     print '\x1b[5;31;46m'+'Warning: used approximation for the inverse'+ '\x1b[0m'
            
            try:
                self.lower_inv = linalg.cholesky(self.inv, lower=1, check_finite=0, overwrite_a=0)
            except np.linalg.LinAlgError:
                v2 = linalg.solve_triangular(self.upper, v1, lower=0,trans=0, overwrite_b=1,check_finite=0)
                self.lower_inv = linalg.cholesky(v2, lower=1, check_finite=0, overwrite_a=0)
                if self.verbose:
                     print '\x1b[5;31;46m'+'Warning: used approximation for the Cholesky of the inverse'+ '\x1b[0m'
    
        else:
            self.inv = linalg.solve_triangular(self.upper, v1, lower=0,trans=0, overwrite_b=1,check_finite=0)
            self.lower_inv = linalg.cholesky(self.inv, lower=1, check_finite=0, overwrite_a=0)
        
        return self.lower_inv
        

    def __inv(self):
        '''return the inverse of A'''
        if self.lower_inv is None:
            self.lower_inv = self.__chol_of_the_inv()
        
        return self.inv
        

    def r_1_update(self, X, pychud_im = pychud_im):
        '''Returns a rank 1 update to the lower Cholesky decomposition of A. 
        Returns the cholesky decomposition of A*, where A*=A+XX' and X is a d-dimensional vector.'''

        chol_A = self.lower

        if pychud_im:
            return pychud.dchud(chol_A.T, X, overwrite_r=False).T
        else:
            for k in xrange(self.d):
                r = math.sqrt((chol_A[k,k]**2)+(X[k]**2))
                c = r/chol_A[k,k]
                s = X[k]/chol_A[k,k]
                chol_A[k,k] = r
                for i in xrange(k+1, self.d):
                    chol_A[i, k] = (chol_A[i,k]+ (s*X[i]))/c
                    X[i] = (X[i]*c)-(s*chol_A[i,k])
        return chol_A
        
    def r_1_downdate(self, X, pychud_im = pychud_im):
        '''Perform a rank 1 downdate to the lower Cholesky decomposition of A. 
        Returns the cholesky decomposition of A*, where A*=A-XX' and X is a d-dimensional vector.
        The pychud method is faster and more stable'''
        chol_A = self.lower
        if self.verbose:
            print '\x1b[5;31;46m'+'Warning: downdate assumed to be positive definite. But not guaranteed'+ '\x1b[0m'
        if pychud_im:
            return pychud.dchdd(chol_A.T, X, overwrite_r=False)[0].T
        else:
            for k in xrange(self.d):
                a=abs(chol_A[k,k])
                b=abs(X[k])
                num = min([a,b])
                den = max([a,b])
                t = num/den
                r = den*math.sqrt(1-(t*t))
                c = r/chol_A[k,k]
                s = X[k]/chol_A[k,k]
                chol_A[k,k] = r
                for i in xrange(k+1, self.d):
                    chol_A[i, k] = (chol_A[i,k]- (s*X[i]))/c
                    X[i] = (X[i]*c)-(s*chol_A[i,k])
        
        
        return chol_A
    
    def log_determinant(self):
        '''returns the log of the determinant of the A computed from its Cholesky decomposition'''
        if self.method=='approximate':
            try:
                self.log_det = np.linalg.slogdet(self.matrix)[1]
            except np.linalg.LinAlgError:
                self.log_det = sum(2*np.log(np.diag(self.lower)))
                print '\x1b[5;31;46m'+'Warning: used approximation for the determinant'+ '\x1b[0m'
        else:
            self.log_det = sum(2*np.log(np.diag(self.lower)))
        
        return self.log_det
    
    def determinant(self):
        '''returns the determinant of the A computed from its Cholesky decomposition'''
        self.det=math.exp(self.log_determinant())
        return self.det
        






    
