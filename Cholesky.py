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
    Must be instantiated with a square matrix.
    Operations on Cholesky decompositions can be called with a user supplied 
    Cholesky decomposition or with a positive definite (or semi-definite)
    matrix A.
    If A is positive definite or semi-definite matrix or approximately positive definite, 
    then:
    > Cholesky(A).upper() #returns the upper triangular Chol. dec. of A.
    > Cholesky(A).lower() #returns the lower triangular Chol. dec. of A.
    
    If cA is the Cholesky decomposition of A, then, for example:
    > Cholesky(cA).chol_of_the_inv(cA) #returns the Chol. dec. of the inverse of A
    From A:
    > Cholesky(A).chol_of_the_inv() #returns the Chol. dec. of the inverse of A
        '''
    
    def __init__(self, A, verbose=True):
        self.matrix = A
        self.d = len(A)
        self.verbose=verbose

    def mat(self, chol_A=None, config='lower'):
        '''return the orginal matrix when given an upper or lower Cholesky decomposition.'''

        if chol_A is None:
            return self.matrix
        elif config=='lower':
            return trm(alpha=1, a=chol_A, b=chol_A.T,lower=1)
        else:
            return trm(alpha=1, a=chol_A.T, b=chol_A,lower=1)

    def upper(self):
        '''return the upper triangular Cholesky decompostion of A.
        If A is a postive definite matrix, and U is its upper triangular Cholesky decomposition.
        Then A = U'U. The function can handle positive semi-definite matrices or matrices approximately
        positive definite, providing an approximation.'''
        try:
            return linalg.cholesky(self.matrix, lower=0, check_finite=0, overwrite_a=0)
        except linalg.LinAlgError:
            return self.lower_semidefinite().T
        
    def lower(self):
        '''return the lower triangular Cholesky decompostion of A.
        If A is a postive definite matrix, and L is its lower triangular Cholesky decomposition.
        Then A = LL'. The function can handle positive semi-definite matrices or matrices approximately
        positive definite, providing an approximation.'''
        
        try:
            return linalg.cholesky(self.matrix, lower=1, check_finite=0, overwrite_a=0)
        except linalg.LinAlgError:
            return self.lower_semidefinite()

    def lower_semidefinite(self):
        '''returns the approximation of the cholesky decomposition for positive semi-definite matrices
        and aproximately positive definite matrices. Notice that this is an approximation to a matrix that is singular, 
        so the inverse will not result in the identity A*inv(A)=I'''
        if self.verbose:
            print '\x1b[5;31;46m'+'Warning: A is not positive definite. Applied positive semidefinite method that can result in a wrong result.'+ '\x1b[0m'
        a,b = linalg.eigh(0.5*(self.matrix+self.matrix.T))
        a[a<0]=0
        a = np.diag(a)
        B = b.dot(a).dot(b.T)
        return linalg.cholesky(B+np.eye(self.d)*0.00000000001, lower=1, check_finite=0,overwrite_a=0)


    
    def __chol_A(self, chol_A=None):
        '''Check if a Cholesky decompostion is provided. Otherwise take the decomposition of A'''
        if chol_A is None:
            try:
                return self.lower()
            except linalg.LinAlgError:
                return self.lower_semidefinite()
        else:
            return chol_A

    def  chol_of_the_inv(self, chol_A=None):
        '''return the Cholesky decompostion of the inverse of A'''
        chol_A = self.__chol_A(chol_A)
        v1 = linalg.solve_triangular(chol_A, np.eye(self.d), lower=1,trans=0, overwrite_b=1,check_finite=0)
        v2 = linalg.solve_triangular(chol_A.T, v1, lower=0,trans=0, overwrite_b=1,check_finite=0)
        return linalg.cholesky(v2, lower=1, check_finite=0, overwrite_a=1)

    def inv(self, chol_A=None):
        '''return the inverse of A'''
        chol_A = self.__chol_A(chol_A)
        self.choliA = self.chol_of_the_inv(chol_A)
        return trm(alpha=1, a=self.choliA, b=self.choliA.T,lower=1)

    def r_1_update(self, X, chol_A=None, pychud_im = pychud_im):
        '''Perform a rank 1 update to the lower Cholesky decomposition of A. 
        Returns the cholesky decomposition of A*, where A*=A+XX' and X is a d-dimensional vector.'''

        chol_A = self.__chol_A(chol_A)
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
        
    def r_1_downdate(self, X, chol_A=None, pychud_im = pychud_im):
        '''Perform a rank 1 downdate to the lower Cholesky decomposition of A. 
        Returns the cholesky decomposition of A*, where A*=A-XX' and X is a d-dimensional vector.
        The pychud method is faster and more stable'''
        chol_A = self.__chol_A(chol_A)
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
    
    def log_determinant(self, chol_A=None):
        '''returns the log of the determinant of the A computed from its Cholesky decomposition'''
        chol_A = self.__chol_A(chol_A)
        return sum(2*np.log(np.diag(chol_A)))
    
    def determinant(self, chol_A=None):
        '''returns the determinant of the A computed from its Cholesky decomposition'''
        chol_A = self.__chol_A(chol_A)
        return math.exp(self.log_determinant(chol_A))
        






    
