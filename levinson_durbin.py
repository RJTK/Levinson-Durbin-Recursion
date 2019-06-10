'''
Implementation of the Levinson Durbin recursion.
'''
def lev(r):
  '''
  We take in the left most row 'r' of a p by p autocorrelation
  (Hermitian Toeplitz positive semi-definite) matrix.  We return p
  coefficients a(1), a(2), ..., a(p) for an all pole model, as well as
  the modeling error epsilon and a sequence of p reflection
  coeffcients Gamma.

  Exactly the procedure described in Statistical Signal Processing by
  Monson Hayes (pg 219)
  '''
  r = [float(ri) for ri in r] #Ensure all the types are floats
  a = [1.]
  Gamma = []
  eps = r[0]
  for j in range(len(r) - 1):
    G = -sum(ai*ri for ai, ri in zip(a[::-1], r[1:j + 2]))/eps
    #Add 'a' and it's reverse multiplied by G
    a = [1.] + \
        [ai + G*aRi.conjugate() for ai, aRi in zip(a[1:], a[:-(j + 2):-1])] + \
        [G]
    eps = eps*(1 - abs(G)**2)
    Gamma.append(G) #Save the reflection coefficient sequence
  return a, Gamma, eps

def lev_inv(Gamma, eps):
  '''
  The inverse Levinson Durbin recursion.  Given the sequence of
  Gamma coefficients and the modeling error eps we reproduce the
  autocorrelation sequence which would have produced the given (Gamma,
  eps)
  '''
  from operator import mul #For product of a list
  rx = [eps / reduce(mul, Gamma, 1)]
  a = [1.]
  for j in range(len(G) - 1):
    #step up recursion here?
  return rx

def test1():
  '''simple sanity check'''
  import numpy as np
  rx = [1.0, 0.5, 0.5, 0.25]
  a, G, eps = lev(rx)
  assert np.allclose(a, [1.0, -3./8, -3./8, 1./8]), \
    'TEST1 FAILED TO PRODUCE CORRECT a VECTOR'
  assert np.allclose(G, [-1./2, -1./3, 1./8]), \
    'TEST1 FAILED TO PRODUCE CORRECT GAMMA VECTOR'
  assert np.isclose(eps, 21./32), \
    'TEST1 FAILED TO GET CORRECT MODELING ERROR'
  return

def test2():
  '''a second simple tests'''
  import numpy as np
  rx = [2., -1., -0.25, 0.125]
  a, G, eps = lev(rx)
  assert np.allclose(a, [1.0, 1.0, 7./8, 1./2]), \
    'TEST2 FAILED TO PRODUCE CORRECT a VECTOR'
  assert np.allclose(G, [0.5, 0.5, 0.5]), \
    'TEST2 FAILED TO PRODUCE CORRECT GAMMA VECTOR'
  assert np.isclose(eps, 2*(3./4)**3), \
    'TEST2 FAILED TO GET CORRECT MODELING ERROR'
  return

if __name__ == '__main__':
  test1()
  test2()
  test3()
