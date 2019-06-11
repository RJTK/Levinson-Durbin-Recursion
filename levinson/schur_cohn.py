'''
Implementation of the Schur Cohn stability test.
'''
import numpy as np

def schur_cohn(a):
  '''
  Applies the Schur-Cohn stability test to the polynomial given by
  a = [1, a(1), ..., a(p)]

  A(z) = 1 + a(1)z^-1 + ... + a(p)z^-p

  This procedure is much more accurate than testing the zeros of a.

  Returns
    True - If 'a' is stable (no roots outside the unit circle)
    False - Otherwise (has roots outside the unit circle)
  '''
  a = np.array(a)
  a = a / a[0] #Ensure that a[0] = 1
  while len(a) > 1:
    a, G = a[:-1], a[-1]
    if abs(G) >= 1:
      return False
    s = 1 / (1 - abs(G)**2)
    a[1:] = s*(a[1:] - G*np.conjugate(a[:0:-1]))
  
  return True

def test1():
  '''simple check on stable polynomial'''
  zeros = [0.25, 0.5, 0.75]
  a = np.array([1.])
  for z in zeros:
    a = np.polymul(a, [1, -z])

  assert schur_cohn(a), 'TEST1 FAILED - POLYNOMIAL SHOULD BE STABLE'
  return

def test2():
  '''simple check on unstable polynomial'''
  zeros = [0.25, 1.25]
  a = np.array([1.])
  for z in zeros:
    a = np.polymul(a, [1, -z])

  assert not schur_cohn(a), 'TEST2 FAILED - POLYNOMIAL SHOULD BE UNSTABLE'
  return

def test3():
  '''randomized tests on stable polynomials'''
  from scipy import stats
  import math
  import cmath
  np.random.seed(314)
  num_trials = 1000
  num_zeros = 50
  for k in range(num_trials): #100 trials
    zeros = []
    for i in range(num_zeros): #10 zeros
      r = math.sqrt(stats.uniform.rvs(loc = 0, scale = 0.9))
      theta = stats.uniform.rvs(loc = -np.pi, scale = 2*np.pi)
      zeros.append(cmath.rect(r, theta))

    from scipy import signal
    a = np.array([1.])
    for z in zeros:
      a = np.polymul(a, [1, -z])
    
    assert schur_cohn(a), 'TEST3 FAILED, POLYNOMIAL SHOULD BE STABLE'
  return

def test4():
  '''randomized tests on unstable polynomials'''
  from scipy import stats
  import cmath
  import math
  np.random.seed(314)
  num_trials = 1000
  num_zeros = 50
  for k in range(num_trials): #100 trials
    zeros = []
    for i in range(num_zeros - 1): #10 zeros
      r = math.sqrt(stats.uniform.rvs(loc = 0, scale = 1.5))
      theta = stats.uniform.rvs(loc = -np.pi, scale = np.pi)
      zeros.append(cmath.rect(r, theta))

    #Should ensure we actually get atleast one unstable root
    r = math.sqrt(stats.uniform.rvs(loc = 1.1, scale = 2))
    theta = stats.uniform.rvs(loc = -np.pi, scale = np.pi)
    zeros.append(cmath.rect(r, theta))

    a = np.array([1.])
    for z in zeros:
      a = np.polymul(a, [1, -z])
    
    assert not schur_cohn(a), 'TEST3 FAILED, POLYNOMIAL SHOULD BE UNSTABLE'
  return


if __name__ == '__main__':
  test1()
  test2()
  test3()
  test4()
