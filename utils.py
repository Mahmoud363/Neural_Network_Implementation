
import numpy as np
import cupy as cp

# generate a seeded random array to ensure all the weights are initialized to the same value
def random_seeded_2d(x,y):
  cp.random.seed(0)
  ret = cp.random.randn(x, y)
  return ret

def random_seeded_4d(h, w, c, n):
  cp.random.seed(0)
  ret = cp.random.randn(h, w, c, n)
  return ret