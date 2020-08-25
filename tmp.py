import numpy as np
import torch
a = np.array([[['t=0,s=0,f=0', 't=0,s=0,f=1'], ['t=0,s=1,f=0', 't=0,s=1,f=1'], ['t=0,s=2,f=0', 't=0,s=2,f=1']],
              [['t=1,s=0,f=0', 't=1,s=0,f=1'], ['t=1,s=1,f=0', 't=1,s=1,f=1'], ['t=1,s=2,f=0', 't=1,s=2, f=1']]])
print(a.reshape((a.shape[0], -1), order='A') == a.reshape((a.shape[0], -1), order='C'))
print(a.reshape((a.shape[0], -1), order='A') == a.reshape((a.shape[0], -1)))
