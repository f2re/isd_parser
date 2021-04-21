import numpy as np
from pprint import pprint

from scipy import interpolate
import matplotlib.pyplot as plt
import plotly.graph_objs as go

pic93V_x = np.array([2.0, 4.0, 5.0, 8.0,16.0,20.0,  
            3.0, 4.5,10.0,16.0,20.0, 
            2.5, 6.0,14.0,16.0,20.0,    
            3.0, 6.0,10.0,14.0,16.0,20.0,    
            3.5, 5.0,12.0,16.0,20.0,    
            4.0,10.5,16.0,20.0,    
            3.0, 4.0,12.0,16.0,20.0,    
            2.0,10.0,14.0,20.0])
pic93V_y = np.array([2.0, 4.0, 5.0, 8.0,16.0,20.0,  
            2.8, 4.0, 8.8,14.0,16.5, 
            2.0, 4.8,10.8,12.1,14.0,   
            2.0, 4.0,6.2, 8.2,9.1,10.0,   
            2.0, 2.8, 5.8, 6.6, 6.9,    
            1.9, 4.0, 5.0, 4.8,    
            1.0, 1.2, 3.0, 3.5, 3.1,    
            0.2, 1.2, 1.8, 1.3])
pic93V_z = np.array([0.0, 0.0, 0.0, 0.0,0.0,0.0,  
            0.5, 0.5, 0.5, 0.5,0.5,  
            1.0, 1.0, 1.0, 1.0,1.0,
            2.0, 2.0,2.0, 2.0, 2.0,2.0,
            3.0, 3.0, 3.0, 3.0,3.0,
            5.0, 5.0, 5.0, 5.0,
           10.0,10.0,10.0,10.0,10.0,
           15.0,15.0,15.0,15.0])
f = interpolate.interp2d(pic93V_x,pic93V_y,pic93V_z, kind='linear')
print(  f([6.0],[1.0]))

exit()

arr     = np.zeros((12))
pprint(arr)
arr = np.reshape( arr, (-1,3) )
arr[:,0]  = 1
arr[:,1]  = 2
arr[:,2]  = 3
pprint(arr)
pprint(arr[:,0])

exit()


arr     = np.zeros((5,1))
arr[0]  = 1
arr2    = np.zeros((5,1))
arr2[0] = 2
m = np.concatenate( (arr, arr2),axis=1 )
# arr = np.reshape(arr,(-1) ) 
pprint(m)

exit()

n=3
# arr = [[[0]*n]*n]*n
arr    = np.zeros((n,10,1))
arr[1] = 1
arr[2] = 2

fl = arr.reshape(3,10)
fl[0] = 11
fl = np.flip( fl )
pprint( fl )

pprint( np.transpose( np.flip( arr.reshape(3,10)) ) )

exit()

arr = np.array([])

for i in range(0,3):
  _i = np.array([])
  _i=_i.reshape(1,-1)
  print(_i)
  for j in range(1,10):
    _j = np.array([i]).reshape(1,-1)
    print(_i,_j.reshape(1,-1) )
    _i=np.vstack(  (_i,_j ) )
    print(_i.reshape(1,-1))
  arr=np.vstack( (arr,_i) )

print(arr)