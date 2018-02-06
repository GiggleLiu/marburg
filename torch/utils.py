from __future__ import division
import numpy as np

def unpackbits(arr,nbit,axis=-1):
    '''unpack numbers to bits.'''
    nd=np.ndim(arr)
    if axis<0: axis=nd+axis
    res = ((arr & (1 << np.arange(nbit-1,-1,-1)).reshape([-1]+[1]*(nd-axis-1)))) > 0
    return 1-2*np.int8(res)

def packbits(arr, axis=-1):
    '''pack bits to numbers.'''
    arr = (1-arr)//2
    nd=np.ndim(arr)
    nbit = np.shape(arr)[axis]
    if axis<0: axis=nd+axis
    res = (arr*(1 << np.arange(nbit-1,-1,-1)).reshape([-1]+[1]*(nd-axis-1))).sum(axis=axis,keepdims=True).astype('int64')
    return res
