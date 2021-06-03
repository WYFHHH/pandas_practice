import numpy as np
import pandas as pd

a=np.array([1,3,5])

b=np.array([[1,2,3],[4,5,6]])
a+b

np.zeros((3, 5))

np.zeros((2,1))

a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

print(a[1:])

a = np.arange(1, 13).reshape(3,4)
a
a[1,1]=2
a
a = np.arange(1, 10).reshape(3,3)
a
b = np.array([['*', 0, '#']])

def mapper(n):
    dicts = {'*': (3, 0), 0:(3,1), '#':(3,2)}
    if n=='*' or n==0 or n=='#':
        return dicts[n]
mapper('*')


A = [0,0]
B = [0,0]
def calc_dist(A, B):
    l = [abs(a-b) for a, b in zip(A, B)]
    print(sum(l))

calc_dist(A, B)


def calc_dist(A, B):
    l=[abs(a-b) for a, b in zip(A, B)]
    return sum(l)


hand = 'right'
main_hand = (hand[0]).upper()
main_hand