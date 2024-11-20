import numpy as np


# 1_ 5 Path at edge level :
def path1(A):
    return A    

def path2(A):
    A2 = A@A
    I = np.eye(A.shape[0])
    J = np.ones(A.shape) - I
    return A2*J

def path3(A):
    A2 = A@A
    A3 = A2@A
    I = np.eye(A.shape[0])
    J = np.ones(A.shape) - I
    A2I =A2*I
    return A3*J - A@A2I - A2I@A + A 

def path4(A):
    A2 = A@A
    A3 = A2@A
    A4 = A3@A
    I = np.eye(A.shape[0])
    J = np.ones(A.shape) - I
    A2I = A2*I
    A2J = A2*J
    A3I = A3*I
    return (A4-A@(A2I)@A)*J + 2*A2J - A2J@A2I - A2I@A2J - A@A3I - A3I@A + 3*A*A2

def path5(A):
    A2 = A@A
    A3 = A2@A
    A4 = A3@A
    A5 = A4@A
    I = np.eye(A.shape[0])
    J = np.ones(A.shape) - I
    A2I = A2*I
    A2ImI = A2I - I
    A2J = A2*J
    AA2 = A*A2
    A3I = A3*I
    P3 = A3*J - A@A2I - A2I@A + A
    AP3 = A*P3
    AP31 = np.diag((AP3).sum(1))
    return (A5-A@(A2I)@(A2ImI) - A2I@A@A2I - (A2ImI)@A2I@A - A@A2ImI@A2J - A2J@A2ImI@A
            - A2I@P3 - P3@A2ImI - A3I@A2J - A2J@A3I - A@A3I@A - AA2
            + 3*(A@AA2 + AA2@A) - A@AP31 - AP31@A + 3*AP3 + 3*AA2*(A2-(A2>0)))*J 

def path(A,n):
    f = [path1,path2,path3,path4,path5]
    return f[n-1](A)


def cycle(A,n):
    return A*path(A,n-1)

def mattovec(A):
    return A.reshape((A.shape[0]*A.shape[0],1))

def pathtovec(A,n):
    return mattovec(path(A,n)) 