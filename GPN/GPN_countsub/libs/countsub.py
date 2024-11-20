
import torch
# import numpy.linalg as lin
import time


# 2_ 5 Path at edge level :
def path1(A):
    return A
def path2(A):
    A2 = A@A
    I = torch.eye(A.shape[0])
    J = torch.ones(A.shape) - I
    return A2*J

def path3(A):
    A2 = A@A
    A3 = A2@A
    I = torch.eye(A.shape[0])
    J = torch.ones(A.shape) - I
    A2I =A2*I
    return A3*J - A@A2I - A2I@A + A 



def path4(A):
    A2 = A@A
    A3 = A2@A
    A4 = A3@A
    I = torch.eye(A.shape[0])
    J = torch.ones(A.shape) - I
    A2I = A2*I
    A2J = A2*J
    A3I = A3*I
    return (A4-A@(A2I)@A)*J + 2*A2J - A2J@A2I - A2I@A2J - A@A3I - A3I@A + 3*A*A2







def path5(A):
    A2 = A@A
    A3 = A2@A
    A4 = A3@A
    A5 = A4@A
    I = torch.eye(A.shape[0])
    J = torch.ones(A.shape) - I
    A2I = A2*I
    A2ImI = A2I - I
    A2J = A2*J
    AA2 = A*A2
    A3I = A3*I
    P3 = A3*J - A@A2I - A2I@A + A
    AP3 = A*P3
    AP31 = torch.diag((AP3).sum(1))
    return (A5-A@(A2I)@(A2ImI) - A2I@A@A2I - (A2ImI)@A2I@A - A@A2ImI@A2J - A2J@A2ImI@A
            - A2I@P3 - P3@A2ImI - A3I@A2J - A2J@A3I - A@A3I@A - AA2
            + 3*(A@AA2 + AA2@A) - A@AP31 - AP31@A + 3*AP3 + 3*AA2*A2 - 3*AA2)*J 





def path6(A):
    A2 = A@A
    A3 = A2@A
    A4 = A3@A
    A5 = A4@A
    I = torch.eye(A.shape[0])
    J = torch.ones(A.shape) - I
    AA2 = A*A2
    A2A2 = A2*A2
    AA3 = A*A3
    A2J = A2*J
    A3J = A3*J
    A3I = A3*I
    A4J = A4*J
    A2I = A2*I
    A2IA = A2I@A
    A3IA = A3I@A
    AA2I = A@A2I
    AA3I = A@A3I
    AA2IA = AA2I@A
    AA2IA2 = AA2IA@A
    A2A2IA = A@AA2IA
    AA3IA = AA3I@A
    IAA2IA = I*AA2IA
    IAA3IA = I*AA3IA
    A4I = A4*I
    A5I = A5*I
    A3I = A3*I
    return (A5@A*J - A5I@A - A@A5I - A4I@A2J - A2J@A4I - A2I@A4J - A4J@A2I  - A@A4I@A*J + 3*A*A4
            - A3I@A3J - A3J@A3I - AA2IA@A2*J - A2@AA2IA*J - A@AA3IA*J - AA3IA@A*J
            + 4*A2I@A3IA + 4*AA3I@A2I + 6*AA2*A3 + A2I@AA3I + A3IA@A2I + 3*AA3@A*J + 3*A@AA3*J
            +A@IAA3IA + IAA3IA@A - A@AA2IA@A*J + 2* A2I@A2IA@A*J + 2* A@AA2I@A2I*J + A2A2*A2J
            +A2I@A2J@A2I + 3*AA2@A2*J + 3*A2@AA2*J + IAA2IA@A2J + A2J@IAA2IA + A2IA@A2IA*J
            + AA2I@AA2I*J + A@(I*A2A2IA) + (I*A2A2IA)@A + A@(I*AA2IA2) + (I*AA2IA2)@A 
            + 3* AA2*A2@A*J + 3*A@(AA2*A2)*J - 12*AA2@A2I - 12*A2I@AA2 + 2*AA2I@A2IA*J
            -4* A2A2*J - 8*A*(A@AA2) - 8* A*(AA2@A) - 3* A*(AA2IA) + 3*A@AA2@A*J
            + A@IAA2IA@A*J - 4*A@AA2*J - 4*AA2@A*J + 4*A4J - 5*A3IA - 5*AA3I
            -4* (I*(A@AA2))@A - 4*A@(I*(A@AA2)) -7*A2I@A2J - 7*A2J@A2I - 10* AA2IA*J 
            -4*(I*(AA2@A))@A -4*A@(I*(AA2@A)) + 44*AA2 + 12*A2J )



def path(A,n):
    f = [path1,path2,path3,path4,path5,path6]
    return f[n-1](A)


def cycle(A,n):
    return A*path(A,n-1)



