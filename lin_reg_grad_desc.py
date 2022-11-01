import numpy as np

A = np.array([[1,0], [0,1]])
b = np.array([[-3],[4]])
c_0 = np.array([[1],[2]])

def areWeThereYet(newc, oldc):
    return np.linalg.norm(newc - oldc) < 0.001

def gradDescentForL(A, b, c_0, lr):
    c_old = np.zeros([1,1])
    c = c_0.copy()
    track = []
    while not areWeThereYet(c, c_old):
        c_old = c
        c = c - lr * (2 * np.matmul(A.T, np.matmul(A, c)) + b)
        track.append(c)
    print("The number of iterations is: ", len(track))
    print("The final value of c is: ", c)
    print("The whole array is: ", track)

gradDescentForL(A, b, c_0, 0.005)