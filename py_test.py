import numpy as np
from datetime import datetime


A = np.random.random((30,1000))
B = np.random.random((256,30))
C = np.random.random((512,256))
D = np.random.random((784,512))


start = datetime.now()

BA = B.dot(A)

CBA = C.dot(BA)

DCBA = D.dot(CBA)

end = datetime.now()
duration = end-start

print(duration.microseconds / 1000)

start = datetime.now()

BA = B.dot(A)

CB = C.dot(B)

CBA = CB.dot(A)

DCB = D.dot(CB)

DCBA = DCB.dot(A)

end = datetime.now()
duration = end-start

print(duration.microseconds / 1000)