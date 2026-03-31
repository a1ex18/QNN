import galois
import numpy as np

GF= galois.GF(2**4)
L = GF.elements[1:]   # size n = 15

# Choose irreducible polynomial g(x) of degree t
x = galois.Poly.Identity(GF)
print("x= ",x)

g = x**2 + x + GF(1)   # degree t = 2
print("g= ",g)

# Build parity-check matrix H
n = len(L)
t = g.degree
H = np.zeros((t, n), dtype=GF)
print("n=",n,"\n degree t=",t,"\n initial H=",H)
for i, alpha in enumerate(L):
    denom = g(alpha)   # evaluate g(alpha)
    print("denom = ",denom)
    if denom == 0:
        raise ValueError("Support element is root of g(x), invalid choice")
    inv = GF(1) / denom
    for j in range(t):
        H[j, i] = (alpha**j) * inv
    print("\nH=", H)

print("\n\n Parity-check matrix H:\n", H)

# From H we can derive generator matrix G
# (standard linear algebra over GF(2))
k = n - t * GF.degree
print("k=",k)