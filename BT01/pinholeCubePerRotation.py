import numpy as np
import matplotlib.pyplot as plt

vec = np.array

A = vec([1, 1, 1])
B = vec([-1, 1, 1])
C = vec([1, -1, 1])
D = vec([-1, -1, 1])
E = vec([1, 1, -1])
F = vec([-1, 1, -1])
G = vec([1, -1, -1])
H = vec([-1, -1, -1])

camera = vec([2, 3, 5])

points = dict(zip("ABCDEFGH", [A, B, C, D, E, F, G, H]))

edges = ["AB", "CD", "EF", "GH", "AC", "BD", "EG", "FH", "AE", "CG", "BF", "DH"]
points = {k: p - camera for k, p in points.items()}


def pinhole(v):
    x, y, z = v
    return vec([x / z, y / z])

def getRotX(angle):
    Rx = np.zeros(shape=(3, 3))
    Rx[0, 0] = 1
    Rx[1, 1] = np.cos(angle)
    Rx[1, 2] = -np.sin(angle)
    Rx[2, 1] = np.sin(angle)
    Rx[2, 2] = np.cos(angle)

    return Rx

def getRotY(angle):
    Ry = np.zeros(shape=(3, 3))
    Ry[0, 0] = np.cos(angle)
    Ry[0, 2] = -np.sin(angle)
    Ry[2, 0] = np.sin(angle)
    Ry[2, 2] = np.cos(angle)
    Ry[1, 1] = 1

    return Ry

def getRotZ(angle):
    Rz = np.zeros(shape=(3, 3))
    Rz[0, 0] = np.cos(angle)
    Rz[0, 1] = -np.sin(angle)
    Rz[1, 0] = np.sin(angle)
    Rz[1, 1] = np.cos(angle)
    Rz[2, 2] = 1

    return Rz


def rotate(R, v):
    return np.matmul(v, R)

plt.figure(figsize=(10, 10))

angles = [15, 30, 45, 60]
for angle in angles:
    Rz = getRotZ(angle)
    ps = {k: rotate(Rz, p) for k, p in points.items()}
    uvs = {k: pinhole(p) for k, p in ps.items()}

    for a, b in edges:
        ua, va = uvs[a]
        ub, vb = uvs[b]
        plt.title("Rotation Rz {}".format(angle))
        plt.plot([ua, ub], [va, vb], "ko-")
    plt.pause(2)
    plt.clf()

for angle in angles:
    Rx = getRotX(angle)
    ps = {k: rotate(Rx, p) for k, p in points.items()}
    uvs = {k: pinhole(p) for k, p in ps.items()}

    for a, b in edges:
        ua, va = uvs[a]
        ub, vb = uvs[b]
        plt.title("Rotation Rx {}".format(angle))
        plt.plot([ua, ub], [va, vb], "ko-")
    plt.pause(2)
    plt.clf()

for angle in angles:
    Ry = getRotY(angle)
    ps = {k: rotate(Ry, p) for k, p in points.items()}
    uvs = {k: pinhole(p) for k, p in ps.items()}

    for a, b in edges:
        ua, va = uvs[a]
        ub, vb = uvs[b]
        plt.title("Rotation Ry {}".format(angle))
        plt.plot([ua, ub], [va, vb], "ko-")
    plt.pause(2)
    plt.clf()
plt.axis("equal")
plt.grid()

plt.show()