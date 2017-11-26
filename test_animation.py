import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

p = Rectangle((-0.7625, -1.37), 1.525, 2.74, alpha=0.5)
ax.add_patch(p)
art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-5, 5)

sca = ax.scatter([0], [1], [0], color='r', s=50)

sca.set_offsets([[1], [0]])
sca.set_3d_properties([[1]], zdir='z')

plt.show()