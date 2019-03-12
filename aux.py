import matplotlib.patches as patches
import polytope as pc
import matplotlib.pyplot as plt
import numpy as np

def vertex_to_poly(v):
    data = pc.quickhull.quickhull(v)
    return pc.Polytope(data[0], data[1])


def plot_region(ax, poly, name, prob,
                color='red', alpha=0.5,
                hatch=False, fill=True):
    ax.add_patch(patches.Polygon(pc.extreme(poly), color=color, alpha=alpha, hatch=hatch, fill=fill))
    _, xc = pc.cheby_ball(poly)
    ax.text(xc[0] - 0.4, xc[1] - 0.43, '${}_{}$\n$p={}$'.format(name[0].upper(), name[1], prob))


def plot_nodes(nodes):
    assert isinstance(nodes, list)
    x = []
    y = []
    for node in nodes:
        x += np.ravel(node.mean)[0]
        y += np.ravel(node.mean)[1]

    plt.scatter(np.array(x), np.array(y), c="g")
