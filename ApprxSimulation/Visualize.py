# Copyright (c) 2013-2017 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.

"""Routines for the plotting of the simulation relations"""
import matplotlib
import numpy as np
import math
import matplotlib.pyplot as plt
import polytope as pc


def patch_ellips(Meps, pos = None, number =20):

    if Meps.shape[0] !=2:
        print('plot_rel() only works for 2 Dimensional spaces')
        return
    # compute singular value decomposition
    # Meps = U*s*V, Meps**.5=U*s**.5
    U, s, V = np.linalg.svd(Meps, full_matrices=True)
    #
    def x(number):
        Z = np.array([[math.cos(alpha), math.sin(alpha)] for alpha in np.linspace(0,2*math.pi,number)] )
        xvalue = (np.diag(s ** -.5)).dot(U.T).dot(Z.T)
        if pos is None:
            return xvalue

        else:
            return xvalue+pos

    print(x(number))
    # xarray = np.zeros((2,1))
    # if pos is None:
    #     xarray = xarray.append(x(alpha) for alpha in np.linspace(0,math.pi,number))
    # else :
    #     xarray = np.array(x(alpha).T+pos.T for alpha in np.linspace(0,math.pi,number))

    #print(xarray.shape)



    return matplotlib.patches.Polygon(x(number).T)

def plot_rel(Meps, pos = None, number =20):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch =patch_ellips(Meps, pos = None, number =number)

    ax.add_patch(patch)
    #
    #plt.tight_layout()
    plt.show()


def plot_regions(regions, xlim, ylim):
    """

    :param regions:  dictionary with name: polytope
    :return: figure
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for name in regions.keys():
        print(name)
        patch = matplotlib.patches.Polygon(pc.extreme(regions[name]))
        lx, ux = pc.bounding_box(regions[name])  # lower and upperbounds over all dimensions

        #patch = patch_ellips(Meps, pos=None, number=number)
        ax.add_patch(patch)
        plt.text(ux[0], ux[1],  name)
    #return

    #ax.add_patch(patch)
    plt.xlim(xlim)
    plt.ylim(ylim)
    # plt.tight_layout()
    plt.show()
