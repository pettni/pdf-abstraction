#!/usr/bin/env python
"""Short title.

Long explanation
"""

# list of imports:
from hVI_fsrm import Spec_Spaths
from hVI_models import Belief_State
import numpy as np
from scipy.spatial import distance
# List of info
__author__ = "Sofie Haesaert"
__copyright__ = "Copyright 2018, TU/e"
__credits__ = ["Sofie Haesaert"]
__license__ = "BSD3"
__version__ = "1.0"
__maintainer__ = "Sofie Haesaert"
__email__ = "s.haesaert@tue.nl"


def BP_local(prod_, max_nodes): # extend belief points locally
    """
    A function to add belief nodes based on the nodes of the ingoing edges
    :param prod_:
    :type prod_:Spec_Spaths
    :param node:
    :return:
    """
    for n in prod_.active: # update locally
        if prod_.active[n] == False:
            continue
        (i_q, i_v) = n
        if len(prod_.val[n].b_prod_points) > max_nodes:
            return
        b_new = np.concatenate(prod_.val[n].b_prod_points,axis=1)
        for b_old in prod_.val[n].b_prod_points:
            b_update = []
            for key, info in prod_.env.regs.iteritems():
                O = prod_.env.get_O_reg_prob(key, i_v.mean)
                if (O[0,:] == O[1,:]).all():
                    continue

                b = np.diag(np.ravel(b_old))*O.T
                b = b*np.diag(np.ravel(sum(b))**-1.0) # two potential new bs
                b_update += [b]

            if len(b_update) > 0:
                b_update = np.hstack(b_update)
                #print('b_update',b_update)

                #print(distance.cdist(b_new.T, b_update.T))
                dis = np.amin(distance.cdist(b_new.T, b_update.T), axis=0)

                if sum(dis) >prod_.epsilon:
                    b_new = np.concatenate([b_new, b_update[:,np.argmax(dis)]],axis=1)
                    #print(b_new)


        # add new belief points
        l = len(prod_.val[n].b_prod_points)

        for col_index in range(l,b_new.shape[1]):
            prod_.val[n].b_prod_points += [b_new[:,col_index]]

    # for (ni,nj) in prod_.edges():
    #     print(ni,nj)
    #     if prod_.active[ni] == False or prod_.active[nj] == False:
    #         continue
    #     disij = distance.cdist(np.concatenate(prod_.val[ni].b_prod_points, axis=1),
    #                            np.concatenate(prod_.val[nj].b_prod_points, axis=1))
    #     print np.amin(disij, axis=1)
    #     print np.amin(disij, axis=0)




