"""
LTI Gaussian processes abstraction (basic)
Given
    -  x^+= A x + Bu + Ew
    -  z = Cx+Fv
Do :
    - Quantify eps,
    - Abstract to finite MDP
    Remark that since there is no order reduction,
     there is also no delta error.

Author Sofie Haesaert
"""
from ApprxSimulation.LTI_simrel import tune_dratio
from ApprxSimulation.Visualize import plot_regions
from Models.MDP import Rpol
from label_abstraction.test_mdp import test_mdp_dfsa, formula_to_mdp, test_ltl_synth2

print('Import packages')
# Import packages:
from Models.Linear import LTI, POMDP
from label_abstraction.mdp import *
import matplotlib.pyplot as plt

from lomap import Fsa
from gdtl import gdtl2ltl, PredicateContext
import polytope as pc



def main():

    print('Initialise values')
    # Define the linear time invariant system
    #A = np.array([[0,-0.8572],[0.1,0.5]])
    dim = 2
    A =np.eye(2) #np.array([[.9,-0.32],[0.1,0.9]])
    B = np.eye(dim)  #array([[1], [0.1]])
    Tr = .5*np.array([[-1,1],[1,-1]])
    W =  2*Tr.dot(np.eye(dim)).dot(Tr)  # noise on transitions
    print(W)

    # Accuracy
    C = np.array([[1, 0]])  # defines metric for error (||y_finite-y||< epsilon with y= cx   )

    sys = LTI(A, B, C, None, W = W)  # LTI system with   D = None

    # Define spaces
    sys.setU(pc.box2poly(np.kron(np.ones((sys.m, 1)), np.array([[-3, 3]])))) # continuous set of inputs
    sys.setX(pc.box2poly(np.kron(np.ones((sys.dim, 1)), np.array([[-10, 10]])))) # X space


    # Define regions of interest for the labeling
    regions = dict()
    regions['target'] = pc.box2poly(np.kron(np.ones((2, 1)), np.array([[5, 10]])))
    # add avoid
    regions['avoid'] = pc.box2poly(np.array([[-5, 3],[-10, 5]]))


    print('1.  Transform to normalised state space')
    sys_n = sys.normalize()


    ## step 1: tune gridding ratio (find optimal horizontal d_1, and vertical d_2)
    # currently only available for 2D
    print('2.  Tune gridding ratio')
    d_opt, d_vals, eps_values = tune_dratio(sys_n)
    # d_opt has optimal ratio with grid diameter of 1
    # choose grid sizes (factor of d_opt)
    d = d_opt #  with distance measure
    print('Choose grid ribs as', d)    # *Grid space

    print('3.  Grid Gaussian process')
    print(sys_n.T2x)
    mdp_grid = sys_n.abstract_io(d, un=7, verbose=False)  # do the gridding
    print('--- done gridding')



    print('4.  Define formula and compute DFA')

    #('output',system.output(4))

    formula = '( ( ! avoid U target ) )'

    # figure out a map dict_input2prop from numeric inputs to name based inputs
    dfsa, init, final, dict_input2prop = formula_to_mdp(formula)

    mdp_grid.map_dfa_inputs(dict_input2prop, regions)
    mdp_grid.setdfa(dfsa,final)

    print('5. Compute recursions')

    V, policy, W = mdp_grid.reach_dfa(recursions = 10)


    print('6. Plot normalized systen')

    xi, yi = np.meshgrid(*mdp_grid.srep)

    plt.pcolor(mdp_grid.sedge[0], mdp_grid.sedge[1],  W[:-1].reshape(xi.shape, order='F'))
    plt.colorbar()
    plt.xlim(np.array([mdp_grid.srep[0][0],mdp_grid.srep[0][-1]]))
    plt.ylim(np.array([mdp_grid.srep[1][0],mdp_grid.srep[1][-1]]))
    #plt.show()

    pol = Rpol(mdp_grid, V, W, policy)

    xi, yi = np.meshgrid(np.linspace(mdp_grid.srep[0][0],mdp_grid.srep[0][-1],10),np.linspace(mdp_grid.srep[1][0],mdp_grid.srep[1][-1],10))

    # compute inputs
    u =sys_n.b.dot(pol(np.block([[xi.flatten()],[yi.flatten()]])))
    delx = (-np.block([[xi.flatten()],[yi.flatten()]])+sys_n.a.dot(np.block([[xi.flatten()],[yi.flatten()]])) + sys_n.b.dot(pol(np.block([[xi.flatten()],[yi.flatten()]]))))
    x_tr = (np.block([[xi.flatten()], [yi.flatten()]]))

    #plt.quiver(xi.flatten(), yi.flatten(),u[0],u[1])
    plt.quiver(x_tr[0],x_tr[1],delx[0],delx[1], color = 'r')
    plt.show()

    print('6. Plot concrete systen')
    x_edge = np.linspace(-10,10,80)
    x_del = np.diff(x_edge).max()
    y_edge = np.linspace(-10,10,80)
    y_del = np.diff(y_edge).max()

    xi, yi = np.meshgrid(x_edge[:-1]+x_del/2,y_edge[:-1]+y_del/2)

    values = pol.val_concrete(np.block([[xi.flatten()], [yi.flatten()]]))
    plt.pcolor(x_edge, y_edge, values.reshape(xi.shape))
    plt.colorbar()
    plt.xlim(np.array([-10,10]))
    plt.ylim(np.array([-10,10]))

    plt.show()

    #
    #
    #
    # # Sensor noise => pomdp
    # H = np.ones((1,2)) # what can be measured
    # V = np.eye(1)
    #     #  x^+= A x + Bu + w
    #     #  y = C x
    #     #  z = Cz x+v
    #
    # P= np.eye(2)
    # mean = np.zeros((2,1))
    #
    # pomdp = POMDP(sys, H, V, P, mean)
    # Pp = P
    # for i in range(15):
    #     x, Pu = pomdp.update(mean, Pp)
    #     x, Pp = pomdp.predict(np.array([[1],[1]]),x=mean, P=Pu)
    #     # updates of x,P
    #
    # L, Pst = pomdp.kalman()
    #
    # belief_mdp = pomdp.beliefmodel()
    #
    #
    #
    # for i in range(15):
    #     (x,P) = belief_mdp.simulate(np.array([[1],[1]]))



if __name__ == '__main__':
    #loglevel = logging.INFO
    #logging.basicConfig(level=loglevel)
    main()

