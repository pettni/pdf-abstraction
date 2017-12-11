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
from ApprxSimulation.Visualize import plot_regions
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
    A = np.eye(dim)
    B = np.eye(dim)  #array([[1], [0.1]])
    W = np.eye(dim)  # noise on transitions

    # Accuracy
    C = np.array([[1, 0]])  # defines metric for error (||y_finite-y||< epsilon with y= cx   )
    sys = LTI(A, B, C, None)  # LTI system with   D = None

    sys.setBw(np.eye(dim))

    # Define spaces
    sys.setU(pc.box2poly(np.kron(np.ones((sys.m, 1)), np.array([[-3, 3]])))) # continuous set of inputs

    poly = pc.box2poly(np.kron(np.ones((sys.dim, 1)), np.array([[-15, 15]])))
    sys.setX(poly)  # X space

    regions = dict()


    # *Grid space
    d = np.array([[3, 4]])  # with distance measure
    mdp_grid = sys.abstract_io(d, un=5, verbose=False, Accuracy=False)  # do the gridding

    ### add labeling
    # add target
    regions['target'] = pc.box2poly(np.kron(np.ones((2, 1)), np.array([[5, 10]])))

    # add avoid
    regions['avoid'] = pc.box2poly(np.array([[-5, -3],[-10, 5]]))

    #plot_regions(regions, np.array([-15, 15]), np.array([-15, 15]))

    def output(n):
        # map Y1 -> 2^(2^AP)
        out =tuple()

        # check target
        for name in regions.keys() :
            if  pc.is_inside(regions[name], mdp_grid.state_fnc(n)):
                out += ( name,)

        return out
    print('output',output(4))


    system = MDP([mdp_grid.P[a] for a in range(len(mdp_grid.P))],  output_name='ap')

    #print('output',system.output(4))

    formula = '( ( ! avoid U target ) )'

    # figure out a map dict_input2prop from numeric inputs to name based inputs
    dfsa, init, final, dict_input2prop = formula_to_mdp(formula)

    act_inputs = mdp_grid.map_dfa_inputs(dict_input2prop, regions)

    print act_inputs
    # compute matrix which gives for each states 0,1 values for allowed inputs




    # system = MDP([T1, T2], output_fcn=output, output_name='ap')


    print('input',dfsa.input(set(('target',))))

    # prod = ProductMDP(system, dfsa)
    # V, _ = prod.solve_reach(accept=lambda s: s[1] in final)


    # for i in range(0, 7):
    #     mdp_grid.reach_bell()  # Bellman recursions
    #
    # xi, yi = np.meshgrid(*mdp_grid.srep)
    #
    # plt.pcolor(mdp_grid.sedge[0], mdp_grid.sedge[1], mdp_grid.V.reshape(xi.shape, order='F'))
    # plt.colorbar()
    # plt.xlim(np.array([-12, 12]))
    # plt.ylim(np.array([-12, 12]))
    # plt.show()















    # Sensor noise => pomdp
    H = np.ones((1,2)) # what can be measured
    V = np.eye(1)
        #  x^+= A x + Bu + w
        #  y = C x
        #  z = Cz x+v

    P= np.eye(2)
    mean = np.zeros((2,1))

    pomdp = POMDP(sys, H, V, P, mean)
    Pp = P
    for i in range(15):
        x, Pu = pomdp.update(mean, Pp)
        x, Pp = pomdp.predict(np.array([[1],[1]]),x=mean, P=Pu)
        # updates of x,P

    L, Pst = pomdp.kalman()

    belief_mdp = pomdp.beliefmodel()



    for i in range(15):
        (x,P) = belief_mdp.simulate(np.array([[1],[1]]))


def getformula():
    import matplotlib.pyplot as plt
    formula = 'F x > 2 || (P<=8 U custom(x) > 2)'
    predicates = {  # define custom predicates here
        'custom': lambda x: x
    }
    context = PredicateContext(predicates)

    fsa, tl, aps = gdtl2fsa(formula)

    print(tl)
    fsa.visualize()
    return fsa, tl, aps, context


def gdtl2fsa(formula):
    tl, ap = gdtl2ltl(formula)
    ltl_formula = tl.formulaString(ltl=True)
    assert tl.isSynCoSafe()
    fsa = Fsa(multi=False)
    fsa.from_formula(ltl_formula)
    # optional add trap state
    fsa.add_trap_state()

    print(fsa)
    return fsa, tl, ap


if __name__ == '__main__':
    #loglevel = logging.INFO
    #logging.basicConfig(level=loglevel)
    main()

