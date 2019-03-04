# Hybrid Value Iteration

## Code Structure
- __hVI_config.py__
Used to define configuration variables

- __hVI_main.py__
Contains the implementation of the backups (equivalent to the ipython notebook)

- __hVI_firm.py__
Implementation of FIRM

- __hVI_model.py__
Contains motion and observations models used by FIRM

- __hVI_types.py__
Contains classes for environment and Gamma set used represent value function

## Conventions

- **_prod** refers to product space (of environment)
- **_reg** refers to individual region or lower dimensional space (of environment)

## To Execute

`python hVI_main.py`

## To change scenario
You can set the scenario in hVI_config to 'toy' or 'rss' 
Regions can be modified by editing the following in hVI_main:
`regs[<key>] = (polytope, initial_belief, output_label, true_state)`






----
# Example of using hVi
 Currently working examples:
 - Rocksample.ipynb (working):
    Make sure that when you run this notebook, you have installed this package first.
 - Toy [hVI_main] (NOT working and deprecated):
    TODO: Sofie figure out what is going on here
 - RSS [hVI_main] (NOT working)
    TODO: Sofie: Figure out what is going on.

## Configure the environment

Construct **the environment**.

    # Construct belief-MDP for env
    env = Env(regs)

with **regs** an ordered dictionary, whose last element is the state space domain. Some examples are given below: obstacles

    regs = OrderedDict()
    pi = rf.vertex_to_poly(np.array([[2, 0], [3, 0], [3, 1], [2, 1]]))
    regs['ri'] = (pi, 1, 'obs')

Potential sample regions for a rover:

    a1 = rf.vertex_to_poly(np.array([[4, 0], [5, 0], [5, 1], [4, 1]]))
    regs['a1'] = (a1, 0.9, 'sample1', 1)

In ``(a1, 0.9, 'sample1', 1)`` the 4th value required when the 2nd is less than 1,
   and it  is either 0 = doesnt exist or 1 = exists.

As a last step define the *null* region, this region encompasses the full state space. Given last, it has lowest priority and is only assigned to states that are not a member of any of the other polytopes.

    # Define Null regions with bounds of the space for null output with lowest priority
    p = rf.vertex_to_poly(np.array([[-3, -5], [-3, 5], [5, -5], [5, 5]]))
    regs['null'] = (p, 1.0, 'null')




## Configure the robot dynamics

For simulations, one typically initialises the random number generator at a specific seed. This enables reproducability.

    ''' Configuration Parameters '''
    random.seed(rand_seed)
    np.random.seed(rand_seed)

Then we also set up the motion model of the robot. In this case, we have assumed that the robot has deterministic dynamics. Therefore ``Det_SI_Model`` is used.

    ''' Setup Motion and Observation Models '''
    # Define Motion and Observation Model
    Wx = np.eye(2)
    Wu = np.eye(2)
    r2_bs = State_Space([-5, -5], [5, 5])
    motion_model = Det_SI_Model(0.1)

The model `` Det_SI_Model(dt)`` is currently a single integrator model with time discretization ``dt``.

Next, we initialize the sampling-based representation of these robot dynamics.
**Initialize Roadmap**
We can currently use two type of road maps:
1. A Feedback state roadmap, [SPath]
2. A Feedback information roadmap.

The former is a more standard roadmap where the samples are drawn
in the state space and the controls between the samples is based on simple state-feedback.
 The latter is the roadmap for POMDP, that is, for partially observable Markov processes.
 Thus this roadmap can be used for robots whose state can not be measured exactly.

Since we'll need a picture to look at, we first initialize that.

    fig = plt.figure(figsize=(14, 14), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111, aspect='equal')


Then, we initialize the SPath object,

    prm = SPaths(r2_bs, motion_model, Wx, Wu, regs, output_color, ax)

which takes as arguments:
- belief_space, motion_model: Refer to classes in models.py
- Wx = quadratic cost matrix of state used in LQR
- Wu = quadratic cost matrix of input used in LQR
- regs = the ordered dictionary with regions
- regs_outputs = Mapping from regs info [2] to integer of output; e.g. regs_output = {'blue':0, 'green':1, 'red':2}-->
- output_color = e.g. output_color = {-1:'black', 0:'blue', 1:'green', 2:'red'}-->
- ax = handle for plotting stuff, can be generated as: fig = plt.figure(0); ax = fig.add_subplot(111, aspect='equal')-->

Afterwards we generate 40 nodes and edges between nodes with distance less than 3.
The initial state is (-4.5,0) and is the first state that is ampled.

    prm.make_nodes_edges(40, 3, init=np.array([[-4.5],[0]]))
    prm.plot(ax)

The resulting graph looks like for example the following figure

<img src="prm_graph_01.png" width="500" title="PRM" alt="PRM figure " md-pos="5027-5046" />


## Define specification

Generate the specification DFA directly, first define the labels

    from best.fsa import Fsa
    props = ['obs', 'sample1', 'sample2']
    props = dict(zip(props, map(lambda x: 2 ** x, range(0, len(props)))))
    print(props)

Then define the DFA

    fsa = Fsa()
    vars(Fsa)
    fsa.g.add_node(0)
    fsa.g.add_node('trap')
    fsa.g.add_node(1)
    fsa.g.add_edge(0,1, weight=0, input={props['sample1'],props['sample2'],props['sample2']+props['sample1']})
    fsa.g.add_edge(0,0, weight=0, input={0})
    fsa.g.add_edge(0,'trap', weight=0, input={props['obs']})

    fsa.props=props
    fsa.final = {1}
    fsa.init = dict({0:1})

    #fsa.g.add_edge(src, dest, weight=0, input=bitmaps, guard=guard, label=guard)

An alternative is to work with Formulas given in LTL

    from best.ltl import formula_to_mdp

    fsaform = Fsa()
    form = '! obs U sample'
    fsaform.from_formula(form)
    vars(fsa.g)
    print(fsaform.g.edges(data=True))

The specification FSA can be passed through a dictionary as follows

    formula_fsa = dict()
    formula_fsa['fsa'] = fsa
    formula_fsa['init'] = dict({0:1})
    formula_fsa['final'] = {1}
    formula_fsa['prop'] = props

This is then used to define the cross product between the roadmap and the specification

    prod_ = spec_Spaths(prm, formula_fsa,env,n=125)

Here
- **spec_SPath** is the class for this specifiation based roadmaps
- **prm** is the roadmap
- **formula_fsm** is either a formula or a dictionary
- **n** is the maximum number of belief points in each of the nodes of the specification-based roadmap.

# Doing back-ups

Initiate the backups and do them untiall all nodes have converged.

    not_converged = True
    max_it = 30   #  do 30 iterations
    i = 0
    n = prod_.init[0]

    while not_converged:
        print('iteration', i)
        not_converged = prod_.full_back_up(opts)
        opt = np.unique(prod_.val[n].best_edge)

        if i > max_it:
            not_converged = False
        i += 1

For the specification roadmap, with given belief points the backups can be done with the command "full_back_up". Opt_old is used eliminate nodes for the next back up.
If node values have not changed, than they will only be backed-up in the next sequence if one of their neighbors values changed.


# Analyzes & plots

Compute

    traj, v_list, vals, act_list =  simulate(prod_, regs)

To take a look at the possible optimal paths us the function

    plot_optimizers(prod_,ax)


# Changing Firm

- **Add a node**: use
prod_.add_firm_node
See example in
tests.test_det_roadmap.TestStringMethods#test_add_node


- **remove unneeded nodes**
    Using

        nodes, edges, visited = plot_optimizers(prod_,ax,  showplot=False)

    the important nodes of the product are computed and put into "visited".
    
        prod_.prune(keep_list=visited)
        
        
# Expanding the belief point set
As in "Point-based value iteration: an anytime algorithm for POMDPs" Pineau et al.,
the belief point set is build up starting from the initial belief b_0
and updated with the single step forward reachable beliefs. 

