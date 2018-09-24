# pdf-abstraction

## Installation

This package is developed for Python 2.7. Install it as follows:

	python setup.py install

For development (create links to source code, no need to reinstall after making changes):

	python setup.py develop

Run tests:

	nosetests

## Workflow

Given a specification written as a syntactically co-safe LTL property and a POMDP model which is
- a finite state partially observable Markov decision process, or
- a LTI system with gaussian disturbances
compute a policy with guaranteed probability that the property is satisfied.


1. Convert syntactically co-safe LTL property to DFA
  * (code manually until cristi is ready)

2. Formal abstraction of POMDP to approximate belief space model
*output:*
  * MDP (Markov decision process)
  * \delta = probabilistic deviation
  * Lset = set valued labeling map
  * R^{-1}(x) = set valued function containing all abstract states related to concrete state

3. Compute delta-robust game over cross product
*output:*
  * Value function for abstract model
  * Policy for abstract model

4. Implementation
  * refine policy  based on Value function, relation, and abstract Policy
  * simulate refined policy with concrete POMDP




----
# Example of using hVi

## Configure the environment

Define Regions

    regs = OrderedDict()
    pi = rf.vertex_to_poly(np.array([[2, 0], [3, 0], [3, 1], [2, 1]]))
    regs['ri'] = (pi, 1, 'obs')

   - Regions that are added first have a higher priority

    a1 = rf.vertex_to_poly(np.array([[4, 0], [5, 0], [5, 1], [4, 1]]))
    regs['a1'] = (a1, 0.9, 'sample1', 1)

  In ``(a1, 0.9, 'sample1', 1)`` the 4th value required when the 2nd is less than 1,
   and it  is either 0 = doesnt exist or 1 = exists.

As a last step define the *null* region, this region encompasses the full state space. Given last, it has lowest priority and is only assigned to states that are not a member of any of the other polytopes.

    # Define Null regions with bounds of the space for null output with lowest priority
    p = rf.vertex_to_poly(np.array([[-3, -5], [-3, 5], [5, -5], [5, 5]]))
    regs['null'] = (p, 1.0, 'null')

Construct **the environment**.

    # Construct belief-MDP for env
    env = Env(regs)

The command `print(env)` will print information about this environment.


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

    firm = SPaths(r2_bs, motion_model, Wx, Wu, regs, output_color, ax)

which takes as arguments:
  - belief_space, motion_model: Refer to classes in models.py
  - Wx = quadratic cost matrix of state used in LQR
  - Wu = quadratic cost matrix of input used in LQR
  - regs = the ordered dictionary with regions
  - regs_outputs = Mapping from regs info [2] to integer of output; e.g. regs_output = {'blue':0, 'green':1, 'red':2}-->
  - output_color = e.g. output_color = {-1:'black', 0:'blue', 1:'green', 2:'red'}-->
  - ax = handle for plotting stuff, can be generated as: fig = plt.figure(0); ax = fig.add_subplot(111, aspect='equal')-->


    firm.make_nodes_edges(40, 3, init=np.array([[-4.5],[0]]))
    firm.compute_output_prob()
    firm.plot(ax)

