import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
import scipy.sparse as sp
from itertools import product

from best.utils import *
from best import DTYPE

class POMDP:
  """Partially Observable Markov Decision Process"""
  def __init__(self,
               T, 
               Z=[],
               input_names=['u'], 
               state_name='x', 
               output_name=[],
               input_fcns=None,
               output_fcn=None):
    '''
    Create a POMDP

    Below, M is the number of actions, N the number of states, O the number of outputs.

    Input arguments:
      T: dict of NxN matrices such that T[m_tuple][n,n'] = P(n' | n, m_tuple). 
      Z: dict of NxO matrices such that Z[m_tuple][n,o] = P(o | n,m_tuple )
        if len(Z) = 1, outputs do not depend on inputs
        if Z = [], it is an MDP (perfect observations)
      input_fcns:  input resolution functions: input_functions[i] : U_i -> range(M_i) 
      output_fcn: output labeling function: range(O) -> Y
      input_names: identifier for inputs
      state_name: identifier for states
      output_name: identifier for outputs

    Alphabets:
      inputs: range(M)
      states: range(N)
      outputs: range(O)

      input  alphabet: U
      output alphabet: Y
    '''
    
    self._input_names = input_names
    self._state_name = state_name
    self._output_name = output_name

    self._input_fcns = input_fcns
    self._output_fcn = output_fcn

    # Transition matrices for each axis
    self._Tmat_csr = {}
    self._Tmat_coo = {}

    for m_tuple in T.keys():
      self._Tmat_csr[m_tuple] = sp.csr_matrix(T[m_tuple], dtype=DTYPE)
      self._Tmat_coo[m_tuple] = sp.coo_matrix(T[m_tuple], dtype=DTYPE)

    if Z == []:
      self._Zmat_csr = []
      self._Zmat_coo = []
    elif len(Z) == 1:
      self._Zmat_csr = [sp.csr_matrix(Z[0], dtype=DTYPE)]
      self._Zmat_coo = [sp.coo_matrix(Z[0], dtype=DTYPE)]
    else:
      self._Zmat_csr = {}
      self._Zmat_coo = {}

      for m_tuple in Z.keys():
        self._Zmat_csr[m_tuple] = sp.csr_matrix(Z[m_tuple], dtype=DTYPE)
        self._Zmat_coo[m_tuple] = sp.coo_matrix(Z[m_tuple], dtype=DTYPE)

    self.check()

  @property
  def N(self):
    return next(iter(self._Tmat_csr.values())).shape[1]

  @property
  def M(self):
    return tuple(1 + np.amax([np.array(m_tuple) for m_tuple in self._Tmat_coo.keys()], axis=0))

  @property
  def O(self):
    if len(self._Zmat_csr) == 0:
      return self.N
    if len(self._Zmat_csr) == 1:
      return self._Zmat_csr[0].shape[1]
    return next(iter(self._Zmat_csr.values())).shape[1]
 
  @property
  def input_names(self):
    return tuple(self._input_names)

  @property
  def state_name(self):
    return self._state_name

  @property
  def output_name(self):
    if self.observable:
      return self._state_name
    else:
      return self._output_name

  @property
  def observable(self):
    return self._Zmat_csr == []

  def output(self, o):
    if self._output_fcn is None:
      return o
    return self._output_fcn(o)

  def input(self, u, m):
    if self._input_fcns is None:
      return u
    return self._input_fcns[m](u)

  def T(self, m_tuple):
    '''transition matrix for action tuple m_tuple'''
    return self._Tmat_csr[m_tuple]

  def Z(self, m_tuple):
    '''transition matrix for action tuple m_tuple'''
    if self._Zmat_csr == []:
      return sp.identity(self.N, dtype=DTYPE, format='csr')
    if type(self._Zmat_csr) is not dict:
      return self._Zmat_csr
    else:
      return self._Zmat_csr[m_tuple]

  def evolve(self, state, m_tuple):
    '''draw a successor for (state, m_tuple)'''
    succ_prob = np.asarray(self.T(m_tuple).getrow(state).todense()).ravel()
    return np.random.choice(range(self.N), 1, p=succ_prob)[0]


  def check(self):
    for m_tuple in self._Tmat_csr.keys():
      if not self.T(m_tuple).shape == (self.N, self.N):
        raise Exception('matrix not square')

      if not self.Z(m_tuple).shape == (self.N, self.O):
        raise Exception('matrix not of size N x O')

    if self.observable and self.output_name != self.state_name:
      raise Exception('MDP can not have distinct output name')

    if len(self.M) != len(self.input_names):
      raise Exception('Input names does not equal inputs')

    if prod(self.M) != len(self._Tmat_csr.keys()):
      raise Exception('Problem with inputs')

  def __str__(self):
    po = '' if self.observable else 'PO'

    ret = '{0}MDP: {1} inputs {2} --> {3} states {4} --> {5} outputs {6}' \
          .format(po, self.M, self.input_names, self.N, self.state_name, self.O, self.output_name)
    return ret

  def bellman_(self, W, d):
    '''compute V(u1, ... um, x1, .., xd, .., xn) = sum_{xd'} P(xd' | xd, u) W(x1, .., xd', .., xn) '''

    ret = np.zeros(self.M + W.shape, dtype=DTYPE)
    for m_tuple in product(*[list(range(k)) for k in self.M]):
      ret[m_tuple] = sparse_tensordot(self.T(m_tuple), W, d)

    return ret


class POMDPNetwork:

  def __init__(self):
    self.graph = nx.MultiDiGraph()

  def __str__(self):
    po = '' if self.observable else 'PO'
    return '{}MDP network: {} inputs {}, {} states {}, {} outputs {}' \
           .format(po, self.M, self.input_names, self.N, self.state_names, 
                   self.O, self.output_names)

  @property
  def N(self):
    return tuple(pomdp.N for pomdp in self.graph.nodes())

  @property
  def O(self):
    return tuple(pomdp.O for pomdp in self.graph.nodes())

  @property
  def M(self):
    in_s = self.input_size()
    if len(in_s):
      _, sizes = zip(*in_s)
      return tuple(sizes)
    return ()

  @property
  def input_names(self):
    '''return list of global inputs'''
    in_s = self.input_size()
    if len(in_s):
      inputs, _ = zip(*self.input_size())
      return tuple(inputs)
    return ()

  @property
  def state_names(self):
    return tuple(pomdp.state_name for pomdp in self.graph.nodes())

  @property
  def output_names(self):
    '''return list of global outputs'''
    return tuple(pomdp.output_name for pomdp in self.graph.nodes())

  @property
  def observable(self):
    return all(pomdp.observable for pomdp in self.graph.nodes())

  def output(self, states):
    return tuple(pomdp.output(state) for (pomdp, state) in zip(self.graph.nodes(), states))

  def add_pomdp(self, pomdp):

    if any(input_name in self.input_names for input_name in pomdp.input_names):
      raise Exception('input name collision')

    if pomdp.output_name in self.output_names:
      raise Exception('output name collision')

    if pomdp.state_name in self.state_names:
      raise Exception('state name collision')

    self.graph.add_node(pomdp)

  def add_connection(self, pomdp1, output, pomdp2, input, conn_fcn):

    if input not in pomdp2.input_names or input not in self.input_names or output not in [pomdp1.output_name]:
      raise Exception('invalid connection')

    nO = pomdp1.O  # number of outputs
    m = pomdp2.input_names.index(input) # input dimension
    nM = pomdp2.M[m] # number of inputs

    # compute bool connection matrix
    conn_matrix = np.zeros([nO, nM], dtype=bool)
    deterministic = True

    for o in range(nO):

      u_list = conn_fcn(pomdp1.output(o))
      if len(u_list) == 0:
        raise Exception('connection empty for output {}'.format(pomdp1.output_name(o)))

      if len(u_list) > 1:
        deterministic = False

      for u in u_list:
        inp = pomdp2.input(u, m)
        if inp < 0 or inp >= nM:
          raise Exception('connection invalid for output {}'.format(pomdp1.output_name(o)))

        conn_matrix[o, inp] = 1

    self.graph.add_edge(pomdp1, pomdp2, output=output, input=input, 
                        conn_mat=conn_matrix, deterministic=deterministic)

    if len(list(nx.simple_cycles(self.graph))) > 0:
      raise Exception('connection graph can not have cycles')

  def input_size(self):
    all_inputs_size = [(input, Mi) for pomdp in self.graph.nodes() for (input, Mi) in zip(pomdp.input_names, pomdp.M)]
    connected_inputs = [attr['input'] for _, _, attr in self.graph.edges(data=True)]
    return [input_size for input_size in all_inputs_size if input_size[0] not in connected_inputs]


  def plot(self):
    # add dummy nodes for drawing

    inputs_pomdp = []
    outputs_pomdp = []

    node_labels = {pomdp: pomdp.state_name for pomdp in self.graph.nodes()}
    node_labels_dummy = {'source': 'in', 'sink': 'out'}

    node_labels.update(node_labels_dummy)

    for pomdp in self.graph.nodes():
      for input_name in pomdp.input_names:
        if input_name in self.input_names:
          inputs_pomdp.append((input_name, pomdp))
      if pomdp.output_name in self.output_names:
        outputs_pomdp.append((pomdp.output_name, pomdp))

    self.graph.add_nodes_from(['source', 'sink'])

    for input_name, pomdp in inputs_pomdp:
      self.graph.add_edge('source', pomdp, input=input_name, output='')
    for output_name, pomdp in outputs_pomdp:
      self.graph.add_edge(pomdp, 'sink', input='', output=output_name)

    pos = graphviz_layout(self.graph, prog='dot')

    nx.draw_networkx(self.graph, pos=pos, with_labels=False)

    edge_labels = {(pomdp1, pomdp2) : '{}'.format(attr['input'])
                   for pomdp1, pomdp2, attr in self.graph.edges(data=True)}

    nx.draw_networkx_edge_labels(self.graph, pos=pos, edge_labels=edge_labels)

    nx.draw_networkx_labels(self.graph, pos=pos, labels=node_labels)

    self.graph.remove_node('source')
    self.graph.remove_node('sink')

  def bottom_up_iter(self):
    '''walk backwards over network'''

    for pomdp in self.graph.nodes():
      self.graph.node[pomdp]['passed'] = False

    try:
      while True:
        next_pomdp = next(pomdp for pomdp in self.graph.nodes()
                          if self.graph.node[pomdp]['passed'] == False and
                          all(self.graph.node[successor]['passed'] == True for
                              successor in self.graph.successors(pomdp)
                             )
                         )
        self.graph.node[next_pomdp]['passed'] = True
        yield next_pomdp

    except StopIteration as e:
      raise e


  def top_down_iter(self):
    '''walk forward over network'''

    for pomdp in self.graph.nodes():
      self.graph.node[pomdp]['passed'] = False

    try:
      while True:
        next_pomdp = next(pomdp for pomdp in self.graph.nodes()
                          if self.graph.node[pomdp]['passed'] == False and
                          all(self.graph.node[successor]['passed'] == False for
                              successor in self.graph.successors(pomdp)
                             )
                         )
        self.graph.node[next_pomdp]['passed'] = True
        yield next_pomdp

    except StopIteration as e:
      raise e


  def evolve(self, state, inputs):

    all_inputs = dict(zip(self.input_names, inputs))

    for pomdp in self.top_down_iter():

      # index of current state
      idx = self.state_names.index(pomdp.state_name)

      # find inputs to current pomdp and evolve
      pomdp_input_tuple = tuple(all_inputs[input_name] for input_name in pomdp.input_names)
      state[idx] = pomdp.evolve(state[idx], pomdp_input_tuple) 

      # add any inputs that are a function of updated state
      for _, _, attr in self.graph.out_edges(pomdp, data=True):

        inputs = np.nonzero(attr['conn_mat'][state[idx]])
        all_inputs[attr['input']] = np.random.choice(inputs[0])

    return state

  # OpenAI baselines fcns

  def step(self, inputs):
    pass

  def reset(self):
    pass
