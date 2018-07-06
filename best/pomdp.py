import networkx as nx
import numpy as np
import scipy.sparse as sp

from best.utils import *

DTYPE = np.float32
DTYPE_ACTION = np.uint8
DTYPE_OUTPUT = np.uint32

class POMDP:
  """Partially Observable Markov Decision Process"""
  def __init__(self,
               T, 
               Z=[],
               input_names=['u'], 
               state_name='x', 
               output_name=[]):
    '''
    Create a POMDP

    Below, M is the number of actions, N the number of states, O the number of outputs.

    Input arguments:
      T: List of M stochastic transition matrices of size N x N such that T[m][n,n'] = P(n' | n, m). 
      Z: List of M stochastic matrices of size N x O such that Z[m][n,o] = P(o | n,m )
        if len(Z) = 1, outputs do not depend on inputs
        if Z = [], it is an MDP (perfect observations)
      input_fcn:  input resolution function: (U1, ... Uk) -> range(M) 
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
    
    self.input_names = input_names
    self.state_name = state_name
    self.output_name = output_name

    # Transition matrices for each axis
    self.Tmat_csr = {}
    self.Tmat_coo = {}

    for key in T.keys():
      self.Tmat_csr[key] = sp.csr_matrix(T[key], dtype=DTYPE)
      self.Tmat_coo[key] = sp.coo_matrix(T[key], dtype=DTYPE)

    if Z == []:
      self.Zmat_csr = []
      self.Zmat_coo = []
    elif len(Z) == 1:
      self.Zmat_csr = [sp.csr_matrix(Z[0], dtype=DTYPE)]
      self.Zmat_coo = [sp.coo_matrix(Z[0], dtype=DTYPE)]
    else:
      self.Zmat_csr = {}
      self.Zmat_coo = {}

      for key in Z.keys():
        self.Zmat_csr[key] = sp.csr_matrix(Z[key], dtype=DTYPE)
        self.Zmat_coo[key] = sp.coo_matrix(Z[key], dtype=DTYPE)

    self.check()

  @property
  def N(self):
    return next(iter(self.Tmat_csr.values())).shape[1]

  @property
  def M(self):
    return list(1 + np.amax([np.array(key) for key in self.Tmat_coo.keys()], axis=0))

  @property
  def O(self):
    if len(self.Zmat_csr) == 0:
      return self.N
    if len(self.Zmat_csr) == 1:
      return self.Zmat_csr[0].shape[1]
    return next(iter(self.Zmat_csr.values())).shape[1]
 
  @property
  def inputs(self):
    return self.input_names

  @property
  def state(self):
    return self.state_name

  @property
  def output(self):
    if self.observable:
      return self.state_name
    else:
      return self.output_name
  
  def T(self, m_tuple):
    '''transition matrix for action tuple m_tuple'''
    return self.Tmat_csr[m_tuple]

  def Z(self, m_tuple):
    '''transition matrix for action tuple m_tuple'''
    if self.Zmat_csr == []:
      return sp.identity(self.N, dtype=DTYPE, format='csr')
    if type(self.Zmat_csr) is not dict:
      return self.Zmat_csr
    else:
      return self.Zmat_csr[m_tuple]

  def check(self):
    for m_tuple in self.Tmat_csr.keys():
      if not self.T(m_tuple).shape == (self.N, self.N):
        raise Exception('matrix not square')

      if not self.Z(m_tuple).shape == (self.N, self.O):
        raise Exception('matrix not of size N x O')

    if self.observable and self.output_name != []:
      raise Exception('MDP can not have distinct output name')

    if len(self.M) != len(self.inputs):
      raise Exception('Input names does not equal inputs')

    if prod(self.M) != len(self.Tmat_csr.keys()):
      raise Exception('Problem with inputs')

  def __str__(self):
    po = '' if self.observable else 'PO'

    ret = '{0}MDP: {1} inputs {2} --> {3} states {4} --> {5} outputs {6}' \
          .format(po, self.M, self.inputs, self.N, self.state, self.O, self.output)
    return ret

  def bellman_(self, W, d):
    '''compute V(u, x1, .., xd, .., xn) = sum_{xd'} P(xd' | xd, u) W(u, x1, .., xd', .., xn) '''
    return np.stack([sparse_tensordot(self.T(m), W, d) for m in range(self.M)])

  @property
  def observable(self):
    return self.Zmat_csr == []


class POMDPNetwork:

  def __init__(self):
    self.graph = nx.DiGraph()

  def __str__(self):
    po = '' if self.observable else 'PO'
    return '{}MDP network: {} inputs {}, {} states {}, {} outputs {}' \
           .format(po, self.M, self.inputs, self.N, self.states, self.O, self.outputs)

  def add_pomdp(self, pomdp):

    if any(input_name in self.inputs for input_name in pomdp.inputs):
      raise Exception('input name collision')

    if pomdp.output_name in self.outputs:
      raise Exception('output name collision')

    if pomdp.state_name in self.states:
      raise Exception('state name collision')

    self.graph.add_node(pomdp)

  def add_connection(self, pomdp1, output, pomdp2, input, conn_fcn):

    if input not in pomdp2.inputs or input not in self.inputs or output not in [pomdp1.output]:
      raise Exception('invalid connection')

    self.graph.add_edge(pomdp1, pomdp2, output=output, input=input, conn_fcn=conn_fcn)

    if len(list(nx.simple_cycles(self.graph))) > 0:
      raise Exception('connection graph can not have cycles')

  @property
  def N(self):
    return [pomdp.N for pomdp in self.graph.nodes()]

  @property
  def O(self):
    return [pomdp.O for pomdp in self.graph.nodes()]

  def input_size(self):
    all_inputs_size = [(input, Mi) for pomdp in self.graph.nodes() for (input, Mi) in zip(pomdp.inputs, pomdp.M)]
    connected_inputs = [attr['input'] for _, _, attr in self.graph.edges(data=True)]
    return [input_size for input_size in all_inputs_size if input_size[0] not in connected_inputs]

  @property
  def M(self):
    in_s = self.input_size()
    if len(in_s):
      _, sizes = zip(*in_s)
      return sizes
    return []

  @property
  def inputs(self):
    '''return list of global inputs'''
    in_s = self.input_size()
    if len(in_s):
      inputs, _ = zip(*self.input_size())
      return inputs
    return []

  @property
  def states(self):
    return [pomdp.state for pomdp in self.graph.nodes()]

  @property
  def outputs(self):
    '''return list of global outputs'''
    return [pomdp.output for pomdp in self.graph.nodes()]

  @property
  def observable(self):
    return all(pomdp.observable for pomdp in self.graph.nodes())

  def plot(self):
    # add dummy nodes for drawing

    inputs_pomdp = []
    outputs_pomdp = []

    for pomdp in self.graph.nodes():
      for input_name in pomdp.inputs:
        if input_name in self.inputs:
          inputs_pomdp.append((input_name, pomdp))
      if pomdp.output in self.outputs:
        outputs_pomdp.append((pomdp.output, pomdp))

    self.graph.add_nodes_from(['source', 'sink'])

    for input_name, pomdp in inputs_pomdp:
      self.graph.add_edge('source', pomdp, input=input_name, output='')
    for output_name, pomdp in outputs_pomdp:
      self.graph.add_edge(pomdp, 'sink', input='', output=output_name)

    pos = nx.spring_layout(self.graph)

    nx.draw_networkx(self.graph, pos=pos, with_labels=False)

    edge_labels = {(pomdp1, pomdp2) : '{} -> {}'.format(attr['output'], attr['input'])
                   for pomdp1, pomdp2, attr in self.graph.edges(data=True)}

    nx.draw_networkx_edge_labels(self.graph, pos=pos, edge_labels=edge_labels)

    self.graph.remove_node('source')
    self.graph.remove_node('sink')

  # OpenAI baselines fcns

  def step(self, inputs):
    pass

  def reset(self):
    pass
