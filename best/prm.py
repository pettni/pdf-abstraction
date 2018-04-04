from best.mdp import MDP
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import scipy.sparse as sp

class FIRM(object):

  def __init__(self, x_low, x_up, regs):
    ''' Create a FIRM graph with n_nodes in [x_low, x_up]; informed sampling in regs '''
    np.random.seed(12)
    self.x_low = x_low
    self.x_up = x_up
    self.regs = regs
    self.nodes = []
    self.edges = {}
    self.T_list = None
    self.sample_nodes(60)
    self.make_edges(2)

  def abstract(self):
    ''' Construct T by treating index of neigh as action number '''
    nodes_start_list=[[] for i in range(self.max_actions)]
    nodes_end_list=[[] for i in range(self.max_actions)]
    vals_list=[[] for i in range(self.max_actions)]
    for i in range(self.nodes.shape[0]):
      neigh=self.edges[i]
      for j in range(len(neigh)):
        nodes_start_list[j].append(i)
        nodes_end_list[j].append(neigh[j])
        vals_list[j].append(1)
    self.T_list = []
    for i in range(self.max_actions):
      self.T_list.append(sp.coo_matrix((vals_list[i],(nodes_start_list[i], nodes_end_list[i])),shape=(self.nodes.shape[0],self.nodes.shape[0])))
    output_fcn = lambda s: self.nodes[s]
    return MDP(self.T_list, output_name='xc', output_fcn=output_fcn)

  def sample_nodes(self, n_nodes, append=False):
    ''' Generate n_nodes Nodes by randomly sampling in [x_low, x_up] '''
    # TODO: Implement append to sample nodes incrementally
    if append == False:
      self.nodes = np.zeros((n_nodes+len(self.regs), len(self.x_low)))
      self.edges = {} # clear previous edges
    for i in range(len(self.x_low)):
      self.nodes[0:n_nodes,i] = self.x_low[i] + (self.x_up[i] - self.x_low[i])*np.random.rand(n_nodes).ravel()
    ''' Generate one node in each region '''
    j=0
    for key in self.regs:
      x_low = self.regs[key][0].bounding_box[0].ravel()
      x_up = self.regs[key][0].bounding_box[1].ravel()
      for i in range(len(self.x_low)):
        self.nodes[n_nodes+j,i] = x_low[i] + (x_up[i] - x_low[i])*np.random.rand(1).ravel()
      j=j+1

  def make_edges(self, dist):
    ''' Construct edges for self.nodes within distance dist '''
    self.max_actions = 1 #'''used to construct T'''
    # Can make more efficient
    for i in range(self.nodes.shape[0]):
      neigh=[]
      for j in range(self.nodes.shape[0]):
        if self.distance(self.nodes[i,:], self.nodes[j,:]) < dist :
          neigh.append(j)
      if len(neigh)>self.max_actions:
        self.max_actions = len(neigh)
      self.edges[i]=neigh

  def distance(self, node1, node2):
      return LA.norm(node1-node2)

  def plot(self, ax):
      ''' Plot the FIRM graph '''
      for i in range(self.nodes.shape[0]):
        neigh=self.edges[i]
        for j in neigh:
          x = [self.nodes[i,0],self.nodes[j,0]]
          y = [self.nodes[i,1],self.nodes[j,1]]
          ax.plot(x, y, 'b')
      ax.plot(self.nodes[...,0], self.nodes[...,1], 'go')
      ax.set_xlim(self.x_low[0], self.x_up[0])
      ax.set_ylim(self.x_low[1], self.x_up[1])

#firm = FIRM([-1,-1],[1,1])
#firm.plot()
#firm.abstract()
