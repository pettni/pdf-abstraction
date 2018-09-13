import numpy as np
from best.models.pomdp import POMDP
import matplotlib.patches as patches
import polytope as pc
from functools import partial


def is_adjacent(poly, x, d):
  # return true x within distance d of poly
  return np.all(poly.A.dot(x) < poly.b + d)

def plot_region(ax, poly, name, prob, color='red', alpha=0.5, hatch=False, fill=True):
  ax.add_patch(patches.Polygon(pc.extreme(poly), color=color, alpha=alpha, hatch=hatch, fill=fill))
  _, xc = pc.cheby_ball(poly)
  ax.text(xc[0], xc[1], '${}_{}$\n$p={:.2f}$'.format(name[0].upper(), name[1], prob))

def get_cassie_env_conn(region):
  name = region[0]
  poly = region[1][0]
  def conn(rx):
    dist = 0.6 if name[0] == 'r' else 0
    if is_adjacent(poly, rx, dist):
      return {1}
    return {0}
  return conn

def get_uav_env_conn(region):
  name = region[0]
  poly = region[1][0]
  def conn(cx):
    if is_adjacent(poly, cx, 0):
      return {1} 
    return {0}
  return conn


def environment_belief_model(p0, name):
  # Create map belief MDP with prior p0
  if p0 == 0:
    # no dynamics
    return POMDP([np.array([1])], input_names=[name+'_u'], state_name=name+'_b', 
           input_trans = lambda n: 0, output_trans = lambda s: 0)
  if p0 == 1:
    return POMDP([np.array([1])], input_names=[name+'_u'], state_name=name+'_b',
           input_trans = lambda n: 0, output_trans = lambda s: 1)
  

  Tnone = np.eye(3);
  Tmeas = np.array([[1.  , 0,  0],
                    [1-p0, 0, p0],
                    [0,    0,  1]]);

  return POMDP([Tnone, Tmeas], input_names=[name+'_u'], state_name=name+'_b',
               output_trans=lambda s: [0, p0, 1][s])

def get_predicates(regions):

  predicates = dict()

  # fail predicate
  risk_names = list(filter(lambda name: name[0] == 'r', regions.keys()))
  risk_inputs = ['c_x'] + ['{}_b'.format(name) for name in risk_names]

  def risk_predicate(names, rx, *nargs):
    conds = [is_adjacent(regions[name][0], rx, 0) and nargs[i] > 0  
             for (i, name) in enumerate(names)]
    return {any(conds)}

  predicates['fail'] = (risk_inputs, partial(risk_predicate, risk_names))

  # sample predicates
  sample_types = set([name[0] for name in regions.keys()]) - set('r')

  for reg_type in sample_types:
    type_names = list(filter(lambda name: name[0] == reg_type, regions.keys()))
    type_inputs = ['c_x'] + ['{}_b'.format(name) for name in type_names]

    # Use functools.partial to avoid using same reg_type for all predicates
    def type_predicate(names, rx, *nargs):
      conds = [is_adjacent(regions[name][0], rx, 0) and nargs[i] == 1  
               for (i, name) in enumerate(names)]
      return {any(conds)}

    ap_name = 'sample{}'.format(reg_type.upper())
    predicates[ap_name] = (type_inputs, partial(type_predicate, type_names))

  return predicates
