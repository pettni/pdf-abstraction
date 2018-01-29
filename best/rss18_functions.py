import numpy as np
from best.mdp import MDP
import matplotlib.patches as patches
import polytope as pc


def vertex_to_poly(V):
    data = pc.quickhull.quickhull(V)
    return pc.Polytope(data[0], data[1])

def plot_region(ax, poly, name, prob, color='red', alpha=0.5, hatch=False, fill=True):
    ax.add_patch(patches.Polygon(pc.extreme(poly), color=color, alpha=alpha, hatch=hatch, fill=fill))
    _, xc = pc.cheby_ball(poly)
    ax.text(xc[0]-0.4, xc[1]-0.43, '${}_{}$\n$p={}$'.format(name[0].upper(), name[1], prob))


def environment_belief_model(p0, levels, name):
    # Create map belief MDP with prior p0 and qw quality of weak measurements
    if p0 == 0:
        # no dynamics
        return MDP([np.array([1])], input_name=name+'_u', output_name=name+'_b', 
                   input_fcn = lambda n: 0, output_fcn = lambda s: 0)
    elif p0 == 1:
        return MDP([np.array([1])], input_name=name+'_u', output_name=name+'_b',
                   input_fcn = lambda n: 0, output_fcn = lambda s: 1)
    else:
        pm = levels[0]
        pp = levels[1]

        Tnone = np.eye(5);
        Tweak = np.array([[1, 0, 0, 0, 0], 
                          [0, 1, 0, 0, 0],
                          [0, 1-p0, 0, p0, 0],
                          [0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 1]])
        Tstrong = np.array([[1,      0, 0, 0, 0],
                            [(1-pm), 0, 0, 0, pm],
                            [(1-p0), 0, 0, 0, p0],
                            [(1-pp), 0, 0, 0, pp],
                            [0,      0, 0, 0, 1]])

        def output_fcn(s):
            return [0, pm, p0, pp, 1][s]
    
    return MDP([Tnone, Tweak, Tstrong], input_name=name+'_u', output_name=name+'_b', output_fcn=output_fcn)

def environment_belief_model2(p0, levels, name):

    pmm = levels[0]
    pm = levels[1]
    pp = levels[2]
    ppp = levels[3]

    if p0 == 0:
        # no dynamics
        return MDP([np.array([1])], input_fcn = lambda n: 0, output_fcn = lambda s: 0)
    elif p0 == 1:
        # no dynamics
        return MDP([np.array([1])], input_fcn = lambda n: 0, output_fcn = lambda s: 1)
    else:
        Tnone = np.eye(7);

        Tweak = np.array([[1, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0],
                          [0, 0, (1-p0), 0, p0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 1]])

        Tstrong = np.array([[1, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0],
                          [0, (1-pm), 0, 0, 0, pm, 0],
                          [0, (1-p0), 0, 0, 0, p0, 0],
                          [0, (1-pp), 0, 0, 0, pp, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 1]])

        Texact = np.array([[1, 0, 0, 0, 0, 0, 0],
                          [(1-pmm), 0, 0, 0, 0, 0, pmm],
                          [(1-pm), 0, 0, 0, 0, 0, pm],
                          [(1-p0), 0, 0, 0, 0, 0, p0],
                          [(1-pp), 0, 0, 0, 0, 0, pp],
                          [(1-ppp), 0, 0, 0, 0, 0, ppp],
                          [0, 0, 0, 0, 0, 0, 1]])

        def output_fcn(s):
            return [0, pmm, pm, p0, pp, ppp, 1][s]
    
    return MDP([Tnone, Tweak, Tstrong], input_name=name+'_u', output_name=name+'_b', output_fcn=output_fcn)

# ROVER-ENVIRONMENT connection
def get_conn_rov_env(regs):
    def conn_rov_env(xr):
        ret = [0 for i in range(len(regs))]
        i = 0
        for (name, info) in regs.iteritems():
            poly = info[0]
            if is_adjacent(poly, xr, 1) and name[0] == 'r':
                ret[i] = 2    # strong measurement of risk region
            elif is_adjacent(poly, xr, 0):
                ret[i] = 2    # strong measurement of target region
            i += 1
        return set([tuple(ret)])
    return conn_rov_env

# COPTER-ENVIRONMENT connection
def get_conn_copt_env(regs, copter_sight):
    def conn_copt_env(xr):
        ret = [0 for i in range(len(regs))]
        i = 0
        for (name, info) in regs.iteritems():
            if is_adjacent(info[0], xr[0:2], 0) and xr[2] <= 2.6:
                ret[i] = 2    # strong measurement
            elif is_adjacent(info[0], xr[0:2], copter_sight) and xr[2] > 2.6:
                ret[i] = 1    # weak measurement
            i += 1
            
        return set([tuple(ret)])
    return conn_copt_env

# (ROVER-ENVIRONMENT)-LTL connection
def get_ltl_connection(regs):
    def ltl_connection(xc_env):
        xc = np.array(xc_env[0]).reshape(2,1)
        env = xc_env[1]
        
        i = 0
        
        ret = set([])
        
        for (name, info) in regs.iteritems():
            poly = info[0]
            
            if name[0] == 'r' and poly.contains(xc) and env[i] > 0:
                # we are in risk reg that is not confirmed safe 
                ret |= set(['fail'])
                
            if name[0] == 'a' and poly.contains(xc) and env[i] == 1:
                # we are in target reg with confirmed sample
                ret |= set(['sampleA'])

            if name[0] == 'b' and poly.contains(xc) and env[i] == 1:
                # we are in target reg with confirmed sample
                ret |= set(['sampleB'])

            if name[0] == 'c' and poly.contains(xc) and env[i] == 1:
                # we are in target reg with confirmed sample
                ret |= set(['sampleC'])
            i += 1

        if env[2] == 0:
            # no sample A
            ret |= set(['emptyA'])

        return set([tuple(ret)])
    return ltl_connection

def is_adjacent(poly, x, distance):
    # return true x within distance 3 of func
    return np.all(poly.A.dot(x) < poly.b + distance)

# Control policies
class RoverPolicy:
    
    def __init__(self, ltlpol, rover_abstr):
        self.ltlpol = ltlpol
        self.rover_abstr = rover_abstr

        self.t = 0
        self.s_ab = None

    def __call__(self, x_rov, s_map, APs):
        self.ltlpol.report_aps(APs)

        s_ab = self.rover_abstr.x_to_s(x_rov)
        
        if s_ab != self.s_ab and self.s_ab != None:
            self.t +=  1
        
        self.s_ab = s_ab
        u_ab, val = self.ltlpol((s_ab, s_map), self.t)

        if u_ab == 0:
            self.t += 1

        return self.rover_abstr.interface(u_ab, s_ab, x_rov), val
    
    def finished(self):
        return self.ltlpol.finished() or self.t > len(self.ltlpol.val)
    
    def reset(self):
        self.ltlpol.reset()
        self.t = 0
        self.s_ab = None    

class CopterPolicy:
    
    def __init__(self, pol_list, val_list, copter_abstr):
        self.pol_list = pol_list
        self.val_list = val_list
        self.ft = False
        self.copter_abstr = copter_abstr

        self.t = 0
        self.s_ab = None
            
    def __call__(self, x_cop, s_map):
                
        s_ab = self.copter_abstr.x_to_s(x_cop)

        if s_ab != self.s_ab and self.s_ab != None:
            self.t +=  1

        if self.t >= len(self.val_list):
            self.ft = True
            u_ab = 0
            val = self.val_list[-1][s_ab, s_map]

        else:
            self.s_ab = s_ab

            val = self.val_list[self.t][s_ab, s_map]
            u_ab = self.pol_list[self.t][s_ab, s_map]
            if u_ab == 0:
                # stay in cell
                self.t += 1 

        return self.copter_abstr.interface(u_ab, s_ab, x_cop), val
    
    def reset(self):
        self.ft = False
        self.t = 0
        self.s_ab = None
    
    def finished(self):
        return self.ft