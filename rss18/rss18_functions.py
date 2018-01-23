import numpy as np
from best.mdp import MDP

def get_mdp(p0, qw, name):
    # Create map belief MDP with prior p0 and qw quality of weak measurements
    if p0 == 0:
        # no dynamics
        Tnone = np.array([1])
        Tweak = np.array([1])
        Tstrong = np.array([1])
        output_fcn = lambda s: 0
    
    elif p0 == 1:
        Tnone = np.array([1])
        Tweak = np.array([1])
        Tstrong = np.array([1])
        output_fcn = lambda s: 1
        
    else:
        pm = p0/2
        pp = p0 + (1-p0)/2
        # levels [0 p- p0 p+ 1]

        Tnone = np.eye(5);
        Tweak = np.array([[1,           0,             0,       0,              0], 
                          [qw*(1-pm),   (1-qw)*(1-pm), 0,       (1-qw)*pm,      qw*pm],
                          [qw*(1-p0),   (1-qw)*p0,     0,       (1-qw)*(1-p0),  qw*p0],
                          [qw*(1-pp),   (1-qw)*(1-pp), 0,       (1-qw)*pp,      qw*pp],
                          [0,           0,             0,       0,              1]])
        Tstrong = np.array([[1,      0, 0, 0, 0],
                            [(1-pm), 0, 0, 0, pm],
                            [(1-p0), 0, 0, 0, p0],
                            [(1-pp), 0, 0, 0, pp],
                            [0,      0, 0, 0, 1]])

        def output_fcn(s):
            return [0, pm, p0, pp, 1][s]
    
    return MDP([Tnone, Tweak, Tstrong], input_name=name+'_u', output_name=name+'_b', output_fcn=output_fcn)


# ROVER-ENVIRONMENT connection
def get_conn_rov_env(regs):
    def conn_rov_env(xr):
        ret = [0 for i in range(len(regs))]
        i = 0
        for (name, info) in regs.iteritems():
            poly = info[0]
            if np.all(poly.A.dot(xr) < poly.b + 2):
                ret[i] = 2    # strong measurement
            i += 1
        return set([tuple(ret)])
    return conn_rov_env


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

# COPTER-ENVIRONMENT connection
def get_conn_copt_env(regs, copter_sight):
    def conn_copt_env(xr):
        ret = [0 for i in range(len(regs))]
        i = 0
        for (name, info) in regs.iteritems():
            if is_adjacent(info[0], xr[0:2], 0) and xr[2] < 2.5:
                ret[i] = 2    # strong measurement
            elif is_adjacent(info[0], xr[0:2], copter_sight) and xr[2] > 2.5:
                ret[i] = 1    # weak measurement
            i += 1
            
        return set([tuple(ret)])
    return conn_copt_env

# Control policies
class RoverPolicy:
    
    def __init__(self, ltlpol, rover_abstr, rover_env_mdp):
        self.ltlpol = ltlpol
        self.rover_abstr = rover_abstr
        self.rover_env_mdp = rover_env_mdp
    
    def __call__(self, x, mapstate, aps):
        self.ltlpol.report_aps(aps)

        s_ab = self.rover_abstr.x_to_s(x)

        s_tot = self.rover_env_mdp.global_state((s_ab, mapstate))
        u_ab, curr_val = self.ltlpol.get_input(s_tot)

        return self.rover_abstr.interface(u_ab, s_ab, x)
    
    def finished(self):
        return self.ltlpol.finished()
    
    def reset(self):
        self.ltlpol.reset()    


class CopterPolicy:
    
    def __init__(self, pol, final_val, copter_abstr):
        self.pol = pol
        self.final_val = final_val
        self.ft = False
        self.copter_abstr = copter_abstr
            
    def __call__(self, x, mapstate):
                
        s_ab = self.copter_abstr.x_to_s(x)

        if self.final_val[s_ab, mapstate] >= 1:
            self.ft = True
            u_ab = 0
        else:
            u_ab = self.pol[s_ab, mapstate]

        return self.copter_abstr.interface(u_ab, s_ab, x)
    
    def reset(self):
        self.ft = False
    
    def finished(self):
        return self.ft