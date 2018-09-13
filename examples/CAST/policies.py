class CassiePolicy:
  
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
    u_ab, val = self.ltlpol((s_ab,) + tuple(s_map), self.t)

    if u_ab == (0,):
      self.t += 1

    return self.rover_abstr.interface(u_ab, s_ab, x_rov), val

  def get_value(self, x, s_map):
    s_ab = self.rover_abstr.x_to_s(x)
    t_act = min(self.t, len(self.ltlpol.val)-1)
    return self.ltlpol.val[t_act][(s_ab,) + tuple(s_map) + (self.ltlpol.dfsa_state,)]

  def finished(self):
    return self.ltlpol.finished() or self.t > len(self.ltlpol.val)
  
  def reset(self):
    self.ltlpol.reset()
    self.t = 0
    self.s_ab = None  


class UAVPolicy:
  
  def __init__(self, pol_list, val_list, uav_abstr):
    self.pol_list = pol_list
    self.val_list = val_list
    self.ft = False
    self.uav_abstr = uav_abstr

    self.t = 0
    self.s_ab = None
      
  def __call__(self, x_cop, s_map):
        
    s_ab = self.uav_abstr.x_to_s(x_cop)

    if s_ab != self.s_ab and self.s_ab != None:
      self.t +=  1

    if self.t >= len(self.pol_list):
      self.ft = True
      u_ab = (0,)
      val = self.val_list[-1][(s_ab,) + tuple(s_map)]

    else:
      self.s_ab = s_ab
      val = self.val_list[self.t][(s_ab,) + tuple(s_map)]
      u_ab = (self.pol_list[self.t][0][(s_ab,) + tuple(s_map)],)  # input is 1-tuple
      if u_ab == (0,):
        # stay in cell
        self.t += 1 
    return self.uav_abstr.interface(u_ab, s_ab, x_cop), val

  def reset(self):
    self.ft = False
    self.t = 0
    self.s_ab = None
  
  def finished(self):
    return self.ft
