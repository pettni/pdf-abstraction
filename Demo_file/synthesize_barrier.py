import numpy as np
from sympy.abc import x,y,v,w

import posipoly as pp

# using this function requires installing the posipoly package from https://github.com/pettni/posipoly

def synthesize_barrier(g, g0, gT, gU, f, Sigma, tot_deg, cone_type):
	''' 
	Lower bound probability
		P [ gU(x(t)) >= 0, t=0,...,T-1, and  gT(x(T)) >= 0 ) | g0(x(0)) >= 0 ]
	for dynamics
		x(t+1) = f_t(x(t), w(t)),     w(t) ~ N(0, Sigma)
	by searching for a barrier certificate with positive polynomial programming.

    Parameters
    ----------
    g : Polynomial in x
      analysis domain (result is only valid inside this set)
    g0 : Polynomial in x
      initial set
    gT : Polynomial in x
      final set
    gU : Polynomial in x
      safe set
    f : list of list of Polynomial in (x, w)
      dynamics: x_i[t+1] = f[t][i](x[t], w[t])
    Sigma: numpy array
      covariance matrix of noise (positive definite symmetric of size len(w) x len(w))
    tot_deg : integer
	  maximal degree of positive polynomial program
	cone_type : 'psd' or 'sdd'
	  cone for positive polynomial programming (positive semidefinite cone or scaled diagonally dominant cone)

    Returns
    -------
    lb : scalar
      lower bound on transition probability
	'''

	T = len(f)     # time horizon
	n_x = len(f[0])       # state dimension
	n_w = f[0][0].n - n_x # noise dimension

	# TODO: add more argument checks
	if len(Sigma.shape) != 2 or np.any(np.array(Sigma.shape) != n_w):
		raise Exception('Sigma must be {}x{}'.format(n_w, n_w))

	deg_B = tot_deg   # need to subtract if deg f larger than 1
	deg_g0 = tot_deg-g0.d
	deg_gT = tot_deg-gT.d
	deg_gU = tot_deg-gU.d
	deg_g = tot_deg-g.d

	prog = pp.PPP()

	prog.add_var('c', n_x, 0, 'coef')      # scalar variable = polynomial of degree 0
	prog.add_var('gamma', n_x, 0, 'coef')  # scalar variable

	for t in range(T+1):
	    prog.add_var('B{}'.format(t), n_x, deg_B, 'coef')

	for t in range(T):
	    prog.add_var('s1{}'.format(t), n_x, deg_g, 'pp')
	    prog.add_var('s2{}'.format(t), n_x, deg_g, 'pp')
	    
	prog.add_var('s3', n_x, deg_g0, 'pp')

	for t in range(1, T):
	    prog.add_var('s4{}'.format(t), n_x, deg_gU, 'pp')

	prog.add_var('s5a', n_x, deg_gT, 'pp')
	prog.add_var('s5b', n_x, deg_g, 'pp')

	for t in range(T+1):
	    prog.add_var('s6{}'.format(t), n_x, deg_g, 'pp')

	# Identity for scalars to tot_deg
	T1 = pp.PTrans.eye(n0=n_x, d0=0, n1=n_x, d1=tot_deg)

	# Identity for scalars to scalar
	T10 = pp.PTrans.eye(n0=n_x, d0=0, n1=n_x, d1=0)

	# B(x,y) -> E_w[ B(ft(x,y,w)) ]
	TBp = list()
	for t in range(T):
		TBp.append(pp.PTrans.gaussian_expectation(n0=n_x+n_w, d0=deg_B, i_list=range(n_x, n_x+n_w), Sigma=Sigma) \
		           * pp.PTrans.composition(n0=n_x, d0=deg_B, g_list=f[t]))

	# Identity for B
	TB  = pp.PTrans.eye(n_x, deg_B)     

	# Multiplication with g, g0, g1, gU
	Tg  = pp.PTrans.mul_pol(n_x, deg_g, g)
	Tg0 = pp.PTrans.mul_pol(n_x, deg_g0, g0)
	TgT = pp.PTrans.mul_pol(n_x, deg_gT, gT)
	TgU = pp.PTrans.mul_pol(n_x, deg_gU, gU)

	# add (1)
	for t in range(T):
	    Aop = {'B{}'.format(t+1): TBp[t],
	           'B{}'.format(t): -TB, 
	           's1{}'.format(t): -Tg}
	    prog.add_constraint(Aop, pp.Polynomial.zero(n_x), 'pp')
	    
	# add (2)
	for t in range(T):
	    Aop = {'c': T1,
	           'B{}'.format(t): TB, 
	           'B{}'.format(t+1): -TBp[t],
	           's2{}'.format(t): -Tg}
	    prog.add_constraint(Aop, pp.Polynomial.zero(n_x), 'pp')
	    
	# add (3)
	prog.add_constraint({'gamma': T1, 'B0': -TB, 's3': -Tg0}, pp.Polynomial.zero(n_x), 'pp')

	# add (4)
	for t in range(1, T):
	    prog.add_constraint({'B{}'.format(t): TB, 's4{}'.format(t): -TgU,},
	                        pp.Polynomial.one(n_x), 'pp')
	    
	# add (5)
	prog.add_constraint({'B{}'.format(T): TB, 's5a': TgT, 's5b': -Tg}, pp.Polynomial.one(n_x), 'pp')

	# add (6)
	for t in range(T+1):
	    Aop = {'B{}'.format(t): TB, 
	           's6{}'.format(t): -Tg}
	    prog.add_constraint(Aop, pp.Polynomial.zero(n_x), 'pp')
	    
	# add inequality: gamma <= 1  (to exclude large solutions)
	prog.add_constraint({'gamma': T10}, pp.Polynomial.one(n_x), 'iq')

	# set c=0
	# prog.add_row({'c': T10}, pp.Polynomial.zero(n_x), 'eq')

	# add objective
	prog.set_objective({'c': pp.PTrans.eval0(n_x, 0)* T, 'gamma': pp.PTrans.eval0(n_x, 0)})

	sol, status = prog.solve(cone_type)

	c = prog.get_poly('c')(0,0)
	gamma = prog.get_poly('gamma')(0,0)

	lb = 1-(gamma+c*T)

	return lb


def main():
	# parameters
	T = 4          # time horizon
	sigma = 0.1    # noise standard deviation
	tot_deg = 8    # overall degree of ppp

	# polynomials defining sets
	g  = pp.Polynomial.from_sympy(1, [x,y])
	g0 = pp.Polynomial.from_sympy(0.25**2 - x**2 - y**2, [x,y])
	gT = pp.Polynomial.from_sympy(0.5**2 - (x-1)**2 - y**2, [x,y])
	gU = pp.Polynomial.from_sympy(0.2**2 - (x-0.4)**2 - (y-0.5)**2, [x,y])

	# dynamics: 
	#  x(t+1) = x(t) + 0.25 + w
	#  y(t+1) = y(t)
	ft_x = pp.Polynomial.from_sympy(x + 0.25 + v, [x,y,v,w])
	ft_y = pp.Polynomial.from_sympy(y + w, [x,y,v,w])
	
	# new
	Sigma = 0.1**2 * np.eye(2)

	f = list()
	for t in range(T):
		f.append([ft_x, ft_y])

	tot_deg = 8

	print (synthesize_barrier(g, g0, gT, gU, f, Sigma, tot_deg, 'psd'))

if __name__ == '__main__':
	main()
