license_text='''
    Module implements a simple example of translating a syntactically co-safe
    GDTL formula to a deterministic finite state automaton.
    Copyright (C) 2017  Cristian Ioan Vasile <cvasile@mit.edu>
    CSAIL, LIDS, Massachusetts Institute of Technology

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
'''
.. module:: gdtl_dfa_example.py
   :synopsis: Module implements a simple example of translating a syntactically
              co-safe GDTL formula to a deterministic finite state automaton.

.. moduleauthor:: Cristian Ioan Vasile <cvasile@mit.edu>

'''
'''
Created on Dec 4, 2017

@author: cristi
'''

import logging
from collections import deque

from lomap import Fsa
from lomap import Markov as MDP
from gdtl import gdtl2ltl, PredicateContext


def gdtl2fsa(formula):
    tl, ap = gdtl2ltl(formula)
    ltl_formula = tl.formulaString(ltl=True)
    assert tl.isSynCoSafe()
    fsa = Fsa(multi=False)
    fsa.from_formula(ltl_formula)
    # optional add trap state
    fsa.add_trap_state()
    
    logging.info('Automaton size: %s', fsa.size())
    logging.info('Automaton: %s', fsa)
    
    return fsa, tl, ap

def getAPs(state, cov, aps, context):
    '''Computes the set of atomic propositions that are true at the given belief
    node.
    '''
    # update predicate evaluation context with custom symbols
    return set([ap for ap, pred in aps.iteritems()
        if context.evalPred(pred, state, cov, state_label='x', cov_label='P')])

def mdp_times_fsa(mdp, fsa, expand_finals=True):
    '''Computes the product automaton between a Markov decision process and an
    FSA.

    Parameters
    ----------
    mdp: LOMAP Markov Decision Process

    fsa: LOMAP deterministic finite state automaton
    '''

    # Create the product_model
    product_model = MDP()
    # Iterate over initial states of the TS
    for init_ts, init_ts_prob in mdp.init.iteritems():
        init_prop = mdp.g.node[init_ts].get('prop', set())
        # Iterate over the initial states of the FSA
        for init_fsa in fsa.init:
            # Add the initial states to the graph and mark them as initial
            act_init_fsa = fsa.next_state(init_fsa, init_prop)
            if act_init_fsa is not None:
                init_state = (init_ts, act_init_fsa)
                product_model.init[init_state] = init_ts_prob
                product_model.g.add_node(init_state)
                if act_init_fsa in fsa.final:
                    product_model.final.add(init_state)

    # Add all initial states to the stack
    stack = deque(product_model.init)
    # Consume the stack
    while stack:
        cur_state = stack.pop()
        mdp_state, fsa_state = cur_state

        # skip processing final beyond final states
        if not expand_finals and fsa_state in fsa.final:
            continue

        for _, ts_next_state, action in mdp.g.edges_iter(mdp_state, keys=True):
            ts_next_prop = mdp.g.node[ts_next_state].get('prop', set())
            fsa_next_state = fsa.next_state(fsa_state, ts_next_prop)
            if fsa_next_state is not None:
                next_state = (ts_next_state, fsa_next_state)
                if next_state not in product_model.g:
                    # Add the new state
                    product_model.g.add_node(next_state)
                    # Add transition
                    product_model.g.add_edge(cur_state, next_state, key=action)
                    # Mark as final if final in fsa
                    if fsa_next_state in fsa.final:
                        product_model.final.add(next_state)
                    # Continue search from next state
                    stack.append(next_state)
                elif next_state not in product_model.g[cur_state]:
                    # Add transition
                    product_model.g.add_edge(cur_state, next_state, key=action)

    return product_model

def computePolicy(product_mdp, epsilon=1e-3):

    g = product_mdp.g

    logging.info("Starting value iteration to find policy")

    # Initialize value iteration
    for _, node_data in g.nodes_iter(data=True):
        node_data['value'] = 0. 

    # Dynp function
    def dynp(node_start, action):
        node_end_list = [node_end for node_end in g[node_start]
                                          if action in g[node_start][node_end]]
        return sum(g[node_start][node_end][action]['prob'] * g.node[node_end]['value']
                                                  for node_end in node_end_list)

    # Do policy/value iteration
    changed = True
    pa_policy = {}
    while changed:
        changed = False
        for node_start, node_data in g.nodes_iter(data=True):
            if node_start in product_mdp.final:   # accept
                new_action = node_start[0]   # stay put
                new_value = 1.
            elif node_start[1] != 'trap': # TODO: name should not be hard-coded
                new_value = 0.
            else:
                action_set = set(act for node_end in g[node_start]
                                            for act in g[node_start][node_end])
                new_action = max(action_set, key=lambda action: dynp(node_start, action))
                new_value = dynp(node_start, new_action)

            if new_value > node_data['value'] + epsilon:
                changed = True
                node_data['value'] = new_value
                pa_policy[node_start] = new_action

#     logging.info("Value iteration finished with solution p=%f", ?)
    return pa_policy

def main():
    formula = 'F x > 2 || (P<=8 U custom(x) > 2)'
    predicates = { # define custom predicates here
        'custom' : lambda x: x
    }
    context = PredicateContext(predicates)
    
    fsa, tl, aps = gdtl2fsa(formula)
    
    mdp = MDP(directed=True, multi=True)
    mdp.name = 'Two rooms'
    
    # states
    mdp.g.add_node('room1', prop=getAPs(1, 3, aps, context))
    mdp.g.add_node('room2', prop=getAPs(4, 7, aps, context))
    mdp.init['room1'] = 0.5
    mdp.init['room2'] = 0.5
    # actions
    mdp.act = {'stay', 'leave'}
    # transitions
    mdp.g.add_edge('room1', 'room1', key='stay', prob=1.0)
    mdp.g.add_edge('room1', 'room1', key='leave', prob=0.3)
    mdp.g.add_edge('room1', 'room2', key='leave', prob=0.7)
    mdp.g.add_edge('room2', 'room1', key='stay', prob=0.1)
    mdp.g.add_edge('room2', 'room2', key='stay', prob=0.9)
    mdp.g.add_edge('room2', 'room1', key='leave', prob=0.4)
    mdp.g.add_edge('room2', 'room2', key='leave', prob=0.6)
    # create product mdp
    product_mdp = mdp_times_fsa(mdp, fsa)
    # solve mdp
    policy_pa = computePolicy(product_mdp)

if __name__ == '__main__':
    loglevel = logging.INFO
    logging.basicConfig(level=loglevel)
    main()