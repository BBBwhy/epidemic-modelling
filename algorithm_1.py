import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
import random
from parameter_list import *

def generate_time(dist, _DECIMAL):
    delta_t = np.random.exponential(dist)
    while delta_t < 5 / 10 ** (_DECIMAL + 1):
        delta_t = np.random.exponential(dist)

    return round(delta_t, _DECIMAL)


def initialize_attributes(Graph, state_param, number_of_infected, _DECIMAL):

    G = Graph.copy()
    attr_dict = dict()
    edge_attr = dict()

    infected_nodes = random.sample(G.nodes, number_of_infected)

    for i in G.nodes:
        attr_dict[i] = {'active': False,
                        'state': 'S',
                        'time_infect': None,
                        'time_carrier_recover': None,
                        'time_carrier_infected': None,
                        'time_infect_recover': None
                        }
        if i in infected_nodes:
            attr_dict[i]['state'] = 'C'

        attr_dict[i]['time_activate'] = generate_time(state_param[attr_dict[i]['state']]['sigma_1'], _DECIMAL)

        attr_dict[i]['time_inactivate'] = round(
            attr_dict[i]['time_activate'] + generate_time(state_param[attr_dict[i]['state']]['sigma_2'], _DECIMAL),
            _DECIMAL)

        assert (attr_dict[i]['time_activate'] < attr_dict[i]['time_inactivate'])

    nx.set_node_attributes(G, attr_dict)

    for edge in G.edges:
        edge_attr[edge] = {'perm': True, 'color': 'b'}

    nx.set_edge_attributes(G, edge_attr)

    return G




def active_nodes_update(Graph, t, node, active_nodes, m):

    G = Graph.copy()
    # G is a graph
    # t is a moment in time
    # node which node is analyzed
    # active_nodes list of active nodes in the moment

    if G.nodes[node]['time_activate'] == t:

        G.nodes[node]['active'] = True  # node is activated
        try:

            new_edge_nodes = random.sample(active_nodes, m)

            for j in new_edge_nodes:
                if j not in G[node]:
                    G.add_edge(node, j, perm=False, color='r')  # G is updated



        except:
            print("############EXCEPT############")  # exception when number of active nodes is lower than m
        active_nodes.append(node)  # active nodes are updated

    assert (type(active_nodes) == list)

    return G, active_nodes


def inactive_nodes_update(Graph, t, node, active_nodes, state_param, _DECIMAL):

    G = Graph.copy()
    rem_list = list()

    if G.nodes[node]['time_inactivate'] == t:

        G.nodes[node]['active'] = False  # node is inactive

        t_act = generate_time(state_param[G.nodes[node]['state']]['sigma_1'], _DECIMAL)
        t_deact = generate_time(state_param[G.nodes[node]['state']]['sigma_2'], _DECIMAL)

        G.nodes[node]['time_activate'] = round(t + t_act, _DECIMAL)
        G.nodes[node]['time_inactivate'] = round(G.nodes[node]['time_activate'] + t_deact, _DECIMAL)
        active_nodes.remove(node)

        for j in G[node]:
            if not G[node][j]['perm']:
                rem_list.append(j)
        G.remove_edges_from([(node, k) for k in rem_list])

    return G, active_nodes


def infect_neighbor(Graph, t, node, _DECIMAL):

    G = Graph.copy()
    if G.nodes[node]['state'] == 'C' or G.nodes[node]['state'] == 'I':
        for j in G[node]:
            try:
                infect_t = min(G.nodes[j]['time_infect'],
                               round(t + generate_time(state_param[G.nodes[node]['state']]['beta'], _DECIMAL),
                                     _DECIMAL))
            except:
                infect_t = round(t + generate_time(state_param[G.nodes[node]['state']]['beta'], _DECIMAL), _DECIMAL)

            G.nodes[j]['time_infect'] = infect_t

    return G

def carrier_next_state(Graph, t, node, _DECIMAL):

    G = Graph.copy()
    if G.nodes[node]['state'] == 'C':
        #carrier node either recovers or gets infected
        #runs from carrier state to I or R
        if (G.nodes[node]['time_carrier_recover'] == None) and (G.nodes[node]['time_carrier_infected'] == None):
            t_i = generate_time(state_param[G.nodes[node]['state']]['nu_I'], _DECIMAL) #time when the node becomes infected
            t_r = generate_time(state_param[G.nodes[node]['state']]['nu_R'], _DECIMAL) #time when the node recovers
            if t_i - t_r >0:
                G.nodes[node]['time_carrier_recover'] = round(t + t_r, _DECIMAL)
            else:
                G.nodes[node]['time_carrier_infected'] = round(t + t_i, _DECIMAL)
    return G

def infected_next_state(Graph, t, node, _DECIMAL):

    G = Graph.copy()
    if G.nodes[node]['state'] == 'I' and G.nodes[node]['time_infect_recover']==None:
        G.nodes[node]['time_infect_recover'] = round(t + generate_time(state_param[G.nodes[node]['state']]['gamma'], _DECIMAL), _DECIMAL)
    return G


if __name__=='__main__':

    G = nx.erdos_renyi_graph(100, 0.1, seed=None, directed=False)
    pos = nx.circular_layout(G)

    active_nodes = list()  # list of active nodes

    state_param = {'S': {'sigma_1': 1 / parameters['sigma_1_H'],
                         'sigma_2': 1 / parameters['sigma_2_H']
                         },

                   'C': {'sigma_1': 1 / parameters['sigma_1_H'],
                         'sigma_2': 1 / parameters['sigma_2_H'],
                         'beta': 1 / parameters['beta_C'],
                         'nu_I': 1 / parameters['nu_I'],
                         'nu_R': 1 / parameters['nu_R']},
                   'I': {'sigma_1': 1 / parameters['sigma_1_I'],
                         'sigma_2': 1 / parameters['sigma_2_I'],
                         'beta': 1 / parameters['beta_I'],
                         'gamma': 1 / parameters['gamma']},
                   'R': {'sigma_1': 1 / parameters['sigma_1_H'],
                         'sigma_2': 1 / parameters['sigma_2_H']}

                   }

    G = initialize_attributes(G, state_param, 1, DECIMAL)

    result_dict = dict()

    for t in range(0, 3 * 10 ** DECIMAL, 1):
        t = t / 10 ** DECIMAL
        current_state_count = {'S': 0, 'C': 0, 'I': 0, 'R': 0}

        for i in G.nodes:

            if G.nodes[i]['state'] == 'S' and G.nodes[i]['time_infect'] == t:
                G.nodes[i]['state'] = 'C'
            if G.nodes[i]['state'] == 'C' and G.nodes[i]['time_carrier_recover'] == t:
                G.nodes[i]['state'] = 'R'
            if G.nodes[i]['state'] == 'C' and G.nodes[i]['time_carrier_infected'] == t:
                G.nodes[i]['state'] = 'I'
            if G.nodes[i]['state'] == 'I' and G.nodes[i]['time_infect_recover'] == t:
                G.nodes[i]['state'] = 'R'

            G, active_nodes = active_nodes_update(G, t, i, active_nodes, m)  # active nodes
            G, active_nodes = inactive_nodes_update(G, t, i, active_nodes, state_param, DECIMAL)  # inactive nodes
            G = infect_neighbor(G, t, i, DECIMAL)  # infecting neighbor nodes
            G = carrier_next_state(G, t, i, DECIMAL)  # C->I C->R
            G = infected_next_state(G, t, i, DECIMAL)

            current_state_count[G.nodes[i]['state']] = current_state_count[G.nodes[i]['state']] + 1

        result_dict[t] = current_state_count

        '''
        colors = [G[x][y]['color'] for x,y in G.edges]
        labels = {i: G.nodes[i]['state'] for i in G.nodes}
        plt.figure()
        plt.clf()
        nx.draw(G, pos=pos, edge_color=colors, labels = labels)
        plt.pause(0.1)
        plt.show()
        '''

    print("here")
    plt.plot(list(result_dict.keys()), [i['S'] for i in result_dict.values()], color = 'red', label = 'Susceptible')
    plt.plot(list(result_dict.keys()), [i['C'] for i in result_dict.values()], color = 'green', label = 'Carrier')
    plt.plot(list(result_dict.keys()), [i['I'] for i in result_dict.values()], color = 'blue', label = 'Infected')
    plt.plot(list(result_dict.keys()), [i['R'] for i in result_dict.values()], color = 'black', label = 'Recovered')
    plt.legend()
    plt.show()