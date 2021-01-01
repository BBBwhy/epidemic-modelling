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


def initialize_attributes(Graph, state_param, infected_nodes, _DECIMAL):
    G = Graph.copy()

    # infected_nodes - list of infected nodes or number of infected nodes. In case of number it is randomly selected from thelist of nodes

    attr_dict = dict()
    edge_attr = dict()

    if type(infected_nodes) == int:
        infected_nodes_list = random.sample(G.nodes, int(infected_nodes))
    if type(infected_nodes) == list:
        infected_nodes_list = infected_nodes

    for i in G.nodes:
        attr_dict[i] = {'active': False,
                        'state': 'S',
                        'time_infect': None,
                        'time_carrier_recover': None,
                        'time_carrier_infected': None,
                        'time_infect_recover': None
                        }
        if i in infected_nodes_list:
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


def active_nodes_update(Graph, t, node, active_nodes, _DECIMAL, m):
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
                    d_t = min(abs(G.nodes[node]['time_activate'] - G.nodes[j]['time_inactivate']),
                              abs(G.nodes[node]['time_activate'] - G.nodes[node]['time_inactivate']))
                    d_t = round(d_t, _DECIMAL)
                    G.add_edge(node, j, perm=False, color='r', T=d_t)  # G is updated




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


def DTDG(Graph, sampling_rate, state_param, T, _DECIMAL, m):
    # G - dynamic graph
    # sampling_rate - snapshot of the graph each sampling_rate
    # T - maximum time of simulation
    active_nodes = list()  # list of active nodes
    G = Graph.copy()
    G_dict = {0: G.copy()}
    count = 0

    assert sampling_rate < T * 10 ** _DECIMAL, "Sampling rate higher than maximum time steps in simulation"

    for t in range(0, T * 10 ** _DECIMAL, 1):
        t = t / 10 ** _DECIMAL

        for i in G.nodes:
            G, active_nodes = active_nodes_update(G, t, i, active_nodes, _DECIMAL, m)  # active nodes
            G, active_nodes = inactive_nodes_update(G, t, i, active_nodes, state_param, _DECIMAL)  # inactive nodes

        count = count + 1
        if count == sampling_rate:
            G_dict[t] = G.copy()
            count = 0

    return G_dict

def print_graph(G, pos):
    colors = [G[x][y]['color'] for x,y in G.edges]
    labels = {i: G.nodes[i]['state'] for i in G.nodes}
    plt.figure()
    plt.clf()
    nx.draw(G, pos=pos, edge_color=colors, labels = labels)
    plt.show()

def carrier_transfer(beta, delta_t, mul=1):
    prob = (1-np.exp(-(1/beta)*delta_t))*mul
    return np.random.uniform(0,1) <= prob

def carrier_infecting(Graph, node, state_param, sampling_rate, _DECIMAL):
    delta_t_l = sampling_rate / 10 ** _DECIMAL
    G = Graph.copy()
    if G.nodes[node]['state'] == 'C':
        for i in G[node]:
            if G[node][i]['perm']:
                if G.nodes[i]['state']=='S' and carrier_transfer(state_param[G.nodes[node]['state']]['beta'],delta_t_l):
                    G.nodes[i]['state'] = 'C'
            else:
                if carrier_transfer(state_param[G.nodes[node]['state']]['beta'],G[node][i]['T']) and G.nodes[i]['state']=='S':
                    G.nodes[i]['state'] = 'C'
    return G

def infected_infecting(Graph, node, state_param, sampling_rate, _DECIMAL):
    delta_t_l = sampling_rate / 10 ** _DECIMAL
    G = Graph.copy()
    if G.nodes[node]['state'] == 'I':
        for i in G[node]:
            if G[node][i]['perm']:
                if G.nodes[i]['state']=='S' and carrier_transfer(state_param[G.nodes[node]['state']]['beta'],delta_t_l, (1/state_param[G.nodes[node]['state']]['sigma_1'])/((1/state_param[G.nodes[node]['state']]['sigma_1'])+(1/state_param[G.nodes[node]['state']]['sigma_2']))):
                    G.nodes[i]['state'] = 'C'
            else:
                if carrier_transfer(state_param[G.nodes[node]['state']]['beta'],G[node][i]['T'], (1/state_param[G.nodes[node]['state']]['sigma_1'])/((1/state_param[G.nodes[node]['state']]['sigma_1'])+(1/state_param[G.nodes[node]['state']]['sigma_2']))) and G.nodes[i]['state']=='S':
                    G.nodes[i]['state'] = 'C'
    return G


def change_node_state(Graph, new_states):
    G = Graph.copy()
    for i in G.nodes:
        G.nodes[i]['state'] = new_states[i]

    return G


def next_state(Graph, sampling_rate, state_param, _DECIMAL):
    G = Graph.copy()
    delta_t_l = sampling_rate / 10 ** _DECIMAL
    new_states = dict()

    for node in G.nodes:
        if G.nodes[node]['state'] == 'C':
            nu_i = state_param[G.nodes[node]['state']]['nu_I']
            nu_r = state_param[G.nodes[node]['state']]['nu_R']
            prob_C = 1 - np.exp(-(1 / nu_i + 1 / nu_r) * delta_t_l)
            prob_I = nu_i / (nu_i + nu_r) * np.exp(-(1 / nu_i + 1 / nu_r) * delta_t_l)
            #prob_R = nu_r / (nu_i + nu_r) * np.exp(-(1 / nu_i + 1 / nu_r) * delta_t_l)

            prob = np.random.uniform(0, 1)

            if prob < prob_C:
                new_states[node] = 'C'
            elif prob_C <= prob < prob_C + prob_I:
                new_states[node] = 'I'
            elif prob_C + prob_I <= prob:
                new_states[node] = 'R'


        elif G.nodes[node]['state'] == 'I':

            gamma_i = state_param[G.nodes[node]['state']]['gamma']
            prob = 1 - np.exp(-(1 / gamma_i) * delta_t_l)

            if np.random.uniform(0, 1) <= prob:
                new_states[node] = 'I'
            else:
                new_states[node] = 'R'

        else:
            new_states[node] = G.nodes[node]['state']

    return new_states


if __name__=='__main__':
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

    G0 = nx.erdos_renyi_graph(100, 0.1, seed=None, directed=False)
    pos = nx.circular_layout(G0)

    S = [0]
    G = initialize_attributes(G0, state_param, S, DECIMAL)
    total = 5 #Q
    sampling_rate = 10 #sampling_time
    T = 10

    cols = int(T*10**DECIMAL/sampling_rate+1)

    node_info = {'S': np.zeros((total, cols)),
                 'C': np.zeros((total, cols)),
                 'I': np.zeros((total, cols)),
                 'R': np.zeros((total, cols))}

    for ind in range(total):


        G_dict = DTDG(G, sampling_rate, state_param, T, DECIMAL, m)

        new_states = list()

        for t, Graph in enumerate(G_dict.values()):

            try:
                Graph = change_node_state(Graph, new_states)
            except:
                print("First iteration")
            for i in S:
                Graph = carrier_infecting(Graph, i, state_param, sampling_rate, DECIMAL)
                Graph = infected_infecting(Graph, i, state_param, sampling_rate, DECIMAL)

            new_states = next_state(Graph, sampling_rate, state_param, DECIMAL)
            S.clear()

            for node, state in new_states.items():
                if state == 'C' or state == 'I':
                    S.append(node)

            for i in Graph.nodes:
                node_info[Graph.nodes[i]['state']][ind, t] = node_info[Graph.nodes[i]['state']][ind, t] + 1

            # print_graph(Graph, pos)

    analysis = {'S': node_info['S'].sum(axis=0) / total,
                'C': node_info['C'].sum(axis=0) / total,
                'I': node_info['I'].sum(axis=0) / total,
                'R': node_info['R'].sum(axis=0) / total}

    plt.figure()
    plt.plot(list(G_dict.keys())[:20], analysis['S'][:20], color='red', label='Susceptible')
    plt.plot(list(G_dict.keys())[:20], analysis['C'][:20], color='green', label='Carrier')
    plt.plot(list(G_dict.keys())[:20], analysis['I'][:20], color='blue', label='Infected')
    plt.plot(list(G_dict.keys())[:20], analysis['R'][:20], color='black', label='Recovered')
    plt.legend()
    plt.show()
