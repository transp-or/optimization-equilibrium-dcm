# Functions used to cluster scenarios

# General
import time
import numpy as np

from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample

# Project
import functions


def opportunityCostScenarios(data):

    data['opportunityCostScenario'] = np.zeros([data['R'],data['R']])

    for s1 in range(data['R']):
        for s2 in range(s1+1, data['R']):
            objOptS1whenS2 = functions.objScenario(data, data['all_y_scenario'][s1], s2)
            objOptS2whenS1 = functions.objScenario(data, data['all_y_scenario'][s2], s1)
            data['opportunityCostScenario'][s1,s2] = -(objOptS1whenS2 - data['OF_scenario'][s2] + objOptS2whenS1 - data['OF_scenario'][s1])
            data['opportunityCostScenario'][s2,s1] = data['opportunityCostScenario'][s1,s2]
    
    # Print opportunity cost distance matrix
    if data['R'] <= 20:
        print('\nOpportunity cost matrix (scenarios):\n         ', end='')
        for r1 in range(data['R']):
            print('     {:3.0f}'.format(r1), end='')
        for r1 in range(data['R']):
            print('\n{:3d}      '.format(r1), end='')
            for r2 in range(data['R']):
                print(' {:7.1f}'.format(data['opportunityCostScenario'][r1,r2]), end='')
        print()

def opportunityCostCustomers(data):

    data['opportunityCostCustomer'] = np.zeros([data['N'],data['N']])

    for n1 in range(data['N']):
        for n2 in range(n1+1, data['N']):
            objOptN1whenN2 = functions.objCustomer(data, data['all_y_customer'][n1], n2)
            objOptN2whenN1 = functions.objCustomer(data, data['all_y_customer'][n2], n1)
            data['opportunityCostCustomer'][n1,n2] = -(objOptN1whenN2 - data['OF_customer'][n2] + objOptN2whenN1 - data['OF_customer'][n1])
            data['opportunityCostCustomer'][n2,n1] = data['opportunityCostCustomer'][n1,n2]

def kMedoidsScenario(data):

    t_in = time.time()

    # K-Medoids algorithm to process distance matrix
    kmedoids_instance =kmedoids(data['opportunityCostScenario'], range(data['nClustersR']), data_type='distance_matrix')

    # Run cluster analysis and obtain results
    kmedoids_instance.process()
    print('\nTime to solve k-medoids: {:8.3f}'.format(time.time() - t_in))

    data['list_cluster_R'] = kmedoids_instance.get_clusters()
    data['list_medoids_R'] = kmedoids_instance.get_medoids()
    data['cluster_R'] = np.empty((data['R']))
    for r in range(data['R']):
        for c in range(data['nClustersR']):
            if r in data['list_cluster_R'][c]:
                data['cluster_R'][r] = c

    print('\nScenario   ', end='')
    for r in range(data['R']):
        print('{:3d}'.format(r), end='')
    print('\nCluster    ', end='')
    for r in range(data['R']):
        print('{:3.0f}'.format(data['cluster_R'][r]), end='')
    print()

    print('List scenario clusters: ', data['list_cluster_R'])
    print('List medoids: ', data['list_medoids_R'])

def kMedoidsCustomer(data):

    t_in = time.time()

    # K-Medoids algorithm to process distance matrix
    kmedoids_instance = kmedoids(data['opportunityCostCustomer'], range(data['nClustersN']), data_type='distance_matrix')

    # Run cluster analysis and obtain results
    kmedoids_instance.process()
    print('\nTime to solve k-medoids: {:8.3f}'.format(time.time() - t_in))

    data['list_cluster_N'] = kmedoids_instance.get_clusters()
    data['list_medoids_N'] = kmedoids_instance.get_medoids()
    data['cluster_N'] = np.empty((data['N']))
    for n in range(data['N']):
        for c in range(data['nClustersN']):
            if n in data['list_cluster_N'][c]:
                data['cluster_N'][n] = c

    print('\nCostumer   ', end='')
    for n in range(data['N']):
        print('{:3d}'.format(n), end='')
    print('\nCluster    ', end='')
    for n in range(data['N']):
        print('{:3.0f}'.format(data['cluster_N'][n]), end='')
    print()

    print('List costumer clusters: ', data['list_cluster_N'])
    print('List medoids: ', data['list_medoids_N'])