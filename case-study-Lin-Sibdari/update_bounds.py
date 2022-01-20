# Calculate bounds on the utilities of the customers
# and on the profits on the suppliers

# General
import sys
import time
import copy
import numpy as np

# Data
import Data_LinSibdari_MNL as data_file


def updateSubgamePriceBounds(data):

    lb_p = copy.deepcopy(data['initial_data']['ub_p'])
    ub_p = copy.deepcopy(data['initial_data']['lb_p'])

    print('\nAlt      LB_P      UB_P')
    for i in range(data['I_opt_out'], data['I_tot']):
        k = data['operator'][i]
        for s in data['list_strategies_opt'][k]:
            lb_p[i] = min(lb_p[i], data['strategies']['prices'][i][s])
            ub_p[i] = max(ub_p[i], data['strategies']['prices'][i][s])
        print(' {:2d}   {:7.2f}   {:7.2f}'.format(i, lb_p[i], ub_p[i]))
    
    data['lb_p'] = copy.deepcopy(lb_p)
    data['ub_p'] = copy.deepcopy(ub_p)
    data['subgame_lb_p'] = copy.deepcopy(lb_p)
    data['subgame_ub_p'] = copy.deepcopy(ub_p)


def updateUtilityBounds(data):
    
    # Utility bounds
    lb_U = np.empty([data['I_tot'], data['N'], data['R']])
    ub_U = np.empty([data['I_tot'], data['N'], data['R']])
    lb_Umin = np.full((data['N'], data['R']), np.inf)
    ub_Umax = np.full((data['N'], data['R']), -np.inf)

    M = np.empty([data['N'], data['R']])

    for n in range(data['N']):
        for r in range(data['R']):
            for i in range(data['I_tot']):
                
                # Lin and Sibdari case study
                if data['DCM'] == 'MixedLogit':
                    if data['endo_coef'][i,n,r] <= 0:
                        lb_U[i, n, r] = data['endo_coef'][i,n,r] * data['ub_p'][i] + data['exo_utility'][i,n,r] + data['xi'][i, n, r]
                        ub_U[i, n, r] = data['endo_coef'][i,n,r] * data['lb_p'][i] + data['exo_utility'][i,n,r] + data['xi'][i, n, r]
                    else:
                        lb_U[i, n, r] = data['endo_coef'][i,n,r] * data['lb_p'][i] + data['exo_utility'][i,n,r] + data['xi'][i, n, r]
                        ub_U[i, n, r] = data['endo_coef'][i,n,r] * data['ub_p'][i] + data['exo_utility'][i,n,r] + data['xi'][i, n, r]

                else:
                    lb_U[i, n, r] = data['beta'] * data['ub_p'][i] + data['exo_utility'][i,n] + data['xi'][i, n, r]
                    ub_U[i, n, r] = data['beta'] * data['lb_p'][i] + data['exo_utility'][i,n] + data['xi'][i, n, r]

            # Bounds for each customer, for each draw
            lb_Umin[n, r] = np.min(lb_U[:, n, r])
            ub_Umax[n, r] = np.max(ub_U[:, n, r])

            # Calcule the big-M values
            M[n, r] = ub_Umax[n, r] - lb_Umin[n, r]

    data['lb_U'] = lb_U
    data['ub_U'] = ub_U
    data['lb_Umin'] = lb_Umin
    data['ub_Umax'] = ub_Umax
    data['M_U'] = M


    # Profit bounds
    data['M_Rev'] = {}
    for k in range(1, data['K'] + 1):
        data['M_Rev'][k] = np.amax(np.multiply(data['ub_p'], data['Pop']))


def updateUtilityBoundsLogit(data):
    
    # Utility bounds
    lb_U = np.empty([data['I_tot'], data['N']])
    ub_U = np.empty([data['I_tot'], data['N']])
    lb_Umin = np.full((data['N']), np.inf)
    ub_Umax = np.full((data['N']), -np.inf)

    M = np.empty([data['N']])

    for n in range(data['N']):
        for i in range(data['I_tot']):
            
            # Parking case study
            if i == 0:
                lb_U[i, n] = data['beta'] * data['ub_p'][i]
                ub_U[i, n] = data['beta'] * data['lb_p'][i]
            elif i == 1:
                lb_U[i, n] = data['beta'] * data['ub_p'][i] + data['a_1']
                ub_U[i, n] = data['beta'] * data['lb_p'][i] + data['a_1']
            elif i == 2:
                lb_U[i, n] = data['beta'] * data['ub_p'][i] + data['a_2']
                ub_U[i, n] = data['beta'] * data['lb_p'][i] + data['a_2']

        # Bounds for each customer, for each draw
        lb_Umin[n] = np.min(lb_U[:, n])
        ub_Umax[n] = np.max(ub_U[:, n])

        # Calcule the big-M values
        M[n] = ub_Umax[n] - lb_Umin[n]

    data['lb_U'] = lb_U
    data['ub_U'] = ub_U
    data['lb_Umin'] = lb_Umin
    data['ub_Umax'] = ub_Umax
    data['M_U'] = M

    # Profit bounds
    data['M_Rev'] = {}
    for k in range(1, data['K'] + 1):
        data['M_Rev'][k] = np.amax(np.multiply(data['ub_p'], data['Pop']))


def updateUtilityBoundsMixedLogit(data):
    
    # Utility bounds
    lb_U = np.empty([data['I_tot'], data['N'], data['R']])
    ub_U = np.empty([data['I_tot'], data['N'], data['R']])
    lb_Umin = np.full((data['N'], data['R']), np.inf)
    ub_Umax = np.full((data['N'], data['R']), -np.inf)

    M = np.empty([data['N'], data['R']])

    for n in range(data['N']):
        for i in range(data['I_tot']):
            for r in range(data['R']):
                # Parking case study
                if i == 0:
                    lb_U[i, n, r] = data['endo_coef'][i,n,r] * data['ub_p'][i]
                    ub_U[i, n, r] = data['endo_coef'][i,n,r] * data['lb_p'][i]
                elif i == 1:
                    lb_U[i, n, r] = data['endo_coef'][i,n,r] * data['ub_p'][i] + data['a_1']
                    ub_U[i, n, r] = data['endo_coef'][i,n,r] * data['lb_p'][i] + data['a_1']
                elif i == 2:
                    lb_U[i, n, r] = data['endo_coef'][i,n,r] * data['ub_p'][i] + data['a_2']
                    ub_U[i, n, r] = data['endo_coef'][i,n,r] * data['lb_p'][i] + data['a_2']

        # Bounds for each customer, for each draw
        lb_Umin[n,r] = np.min(lb_U[:,n,r])
        ub_Umax[n,r] = np.max(ub_U[:,n,r])

        # Calcule the big-M values
        M[n,r] = ub_Umax[n,r] - lb_Umin[n,r]

    data['lb_U'] = lb_U
    data['ub_U'] = ub_U
    data['lb_Umin'] = lb_Umin
    data['ub_Umax'] = ub_Umax
    data['M_U'] = M

    # Profit bounds
    data['M_Rev'] = {}
    for k in range(1, data['K'] + 1):
        data['M_Rev'][k] = np.amax(np.multiply(data['ub_p'], data['Pop']))


def updateMaxProfit(data):

    for k in range(1, data['K'] + 1):
        unreachableCustomers = 0
        for i in range(data['I_tot']):
            if data['operator'][i] != k:
                for n in range(data['N']):
                    for r in range(data['R']):
                        if data['w_pre'][i,n,r] == 1:
                            unreachableCustomers += data['popN'][n]/data['R']
        data['M_Rev'][k] = np.amax(np.multiply(data['ub_p'], (data['Pop']-unreachableCustomers)) - data['fixed_cost'])


if __name__ == '__main__':
    
    # Get the data and preprocess
    data = data_file.getData()
            
    #Calculate utility bounds
    updateUtilityBounds(data)
