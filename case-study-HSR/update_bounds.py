# Calculate bounds on the utilities of the customers
# and on the profits on the suppliers

# General
import copy
import numpy as np

# Project
import nested_logit

# Data
import data_HSR as data_file


def updateSubgamePriceBounds(data):

    lb_p_U = copy.deepcopy(data['initial_data']['ub_p_urban'])
    ub_p_U = copy.deepcopy(data['initial_data']['lb_p_urban'])
    lb_p_R = copy.deepcopy(data['initial_data']['ub_p_rural'])
    ub_p_R = copy.deepcopy(data['initial_data']['lb_p_rural'])

    print('\nAlt  LB_P Urban  UB_P Urban  LB_P Rural  UB_P Rural')
    for i in range(data['I_opt_out'], data['I_tot']):
        k = data['operator'][i]
        for s in data['list_strategies_opt'][k]:
            lb_p_U[i] = min(lb_p_U[i], data['strategies']['prices_urban'][i][s])
            ub_p_U[i] = max(ub_p_U[i], data['strategies']['prices_urban'][i][s])
            lb_p_R[i] = min(lb_p_R[i], data['strategies']['prices_rural'][i][s])
            ub_p_R[i] = max(ub_p_R[i], data['strategies']['prices_rural'][i][s])
        print(' {:2d}     {:7.2f}     {:7.2f}     {:7.2f}     {:7.2f}'
            .format(i, lb_p_U[i], ub_p_U[i], lb_p_R[i], ub_p_R[i]))
    
    data['lb_p_urban'] = copy.deepcopy(lb_p_U)
    data['ub_p_urban'] = copy.deepcopy(ub_p_U)
    data['lb_p_rural'] = copy.deepcopy(lb_p_R)
    data['ub_p_rural'] = copy.deepcopy(ub_p_R)
    data['subgame_lb_p_urban'] = copy.deepcopy(lb_p_U)
    data['subgame_ub_p_urban'] = copy.deepcopy(ub_p_U)
    data['subgame_lb_p_rural'] = copy.deepcopy(lb_p_R)
    data['subgame_ub_p_rural'] = copy.deepcopy(ub_p_R)


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

                if data['endo_coef'][i, n] <= 0:
                    if data['ORIGIN'][n] == 1: 
                        lb_U[i, n, r] = data['endo_coef'][i, n] * data['ub_p_urban'][i] + data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r]
                        ub_U[i, n, r] = data['endo_coef'][i, n] * data['lb_p_urban'][i] + data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r]
                    elif data['ORIGIN'][n] == 0: 
                        lb_U[i, n, r] = data['endo_coef'][i, n] * data['ub_p_rural'][i] + data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r]
                        ub_U[i, n, r] = data['endo_coef'][i, n] * data['lb_p_rural'][i] + data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r]
                else:
                    raise Exception('\nERROR! Positive beta_cost coefficient!')

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
        data['M_Rev'][k] = np.amax(np.multiply(data['ub_p_urban'], data['Pop']) - data['fixed_cost'])


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
    
    #Precompute exogenous terms
    data_file.preprocessUtilities(data)
    
    if data['DCM'] == 'NestedLogit':
        #Calculate initial values of logsum terms
        nested_logit.logsumNestedLogit(data)
            
    #Calculate utility bounds
    updateUtilityBounds(data)