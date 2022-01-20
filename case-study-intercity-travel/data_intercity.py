# Instance intercity travel market with regulator
import copy
import numpy as np
import math

import nested_logit

# Data
import data_intercity_nested_logit


def setAlgorithmParameters(dict):
    
    ##########################################################
    # Parameters needed in the algorithmic framework
    ##########################################################

    # Initial random seed
    dict['Seed'] = 1
    
    # Number of endogenous operators
    dict['K'] = 2

    # Number of draws
    dict['R'] = 50

    # Number of equilibria
    dict['nEquilibria'] = 1

    #### Parameters of the fixed-point iteration algorithm

    # Initial optimizer
    dict['optimizer'] = 1
    # Max iter
    dict['max_iter'] = 20                          #Modify here for testing (ideally >= 100)
    dict['max_iter_smoothing'] = 2
    # Tolerance to identify identical solutions
    dict['tolerance'] = 0.10

    #### Parameters of the choice-based optimization model

    dict['lb_profit'] = None

    #### Parameters of the nested logit fixed-point

    dict['max_iter_nested_logit'] = 1           # Number of iterations allowed to reach convergence
    dict['tolerance_nested_logit'] = 5.0        # Price tolerance accepted for convergence 
    dict['smoothing'] = 0.333                   # If smoothing = 0, there is no smoothing (update to last found prices)
                                                # For low values of mu_nest, smoothing can be quite low (e.g. 0.2-0.4)

    #### Parameters for the eps-equilibrium conditions

    dict['eps_equilibrium_profit'] = 0.01   #Accepted % of profit increase      #Modify here for testing


def getData(dict):

    # Name of the instance
    dict['Instance'] = 'Schedules_NestedLogit'

    # Set random seed
    np.random.seed(dict['Seed'])

    # 1) Read discrete choice model parameters
    # 2) Read supply data
    # 3) Read demand data
    # 4) Generate groups of customers
    # 5) Generate arrival times

    data_intercity_nested_logit.data_instance(dict)
    data_intercity_nested_logit.regulator(dict)

    # Random term (Gumbel distributed 0,1)
    dict['xi'] = np.random.gumbel(size=(dict['I_tot'], dict['N'], dict['R']))


    ##########################################################
    # Deepcopy of the initial data (for restarts)
    ##########################################################
    dict['initial_data'] = copy.deepcopy(dict)


def preprocessUtilities(data):

    ##########################################################
    # Exogenous utilities and endogenous parameters
    ##########################################################
    exo_utility = np.empty([data['I_tot'], data['N'], data['R']])
    endo_coef = np.full([data['I_tot'], data['N']], 0.0)

    for n in range(data['N']):
        group = int(data['BUSINESS'][n])
        for r in range(data['R']):
            for i in range(data['I_tot']):

                ### Model from Cascetta and Coppola (2012)
                if data['alternatives'][i]['Mode'] == 'Car':
                    exo_utility[i,n,r] = data['ASC_CAR'][group] +\
                                        data['BETA_TTIME'][group] * data['alternatives'][i]['TT']# +\
                                        #data['BETA_MALE'][group] * data['MALE'][n]
                    if data['REIMBURSEMENT'][n] == 1:
                        endo_coef[i, n] = data['BETA_COST_CAR_REIMBURSED'][group]
                    elif data['INCOME'][n] == 1:
                        endo_coef[i, n] = data['BETA_COST_CAR_HIGH_INC'][group]
                    else:
                        endo_coef[i, n] = data['BETA_COST_CAR_LOW_INC'][group]
                elif data['alternatives'][i]['Mode'] == 'Plane':
                    exo_utility[i,n,r] = data['ASC_PLANE'][group] +\
                                        data['BETA_TTIME'][group] * data['alternatives'][i]['TT'] +\
                                        data['BETA_EARLY'][group] * data['EARLY'][i,n,r] +\
                                        data['BETA_LATE'][group] * data['LATE'][i,n,r] +\
                                        data['BETA_ACC_EGR'][group] * (data['ACCESS'][i,n,r] + data['EGRESS'][i,n,r])
                    if data['REIMBURSEMENT'][n] == 1:
                        endo_coef[i, n] = data['BETA_COST_PLANE_REIMBURSED'][group]
                    elif data['INCOME'][n] == 1:
                        endo_coef[i, n] = data['BETA_COST_PLANE_HIGH_INC'][group]
                    else:
                        endo_coef[i, n] = data['BETA_COST_PLANE_LOW_INC'][group]
                elif data['alternatives'][i]['Operator'] == 'IC':
                    exo_utility[i,n,r] = data['ASC_IC'][group] +\
                                        data['BETA_TTIME'][group] * data['alternatives'][i]['TT'] +\
                                        data['BETA_EARLY'][group] * data['EARLY'][i,n,r] +\
                                        data['BETA_LATE'][group] * data['LATE'][i,n,r]+\
                                        data['BETA_ACC_EGR'][group] * (data['ACCESS'][i,n,r] + data['EGRESS'][i,n,r])
                    if data['REIMBURSEMENT'][n] == 1:
                        endo_coef[i, n] = data['BETA_COST_IC_REIMBURSED'][group]
                    elif data['INCOME'][n] == 1:
                        endo_coef[i, n] = data['BETA_COST_IC_HIGH_INC'][group]
                    else:
                        endo_coef[i, n] = data['BETA_COST_IC_LOW_INC'][group]
                elif data['alternatives'][i]['Operator'] == 'HSR_Supplier1':
                    exo_utility[i,n,r] = data['ASC_AV'][group] +\
                                        data['BETA_TTIME'][group] * data['alternatives'][i]['TT'] +\
                                        data['BETA_EARLY'][group] * data['EARLY'][i,n,r] +\
                                        data['BETA_LATE'][group] * data['LATE'][i,n,r] +\
                                        data['BETA_ACC_EGR'][group] * (data['ACCESS'][i,n,r] + data['EGRESS'][i,n,r])
                    if data['REIMBURSEMENT'][n] == 1:
                        endo_coef[i, n] = data['BETA_COST_HSR_REIMBURSED'][group]
                    elif data['INCOME'][n] == 1:
                        endo_coef[i, n] = data['BETA_COST_HSR_HIGH_INC'][group]
                    else:
                        endo_coef[i, n] = data['BETA_COST_HSR_LOW_INC'][group]

    data['exo_utility'] = exo_utility
    data['endo_coef'] = endo_coef



def printCustomers(data):

    ### Print general information
    print('\nI   = {:5d} \nN   = {:5d} \nPop = {:5d} \nR   = {:5d}'.format(data['I_tot'], data['N'], data['Pop'], data['R']))
    print('\nRegulator budget     = {:8.2f}'.format(data['Budget']))
    print('Max tax level H      = {:8.2f}'.format(np.amax(data['ub_tax_highincome'])))
    print('Max tax level L      = {:8.2f}'.format(np.amax(data['ub_tax_lowincome'])))
    print('Max subsidy level H  = {:8.2f}'.format(np.amax(data['ub_subsidy_highincome'])))
    print('Max subsidy level L  = {:8.2f}'.format(np.amax(data['ub_subsidy_lowincome'])))

    ### Print aggregate customer data by travel purpose and origin
    NB0 = int(sum([data['popN'][n] for n in range(data['N']) if data['BUSINESS'][n] == 0 and data['ORIGIN'][n] == 0]))
    NB1 = int(sum([data['popN'][n] for n in range(data['N']) if data['BUSINESS'][n] == 0 and data['ORIGIN'][n] == 1]))
    B0  = int(sum([data['popN'][n] for n in range(data['N']) if data['BUSINESS'][n] == 1 and data['ORIGIN'][n] == 0]))
    B1  = int(sum([data['popN'][n] for n in range(data['N']) if data['BUSINESS'][n] == 1 and data['ORIGIN'][n] == 1]))
    NBL = int(sum([data['popN'][n] for n in range(data['N']) if data['BUSINESS'][n] == 0 and data['INCOME'][n] == 0]))
    NBH = int(sum([data['popN'][n] for n in range(data['N']) if data['BUSINESS'][n] == 0 and data['INCOME'][n] == 1]))
    BL  = int(sum([data['popN'][n] for n in range(data['N']) if data['BUSINESS'][n] == 1 and data['INCOME'][n] == 0]))
    BH  = int(sum([data['popN'][n] for n in range(data['N']) if data['BUSINESS'][n] == 1 and data['INCOME'][n] == 1]))
    BR  = int(sum([data['popN'][n] for n in range(data['N']) if data['BUSINESS'][n] == 1 and data['REIMBURSEMENT'][n] == 1]))
    BN  = int(sum([data['popN'][n] for n in range(data['N']) if data['BUSINESS'][n] == 1 and data['REIMBURSEMENT'][n] == 0]))
    LR  = int(sum([data['popN'][n] for n in range(data['N']) if data['INCOME'][n] == 0 and data['ORIGIN'][n] == 0]))
    LU  = int(sum([data['popN'][n] for n in range(data['N']) if data['INCOME'][n] == 0 and data['ORIGIN'][n] == 1]))
    HR  = int(sum([data['popN'][n] for n in range(data['N']) if data['INCOME'][n] == 1 and data['ORIGIN'][n] == 0]))
    HU  = int(sum([data['popN'][n] for n in range(data['N']) if data['INCOME'][n] == 1 and data['ORIGIN'][n] == 1]))

    print('\nAGGREGATE CUSTOMER DATA (TRAVEL PURPOSE, ORIGIN, INCOME, REIMBURSEMENT):\n\nTotal customers   {:6d}'.format(data['Pop']))
    print(' Business          {:5d}   {:5.3f}\n ..reimbursed      {:5d}   {:5.3f}\n ..non-reimbursed  {:5d}   {:5.3f}'
            .format(BR+BN, (BR+BN)/data['Pop'], BR, BR/data['Pop'], BN, BN/data['Pop']))
    print(' ..urban           {:5d}   {:5.3f}\n ..rural           {:5d}   {:5.3f}'.format(B1, B1/data['Pop'], B0, B0/data['Pop']))
    print(' ..high income     {:5d}   {:5.3f}\n ..low income      {:5d}   {:5.3f}'.format(BH, BH/data['Pop'], BL, BL/data['Pop']))
    print(' Non-business      {:5d}   {:5.3f}\n ..urban           {:5d}   {:5.3f}\n ..rural           {:5d}   {:5.3f}'
            .format(NB1+NB0, (NB1+NB0)/data['Pop'], NB1, NB1/data['Pop'], NB0, NB0/data['Pop']))
    print(' ..high income     {:5d}   {:5.3f}\n ..low income      {:5d}   {:5.3f}'.format(NBH, NBH/data['Pop'], NBL, NBL/data['Pop']))
    print(' Low income        {:5d}   {:5.3f}\n ..urban           {:5d}   {:5.3f}\n ..rural           {:5d}   {:5.3f}'
            .format(LR+LU, (LR+LU)/data['Pop'], LU, LU/data['Pop'], LR, LR/data['Pop']))
    print(' High income       {:5d}   {:5.3f}\n ..urban           {:5d}   {:5.3f}\n ..rural           {:5d}   {:5.3f}'
            .format(HR+HU, (HR+HU)/data['Pop'], HU, HU/data['Pop'], HR, HR/data['Pop']))

    ### Print list of customers
    print('\nLIST OF CUSTOMERS:\n')
    print('  N   R  Business Reimbursed  Income  Origin  DesiredArrTime   ', end=" ")
    for i in range(data['I_tot']):
        print('    {:2d}    '.format(i), end=" ")
    for n in range(data['N']):
        for r in range(min(5,data['R'])):
            if data['DAT'][n,r] != -1:
                print('\n{:3d} {:3d}       {:3.0f}        {:3.0f}     {:3.0f}     {:3.0f}           {:2.0f}:{:02d}  '
                    .format(n, r, data['BUSINESS'][n], data['REIMBURSEMENT'][n], data['INCOME'][n], data['ORIGIN'][n],
                    math.floor(data['DAT'][n,r]/60), int(data['DAT'][n,r]%60)), end =" ")
            else:   
                print('\n{:3d} {:3d}       {:3.0f}        {:3.0f}     {:3.0f}     {:3.0f}            None  '
                    .format(n, r, data['BUSINESS'][n], data['REIMBURSEMENT'][n], data['INCOME'][n], data['ORIGIN'][n]), end =" ")
            for i in range(data['I_tot']):
                print(' {:7.3f}  '.format(data['endo_coef'][i,n] * data['price'][i] + data['exo_utility'][i,n,r] + data['Logsum'][i,n,r]), end=" ")
    print()


if __name__ == '__main__':

    # Initialize the dictionary 'dict' containing all the input/output data
    data = {}

    # Define parameters of the algorithm
    setAlgorithmParameters(data)

    # Read instance
    getData(data)
    # Precompute exogenous part of the utility and beta_cost parameters
    preprocessUtilities(data)

    if data['DCM'] == 'NestedLogit':
        #Calculate initial values of logsum terms
        nested_logit.logsumNestedLogitRegulator(data)

    # Print list of customers
    printCustomers(data)
