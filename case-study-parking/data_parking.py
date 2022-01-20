'''
Data used in Section 5.2 of the following article:
S. Bortolomiol, V. Lurkin, M. Bierlaire (2021).
A Simulation-Based Heuristic to Find Approximate Equilibria with
Disaggregate Demand Models. Transportation Science 55(5):1025-1045.
https://doi.org/10.1287/trsc.2021.1071

Discrete choice model derived from the following article:
Ibeas A, Dell'Olio L, Bordagaray M, Ortuzar Jd D (2014).
Modelling parking choices considering user heterogeneity.
Transportation Res. Part A Policy Practice 70:41-49.
'''
import copy
import numpy as np

def discrete_choice_model(dict):
    
    # Define the type of discrete choice model
    dict['DCM'] = 'MixedLogit'
    dict['N'] = 11

    ##########################################################
    # DISCRETE CHOICE MODEL PARAMETERS
    ##########################################################

    # Alternative Specific Coefficients
    dict['ASC_FSP'] = 0.0   # Free Street Parking
    dict['ASC_PSP'] = 32.0  # Paid Street Parking
    dict['ASC_PUP'] = 34.0  # Paid Underground Parking

    # Beta coefficients
    dict['Beta_TD'] = -0.612
    dict['Beta_Origin'] = -5.762
    dict['Beta_Age_Veh'] = 4.037
    dict['Beta_FEE_INC_PSP'] = -10.995
    dict['Beta_FEE_RES_PSP'] = -11.440
    dict['Beta_FEE_INC_PUP'] = -13.729
    dict['Beta_FEE_RES_PUP'] = -10.668

    # Access time coefficient (random parameter with normal distribution)
    dict['Beta_AT'] = np.random.normal(-0.788, 1.064, size=(dict['N'], dict['R']))
    # Fee coefficient (random parameter with normal distribution)
    dict['Beta_FEE'] = np.random.normal(-32.328, 14.168, size=(dict['N'], dict['R']))

    ### Alternatives' features

    # Access times to parking (AT)
    dict['AT_FSP'] = 10
    dict['AT_PSP'] = 10
    dict['AT_PUP'] = 5

    # Access time to final destination from the parking space (TD)
    dict['TD_FSP'] = 10
    dict['TD_PSP'] = 10
    dict['TD_PUP'] = 10


def supply(dict):
    
    # Number of endogenous operators
    dict['K'] = 2

    ##########################################################
    # Alternatives
    ##########################################################

    # Number of endogenous alternatives in the choice set
    dict['I'] = 2
    # Number of opt-out alternatives in the choice set
    dict['I_opt_out'] = 1
    # Size of the universal choice set
    dict['I_tot'] = dict['I'] + dict['I_opt_out']

    ##########################################################
    # Attributes of the alternatives
    ##########################################################

    # Identify which supplier controls which alternatives (0 = opt-out options)
    # Opt-out options must be listed before endogenous alternatives!
    dict['operator'] = np.array([0, 1, 2])

    # Generate list of alternatives belonging to supplier k
    dict['list_alt_supplier'] = {}
    for k in range(dict['K'] + 1):
        dict['list_alt_supplier'][k] = [i for i, op in enumerate(dict['operator']) if op == k]

    # Mapping between alternatives index and their names
    dict['name_mapping'] = {0: 'FSP', 1: 'PSP', 2: 'PUP'}

    ##########################################################
    # Supply costs, prices, bounds
    ##########################################################

    # Fixed costs (running a service)
    dict['fixed_cost'] = np.zeros([dict['I_tot']])
    # Variable costs (customer service)
    #dict['customer_cost'] = np.zeros([dict['I_tot']])
    dict['customer_cost'] = np.array([0.0, 0.25, 0.50])

    # Lower and upper bound on prices
    dict['lb_p'] = np.array([0, 0.25, 0.50]) # lower bound (FSP, PSP, PUP)
    dict['ub_p'] = np.array([0, 1.00, 1.50]) # upper bound (FSP, PSP, PUP)

    # Initial price
    dict['price'] = (dict['ub_p'] + dict['lb_p']) / 2.0

    # Initial supply strategies
    dict['p_fixed'] = copy.deepcopy(dict['price'])

    dict['best_response_lb_p'] = copy.deepcopy(dict['lb_p'])
    dict['best_response_ub_p'] = copy.deepcopy(dict['ub_p'])

    #dict['disc_residents_PUP'] = 0.25    #0.25 = 25%
    dict['disc_residents_PUP'] = 0


def demand(dict):

    # Number of customers
    dict['Pop'] = 50

    ##########################################################
    # Customer socio-economic characteristics:
    # origin, age of vehicle, income, resident
    ##########################################################
    #                           1  2  3  4  5             10                            20                            30                            40                            50
    dict['Origin'] =  np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1])
    dict['Age_veh'] = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0])
    dict['Low_inc'] = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0])
    dict['Res'] =     np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1])


def groups(data):

    # Define number of segments
    # Origin (2) * AgeVeh (2) * Income (2) * Resident (2)
    data['N'] = 11

    # n     Origin      AgeVeh      LowInc          Resident
    # 0     0           0           0               0 
    # 1     0           0           1               0 
    # 2     0           0           1               1 
    # 3     0           1           0               0 
    # 4     0           1           0               1 
    # 5     0           1           1               0
    # 6     0           1           1               1 
    # 7     1           0           0               1 
    # 8     1           0           1               1 
    # 9     1           1           0               1 
    # 10     1           1           1               1 

    # Count population in each group
    data['popN'] = np.zeros((data['N']))
    data['ORIGIN'] = np.zeros((data['N']))
    data['AGE_VEH'] = np.zeros((data['N']))
    data['LOW_INC'] = np.zeros((data['N']))
    data['RESIDENT'] = np.zeros((data['N']))

    # Assign people to segments
    for n in range(data['Pop']):
        if data['Origin'][n] == 0:
            if data['Age_veh'][n] == 0:
                if data['Low_inc'][n] == 0:
                    data['popN'][0] += 1
                    data['ORIGIN'][0] = 0
                    data['AGE_VEH'][0] = 0
                    data['LOW_INC'][0] = 0
                    data['RESIDENT'][0] = 0
                elif data['Low_inc'][n] == 1 and data['Res'][n] == 0:
                    data['popN'][1] += 1
                    data['ORIGIN'][1] = 0
                    data['AGE_VEH'][1] = 0
                    data['LOW_INC'][1] = 1
                    data['RESIDENT'][1] = 0 
                elif data['Low_inc'][n] == 1 and data['Res'][n] == 1:
                    data['popN'][2] += 1
                    data['ORIGIN'][2] = 0
                    data['AGE_VEH'][2] = 0
                    data['LOW_INC'][2] = 1
                    data['RESIDENT'][2] = 1
            elif data['Age_veh'][n] == 1:
                if data['Low_inc'][n] == 0 and data['Res'][n] == 0:
                    data['popN'][3] += 1
                    data['ORIGIN'][3] = 0
                    data['AGE_VEH'][3] = 1
                    data['LOW_INC'][3] = 0
                    data['RESIDENT'][3] = 0
                elif data['Low_inc'][n] == 0 and data['Res'][n] == 1:
                    data['popN'][4] += 1
                    data['ORIGIN'][4] = 0
                    data['AGE_VEH'][4] = 1
                    data['LOW_INC'][4] = 0
                    data['RESIDENT'][4] = 1                
                elif data['Low_inc'][n] == 1 and data['Res'][n] == 0:
                    data['popN'][5] += 1
                    data['ORIGIN'][5] = 0
                    data['AGE_VEH'][5] = 1
                    data['LOW_INC'][5] = 1
                    data['RESIDENT'][5] = 0
                elif data['Low_inc'][n] == 1 and data['Res'][n] == 1:
                    data['popN'][6] += 1
                    data['ORIGIN'][6] = 0
                    data['AGE_VEH'][6] = 1
                    data['LOW_INC'][6] = 1
                    data['RESIDENT'][6] = 1
        elif data['Origin'][n] == 1:
            if data['Age_veh'][n] == 0 and data['Low_inc'][n] == 0:
                    data['popN'][7] += 1
                    data['ORIGIN'][7] = 1
                    data['AGE_VEH'][7] = 0
                    data['LOW_INC'][7] = 0
                    data['RESIDENT'][7] = 1
            elif data['Age_veh'][n] == 0 and data['Low_inc'][n] == 1:
                    data['popN'][8] += 1
                    data['ORIGIN'][8] = 1
                    data['AGE_VEH'][8] = 0
                    data['LOW_INC'][8] = 1
                    data['RESIDENT'][8] = 1
            elif data['Age_veh'][n] == 1 and data['Low_inc'][n] == 0:
                    data['popN'][9] += 1
                    data['ORIGIN'][9] = 1
                    data['AGE_VEH'][9] = 1
                    data['LOW_INC'][9] = 0
                    data['RESIDENT'][9] = 1
            elif data['Age_veh'][n] == 1 and data['Low_inc'][n] == 1:
                    data['popN'][10] += 1
                    data['ORIGIN'][10] = 1
                    data['AGE_VEH'][10] = 1
                    data['LOW_INC'][10] = 1
                    data['RESIDENT'][10] = 1


def setAlgorithmParameters(dict):

    ##########################################################
    # Parameters needed in the algorithmic framework
    ##########################################################

    dict['nEquilibria'] = 5

    #### Parameters of the fixed-point iteration algorithm

    # Initial optimizer
    dict['optimizer'] = 1
    # Max iter
    dict['max_iter'] = 20                          #Modify here for testing
    # Tolerance for equilibrium convergence
    dict['tolerance_equilibrium'] = 0.001
    # Tolerance for cycle convergence
    dict['tolerance_cyclic_equilibrium'] = 0.01
    # Counter of how many times the fixed-point iteration algorithm is used
    dict['countFixedPointIter'] = 1     #Initialized to 1

    #### Parameters of the choice-based optimization model

    dict['lb_profit'] = None

    #### Parameters for the eps-equilibrium conditions

    dict['eps_equilibrium_profit'] = 0.01   #Accepted % of profit increase      #Modify here for testing
    dict['eps_equilibrium_price'] = 0.20    #25% of price change

    #### Parameters of the fixed-point MIP model

    # Number of strategies for each supplier in the initial fixed-point game
    #                                    0  1  2 
    dict['n_strategies'] =     np.array([0, 5, 5])
    
    dict['min_strategies'] = 5
    dict['max_strategies'] = 10


def getData():
    '''Construct a dictionary 'dict' containing all the input data'''

    # Initialize the output dictionary
    dict = {}

    # Name of the instance
    dict['Instance'] = 'Parking_MixedLogit'

    # Number of draws
    dict['R'] = 100

    # Set random seed
    np.random.seed(10)
    
    # 1) Read discrete choice model parameters
    # 2) Read supply data
    # 3) Read demand data
    # 4) Generate groups of customers
    discrete_choice_model(dict)
    supply(dict)
    demand(dict)
    groups(dict)
    
    # Random term (Gumbel distributed 0,1)
    dict['xi'] = np.random.gumbel(size=(dict['I_tot'], dict['N'], dict['R']))

    # Define parameters of the algorithm
    setAlgorithmParameters(dict)

    ##########################################################
    # Deepcopy of the initial data (for restarts)
    ##########################################################
    dict['initial_data'] = copy.deepcopy(dict)

    return dict


def preprocessUtilities(data):    

    ##########################################################
    # Exogenous utilities and endogenous parameters
    ##########################################################
    exo_utility = np.empty([data['I_tot'], data['N'], data['R']])
    endo_coef = np.full([data['I_tot'], data['N'], data['R']], 0.0)

    for n in range(data['N']):
        for i in range(data['I_tot']):
            for r in range(data['R']):
    
                if i == 0:
                    # Opt-Out
                    exo_utility[i, n] = (data['Beta_AT'][n,r] * data['AT_FSP'] +
                                        data['Beta_TD'] * data['TD_FSP'] +
                                        data['Beta_Origin'] * data['Origin'][n])
                elif i == 1:
                    # PSP
                    exo_utility[i, n] = (data['ASC_PSP'] +
                                        data['Beta_AT'][n,r] * data['AT_PSP'] +
                                        data['Beta_TD'] * data['TD_PSP'])
                else:
                    # PUP
                    exo_utility[i, n] = (data['ASC_PUP'] +
                                        data['Beta_AT'][n,r] * data['AT_PUP'] +
                                        data['Beta_TD'] * data['TD_PUP'] +
                                        data['Beta_Age_Veh'] * data['Age_veh'][n])
        
    data['exo_utility'] = exo_utility

    # Beta coefficient for endogenous variables
    beta_FEE_PSP = np.empty([data['N'], data['R']])
    beta_FEE_PUP = np.empty([data['N'], data['R']])
    for n in range(data['N']):
        for r in range(data['R']):
            beta_FEE_PSP[n,r] = (data['Beta_FEE'][n,r] +
                            data['Beta_FEE_INC_PSP'] * data['Low_inc'][n] +
                            data['Beta_FEE_RES_PSP'] * data['Res'][n])
            beta_FEE_PUP[n,r] = (data['Beta_FEE'][n,r] +
                            data['Beta_FEE_INC_PUP'] * data['Low_inc'][n] +
                            data['Beta_FEE_RES_PUP'] * data['Res'][n])
        
    data['endo_coef'] = np.array([np.zeros([data['N'], data['R']]), beta_FEE_PSP, beta_FEE_PUP])


def printCustomers(f, data):
    
    ### Print general information
    print('\nI   = {:5d} \nN   = {:5d} \nPop = {:5d} \nR   = {:5d}'.format(data['I_tot'], data['N'], data['Pop'], data['R']))

    ### Print aggregate customer data by travel purpose and origin

    ORIGIN1 = int(sum([data['popN'][n] for n in range(data['N']) if data['ORIGIN'][n] == 1]))
    ORIGIN0 = int(sum([data['popN'][n] for n in range(data['N']) if data['ORIGIN'][n] == 0]))
    AGEVEH1 = int(sum([data['popN'][n] for n in range(data['N']) if data['AGE_VEH'][n] == 1]))
    AGEVEH0 = int(sum([data['popN'][n] for n in range(data['N']) if data['AGE_VEH'][n] == 0]))
    LOWINC1 = int(sum([data['popN'][n] for n in range(data['N']) if data['LOW_INC'][n] == 1]))
    LOWINC0 = int(sum([data['popN'][n] for n in range(data['N']) if data['LOW_INC'][n] == 0]))
    RESIDENT1 = int(sum([data['popN'][n] for n in range(data['N']) if data['RESIDENT'][n] == 1]))
    RESIDENT0 = int(sum([data['popN'][n] for n in range(data['N']) if data['RESIDENT'][n] == 0]))
    RESIDENTHIGH = int(sum([data['popN'][n] for n in range(data['N']) if data['RESIDENT'][n] == 1 and data['LOW_INC'][n] == 0]))
    RESIDENTLOW = int(sum([data['popN'][n] for n in range(data['N']) if data['RESIDENT'][n] == 1 and data['LOW_INC'][n] == 1]))
    NONRESHIGH = int(sum([data['popN'][n] for n in range(data['N']) if data['RESIDENT'][n] == 0 and data['LOW_INC'][n] == 0]))
    NONRESLOW = int(sum([data['popN'][n] for n in range(data['N']) if data['RESIDENT'][n] == 0 and data['LOW_INC'][n] == 1]))

    print('\nAGGREGATE CUSTOMER DATA (ORIGIN, AGE VEHICLE, INCOME, RESIDENCY):\n\nTotal customers    {:6d}'.format(data['Pop']))
    print(' Origin internal    {:5d}   {:5.3f}\n Origin external    {:5d}   {:5.3f}'
            .format(ORIGIN1, ORIGIN1/data['Pop'], ORIGIN0, ORIGIN0/data['Pop']))
    print(' Vehicle new (<3)   {:5d}   {:5.3f}\n Vehicle old (>3)   {:5d}   {:5.3f}'
            .format(AGEVEH1, AGEVEH1/data['Pop'], AGEVEH0, AGEVEH0/data['Pop']))
    print(' Income low (<1200) {:5d}   {:5.3f}\n Income high (>1200){:5d}   {:5.3f}'
            .format(LOWINC1, LOWINC1/data['Pop'], LOWINC0, LOWINC0/data['Pop']))
    print(' Resident           {:5d}   {:5.3f}\n Non-resident       {:5d}   {:5.3f}'
            .format(RESIDENT1, RESIDENT1/data['Pop'], RESIDENT0, RESIDENT0/data['Pop']))
    print(' Resident high      {:5d}   {:5.3f}\n Resident low       {:5d}   {:5.3f}'
            .format(RESIDENTHIGH, RESIDENTHIGH/data['Pop'], RESIDENTLOW, RESIDENTLOW/data['Pop']))
    print(' Non-resident high  {:5d}   {:5.3f}\n Non-resident low   {:5d}   {:5.3f}'
            .format(NONRESHIGH, NONRESHIGH/data['Pop'], NONRESLOW, NONRESLOW/data['Pop']))


if __name__ == '__main__':
            
    # Read instance
    data = getData()
    # Precompute exogenous part of the utility and beta_cost parameters
    preprocessUtilities(data)

    # Print list of customers
    printCustomers(data)
