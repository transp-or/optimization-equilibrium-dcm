'''
Data used in Section 5.3 of the following article:
S. Bortolomiol, V. Lurkin, M. Bierlaire (2021).
A Simulation-Based Heuristic to Find Approximate Equilibria with
Disaggregate Demand Models. Transportation Science 55(5):1025-1045.
https://doi.org/10.1287/trsc.2021.1071

Discrete choice model derived from the following article:
Cascetta E, Coppola P (2012) An elastic demand schedule-based
multimodal assignment model for the simulation of high speed
rail (HSR) systems. EURO J. Transportation Logist. 1(1-2):3-27.
'''

# General
import copy
import numpy as np
import nested_logit

# Data
import data_HSR as data_file


def discrete_choice_model(dict):

    # Define the type of discrete choice model
    dict['DCM'] = 'NestedLogit'
    dict['Nests'] = 5

    ##########################################################
    # DISCRETE CHOICE MODEL PARAMETERS
    ##########################################################

    # From Cascetta and Coppola (2012)
                                        #Non-business Business
    dict['BETA_ACC_EGR'] =              [-0.0103,    -0.0056]
    dict['BETA_TTIME'] =                [-0.0054,    -0.0133]
    dict['BETA_EARLY'] =                [-0.0068,    -0.0019]
    dict['BETA_LATE'] =                 [-0.0062,    -0.0130]
    dict['BETA_MALE'] =                 [ 0.5500,     1.4000]

    dict['BETA_COST_CAR_LOW_INC'] =     [-0.0405,    -0.0527]
    dict['BETA_COST_PLANE_LOW_INC'] =   [-0.0194,    -0.0201]
    dict['BETA_COST_IC_LOW_INC'] =      [-0.0172,    -0.0377]
    dict['BETA_COST_HSR_LOW_INC'] =     [-0.0256,    -0.0284]
    
    dict['BETA_COST_CAR_HIGH_INC'] =    [-0.0228,    -0.0296]
    dict['BETA_COST_PLANE_HIGH_INC'] =  [-0.0109,    -0.0113]
    dict['BETA_COST_IC_HIGH_INC'] =     [-0.0097,    -0.0212]
    dict['BETA_COST_HSR_HIGH_INC'] =    [-0.0144,    -0.0160]        
    
    dict['BETA_COST_CAR_REIMBURSED'] =  [-0.0000,    -0.0222]
    dict['BETA_COST_PLANE_REIMBURSED'] =[-0.0000,    -0.0109]
    dict['BETA_COST_IC_REIMBURSED'] =   [-0.0000,    -0.0158]
    dict['BETA_COST_HSR_REIMBURSED'] =  [-0.0000,    -0.0120]

    # MU (nested logit parameter)
    dict['MU'] =                    [   [1.000,     1.000],  #Nest 0 - Car
                                        [1.106,     1.086],  #Nest 1 - Plane
                                        [1.000,     1.000],  #Nest 2 - IC
                                        [1.333,     1.190],  #Nest 3 - HSR1
                                        [1.299,     1.134]]  #Nest 4 - HSR2
    
    # Alternative specific coefficients (after calibration)
    #                                   Non-business Business
    dict['ASC_CAR'] =                   [ 0.000,     0.000]
    dict['ASC_PLANE'] =                 [-1.781,    -3.060]
    dict['ASC_IC'] =                    [-2.046,    -1.433]
    dict['ASC_AV'] =                    [-0.653,    -0.993]
    dict['ASC_NTV'] =                   [-0.813,    -1.095]


def supply(dict):
    
    # Number of endogenous operators
    dict['K'] = 2

    # Distance from origin to destination (in km)
    dict['Distance'] = 600

    ##########################################################
    # Schedule of all alternatives
    ##########################################################

    # ------ PLANE ------ # #1 hours 10 minutes
    air_dep = np.array([7*60+10, 8*60+10])
    air_arr = np.array([8*60+20, 9*60+20])
    air_tt = 60 + np.subtract(air_arr, air_dep) #waiting time at airport = 60

    # ------ HSR SUPPLIER 1 ------ # #3 hours
    HSR1_dep = np.array([5*60+45, 6*60+45])
    HSR1_arr = np.array([8*60+45, 9*60+45])
    HSR1_tt = np.subtract(HSR1_arr, HSR1_dep)

    # ------ HSR SUPPLIER 2 ------ # #3 hours 20 minutes
    HSR2_dep = np.array([5*60+40, 6*60+40])
    HSR2_arr = np.array([9*60,   10*60])
    HSR2_tt = np.subtract(HSR2_arr, HSR2_dep)

    # ------ IC ------ # #8 hours
    ic_dep = np.array([2*60])
    ic_arr = np.array([10*60])
    ic_tt = np.subtract(ic_arr, ic_dep)

    # ------ CAR ------ # #6 hours
    car_tt = 360

    ##########################################################
    # Create a list of all alternatives (universal choice set)
    ##########################################################

    dict['alternatives'] = []

    #CAR
    d = {'Mode': 'Car', 'Operator': '-', 'DepTime': None, 'ArrTime': None, 'TT': car_tt, 'Endogenous': 0}
    dict['alternatives'].append(d)
    #TRAIN IC
    for i in range(len(ic_dep)):
        d = {'Mode': 'Train', 'Operator': 'IC', 'DepTime': ic_dep[i], 'ArrTime': ic_arr[i], 'TT': ic_tt[i], 'Endogenous': 0}
        dict['alternatives'].append(d)
    #PLANE
    for i in range(len(air_dep)):
        d = {'Mode': 'Plane', 'Operator': 'Airline', 'DepTime': air_dep[i], 'ArrTime': air_arr[i], 'TT': air_tt[i], 'Endogenous': 0}
        dict['alternatives'].append(d)
    #TRAIN HSR SUPPLIER 1
    for i in range(len(HSR1_dep)):
        d = {'Mode': 'Train', 'Operator': 'HSR_Supplier1', 'DepTime': HSR1_dep[i], 'ArrTime': HSR1_arr[i], 'TT': HSR1_tt[i], 'Endogenous': 1}
        dict['alternatives'].append(d)
    #TRAIN HSR SUPPLIER 2
    for i in range(len(HSR2_dep)):
        d = {'Mode': 'Train', 'Operator': 'HSR_Supplier2', 'DepTime': HSR2_dep[i], 'ArrTime': HSR2_arr[i], 'TT': HSR2_tt[i], 'Endogenous': 1}
        dict['alternatives'].append(d)

    # Number of endogenous alternatives in the choice set
    dict['I'] = sum(i.get('Endogenous') == 1 for i in dict['alternatives'])
    # Number of opt-out alternatives in the choice set
    dict['I_opt_out'] = len(dict['alternatives']) - dict['I']
    # Size of the universal choice set
    dict['I_tot'] = len(dict['alternatives'])

    ##########################################################
    # Define the attributes of the alternatives
    ##########################################################

    # Identify which supplier controls which alternatives (0 = opt-out options)
    # Opt-out options must be listed before endogenous alternatives!
    dict['operator'] = np.zeros([dict['I_tot']], dtype=int)
    for i in range(dict['I_tot']):
        if dict['alternatives'][i]['Operator'] == 'HSR_Supplier1':
            dict['operator'][i] = 1
        elif dict['alternatives'][i]['Operator'] == 'HSR_Supplier2':
            dict['operator'][i] = 2
        elif dict['alternatives'][i]['Operator'] == 'Airline' and dict['alternatives'][i]['Endogenous'] == 1:
            dict['operator'][i] = 3

    # Generate list of alternatives belonging to supplier k
    dict['list_alt_supplier'] = {}
    for k in range(dict['K'] + 1):
        dict['list_alt_supplier'][k] = [i for i, op in enumerate(dict['operator']) if op == k]

    # Mapping between alternatives index and their names
    dict['name_mapping'] = {}
    dict['name_mapping'][0] = 'Car'
    for i in range(1, dict['I_tot']):
        dict['name_mapping'][i] = dict['alternatives'][i]['Mode'] + '_' + str(dict['operator'][i]) + \
            '_' + str(int(np.floor(dict['alternatives'][i]['DepTime']/60.0))) + \
            '_' + \
            str(int(np.remainder(dict['alternatives'][i]['DepTime'], 60.0)))

    # Nests
    dict['nest'] = np.zeros(dict['I_tot'], dtype=int)
    # Nest 0: Car. Nest 1: Plane. Nest 2: IC. Nest 3: Supplier 1. Nest 4: Supplier 2.
    for i in range(dict['I_tot']):
        if dict['alternatives'][i]['Mode'] == 'Car':
            dict['nest'][i] = 0
        elif dict['alternatives'][i]['Operator'] == 'Airline':
            dict['nest'][i] = 1
        elif dict['alternatives'][i]['Operator'] == 'IC':
            dict['nest'][i] = 2
        elif dict['alternatives'][i]['Operator'] == 'HSR_Supplier1':
            dict['nest'][i] = 3
        elif dict['alternatives'][i]['Operator'] == 'HSR_Supplier2':
            dict['nest'][i] = 4

    # Generate list of alternatives belonging to nest n
    dict['list_alt_nest'] = {}
    for n in range(dict['Nests']):
        dict['list_alt_nest'][n] = [i for i, val in enumerate(dict['nest']) if val == n]


    ##########################################################
    # Supply costs, prices, bounds
    ##########################################################

    # Fixed costs (running a service)
    dict['fixed_cost'] = np.zeros([dict['I_tot']])
    # Variable costs (customer service)
    dict['customer_cost'] = np.zeros([dict['I_tot']])

    # Initial prices 
    dict['price'] = np.empty(dict['I_tot'])
    for i in range(dict['I_tot']):
        if dict['alternatives'][i]['Mode'] == 'Car':
            dict['price'][i] = 100.0
        elif dict['alternatives'][i]['Operator'] == 'Airline':
            dict['price'][i] = 60.0
        elif dict['alternatives'][i]['Operator'] == 'IC':
            dict['price'][i] = 30.0
        elif dict['alternatives'][i]['Operator'] == 'HSR_Supplier1':
            dict['price'][i] = 90.0
        elif dict['alternatives'][i]['Operator'] == 'HSR_Supplier2':
            dict['price'][i] = 80.0
    
    # Lower and upper bound on prices
    dict['lb_p_urban'] = np.zeros([dict['I_tot']])
    dict['ub_p_urban'] = np.zeros([dict['I_tot']])
    dict['lb_p_rural'] = np.zeros([dict['I_tot']])
    dict['ub_p_rural'] = np.zeros([dict['I_tot']])
    for i in range(dict['I_tot']):
        if dict['alternatives'][i]['Endogenous'] == 0:
            dict['lb_p_urban'][i] = dict['price'][i]
            dict['ub_p_urban'][i] = dict['price'][i]
            dict['lb_p_rural'][i] = dict['price'][i]
            dict['ub_p_rural'][i] = dict['price'][i]
        elif dict['alternatives'][i]['Operator'] == 'HSR_Supplier1':
            dict['lb_p_urban'][i] = 50.0
            dict['ub_p_urban'][i] = 150.0
            dict['lb_p_rural'][i] = 50.0
            dict['ub_p_rural'][i] = 150.0
        elif dict['alternatives'][i]['Operator'] == 'HSR_Supplier2':
            dict['lb_p_urban'][i] = 50.0
            dict['ub_p_urban'][i] = 150.0
            dict['lb_p_rural'][i] = 50.0
            dict['ub_p_rural'][i] = 150.0

    # Initial supply strategies
    dict['p_fixed'] = copy.deepcopy(dict['price'])
    dict['p_urban_fixed'] = copy.deepcopy(dict['price'])
    dict['p_rural_fixed'] = copy.deepcopy(dict['price'])

    dict['best_response_lb_p_urban'] = copy.deepcopy(dict['lb_p_urban'])
    dict['best_response_ub_p_urban'] = copy.deepcopy(dict['ub_p_urban'])
    dict['best_response_lb_p_rural'] = copy.deepcopy(dict['lb_p_rural'])
    dict['best_response_ub_p_rural'] = copy.deepcopy(dict['ub_p_rural'])


def demand(dict):

    # Number of customers
    dict['Pop'] = 1000

    ##########################################################
    # Customer socio-economic characteristics:
    # gender, income, trip purpose, origin, desired arrival
    ##########################################################

    # GENDER

    # 50-50 at random
    dict['MALE_pop'] = np.zeros((dict['Pop']))
    for n in range(dict['Pop']):
        if np.random.random_sample() > 0.5:
            dict['MALE_pop'][n] = 1    

    # TRIP PURPOSE

    # 1 = business, 0 = non-business
    # 50% are business travellers, 50% are not
    dict['BUSINESS_pop'] = np.zeros((dict['Pop']))
    for n in range(dict['Pop']):
        r = np.random.random_sample()  #random reason for travel
        if r < 0.5:
            dict['BUSINESS_pop'][n] = 1
        else:
            dict['BUSINESS_pop'][n] = 0

    # REIMBURSEMENT

    # 1 = business traveller is reimbursed, 0 = no reimbursement
    # 75% of business travellers are reimbursed, 25% are not
    dict['REIMBURSEMENT_pop'] = np.zeros((dict['Pop']))
    for n in range(dict['Pop']):
        r = np.random.random_sample()  # random reimbursement
        if dict['BUSINESS_pop'][n] == 1 and r < 0.75:
            dict['REIMBURSEMENT_pop'][n] = 1
        else:
            dict['REIMBURSEMENT_pop'][n] = 0

    # INCOME

    # 1 = high, 0 = low
    # 25% of business travellers and 10% of non-business travellers have high income
    dict['INCOME_pop'] = np.zeros((dict['Pop']))
    for n in range(dict['Pop']):
        r = np.random.random_sample()  # random income
        if (dict['BUSINESS_pop'][n] == 0 and r < 0.1) or (dict['BUSINESS_pop'][n] == 1 and r < 0.25):
            dict['INCOME_pop'][n] = 1
        else:
            dict['INCOME_pop'][n] = 0

    # ORIGIN
    
    # 1 = urban (direct access to HSR), 0 = rural (no direct access to HSR)
    # 80% of business travellers and 20% of non-business travellers are urban
    dict['ORIGIN_pop'] = np.zeros((dict['Pop']))
    for n in range(dict['Pop']):
        r = np.random.random_sample()  # random origin
        if (dict['BUSINESS_pop'][n] == 0 and r < 0.8) or (dict['BUSINESS_pop'][n] == 1 and r < 0.2):
            dict['ORIGIN_pop'][n] = 0
        else:
            dict['ORIGIN_pop'][n] = 1


def groups(data):
    
    # Define number of segments and names
    # Income (2) * Business (3) * Origin (2)
    data['N'] = 12

    # n     Income      Origin      Trip purpose    Reimbursement
    # 0     0           0           0               0
    # 1     0           1           0               0
    # 2     1           0           0               0
    # 3     1           1           0               0
    # 4     0           0           1               0
    # 5     0           0           1               1
    # 6     0           1           1               0
    # 7     0           1           1               1
    # 8     1           0           1               0
    # 9     1           0           1               1
    # 10    1           1           1               0
    # 11    1           1           1               1

    # Count population in each group
    data['popN'] = np.zeros((data['N']))
    data['BUSINESS'] = np.zeros((data['N']))
    data['REIMBURSEMENT'] = np.zeros((data['N']))
    data['INCOME'] = np.zeros((data['N']))
    data['ORIGIN'] = np.zeros((data['N']))

    # Assign people to segments
    for n in range(data['Pop']):
        if data['BUSINESS_pop'][n] == 0:
            if data['INCOME_pop'][n] == 0 and data['ORIGIN_pop'][n] == 0:
                data['popN'][0] += 1
                data['BUSINESS'][0] = 0
                data['REIMBURSEMENT'][0] = 0
                data['INCOME'][0] = 0
                data['ORIGIN'][0] = 0
            elif data['INCOME_pop'][n] == 0 and data['ORIGIN_pop'][n] == 1:
                data['popN'][1] += 1
                data['BUSINESS'][1] = 0
                data['REIMBURSEMENT'][1] = 0
                data['INCOME'][1] = 0
                data['ORIGIN'][1] = 1
            elif data['INCOME_pop'][n] == 1 and data['ORIGIN_pop'][n] == 0:
                data['popN'][2] += 1
                data['BUSINESS'][2] = 0
                data['REIMBURSEMENT'][2] = 0
                data['INCOME'][2] = 1
                data['ORIGIN'][2] = 0
            elif data['INCOME_pop'][n] == 1 and data['ORIGIN_pop'][n] == 1:
                data['popN'][3] += 1
                data['BUSINESS'][3] = 0
                data['REIMBURSEMENT'][3] = 0
                data['INCOME'][3] = 1
                data['ORIGIN'][3] = 1
        elif data['REIMBURSEMENT_pop'][n] == 0:
            if data['INCOME_pop'][n] == 0 and data['ORIGIN_pop'][n] == 0:
                data['popN'][4] += 1
                data['BUSINESS'][4] = 1
                data['REIMBURSEMENT'][4] = 0
                data['INCOME'][4] = 0
                data['ORIGIN'][4] = 0                
            elif data['INCOME_pop'][n] == 0 and data['ORIGIN_pop'][n] == 1:
                data['popN'][6] += 1
                data['BUSINESS'][6] = 1
                data['REIMBURSEMENT'][6] = 0
                data['INCOME'][6] = 0
                data['ORIGIN'][6] = 1                
            elif data['INCOME_pop'][n] == 1 and data['ORIGIN_pop'][n] == 0:
                data['popN'][8] += 1
                data['BUSINESS'][8] = 1
                data['REIMBURSEMENT'][8] = 0
                data['INCOME'][8] = 1
                data['ORIGIN'][8] = 0              
            elif data['INCOME_pop'][n] == 1 and data['ORIGIN_pop'][n] == 1:
                data['popN'][10] += 1
                data['BUSINESS'][10] = 1
                data['REIMBURSEMENT'][10] = 0
                data['INCOME'][10] = 1
                data['ORIGIN'][10] = 1                
        else:
            if data['INCOME_pop'][n] == 0 and data['ORIGIN_pop'][n] == 0:
                data['popN'][5] += 1
                data['BUSINESS'][5] = 1
                data['REIMBURSEMENT'][5] = 1
                data['INCOME'][5] = 0
                data['ORIGIN'][5] = 0
            elif data['INCOME_pop'][n] == 0 and data['ORIGIN_pop'][n] == 1:
                data['popN'][7] += 1
                data['BUSINESS'][7] = 1
                data['REIMBURSEMENT'][7] = 1
                data['INCOME'][7] = 0
                data['ORIGIN'][7] = 1                  
            elif data['INCOME_pop'][n] == 1 and data['ORIGIN_pop'][n] == 0:
                data['popN'][9] += 1
                data['BUSINESS'][9] = 1
                data['REIMBURSEMENT'][9] = 1
                data['INCOME'][9] = 1
                data['ORIGIN'][9] = 0                 
            elif data['INCOME_pop'][n] == 1 and data['ORIGIN_pop'][n] == 1:
                data['popN'][11] += 1
                data['BUSINESS'][11] = 1
                data['REIMBURSEMENT'][11] = 1
                data['INCOME'][11] = 1
                data['ORIGIN'][11] = 1


def arrival_times(data):

    # DESIRED ARRIVAL TIMES

    # Between 9:00 (=540) and 11:00 (=660) differentiated by travel purpose
    data['DAT'] = np.zeros((data['N'],data['R']))
    data['slot'] = np.random.random_sample((data['N'],data['R']))
    for n in range(data['N']):
        for r in range(data['R']):
            rand = np.random.random_sample()
            if data['BUSINESS'][n] == 1:            # GROUP 1: BUSINESS CUSTOMERS
                if data['slot'][n,r] < 0.50:        # 09:00-09:30: 50% of arrivals
                    data['DAT'][n,r] = 9 * 60 + rand*30
                elif data['slot'][n,r] < 0.85:      # 09:30-10:00: 35% of arrivals
                    data['DAT'][n,r] = 9 * 60 + 30 + rand*30
                elif data['slot'][n,r] < 0.95:      # 10:00-10:30: 10% of arrivals
                    data['DAT'][n,r] = 10 * 60 + rand*30
                else:                               # 10:30-11:00: 5% of arrivals
                    data['DAT'][n,r] = 10 * 60 + 30 + rand*30
            else:                                   # GROUP 2: NON-BUSINESS CUSTOMERS
                if data['slot'][n,r] < 0.50:        # 50% have no desired arrival time
                    data['DAT'][n,r] = -1
                elif data['slot'][n,r] < 0.70:      # 50% follow a distribution 20-15-10-5
                    data['DAT'][n,r] = 9 * 60 + rand*30
                elif data['slot'][n,r] < 0.85:
                    data['DAT'][n,r] = 9 * 60 + 30 + rand*30
                elif data['slot'][n,r] < 0.95:
                    data['DAT'][n,r] = 10 * 60 + rand*30
                else:
                    data['DAT'][n,r] = 10 * 60 + 30 + rand*30


    # ACCESS/EGRESS TIMES TO/FROM TERMINALS

    data['ACCESS'] = np.zeros((data['I_tot'], data['N'], data['R']))
    data['EGRESS'] = np.zeros((data['I_tot'], data['N'], data['R']))

    for n in range(data['N']):
        for r in range(data['R']):
            r_acc_TrainStation = np.random.random_sample()
            r_egr_TrainStation = np.random.random_sample()
            r_acc_Airport = np.random.random_sample()
            r_egr_Airport = np.random.random_sample()
            for i in range(data['I_tot']):
                if data['alternatives'][i]['Mode'] == 'Car':  # No access/egress times for cars
                    data['ACCESS'][i,n,r] = 0
                    data['EGRESS'][i,n,r] = 0
                elif data['alternatives'][i]['Operator'] == 'Airline':  # Flights: 30-60 minutes
                    data['ACCESS'][i,n,r] = 30 + r_acc_Airport * 30
                    data['EGRESS'][i,n,r] = 30 + r_egr_Airport * 30
                else:  # TRAINS
                    if data['ORIGIN'][n] == 1:  # Train access urban: 0-30 minutes
                        data['ACCESS'][i,n,r] = r_acc_TrainStation * 30
                    elif data['ORIGIN'][n] == 0:  # Train access rural: 30-60 minutes
                        data['ACCESS'][i,n,r] = 30 + r_acc_TrainStation * 30
                    data['EGRESS'][i,n,r] = r_egr_TrainStation * 30  # Train egress: 0-30 minutes


    # EARLINESS/LATENESS FOR ALL SCHEDULED SERVICES

    data['EARLY'] = np.zeros((data['I_tot'], data['N'], data['R']))
    data['LATE'] = np.zeros((data['I_tot'], data['N'], data['R']))
    for i in range(1, data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                if data['DAT'][n,r] > 0:
                    if data['DAT'][n,r] - (data['alternatives'][i]['ArrTime'] + data['EGRESS'][i, n, r]) > 0:
                        data['EARLY'][i,n,r] = data['DAT'][n,r] - (data['alternatives'][i]['ArrTime'] + data['EGRESS'][i,n,r])
                    elif data['DAT'][n,r] - (data['alternatives'][i]['ArrTime'] + data['EGRESS'][i,n,r]) < 0:
                        data['LATE'][i,n,r] = (data['alternatives'][i]['ArrTime'] + data['EGRESS'][i,n,r]) - data['DAT'][n,r]


def data_instance(data):
    
    discrete_choice_model(data)
    supply(data)
    demand(data)
    groups(data)
    arrival_times(data)


if __name__ == '__main__':
    
    #Read instance and precompute exogenous terms
    data = data_file.getData()
    data_file.preprocessUtilities(data)

    if data['DCM'] == 'NestedLogit':
        #Calculate initial values of logsum terms
        nested_logit.logsumNestedLogit(data)

    # Print list of customers
    data_file.printCustomers(data)
