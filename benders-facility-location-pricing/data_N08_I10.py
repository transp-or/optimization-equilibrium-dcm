'''
Data used in Chapter 4 of the following PhD Thesis:
S. Bortolomiol (2021). Optimization and equilibrium problems
with discrete choice models.

Discrete choice model derived from the following article:
Ibeas A, Dell'Olio L, Bordagaray M, Ortuzar Jd D (2014).
Modelling parking choices considering user heterogeneity.
Transportation Res. Part A Policy Practice 70:41-49.
'''
import math
import numpy as np


def getData(seed):
    
    ''' Construct a dictionary 'dict' containing all the input data '''
    dict = {}

    # Number of draws
    dict['R'] = 5

    # Number of price discretizations
    dict['Q_i'] = 6.0
    dict['discreteStep'] = (3.0 / dict['Q_i']) - 0.000000001

    # Rounding to avoid ill-conditioned problem
    dict['round'] = 8

    # Set random seed
    np.random.seed(seed)
    
    # 1) Read discrete choice model parameters
    # 2) Read supply data
    # 3) Read demand data
    discrete_choice_model(dict)
    supply(dict)
    demand(dict)
    
    # Random term (Gumbel distributed 0,1)
    dict['xi'] = np.random.gumbel(size=(dict['I_tot'], dict['N'], dict['R']))

    return dict


def discrete_choice_model(data):
    
    # Define the type of discrete choice model
    data['DCM'] = 'MixedLogit'
    data['N'] = 8

    ##########################################################
    # DISCRETE CHOICE MODEL PARAMETERS
    ##########################################################

    # Alternative specific coefficients (after calibration)
    #                                   High income  Low income
    data['ASC_Walk'] =                      [  0.000,     0.000]
    data['ASC_PT']   =                      [-15.819,   -17.455]
    data['ASC_PSP']  =                      [ 19.565,    19.552]
    data['ASC_PUP']  =                      [ 30.822,    36.444]

    # Beta coefficients
    data['Beta_TimeToDestination'] = -0.612
    data['Beta_Origin'] = -5.762
    data['Beta_Age_Veh'] = 4.037
    data['Beta_Fee_Income_PSP'] = -10.995
    data['Beta_Fee_Income_PUP'] = -13.729
    data['Beta_Fee_Resident_PSP'] = -11.440
    data['Beta_Fee_Resident_PUP'] = -10.668

    # Access time coefficient (random parameter with normal distribution)
    data['Beta_AccessTime'] = np.random.normal(-0.788, 1.064, size=(data['N'], data['R']))
    # Fee coefficient (random parameter with normal distribution)
    data['Beta_Fee'] = np.random.normal(-32.328, 14.168, size=(data['N'], data['R']))


def supply(data):

    # Number of hubs
    #data['nHubs'] = 4

    ##########################################################
    # Alternatives
    ##########################################################

    # Number of endogenous alternatives in the choice set:
    data['I'] = 8  #4 underground + 4 on-street 
    # Number of opt-out alternatives in the choice set
    data['I_opt_out'] = 2  # parking + PT from origin
    # Size of the universal choice set
    data['I_tot'] = data['I'] + data['I_opt_out']

    ##########################################################
    # Attributes of the alternatives
    ##########################################################

    # Access time to parking (AT) (from 8 origins n to 10 alternatives i)
    #                                         OW  OE  ON  OS INW ISW INE ISE   
    data['AccessTimeToParking'] = np.array([[  0,  0,  0,  0,  0,  0,  0,  0], #Walk
                                            [  0,  0,  0,  0,  0,  0,  0,  0], #PT
                                            [  9, 33, 21, 21,  9,  9, 27, 27], #IW
                                            [ 33,  9, 21, 21, 27, 27,  9,  9], #IE
                                            [ 21, 21,  9, 33,  9, 27,  9, 27], #IN
                                            [ 21, 21, 33,  9, 27,  9, 27,  9], #IS
                                            [ 17, 33, 25, 25, 17, 17, 25, 25], #UW
                                            [ 33, 17, 25, 25, 25, 25, 17, 17], #UE
                                            [ 25, 25, 17, 33, 17, 25, 17, 25], #UN
                                            [ 25, 25, 33, 17, 25, 17, 25, 17]])#US

    # Time from parking to final destination (TD) (from 10 alternatives i to 1 final destination)
    #                                              Walk PT IW2 IE2 IN  IS  UW  UE  UN  US
    data['TimeParkingToDestinationPT'] = np.array([  0,  0, 17, 17,17, 17,  9,  9,  9,  9])
    data['TimeParkingToDestinationWalk'] = np.array([0,  0, 45, 45,45, 45, 15, 15, 15, 15])

    # Time from origin to final destination (by public transport) (from 8 origins n to 10 alternatives i)
    #                                             OW  OE  ON  OS INW ISW INE ISE   
    data['TimeOriginToDestination'] = np.array([[ 90, 90, 90, 90, 90, 90, 90, 90], #Walk
                                                [ 29, 29, 29, 29, 29, 29, 29, 29], #PT
                                                [  0,  0,  0,  0,  0,  0,  0,  0], #IW
                                                [  0,  0,  0,  0,  0,  0,  0,  0], #IE
                                                [  0,  0,  0,  0,  0,  0,  0,  0], #IN
                                                [  0,  0,  0,  0,  0,  0,  0,  0], #IS
                                                [  0,  0,  0,  0,  0,  0,  0,  0], #UW
                                                [  0,  0,  0,  0,  0,  0,  0,  0], #UE
                                                [  0,  0,  0,  0,  0,  0,  0,  0]])#UN

    # Time matrix of parking facilities (10x10 alternatives i)
    #                               Walk  PT  IW  IE  IN  IS  UW  UE  UN  US
    data['distParking'] = np.array([[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], #Walk
                                    [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], #PT
                                    [  0,  0,  0, 32, 18, 18,  8, 24, 16, 16], #IW
                                    [  0,  0, 32,  0, 18, 18, 24,  8, 16, 16], #IE
                                    [  0,  0, 18, 18,  0, 32, 16, 16,  8, 24], #IN
                                    [  0,  0, 18, 18, 32,  0, 16, 16, 24,  8], #IS
                                    [  0,  0,  8, 24, 16, 16,  0, 16,  8,  8], #UW
                                    [  0,  0, 24,  8, 16, 16, 16,  0,  8,  8], #UE
                                    [  0,  0, 16, 16,  8, 24,  8,  8,  0, 16], #UN
                                    [  0,  0, 16, 16, 24,  8,  8,  8, 16,  0]])#US

    # Identify which supplier controls which alternatives (0 = opt-out options)
    # Opt-out options must be listed before endogenous alternatives!
    #                           Walk  PT  IW  IE  IN  IS  UW  UE  UN  US
    data['operator'] = np.array([  0,  0,  1,  1,  1,  1,  1,  1,  1,  1])

    # Mapping between alternatives index and their names
    data['name_mapping'] = {0: 'Walk', 1: 'PT', 2: 'IW', 3: 'IE', 4: 'IN', 5: 'IS', 6: 'UW', 7: 'UE',
                            8: 'UN', 9: 'US'}

    ##########################################################
    # Supply costs, prices, bounds
    ##########################################################
    #                                 Walk   PT    IW    IE    IN    IS    UW    UE    UN    US
    # Fixed costs (running a service)
    #data['fixed_cost'] =    np.array([0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0])
    data['fixed_cost'] =    np.array([0.0,  0.0, 25.0, 25.0, 25.0, 25.0,100.0,100.0,100.0,100.0])

    # Variable costs (customer service)
    data['customer_cost'] = np.array([0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0])

    ##########################################################
    # Lower and upper bound on prices and price discretization
    ##########################################################
    #                        Walk   PT    IW    IE    IN    IS    UW    UE    UN    US
    data['lb_p'] = np.array([ 0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0])
    data['ub_p'] = data['lb_p'] + 3.0
    for i in range(data['I_opt_out']):      #Opt-out alternatives have fixed prices
        data['ub_p'][i] = data['lb_p'][i]

    # Initial/fixed price
    data['price'] = data['lb_p']

    # Initial/fixed locations
    #                     Walk    PT   IW    IE    IN    IS    UW    UE    UN    US
    data['y'] = np.array([ 1.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0])

    # Discretized prices
    data['discr_price'] = {}
    for i in range(data['I_opt_out']):
        data['discr_price'][i] = []
        data['discr_price'][i].append(data['lb_p'][i])
        data['customer_cost'][i] = data['lb_p'][i]
    for i in range(data['I_opt_out'], data['I_tot']):
        data['discr_price'][i] = []
        if data['lb_p'][i] == 0:
            p = data['discreteStep']
        else:
            p = data['lb_p'][i]
        while p <= data['ub_p'][i]:
            data['discr_price'][i].append(round(p,2))
            p += data['discreteStep']
    for i in range(data['I_tot']):
        print(i, data['discr_price'][i])


def demand(data):
    
    #  8 people  OW  OE  ON  OS INW ISW INE ISE 
    data['OD'] = np.array([ 200, #OW - 0
                            200, #OE - 1
                            500, #ON - 2
                            500, #OS - 3
                            400, #INW - 4
                            400, #ISW - 5
                            400, #INE - 6
                            400])#ISE - 7                                          
    # Number of customers
    data['Pop'] = 3000

    ##########################################################
    # Customer socio-economic characteristics:
    # origin, age of vehicle, income
    ##########################################################

    # Define number of segments: Origin (8)
    # We assume origin outside city = non-resident
    data['N'] = 8   

    # Count population in each group
    data['popN'] = np.zeros((data['N']))
    data['ORIGIN_ZONE'] = np.zeros((data['N']), dtype=int)
    data['ORIGIN'] = np.zeros((data['N']), dtype=int)
    data['AGE_VEH'] = np.zeros((data['N']), dtype=int)
    data['LOW_INC'] = np.zeros((data['N']), dtype=int)

    # Assign people to segments
    for n in range(data['N']):
        data['ORIGIN_ZONE'][n] = int(math.floor(n/4))
        if n < 4:
            data['ORIGIN'][n] = 0
        else:
            data['ORIGIN'][n] = 1
        if n == 3 or n == 5 or n == 7:
            data['AGE_VEH'][n] = 1
        else:
            data['AGE_VEH'][n] = 0
        if n == 2 or n == 3 or n == 5:
            data['LOW_INC'][n] = 0
        else:
            data['LOW_INC'][n] = 1
    
    # Define segment sizes
    for n in range(data['N']):
        data['popN'][n] = data['OD'][n]


def preprocessUtilities(data):

    exo_utility = np.empty([data['I_tot'], data['N'], data['R']])
    endo_coef = np.empty([data['I_tot'], data['N'], data['R']])
    
    if data['I'] > 2:
        # Exogenous part of the utility function
        for n in range(data['N']):
            for i in range(data['I_tot']):
                for r in range(data['R']):
                    if i == 0:
                        exo_utility[i,n,r] = data['ASC_Walk'][data['LOW_INC'][n]] +\
                                             data['Beta_TimeToDestination'] * data['TimeOriginToDestination'][i,data['ORIGIN_ZONE'][n]]
                    elif i == 1:
                        exo_utility[i,n,r] = data['ASC_PT'][data['LOW_INC'][n]] +\
                                             data['Beta_TimeToDestination'] * data['TimeOriginToDestination'][i,data['ORIGIN_ZONE'][n]]
                    elif i >= 2 and i <= 5:
                        exo_utility[i,n,r] = data['ASC_PSP'][data['LOW_INC'][n]] +\
                                             data['Beta_AccessTime'][n,r] * data['AccessTimeToParking'][i,data['ORIGIN_ZONE'][n]] +\
                                             data['Beta_TimeToDestination'] * data['TimeParkingToDestinationPT'][i] +\
                                             data['Beta_Origin'] * data['ORIGIN'][n]
                    elif i >= 6:
                        exo_utility[i,n,r] = data['ASC_PUP'][data['LOW_INC'][n]] +\
                                             data['Beta_AccessTime'][n,r] * data['AccessTimeToParking'][i,data['ORIGIN_ZONE'][n]] +\
                                             data['Beta_TimeToDestination'] * data['TimeParkingToDestinationPT'][i] +\
                                             data['Beta_Age_Veh'] * data['AGE_VEH'][n]

        # Beta coefficient for endogenous price variables
        for n in range(data['N']):
            for r in range(data['R']):
                for i in range(data['I_tot']):
                    if i == 0 or i == 1:
                        endo_coef[i,n,r] = data['Beta_Fee'][n,r]
                    if i >= 2 and i <= 5:
                        endo_coef[i,n,r] = data['Beta_Fee'][n,r] + data['Beta_Fee_Income_PSP'] * data['LOW_INC'][n]
                    elif i >= 6:
                        endo_coef[i,n,r] = data['Beta_Fee'][n,r] + data['Beta_Fee_Income_PUP'] * data['LOW_INC'][n]

    data['exo_utility'] = exo_utility
    data['endo_coef'] = endo_coef


def printCustomers(data):
    
    ### Print general information
    print('\nI   = {:5d} \nN   = {:5d} \nPop = {:5d} \nR   = {:5d}'.format(data['I_tot'], data['N'], data['Pop'], data['R']))

    ### Print aggregate customer data by travel purpose and origin

    ORIGIN1 = int(sum([data['popN'][n] for n in range(data['N']) if data['ORIGIN'][n] == 1]))
    ORIGIN0 = int(sum([data['popN'][n] for n in range(data['N']) if data['ORIGIN'][n] == 0]))
    AGEVEH1 = int(sum([data['popN'][n] for n in range(data['N']) if data['AGE_VEH'][n] == 1]))
    AGEVEH0 = int(sum([data['popN'][n] for n in range(data['N']) if data['AGE_VEH'][n] == 0]))
    LOWINC1 = int(sum([data['popN'][n] for n in range(data['N']) if data['LOW_INC'][n] == 1]))
    LOWINC0 = int(sum([data['popN'][n] for n in range(data['N']) if data['LOW_INC'][n] == 0]))

    print('\nAGGREGATE CUSTOMER DATA (ORIGIN, AGE VEHICLE, INCOME):\n\nTotal customers    {:6d}'.format(data['Pop']))
    print(' Origin internal    {:5d}   {:5.3f}\n Origin external    {:5d}   {:5.3f}'
            .format(ORIGIN1, ORIGIN1/data['Pop'], ORIGIN0, ORIGIN0/data['Pop']))
    print(' Vehicle new (<3)   {:5d}   {:5.3f}\n Vehicle old (>3)   {:5d}   {:5.3f}'
            .format(AGEVEH1, AGEVEH1/data['Pop'], AGEVEH0, AGEVEH0/data['Pop']))
    print(' Income low (<1200) {:5d}   {:5.3f}\n Income high (>1200){:5d}   {:5.3f}'
            .format(LOWINC1, LOWINC1/data['Pop'], LOWINC0, LOWINC0/data['Pop']))


if __name__ == '__main__':
    
    nSimulations = 1

    for seed in range(1,nSimulations+1):

        if nSimulations > 1:
            print('\n\n\n\n\n---------\nSEED ={:3d}\n---------\n\n'.format(seed))
        
        #Read instance
        data = getData(seed)

        # Print aggregate customer data
        printCustomers(data)

