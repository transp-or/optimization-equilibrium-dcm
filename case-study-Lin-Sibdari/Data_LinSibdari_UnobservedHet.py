'''
Data used in Section 5.1 of the following article:
S. Bortolomiol, V. Lurkin, M. Bierlaire (2021).
A Simulation-Based Heuristic to Find Approximate Equilibria with
Disaggregate Demand Models. Transportation Science 55(5):1025-1045.
https://doi.org/10.1287/trsc.2021.1071

Case study derived from the following article:
Lin KY, Sibdari SY (2009). Dynamic price competition with discrete
customer choices. Eur. J. Oper. Res. 197(3):969–980.

Accounting for unobserved heterogeneity
'''
import copy
import numpy as np

def discrete_choice_model(dict):
    
    # Define the type of discrete choice model
    dict['DCM'] = 'MixedLogit'
    dict['N'] = 1

    ##########################################################
    # DISCRETE CHOICE MODEL PARAMETERS
    ##########################################################

    '''
    # Alternative Specific Parameters
    dict['a_1'] = 5.0
    dict['a_2'] = 4.0
    '''
    # Alternative Specific Parameters (random parameter with normal distribution)
    dict['a_1'] = np.random.normal(5.0, 2.0, size=(dict['N'], dict['R']))
    dict['a_2'] = np.random.normal(4.0, 1.0, size=(dict['N'], dict['R']))

    # Beta coefficients
    dict['beta'] = -0.1
    '''
    # Cost coefficient (random parameter with normal distribution)
    lower, upper = -0.6, -0.4
    mu, sigma = -0.5, 0.1
    dict['beta'] = stats.truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=(dict['N'], dict['R']))
    '''
    dict['endo_coef'] = np.full((3, dict['N'], dict['R']), 0.0)
    dict['exo_utility'] = np.full((3, dict['N'], dict['R']), 0.0)

    for n in range(dict['N']):
        for r in range(dict['R']):
            dict['endo_coef'][1,n,r] = dict['beta']
            dict['endo_coef'][2,n,r] = dict['beta']
            dict['exo_utility'][1,n,r] = dict['a_1'][n,r]
            dict['exo_utility'][2,n,r] = dict['a_2'][n,r]     
    print(dict['endo_coef'])
    '''
    dict['exo_utility'][1,0] = dict['a_1']
    dict['exo_utility'][2,0] = dict['a_2']
    '''


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
    dict['name_mapping'] = {0: 'Alt0', 1: 'Alt1', 2: 'Alt2'}

    ##########################################################
    # Supply costs, prices, bounds
    ##########################################################

    # Lower and upper bound on prices
    dict['lb_p'] = np.array([0, 0.0, 0.0]) # lower bound (0, 1, 2)
    dict['ub_p'] = np.array([0, 80.0, 80.0]) # upper bound (0, 1, 2)

    # Initial price
    #dict['price'] = (dict['ub_p'] + dict['lb_p']) / 2.0
    dict['price'] = np.array([0, 23.02, 16.57]) #analytical

    # Initial supply strategies
    dict['p_fixed'] = copy.deepcopy(dict['price'])

    dict['best_response_lb_p'] = copy.deepcopy(dict['lb_p'])
    dict['best_response_ub_p'] = copy.deepcopy(dict['ub_p'])


def demand(dict):

    # Number of customers
    dict['Pop'] = 1

    #Number of customers per group N
    dict['popN'] = np.array([1])


def setAlgorithmParameters(dict):

    ##########################################################
    # Parameters needed in the algorithmic framework
    ##########################################################

    dict['nEquilibria'] = 5

    #### Parameters of the fixed-point iteration algorithm

    # Initial optimizer
    dict['optimizer'] = 2
    # Max iter
    dict['max_iter'] = 20                          #Modify here for testing
    # Tolerance for equilibrium convergence
    dict['tolerance_equilibrium'] = 0.001          #0.1%
    # Tolerance for cycle convergence
    dict['tolerance_cyclic_equilibrium'] = 0.05
    # Counter of how many times the fixed-point iteration algorithm is used
    dict['countFixedPointIter'] = 1     #Initialized to 1

    #### Parameters of the choice-based optimization model

    dict['lb_profit'] = None

    #### Parameters for the eps-equilibrium conditions

    dict['eps_equilibrium_profit'] = 0.005   #Accepted % of profit increase      #Modify here for testing
    dict['eps_equilibrium_price'] = 0.25    #25% of price change

    #### Parameters of the fixed-point MIP model

    # Number of strategies for each supplier in the initial fixed-point game
    #                                    0  1  2 
    dict['n_strategies'] =     np.array([0, 5, 5])
    
    dict['min_strategies'] = 3
    dict['max_strategies'] = 6      #Always > initial strategies


def getData():
    '''Construct a dictionary 'dict' containing all the input data'''

    # Initialize the output dictionary
    dict = {}

    # Name of the instance
    dict['Instance'] = 'LinSibdari'

    # Number of draws
    dict['R'] = 1000

    # 1) Read discrete choice model parameters
    # 2) Read supply data
    # 3) Read demand data
    # 4) Generate groups of customers
    discrete_choice_model(dict)
    supply(dict)
    demand(dict)

    # Random term (Gumbel distributed 0,1)
    dict['xi'] = np.random.gumbel(size=(dict['I_tot'], dict['N'], dict['R']))

    # Define parameters of the algorithm
    setAlgorithmParameters(dict)

    ##########################################################
    # Deepcopy of the initial data (for restarts)
    ##########################################################
    dict['initial_data'] = copy.deepcopy(dict)

    return dict
    

if __name__ == '__main__':
            
    # Read instance
    data = getData()
