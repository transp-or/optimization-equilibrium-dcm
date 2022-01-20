'''
MAIN.PY: run this to test the simulation-based heuristic algorithm
to find approximate equilibria, described in the following article:
S. Bortolomiol, V. Lurkin, M. Bierlaire (2021).
A Simulation-Based Heuristic to Find Approximate Equilibria with
Disaggregate Demand Models. Transportation Science 55(5):1025-1045.
https://doi.org/10.1287/trsc.2021.1071

CASE STUDY: PARKING CHOICE
Case study taken from the following article:
Ibeas A, Dell'Olio L, Bordagaray M, Ortuzar Jd D (2014).
Modelling parking choices considering user heterogeneity.
Transportation Res. Part A Policy Practice 70:41-49.
'''

# General
import sys
import time
import copy
import math
import random
import numpy as np

# Project
import supply_opt
import fixed_point_iteration_algorithm
import fixed_point_MIP

import generate_strategies
import update_bounds
import choice_preprocess
import postprocessing

# Data
import data_parking as data_file


def restrictedStrategySets(data, fixed_point_it_results):
    ''' 
        If the fixed-point iteration algorithm finds a cyclic equilibrium,
        use the results as input to generate the initial restricted strategy sets
        for the fixed-point MIP model
    '''

    # Reinitialize strategy sets
    data['strategies'] = {}

    # Derive LB and UB for the prices in the cycle found by the fixed-point iteration algorithm
    lb = []
    ub = []
    print('\n\nPRICE BOUNDS IN THE FIXED-POINT MIP MODEL: \nAlt      lb_p      ub_p')
    for i in range(data['I_tot']):
        p_history = [p for p in fixed_point_it_results['p_history'][fixed_point_it_results['cycle_start']:fixed_point_it_results['cycle_end']][:, i]]
        lb.append(np.amin(p_history))
        ub.append(np.amax(p_history))
        print(' {:2d}   {:6.4f}    {:6.4f}'.format(i, lb[-1], ub[-1]))

    # Update the bounds of the candidate equilibrium region (prices only)
    data['lb_p'] = lb
    data['ub_p'] = ub

    # Generate strategy sets in the equilibrium region identified by the cycles
    generate_strategies.generateStrategySets(data)


def updateListEquilibriumFixedPointIteration(data, fixed_point_it_results):

    data['equilibrium']['prices'].append(fixed_point_it_results['p_history'][-1][:])
    data['equilibrium']['profits'].append(fixed_point_it_results['profit'][-1][:])
    data['equilibrium']['demand'].append(fixed_point_it_results['demand_history'][-1][:])
    data['equilibrium']['eps_price'].append(0.0)
    data['equilibrium']['eps_profit'].append(0.0)


def updateListEquilibriumColumnGeneration(data, equilibrium):

    data['equilibrium']['prices'].append(equilibrium['prices_in'])
    data['equilibrium']['profits'].append(equilibrium['max_profit_in'])
    data['equilibrium']['demand'].append(equilibrium['demand'])
    data['equilibrium']['eps_price'].append(equilibrium['eps_price'])
    data['equilibrium']['eps_profit'].append(equilibrium['eps_profit'])


def newRegionSolutionSpace(data):

    r = np.random.random_sample()

    # Explore pricing strategies above current eps-equilibrium prices
    if r > 0.5:
        ub = np.full((data['I_tot']), -1.0)
        for l in range(len(data['equilibrium']['prices'])):
            for i in range(data['I_tot']):
                if data['equilibrium']['prices'][l][i] > ub[i]:
                    ub[i] = data['equilibrium']['prices'][l][i]                                
        data['initial_data']['lb_p'] = ub
        data['initial_data']['ub_p'] = copy.deepcopy(data['initialUb'])              
    # Explore pricing strategies below current eps-equilibrium prices
    else:
        lb = np.full((data['I_tot']), 1000.0)
        for l in range(len(data['equilibrium']['prices'])):
            for i in range(data['I_tot']):
                if data['equilibrium']['prices'][l][i] < lb[i]:
                    lb[i] = data['equilibrium']['prices'][l][i]   
        data['initial_data']['lb_p'] = copy.deepcopy(data['initialLb'])
        data['initial_data']['ub_p']  = lb


def NewInitialSolution(data):

    data['strategies'] = {}
    data['n_strategies'] = copy.deepcopy(data['initial_data']['n_strategies'])

    data['lb_p'] = copy.deepcopy(data['initial_data']['lb_p'])
    data['ub_p'] = copy.deepcopy(data['initial_data']['ub_p'])

    data['eps_equilibrium_profit'] = copy.deepcopy(data['initial_data']['eps_equilibrium_profit'])

    data['p_fixed'] = np.add(data['lb_p'], np.random.random_sample(data['I_tot']) * np.subtract(data['ub_p'],data['lb_p']))
    data['optimizer'] = np.random.randint(1, high = data['K'] + 1)

    print('\n\nNEW INITIAL PRICES AND BOUNDS :\nAlt      LB_P    UB_P     p_fixed')
    for i in range(data['I_tot']):
        print('{:3d}    {:6.4f}  {:6.4f}     {:6.4f}'
              .format(i, data['initial_data']['lb_p'][i], data['initial_data']['ub_p'][i], data['p_fixed'][i]))


def columnGenerationMethod(data):
    ''' 
        Use the fixed-point MIP model to find a subgame equilibrium.
        Check the solution on the original game by solving best-response problems.
        If not all eps-equilibrium conditions are satisfied,
        best-response strategies are added to the strategy sets.
    '''

    ### INITIALIZE VARIABLES FOR THE METHOD
    iter = 0
    game_equilibrium = False
    hist_best_response = []
    hist_fixed_point = []

    ### MAIN LOOP
    while game_equilibrium is False:

        # Identify captive customers for the current strategy sets
        update_bounds.updateUtilityBounds(data)
        choice_preprocess.choicePreprocess(data)
        choice_preprocess.choicePreprocessStrategies(data)

        ### Solve the fixed-point MIP model to find a subgame equilibrium
        model, nameToIndex = fixed_point_MIP.getModel(data)
        FP_results = fixed_point_MIP.solveModel(model, nameToIndex, data)

        ### Save results of fixed-point MIP
        data['p_fixed'] = copy.deepcopy(FP_results['prices_in'])
        
        BR_results = {}
        BR_results['prices'] = copy.deepcopy(FP_results['prices_in'])
        BR_results['profits'] = copy.deepcopy(FP_results['max_profit_in'])
        
        ### Launch a best-response problem for each operator
        print('\nBEST RESPONSE PROBLEMS:')
        for k in range(1, data['K'] + 1):

            data['optimizer'] = k
            data['lb_profit'] = FP_results['max_profit_in'][k]

            # Set price bounds to either initial bounds (optimizer) or prices of the subgame equilibrium
            for i in range(data['I_opt_out'], data['I_tot']):
                if data['operator'][i] == k:
                    data['lb_p'][i] = copy.deepcopy(data['best_response_lb_p'][i])
                    data['ub_p'][i] = copy.deepcopy(data['best_response_ub_p'][i])
                else:
                    data['lb_p'][i] = copy.deepcopy(data['p_fixed'][i])
                    data['ub_p'][i] = copy.deepcopy(data['p_fixed'][i])

            update_bounds.updateUtilityBounds(data)
            choice_preprocess.choicePreprocess(data)

            # Solve the best response problem
            model = supply_opt.getModel(data)
            results = supply_opt.solveModel(data, model)

            # Update prices and profits after the Stackelberg games
            BR_results['profits'][k] = results['profits'][k]
            for i in data['list_alt_supplier'][k]:
                BR_results['prices'][i] = results['prices'][i]

        ### Verify price differences (absolute and in %)
        price_diff = [abs(BR_results['prices'][i] - FP_results['prices_in'][i]) for i in range(data['I_tot'])]
        price_perc = np.zeros([data['I_tot']])
        for i in range(data['I_opt_out'], data['I_tot']):
            price_perc[i] = price_diff[i] / FP_results['prices_in'][i]
        FP_results['eps_price'] = np.max(price_perc)
        ### Verify profit differences (absolute and in %)
        profit_diff = [BR_results['profits'][k] - FP_results['max_profit_in'][k] for k in range(data['K'] + 1)]
        profit_perc = np.zeros([data['K'] + 1])
        for k in range(1, data['K'] + 1):
            profit_perc[k] = profit_diff[k] / FP_results['max_profit_in'][k]
        FP_results['eps_profit'] = np.max(profit_perc)

        ### Print prices and profits in the fixed-point MIP (FP) and in the best response problems (BR)
        print('\n Alt     PriceFP     PriceBR')
        for i in range(data['I_opt_out'], data['I_tot']):
            print('  {:2d}    {:8.3f}    {:8.3f}'.format(i, FP_results['prices_in'][i], BR_results['prices'][i]))
        print('\nSupp     ProfitFP   ProfitBR    epsilon')
        for k in range(1, data['K'] + 1):
            print('  {:2d}    {:9.3f}  {:9.3f}    {:7.4f}'.format(k, FP_results['max_profit_in'][k], BR_results['profits'][k], profit_perc[k]))

        # If profits satisfy the tolerance threshold, the solution is considered an epsilon-equilibrium
        if all(k < data['eps_equilibrium_profit'] for k in profit_perc):
            updateListEquilibriumColumnGeneration(data, FP_results)

            print('\nProfit tolerance satisfied. Epsilon-equilibrium found.\nEps = {:7.4f}'.format(FP_results['eps_profit']))
            '''
            # Case (1): eps-equilibrium found + price tolerances satisfied => we restart the whole algorithm
            if all(i < data['eps_equilibrium_price'] for i in price_perc) or len(data['equilibrium']['prices']) >= data['nEquilibria']:
                game_equilibrium = True
                print('\nPrice tolerance satisfied. Restart algorithmic framework.\n')
            # Case (2): eps-equilibrium found + price tolerances not satisfied => we search other eps-equilibria with same restricted strategy sets
            else:
                print('\nPrice tolerance not satisfied. Add/remove strategies.')
            '''
            game_equilibrium = True
            print('\nRestart algorithmic framework.\n\n')
        # Case (3): eps-equilibrium not found => we search for eps-equilibria with same restricted strategy sets
        else:
            print('\nProfit tolerance not satisfied. Add/remove strategies.')            

        # If not, the algorithm continues
        if game_equilibrium is False:

            # Randomly choose to increase eps to increase chance of finding an eps-equilibrium
            if random.random() > 0.25:
                data['eps_equilibrium_profit'] = 1.1*data['eps_equilibrium_profit']
                print('eps = {:6.4f}\n'.format(data['eps_equilibrium_profit']))

            # Save best-response strategies to be added to the restricted strategy sets
            data['best_response_prices'] = copy.deepcopy(BR_results['prices'])
            # Save fixed-point strategies (potentially to be removed from the restricted strategy sets?)
            data['fixed_point_prices'] = copy.deepcopy(FP_results['prices_in'])

            # Archive results of current iteration
            hist_best_response.append(BR_results)
            hist_fixed_point.append(FP_results)

            # Check whether the best response strategy for one or more suppliers was already a best response
            # If so, set BRinSet to true
            BRinSet = [False for k in range(data['K'] + 1)]

            for k in range(1, data['K'] + 1):
                for it in range(iter - 1, -1, -1):
                    BR_diff = [abs(BR_results['prices'][i] - hist_best_response[it]['prices'][i]) for i in data['list_alt_supplier'][k]]
                    if all(BR_diff[i] < data['tolerance_equilibrium'] for i in range(len(BR_diff))):
                        BRinSet[k] = True

            # Update existing strategy sets
            generate_strategies.updateStrategySets(data, BRinSet)

            # Update price bounds
            update_bounds.updateSubgamePriceBounds(data)

            iter += 1


def main():
        
    t_0 = time.time()

    #Fix seed for pseudorandom draws
    seed = 10
    np.random.seed(seed)

    #Read instance and precompute exogenous terms
    data = data_file.getData()
    data_file.preprocessUtilities(data)

    data_file.printCustomers(data)

    #Initialize the list of eps-equilibrium solutions (line 1 of Algorithm 1)
    data['equilibrium'] = {'prices': [], 'profits': [], 'demand': [], 'eps_price': [], 'eps_profit': []}
    
    #Initial bounds to use when restarting exploration from different region (loop lines 2-18 of Algorithm 1)
    data['initialLb'] = copy.deepcopy(data['initial_data']['lb_p'])
    data['initialUb'] = copy.deepcopy(data['initial_data']['ub_p'])

    while True:
        #Calculate utility bounds
        update_bounds.updateUtilityBounds(data)
        #Preprocess captive choices
        choice_preprocess.choicePreprocess(data)

        ### FIRST PHASE: Fixed-point iteration algorithm (lines 4-8 of Algorithm 1)
        print('\n\nFIXED-POINT ITERATION ALGORITHM:')

        fixed_point_it_results = fixed_point_iteration_algorithm.fixedPointIterationAlg(data)
        fixed_point_iteration_algorithm.plotGraphs('FixedPoint_Iteration_Algorithm', data, fixed_point_it_results)

        t_1 = time.time()

        '''
        # If a Nash equilibrium is found, save the solution and restart from another initial solution
        if fixed_point_it_results['cycle_type'] == 'NashEquilibrium':
            updateListEquilibriumFixedPointIteration(data, fixed_point_it_results)

        # SECOND PHASE: Fixed-point MIP + best response problem (lines 9-17 of Algorithm 1)
        elif fixed_point_it_results['cycle_type'] == 'CyclicEquilibrium':
    
            restrictedStrategySets(data, fixed_point_it_results)        
            columnGenerationMethod(data)
        '''
        restrictedStrategySets(data, fixed_point_it_results)        
        columnGenerationMethod(data)

        # Stopping criterion (line 18 of Algorithm 1)
        if len(data['equilibrium']['prices']) >= data['nEquilibria']:
            break
        
        else:
            seed += 1
            np.random.seed(seed)

            # Identify a different region of the solution space (line 2 of Algorithm 1)
            newRegionSolutionSpace(data)
            
            # Restart the algorithm with random fixed prices and optimizer (line 3 of Algorithm 1)
            newInitialSolution(data)   

    ##################################################
    # POST-PROCESS: PRINT ALL FOUND EPSILON-EQUILIBRIA
    ##################################################

    print('\n\n---------------------\nEQUILIBRIUM SOLUTIONS \n---------------------')
    print('\n\nPRICES :\nID      epsilon      ', end=' ')
    for i in range(data['I_tot']):
        print('  Alt {:2d} '.format(i), end=' ')
    for j in range(len(data['equilibrium']['prices'])):
        print('\n{:2d}       {:6.3f}      '.format(j+1, data['equilibrium']['eps_profit'][j]), end=' ')
        for i in range(data['I_tot']):
            print('{:8.3f} '.format(data['equilibrium']['prices'][j][i]), end = ' ')
    print('\n\nMARKET SHARES :\nID      epsilon      ', end=' ')
    for i in range(data['I_tot']):
        print(' Alt {:2d} '.format(i), end=' ')
    for j in range(len(data['equilibrium']['prices'])):
        print('\n{:2d}       {:6.3f}      '.format(j+1, data['equilibrium']['eps_profit'][j]), end=' ')
        for i in range(data['I_tot']):
            print('{:7.4f} '.format(data['equilibrium']['demand'][j][i]/data['Pop']), end = ' ')
    print('\n\nPROFITS :\nID      epsilon      ', end=' ')
    for k in range(data['K']+1):
        print('   Supp {:1d} '.format(k), end=' ')
    for j in range(len(data['equilibrium']['prices'])):
        print('\n{:2d}       {:6.3f}      '.format(j+1, data['equilibrium']['eps_profit'][j]), end=' ')
        for k in range(data['K']+1):
            print('{:9.2f} '.format(data['equilibrium']['profits'][j][k]), end = ' ')

    ###########################################
    # POST-PROCESS: ANALYSIS OF MARKET SEGMENTS
    ###########################################
        
    for j in range(len(data['equilibrium']['prices'])):
        
        print('\n\n------------------------\n---- EQUILIBRIUM {:2d} ----'.format(j))
        print('------------------------')

        data['p_fixed'] = data['equilibrium']['prices'][j]

        postprocessing.calculation(data)
        postprocessing.printOutputAnalysis(data)
        postprocessing.segments(data)
        

##### MAIN
if __name__ == '__main__':

    main()
