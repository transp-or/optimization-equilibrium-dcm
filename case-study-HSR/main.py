'''
MAIN.PY: run this to test the simulation-based heuristic algorithm
to find approximate equilibria, described in the following article:
S. Bortolomiol, V. Lurkin, M. Bierlaire (2021).
A Simulation-Based Heuristic to Find Approximate Equilibria with
Disaggregate Demand Models. Transportation Science 55(5):1025-1045.
https://doi.org/10.1287/trsc.2021.1071

CASE STUDY: HIGH-SPEED RAIL COMPETITION
Case study derived from the following article:
Cascetta E, Coppola P (2012) An elastic demand schedule-based
multimodal assignment model for the simulation of high speed
rail (HSR) systems. EURO J. Transportation Logist. 1(1-2):3-27.
'''

# General
import time
import copy
import random
import numpy as np

# Project
import supply_opt
import fixed_point_iteration_algorithm
import fixed_point_MIP

import generate_strategies
import update_bounds
import choice_preprocess
import nested_logit
import postprocessing

# Data
import data_HSR as data_file


def restrictedStrategySets(data, fixed_point_it_results):
    ''' 
        If the fixed-point iteration algorithm finds a cyclic equilibrium,
        use the results as input to generate the initial restricted strategy sets
        for the fixed-point MIP model
    '''

    # Reinitialize strategy sets
    data['strategies'] = {}

    # Derive LB and UB for the prices in the cycle found by the fixed-point iteration algorithm
    lb_Urban = []
    ub_Urban = []
    lb_Rural = []
    ub_Rural = []
    print('\n\nPRICE BOUNDS IN THE FIXED-POINT MIP MODEL: \nAlt    lb_p U  ub_p U    lb_p R  ub_p R')
    for i in range(data['I_tot']):
        p_U_hist = [p for p in fixed_point_it_results['p_urban_history'][fixed_point_it_results['cycle_start']:fixed_point_it_results['cycle_end']][:, i]]
        p_R_hist = [p for p in fixed_point_it_results['p_rural_history'][fixed_point_it_results['cycle_start']:fixed_point_it_results['cycle_end']][:, i]]
        lb_Urban.append(np.amin(p_U_hist))
        ub_Urban.append(np.amax(p_U_hist))
        lb_Rural.append(np.amin(p_R_hist))
        ub_Rural.append(np.amax(p_R_hist))
        print(' {:2d}    {:6.2f}  {:6.2f}    {:6.2f}  {:6.2f}'.format(i, lb_Urban[-1], ub_Urban[-1], lb_Rural[-1], ub_Rural[-1]))

    # Update the bounds of the candidate equilibrium region
    data['lb_p_urban'] = lb_Urban
    data['ub_p_urban'] = ub_Urban
    data['lb_p_rural'] = lb_Rural
    data['ub_p_rural'] = ub_Rural

    # Generate strategy sets in the equilibrium region identified by the cycles
    generate_strategies.generateStrategySets(data)


def updateListEquilibriumFixedPointIteration(data, fixed_point_it_results):

    data['equilibrium']['prices_urban'].append(fixed_point_it_results['p_urban_history'][-1][:])
    data['equilibrium']['prices_rural'].append(fixed_point_it_results['p_rural_history'][-1][:])
    data['equilibrium']['profits'].append(fixed_point_it_results['profit'][-1][:])
    data['equilibrium']['demand'].append(fixed_point_it_results['demand'][-1][:])
    data['equilibrium']['eps_price'].append(0.0)
    data['equilibrium']['eps_profit'].append(0.0)


def updateListEquilibriumColumnGeneration(data, equilibrium):

    data['equilibrium']['prices_urban'].append(equilibrium['prices_urban_in'])
    data['equilibrium']['prices_rural'].append(equilibrium['prices_rural_in'])
    data['equilibrium']['profits'].append(equilibrium['max_profit_in'])
    data['equilibrium']['demand'].append(equilibrium['demand'])
    data['equilibrium']['eps_price'].append(equilibrium['eps_price'])
    data['equilibrium']['eps_profit'].append(equilibrium['eps_profit'])


def newRegionSolutionSpace(data):

    r = np.random.random_sample()

    # Explore pricing strategies above current eps-equilibrium prices
    if r > 0.5:
        ub = np.full((data['I_tot']), -1.0)
        for l in range(len(data['equilibrium']['prices_urban'])):
            for i in range(data['I_tot']):
                if data['equilibrium']['prices_urban'][l][i] > ub[i]:
                    ub[i] = data['equilibrium']['prices_urban'][l][i]
        data['initial_data']['lb_p_urban'] = ub
        data['initial_data']['ub_p_urban'] = copy.deepcopy(data['initialUb'])    
        data['initial_data']['lb_p_rural'] = copy.deepcopy(data['initial_data']['lb_p_urban'])
        data['initial_data']['ub_p_rural'] = copy.deepcopy(data['initial_data']['ub_p_urban'])
    # Explore pricing strategies below current eps-equilibrium prices
    else:
        lb = np.full((data['I_tot']), 1000.0)
        for l in range(len(data['equilibrium']['prices_urban'])):
            for i in range(data['I_tot']):
                if data['equilibrium']['prices_urban'][l][i] < lb[i]:
                    lb[i] = data['equilibrium']['prices_urban'][l][i]   
        data['initial_data']['lb_p_urban'] = copy.deepcopy(data['initialLb'])
        data['initial_data']['ub_p_urban'] = lb
        data['initial_data']['lb_p_rural'] = copy.deepcopy(data['initial_data']['lb_p_urban'])
        data['initial_data']['ub_p_rural'] = copy.deepcopy(data['initial_data']['ub_p_urban'])


def newInitialSolution(data):

    data['strategies'] = {}
    data['n_strategies'] = copy.deepcopy(data['initial_data']['n_strategies'])

    data['lb_p_urban'] = copy.deepcopy(data['initial_data']['lb_p_urban'])
    data['ub_p_urban'] = copy.deepcopy(data['initial_data']['ub_p_urban'])
    data['lb_p_rural'] = copy.deepcopy(data['initial_data']['lb_p_rural'])
    data['ub_p_rural'] = copy.deepcopy(data['initial_data']['ub_p_rural'])

    data['eps_equilibrium_profit'] = copy.deepcopy(data['initial_data']['eps_equilibrium_profit'])

    data['p_urban_fixed'] = np.add(data['lb_p_urban'], np.random.random_sample(data['I_tot']) * np.subtract(data['ub_p_urban'],data['lb_p_urban']))
    data['p_rural_fixed'] = copy.deepcopy(data['p_urban_fixed'])
    data['optimizer'] = np.random.randint(1, high = data['K'] + 1)

    print('\n\nNEW INITIAL PRICES AND BOUNDS :\nAlt    LB_P U  UB_P U   p_fixed U    LB_P R  UB_P R   p_fixed R')
    for i in range(data['I_tot']):
        print('{:3d}    {:6.2f}  {:6.2f}      {:6.2f}    {:6.2f}  {:6.2f}      {:6.2f}'
              .format(i, data['initial_data']['lb_p_urban'][i], data['initial_data']['ub_p_urban'][i], data['p_urban_fixed'][i],
              data['initial_data']['lb_p_rural'][i], data['initial_data']['ub_p_rural'][i], data['p_rural_fixed'][i]))


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

        if data['DCM'] == 'NestedLogit':
            #Calculate initial values of logsum terms
            nested_logit.logsumNestedLogit(data)

        # Identify captive customers for the current strategy sets
        update_bounds.updateUtilityBounds(data)
        choice_preprocess.choicePreprocess(data)
        choice_preprocess.choicePreprocessStrategies(data)

        ### Solve the fixed-point MIP model to find a subgame equilibrium
        model, nameToIndex = fixed_point_MIP.getModel(data)
        FP_results = fixed_point_MIP.solveModel(model, nameToIndex, data)

        ### Save results of fixed-point MIP
        data['p_urban_fixed'] = copy.deepcopy(FP_results['prices_urban_in'])
        data['p_rural_fixed'] = copy.deepcopy(FP_results['prices_rural_in'])
        
        ### Update logsum term and compute real profits, if needed
        if data['DCM'] == 'NestedLogit':
            nested_logit.logsumNestedLogit(data)
            '''
            postprocessing.calculation(data)
            FP_results['max_profit_in'] = data['output']['profit']
            FP_results['demand'] = data['output']['demand']
            '''
        BR_results = {}
        BR_results['prices_urban'] = copy.deepcopy(FP_results['prices_urban_in'])
        BR_results['prices_rural'] = copy.deepcopy(FP_results['prices_rural_in'])
        BR_results['profits'] = copy.deepcopy(FP_results['max_profit_in'])
        

        ### Launch a best-response problem for each operator
        print('\nBEST RESPONSE PROBLEMS:')
        for k in range(1, data['K'] + 1):

            data['optimizer'] = k
            data['lb_profit'] = FP_results['max_profit_in'][k]

            # Set price bounds to either initial bounds (optimizer) or prices of the subgame equilibrium
            for i in range(data['I_opt_out'], data['I_tot']):
                if data['operator'][i] == k:
                    data['lb_p_urban'][i] = copy.deepcopy(data['best_response_lb_p_urban'][i])
                    data['ub_p_urban'][i] = copy.deepcopy(data['best_response_ub_p_urban'][i])
                    data['lb_p_rural'][i] = copy.deepcopy(data['best_response_lb_p_rural'][i])
                    data['ub_p_rural'][i] = copy.deepcopy(data['best_response_ub_p_rural'][i])
                else:
                    data['lb_p_urban'][i] = copy.deepcopy(data['p_urban_fixed'][i])
                    data['ub_p_urban'][i] = copy.deepcopy(data['p_urban_fixed'][i])
                    data['lb_p_rural'][i] = copy.deepcopy(data['p_rural_fixed'][i])
                    data['ub_p_rural'][i] = copy.deepcopy(data['p_rural_fixed'][i])

            update_bounds.updateUtilityBounds(data)
            choice_preprocess.choicePreprocess(data)

            # Solve the best response problem
            # HSR
            if data['DCM'] == 'NestedLogit':
                model = supply_opt.getModel(data)
                results = supply_opt.solveModel(data, model)

            # Update prices and profits after the Stackelberg games
            BR_results['profits'][k] = results['profits'][k]
            for i in data['list_alt_supplier'][k]:
                BR_results['prices_urban'][i] = results['prices_urban'][i]
                BR_results['prices_rural'][i] = results['prices_rural'][i]

        ### Verify price differences (in %)
        price_diff_U = [abs(BR_results['prices_urban'][i] - FP_results['prices_urban_in'][i]) for i in range(data['I_tot'])]
        price_diff_R = [abs(BR_results['prices_rural'][i] - FP_results['prices_rural_in'][i]) for i in range(data['I_tot'])]
        price_perc_U = np.zeros([data['I_tot']])
        price_perc_R = np.zeros([data['I_tot']])
        for i in range(data['I_opt_out'], data['I_tot']):
            price_perc_U[i] = price_diff_U[i] / FP_results['prices_urban_in'][i]
            price_perc_R[i] = price_diff_R[i] / FP_results['prices_rural_in'][i]
        FP_results['eps_price_U'] = np.max(price_perc_U)
        FP_results['eps_price_R'] = np.max(price_perc_R)
        FP_results['eps_price'] = max(FP_results['eps_price_U'],FP_results['eps_price_R'])
        ### Verify profit differences (in %)
        profit_diff = [BR_results['profits'][k] - FP_results['max_profit_in'][k] for k in range(data['K'] + 1)]
        profit_perc = np.zeros([data['K'] + 1])
        for k in range(1, data['K'] + 1):
            profit_perc[k] = profit_diff[k] / FP_results['max_profit_in'][k]
        FP_results['eps_profit'] = np.max(profit_perc)

        ### Print prices and profits in the fixed-point MIP (FP) and in the best response problems (BR)
        print('\nAlt     PriceU FP   PriceU BR     PriceR FP   PriceR BR')
        for i in range(data['I_opt_out'], data['I_tot']):
            print(' {:2d}      {:8.3f}    {:8.3f}      {:8.3f}    {:8.3f}'
                .format(i, FP_results['prices_urban_in'][i], BR_results['prices_urban'][i],
                FP_results['prices_rural_in'][i], BR_results['prices_rural'][i]))
        print('\nSupplier   ProfitFP   ProfitBR    epsilon')
        for k in range(1, data['K'] + 1):
            print('   {:2d}     {:9.3f}  {:9.3f}    {:7.4f}'.format(k, FP_results['max_profit_in'][k], BR_results['profits'][k], profit_perc[k]))

        # If profits satisfy the tolerance threshold, the solution is considered an epsilon-equilibrium
        if all(k < data['eps_equilibrium_profit'] for k in profit_perc):

            newEquilibrium = True

            # Verify if that equilibrium had already been found
            for j in range(len(data['equilibrium']['prices_urban'])):
                equilibrium_diff_U = [abs(data['equilibrium']['prices_urban'][j][i] - FP_results['prices_urban_in'][i]) for i in range(data['I_tot'])]
                equilibrium_diff_R = [abs(data['equilibrium']['prices_rural'][j][i] - FP_results['prices_rural_in'][i]) for i in range(data['I_tot'])]
                maxDiffU = np.max(equilibrium_diff_U)
                maxDiffR = np.max(equilibrium_diff_R)

                if maxDiffU < data['tolerance_equilibrium'] and maxDiffR < data['tolerance_equilibrium']:
                    newEquilibrium = False
            
            if newEquilibrium == True:
                updateListEquilibriumColumnGeneration(data, FP_results)
                print('\nProfit tolerance satisfied. Epsilon-equilibrium found.\nEps = {:7.4f}'.format(FP_results['eps_profit']))

                game_equilibrium = True
                print('\nRestart algorithmic framework.\n\n')
            
            else:
                print('\nThis epsilon-equilibrium had already been found!')

        # Case (3): eps-equilibrium not found => we search for eps-equilibria with same restricted strategy sets
        else:
            print('\nProfit tolerance not satisfied. Add/remove strategies.')            

        # If not, the algorithm continues
        if game_equilibrium is False:

            # Randomly choose to increase eps to increase chance of finding an eps-equilibrium
            if random.random() > 0.5:
                data['eps_equilibrium_profit'] = 1.05*data['eps_equilibrium_profit']
                print('eps = {:6.4f}\n'.format(data['eps_equilibrium_profit']))

            # Save best-response strategies to be added to the restricted strategy sets
            data['best_response_prices_urban'] = copy.deepcopy(BR_results['prices_urban'])
            data['best_response_prices_rural'] = copy.deepcopy(BR_results['prices_rural'])
            # Save fixed-point strategies (potentially to be removed from the restricted strategy sets?)
            data['fixed_point_prices_urban'] = copy.deepcopy(FP_results['prices_urban_in'])
            data['fixed_point_prices_rural'] = copy.deepcopy(FP_results['prices_rural_in'])

            # Archive results of current iteration
            hist_best_response.append(BR_results)
            hist_fixed_point.append(FP_results)

            # Check whether the best response strategy for one or more suppliers was already a best response
            # If so, set BRinSet to true
            BRinSet = [False for k in range(data['K'] + 1)]

            for k in range(1, data['K'] + 1):
                for it in range(iter - 1, -1, -1):
                    BR_diff_U = [abs(BR_results['prices_urban'][i] - hist_best_response[it]['prices_urban'][i]) for i in data['list_alt_supplier'][k]]
                    BR_diff_R = [abs(BR_results['prices_rural'][i] - hist_best_response[it]['prices_rural'][i]) for i in data['list_alt_supplier'][k]]
                    if all(BR_diff_U[i] < data['tolerance_equilibrium'] for i in range(len(BR_diff_U))):
                        BRinSet[k] = True

            # Update existing strategy sets
            generate_strategies.updateStrategySets(data, BRinSet)

            # Update price bounds
            update_bounds.updateSubgamePriceBounds(data)

            iter += 1


def main():
        
    t_0 = time.time()

    #Fix seed for pseudorandom draws
    seed = 3873265
    np.random.seed(seed)

    #Read instance and precompute exogenous terms
    data = data_file.getData()
    data_file.preprocessUtilities(data)

    if data['DCM'] == 'NestedLogit':
        #Calculate initial values of logsum terms
        nested_logit.logsumNestedLogit(data)

    data_file.printCustomers(data)

    #Initialize the list of eps-equilibrium solutions (line 1 of Algorithm 1)
    data['equilibrium'] = {'prices_urban': [], 'prices_rural': [
                        ], 'profits': [], 'demand': [], 'eps_price': [], 'eps_profit': []}
    
    #Initial bounds to use when restarting exploration from different region (loop lines 2-18 of Algorithm 1)
    data['initialLb'] = copy.deepcopy(data['initial_data']['lb_p_urban'])
    data['initialUb'] = copy.deepcopy(data['initial_data']['ub_p_urban'])

    while True:
        
        ### PRE-COMPUTATIONS
        nested_logit.logsumNestedLogit(data)

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
        if len(data['equilibrium']['prices_urban']) >= data['nEquilibria']:
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
    print('\nPRICES URBAN :\nID      epsilon      ', end=' ')
    for i in range(data['I_tot']):
        print('  Alt {:2d} '.format(i), end=' ')
    for j in range(len(data['equilibrium']['prices_urban'])):
        print('\n{:2d}       {:6.3f}      '.format(j+1, data['equilibrium']['eps_profit'][j]), end=' ')
        for i in range(data['I_tot']):
            print('{:8.2f} '.format(data['equilibrium']['prices_urban'][j][i]), end = ' ')
    print('\n\nPRICES RURAL :\nID      epsilon      ', end=' ')
    for i in range(data['I_tot']):
        print('  Alt {:2d} '.format(i), end=' ')
    for j in range(len(data['equilibrium']['prices_urban'])):
        print('\n{:2d}       {:6.3f}      '.format(j+1, data['equilibrium']['eps_profit'][j]), end=' ')
        for i in range(data['I_tot']):
            print('{:8.2f} '.format(data['equilibrium']['prices_rural'][j][i]), end = ' ')
    print('\n\nMARKET SHARES :\nID      epsilon      ', end=' ')
    for i in range(data['I_tot']):
        print(' Alt {:2d} '.format(i), end=' ')
    for j in range(len(data['equilibrium']['prices_urban'])):
        print('\n{:2d}       {:6.3f}      '.format(j+1, data['equilibrium']['eps_profit'][j]), end=' ')
        for i in range(data['I_tot']):
            print('{:7.4f} '.format(data['equilibrium']['demand'][j][i]/data['Pop']), end = ' ')
    print('\n\nPROFITS :\nID      epsilon      ', end=' ')
    for k in range(data['K']+1):
        print('   Supp {:1d} '.format(k), end=' ')
    for j in range(len(data['equilibrium']['prices_urban'])):
        print('\n{:2d}       {:6.3f}      '.format(j+1, data['equilibrium']['eps_profit'][j]), end=' ')
        for k in range(data['K']+1):
            print('{:9.2f} '.format(data['equilibrium']['profits'][j][k]), end = ' ')

    print('\n\nID      epsilon           Car     Air      IC    HSR1    HSR2   Train ', end=' ')
    for j in range(len(data['equilibrium']['prices_urban'])):
        car = data['equilibrium']['demand'][j][0]/data['Pop']
        air = (data['equilibrium']['demand'][j][2]+data['equilibrium']['demand'][j][3])/data['Pop']
        ic = data['equilibrium']['demand'][j][1]/data['Pop']
        HSR1 = (data['equilibrium']['demand'][j][4]+data['equilibrium']['demand'][j][5])/data['Pop']
        HSR2 = (data['equilibrium']['demand'][j][6] +
                data['equilibrium']['demand'][j][7])/data['Pop']
        train = ic + HSR1 + HSR2
        print('\n{:2d}       {:6.3f}      '.format(j+1, data['equilibrium']['eps_profit'][j]), end=' ')
        print('{:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} '.format(car, air, ic, HSR1, HSR2, train), end = ' ')

    ###########################################
    # POST-PROCESS: ANALYSIS OF MARKET SEGMENTS
    ###########################################
        
    for j in range(len(data['equilibrium']['prices_urban'])):
        print('\n\n------------------------\n---- EQUILIBRIUM {:2d} ----'.format(j))
        print('------------------------')

        data['p_urban_fixed'] = data['equilibrium']['prices_urban'][j]
        data['p_rural_fixed'] = data['equilibrium']['prices_rural'][j]

        postprocessing.calculation(data)
        postprocessing.printOutputAnalysis(data)
        postprocessing.segments(data)
                

##### MAIN
if __name__ == '__main__':

    main()
