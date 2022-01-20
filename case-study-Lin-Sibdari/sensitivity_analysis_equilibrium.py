# Sensitivity analysis of equilibrium results

# General
import time
import copy
import numpy as np

# Project
import supply_opt
import choice_preprocess
import update_bounds
import postprocessing

# Data
import Data_LinSibdari_MNL as data_file
import Data_LinSibdari_ObservedHet as data_file
import Data_LinSibdari_UnobservedHet as data_file


def sensitivityAnalysisEquilibrium():

    t_0 = time.time()

    replications = 100

    demand = np.zeros((replications,3))
    marketshares = np.zeros((replications,3))
    profit = np.zeros((replications, 3))
    
    epsilon = np.zeros(replications)

    for seed in range(replications):

        print('\n\n\n---------------\nREPLICATION {:3d}\n---------------'.format(seed+1))

        # Set random seed
        np.random.seed(seed)

        # Read instance
        data = data_file.getData()

        data_file.printCustomers(data)

        postprocessing.calculation(data)
        postprocessing.printOutputAnalysis(data)

        demand[seed] = data['output']['demand']
        marketshares[seed] = data['output']['market_share']
        profit[seed] = data['output']['profit']

        BR_profits = data['output']['profit']

        print('\nBEST RESPONSE PROBLEMS:')
        for k in range(1, data['K'] + 1):

            data['optimizer'] = k
            data['lb_profit'] = profit[seed][k]

            # Set price bounds to either initial bounds (optimizer) or fixed prices
            for i in range(data['I_opt_out'], data['I_tot']):
                if data['operator'][i] != k:
                    data['lb_p'][i] = copy.deepcopy(data['p_fixed'][i])
                    data['ub_p'][i] = copy.deepcopy(data['p_fixed'][i])

            update_bounds.updateUtilityBounds(data)
            choice_preprocess.choicePreprocess(data)

            # Solve the best response problem
            model = supply_opt.getModel(data)
            results = supply_opt.solveModel(data, model)

            # Update prices and profits after the Stackelberg games
            BR_profits[k] = results['profits'][k]

        ### Verify profit differences (absolute and in %)
        profit_diff = [BR_profits[k] - profit[seed][k] for k in range(data['K'] + 1)]
        profit_perc = np.zeros([data['K'] + 1])
        for k in range(1, data['K'] + 1):
            profit_perc[k] = profit_diff[k] / profit[seed][k]
        epsilon[seed] = np.max(profit_perc)

    ###########################################
    # POST-PROCESS: ANALYSIS
    ###########################################
    for seed in range(replications):
        print('\n{:4d}  '.format(seed+1), end =" ")
        print('  {:6.4f}    '.format(epsilon[seed]), end=" ")
        for i in range(3):
            print(' {:6.4f}'.format(marketshares[seed][i]), end=" ")

##### MAIN
if __name__ == '__main__':
    sensitivityAnalysisEquilibrium()
