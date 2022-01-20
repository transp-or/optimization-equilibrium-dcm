# Code for the fixed-point iteration algorithm with regulator
# (simultaneous game solved in a sequential manner)

# General
import time
import copy
import random
import numpy as np

# Models
import supplier_opt
import regulator_opt

import update_bounds
import choice_preprocess
import nested_logit
import postprocessing

# Data
import data_intercity as data_file


def printResults(data, output):
    
    print('\n\n----------------------------------------------')
    print('------------ SUMMARY OF ITERATIONS -----------')
    print('----------------------------------------------\n')

    # Print profits, prices, taxes, market shares, segment analysis, utilities, emissions, policy costs
    print('\nPRICES (URBAN) : \nIter   Eps     ', end=" ")
    for i in range(data['I_tot']):
        print('   {:2d}    '.format(i), end=" ")
    for it in range(1, data['max_iter'] + 1):
        print('\n{:3d}   {:6.4f}  '.format(it, output['eps_history'][it]), end =" ")
        for i in range(data['I_tot']):
            print(' {:8.2f}'.format(output['prices_urban'][it,i]), end=" ")

    print('\nPRICES (RURAL) : \nIter   Eps     ', end=" ")
    for i in range(data['I_tot']):
        print('   {:2d}    '.format(i), end=" ")
    for it in range(1, data['max_iter'] + 1):
        print('\n{:3d}   {:6.4f}  '.format(it, output['eps_history'][it]), end =" ")
        for i in range(data['I_tot']):
            print(' {:8.2f}'.format(output['prices_rural'][it,i]), end=" ")
    
    print('\n\nTAXES/SUBSIDIES (HIGH INCOME) : \nIter   Eps     ', end=" ")
    for i in range(data['I_tot']):
        print('   {:2d}    '.format(i), end=" ")
    for it in range(1, data['max_iter'] + 1):
        print('\n{:3d}   {:6.4f}  '.format(it, output['eps_history'][it]), end =" ")
        for i in range(data['I_tot']):
            print(' {:8.2f}'.format(output['taxsubsidy_highinc'][it,i]), end =" ")

    print('\n\nTAXES/SUBSIDIES (LOW INCOME) : \nIter   Eps     ', end=" ")
    for i in range(data['I_tot']):
        print('   {:2d}    '.format(i), end=" ")
    for it in range(1, data['max_iter'] + 1):
        print('\n{:3d}   {:6.4f}  '.format(it, output['eps_history'][it]), end =" ")
        for i in range(data['I_tot']):
            print(' {:8.2f}'.format(output['taxsubsidy_lowinc'][it,i]), end =" ")

    print('\n\nMARKET SHARES : \nIter   Eps     ', end=" ")
    for i in range(data['I_tot']):
        print('   {:2d}    '.format(i), end=" ")
    for it in range(1, data['max_iter'] + 1):
        print('\n{:3d}   {:6.4f}  '.format(it, output['eps_history'][it]), end =" ")
        for i in range(data['I_tot']):
            print(' {:8.3f}'.format(output['demand'][it,i]/data['Pop']), end =" ")
    
    print('\n\nPROFIT : \nIter   Eps     ', end=" ")
    for k in range(1, data['K'] + 1):
        print('   {:1d}      '.format(k), end=" ")
    for it in range(1, data['max_iter'] + 1):
        print('\n{:3d}   {:6.4f}  '.format(it, output['eps_history'][it]), end =" ")
        for k in range(1, data['K'] + 1):
            print('{:8.2f}  '.format(output['profit'][it,k]), end =" ")
    
    print('\n\nEMISSIONS AND UTILITIES : \nIter   Eps    Emissions    ', end=" ")
    for n in range(data['N']):
        print('  {:2d}    '.format(n), end=" ")
    for it in range(1, data['max_iter'] + 1):
        print('\n{:3d}   {:6.4f}   {:8.2f}  '
            .format(it, output['eps_history'][it], output['emissions'][it]/1000), end =" ")
        for n in range(data['N']):
            print(' {:6.2f} '.format(output['EMU'][it,n]), end =" ")
    print()


def heuristic_algorithm_regulation(data):
    
    output = {}

    # Price, profit, market share and demand at each iteration
    output['prices_urban'] = np.full((data['max_iter']+1, data['I_tot']), -1.0)
    output['prices_urban_BR'] = np.full((data['max_iter']+1, data['I_tot']), -1.0)
    output['prices_rural'] = np.full((data['max_iter']+1, data['I_tot']), -1.0)
    output['prices_rural_BR'] = np.full((data['max_iter']+1, data['I_tot']), -1.0)
    output['taxsubsidy_highinc'] = np.full((data['max_iter']+1, data['I_tot']), -1.0)
    output['taxsubsidy_lowinc'] = np.full((data['max_iter']+1, data['I_tot']), -1.0)
    output['profit'] = np.full((data['max_iter']+1, data['K'] + 1), -1.0)
    output['profit_BR'] = np.full((data['max_iter']+1, data['K'] + 1), -1.0)
    output['demand'] = np.full((data['max_iter']+1, data['I_tot']), -1.0)
    output['demand_high'] = np.full((data['max_iter']+1, data['I_tot']), -1.0)
    output['demand_low'] = np.full((data['max_iter']+1, data['I_tot']), -1.0)
    output['market_share'] = np.full((data['max_iter']+1, data['I_tot']), -1.0)
    output['market_share_high'] = np.full((data['max_iter']+1, data['I_tot']), -1.0)
    output['market_share_low'] = np.full((data['max_iter']+1, data['I_tot']), -1.0)
    output['emissions'] = np.full((data['max_iter']+1), -1.0)
    output['EMU'] = np.full((data['max_iter']+1, data['N']), -1.0)
    output['eps_history'] = np.full((data['max_iter']+1), 100.0)

    # Initialize iteration count
    iter = 0

    # Main loop
    while iter < data['max_iter'] and output['eps_history'][iter] > data['eps_equilibrium_profit']:
        
        iter += 1
        print('\n\n-------------\nITERATION %r\n-------------' %iter)

        ##### (1) Each supplier updates its strategy sequentially (smoothing procedure)

        print('\nSmoothing through sequential solving:')

        data['optimizer'] = np.random.randint(1, high = data['K'] + 1)
        
        for it in range(data['max_iter_smoothing']):
            print('\nIteration {:3d}'.format(it+1))

            for k in range(1, data['K'] + 1):
                
                data['optimizer'] = (data['optimizer'] % data['K']) + 1
                
                ##### Preprocess

                # Update price bounds
                for i in range(data['I_opt_out'], data['I_tot']):
                    if data['operator'][i] != data['optimizer']:
                        data['lb_p_urban'][i] = copy.deepcopy(data['p_urban_fixed'][i])
                        data['ub_p_urban'][i] = copy.deepcopy(data['p_urban_fixed'][i])
                        data['lb_p_rural'][i] = copy.deepcopy(data['p_rural_fixed'][i])
                        data['ub_p_rural'][i] = copy.deepcopy(data['p_rural_fixed'][i])                    
                    else:
                        data['lb_p_urban'][i] = copy.deepcopy(data['initial_data']['lb_p_urban'][i])
                        data['ub_p_urban'][i] = copy.deepcopy(data['initial_data']['ub_p_urban'][i])
                        data['lb_p_rural'][i] = copy.deepcopy(data['initial_data']['lb_p_rural'][i])
                        data['ub_p_rural'][i] = copy.deepcopy(data['initial_data']['ub_p_rural'][i])

                # Update utility bounds and preprocess choices
                update_bounds.updateUtilityBoundsFixedRegulator(data)
                choice_preprocess.choicePreprocess(data)

                ##### Run the best response (BR) problem for the current optimizer
                model = supplier_opt.getModel(data)
                BR_results = supplier_opt.solveModel(data, model)
                    
                ##### Postprocess

                # Update the prices and profits (with smoothing)
                for i in data['list_alt_supplier'][data['optimizer']]:
                    data['p_urban_fixed'][i] = 0.5 * data['p_urban_fixed'][i] + 0.5 * BR_results['prices_urban'][i]
                    data['p_rural_fixed'][i] = 0.5 * data['p_rural_fixed'][i] + 0.5 * BR_results['prices_rural'][i]

        ##### (2) The regulator updates its policies
        
        # Calculate logsum term for nested logit
        nested_logit.logsumNestedLogitRegulator(data)

        # Calculate utility bounds and preprocess choices
        update_bounds.updateUtilityBoundsFixedPrice(data)
        choice_preprocess.choicePreprocess(data)

        # Optimization model for the regulator
        t_reg_start = time.time()
        model = regulator_opt.getModel(data)
        results = regulator_opt.solveModel(data, model)
        print('Time to solve the regulator model:     {:8.2f}'.format(time.time() - t_reg_start))

        output['prices_urban'][iter, :] = copy.deepcopy(data['p_urban_fixed'])
        output['prices_rural'][iter, :] = copy.deepcopy(data['p_rural_fixed'])
        output['taxsubsidy_highinc'][iter, :] = copy.deepcopy(results['taxsubsidy_highinc'])
        output['taxsubsidy_lowinc'][iter, :] = copy.deepcopy(results['taxsubsidy_lowinc'])
        data['fixed_taxsubsidy_highinc'] = copy.deepcopy(results['taxsubsidy_highinc'])
        data['fixed_taxsubsidy_lowinc'] = copy.deepcopy(results['taxsubsidy_lowinc'])

        ##### (3) Profits and demand are computed for the current solution

        # Update logsum term for nested logit
        nested_logit.logsumNestedLogitRegulator(data)

        # Ex-post calculations
        postprocessing.calculation(data)
        output['demand'][iter, :] = copy.deepcopy(data['output']['demand'])
        output['demand_high'][iter, :] = copy.deepcopy(data['output']['demand_high'])
        output['demand_low'][iter, :] = copy.deepcopy(data['output']['demand_low'])
        output['market_share'][iter, :] = copy.deepcopy(data['output']['market_share'])
        output['market_share_high'][iter, :] = copy.deepcopy(data['output']['market_share_high'])
        output['market_share_low'][iter, :] = copy.deepcopy(data['output']['market_share_low'])

        output['emissions'][iter] = copy.deepcopy(data['output']['emissions'])
        output['EMU'][iter, :] = copy.deepcopy(data['output']['EMU'])

        output['profit'][iter, :] = copy.deepcopy(data['output']['profit'])

        ##### (4) Verify best-response conditions

        for k in range(1, data['K'] + 1):
            data['optimizer'] = k
            
            ##### Preprocess

            # Update price bounds
            for i in range(data['I_opt_out'], data['I_tot']):
                if data['operator'][i] != data['optimizer']:
                    data['lb_p_urban'][i] = copy.deepcopy(data['p_urban_fixed'][i])
                    data['ub_p_urban'][i] = copy.deepcopy(data['p_urban_fixed'][i])
                    data['lb_p_rural'][i] = copy.deepcopy(data['p_rural_fixed'][i])
                    data['ub_p_rural'][i] = copy.deepcopy(data['p_rural_fixed'][i])                    
                else:
                    data['lb_p_urban'][i] = copy.deepcopy(data['initial_data']['lb_p_urban'][i])
                    data['ub_p_urban'][i] = copy.deepcopy(data['initial_data']['ub_p_urban'][i])
                    data['lb_p_rural'][i] = copy.deepcopy(data['initial_data']['lb_p_rural'][i])
                    data['ub_p_rural'][i] = copy.deepcopy(data['initial_data']['ub_p_rural'][i])
            
            # Update utility bounds and preprocess choices
            update_bounds.updateUtilityBoundsFixedRegulator(data)
            choice_preprocess.choicePreprocess(data)

            # Print fixed prices of non-optimizing suppliers
            print('\nAlt  Supplier        Fixed price urban   Fixed price rural')
            for i in range(data['I_opt_out'], data['I_tot']):
                if data['operator'][i] != data['optimizer']:
                    print(' {:2d}        {:2d}          {:15.2f}     {:15.2f}'
                    .format(i, data['operator'][i], data['p_urban_fixed'][i], data['p_rural_fixed'][i]))

            ##### Run the best response (BR) problem for the current optimizer
            if data['DCM'] == 'NestedLogit':
                model = supplier_opt.getModel(data)
                BR_results = supplier_opt.solveModel(data, model)
                
            ##### Postprocess

            # Update the prices and profits
            output['profit_BR'][iter,k] = 0.0
            for i in data['list_alt_supplier'][k]:
                output['prices_urban_BR'][iter,i] = BR_results['prices_urban'][i]
                output['prices_rural_BR'][iter,i] = BR_results['prices_rural'][i]
                output['profit_BR'][iter,k] +=\
                  (BR_results['demand_urban'][i]*BR_results['prices_urban'][i] +\
                   BR_results['demand_rural'][i]*BR_results['prices_rural'][i])                   #Revenues
                output['profit_BR'][iter,k] += -data['fixed_cost'][i]                             #Fixed costs
                output['profit_BR'][iter,k] += -data['customer_cost'][i]*BR_results['demand'][i]  #Var costs

        ##### (5) Check for eps-equilibrium (profit differences between (2) and (4))

        profit_diff = [output['profit_BR'][iter,k] - output['profit'][iter,k] for k in range(data['K'] + 1)]
        profit_perc = np.zeros([data['K'] + 1])
        for k in range(1, data['K'] + 1):
            profit_perc[k] = profit_diff[k] / output['profit'][iter,k]
        output['eps_history'][iter] = np.max(profit_perc)

        # Print profits and epsilons
        print('\nSUMMARY : \n\nSupplier     Profit  ProfitBR        Eps')
        for k in range(1, data['K'] + 1):
            print('   {:2d}      {:8.2f}  {:8.2f}    {:7.4f}'
                .format(k, output['profit'][iter,k], output['profit_BR'][iter,k], profit_perc[k]))

        ##### (6) Update for next iteration
        
        # Check if a solution is repeated 
        repeat = False

        for j in range(1,iter):
            diff_price_U = [abs(output['prices_urban_BR'][j,i] - output['prices_urban_BR'][iter,i]) for i in range(data['I_tot'])]
            diff_price_R = [abs(output['prices_rural_BR'][j,i] - output['prices_rural_BR'][iter,i]) for i in range(data['I_tot'])]
            maxDiffU = np.max(diff_price_U)
            maxDiffR = np.max(diff_price_R)

            if maxDiffU < data['tolerance'] and maxDiffR < data['tolerance']:
                repeat = True
        
        # Update fixed prices
        if repeat == False:
            for i in range(data['I_opt_out'], data['I_tot']):
                data['p_urban_fixed'][i] = copy.deepcopy(output['prices_urban_BR'][iter,i])
                data['p_rural_fixed'][i] = copy.deepcopy(output['prices_rural_BR'][iter,i])
        
        elif repeat == True:
            print('\n\nRepeated solution!\n')
            for i in range(data['I_opt_out'], data['I_tot']):
                if random.random() > 0.5:
                    data['p_urban_fixed'][i] = output['prices_urban_BR'][iter,i] - 1.0
                    data['p_rural_fixed'][i] = output['prices_rural_BR'][iter,i] - 1.0
                else:
                    data['p_urban_fixed'][i] = output['prices_urban_BR'][iter,i] + 1.0
                    data['p_rural_fixed'][i] = output['prices_rural_BR'][iter,i] + 1.0

    ##### Print results
    printResults(data, output)

    ##### Run analysis on 'best' eps-equilibrium solution
    bestEps = np.argmin(output['eps_history'])
    print('\nLowest epsilon in iteration {:3d}'.format(bestEps))

    data['p_urban_fixed'] = output['prices_urban'][bestEps,:]
    data['p_rural_fixed'] = output['prices_rural'][bestEps,:]
    data['fixed_taxsubsidy_highinc'] = output['taxsubsidy_highinc'][bestEps,:]
    data['fixed_taxsubsidy_lowinc'] = output['taxsubsidy_lowinc'][bestEps,:]

    nested_logit.logsumNestedLogitRegulator(data)
    
    postprocessing.calculation(data)
    postprocessing.segments(data)
    postprocessing.printOutputAnalysis(data)

    return output


if __name__ == '__main__':
    
    # Initialize the dictionary 'dict' containing all the input/output data
    data = {}

    # Define parameters of the algorithm
    data_file.setAlgorithmParameters(data)

    #Read instance
    data_file.getData(data)

    #Precompute exogenous terms
    data_file.preprocessUtilities(data)

    #Calculate initial values of logsum terms
    nested_logit.logsumNestedLogitRegulator(data)
    
    #Print demand data
    data_file.printCustomers(data)

    #Run Algorithm 1
    results = heuristic_algorithm_regulation(data)
