# Code for the fixed-point iteration algorithm
# (simultaneous game solved in a sequential manner)

# General
import sys
import time
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import cycle
import numpy as np

# Models
import supply_opt

import update_bounds
import choice_preprocess

# Data
import data_parking as data_file


def fixedPointIterationAlg(data):

    output = {}

    # Price, profit, market share and demand at each iteration
    output['p_history'] = np.full((data['max_iter']+1, data['I_tot']), -1.0)
    output['profit'] = np.full((data['max_iter']+1, data['K'] + 1), -1.0)
    output['market_share'] = np.full((data['max_iter']+1, data['I_tot']), -1.0)
    output['demand_history'] = np.full((data['max_iter']+1, data['I_tot']), -1.0)
    # Information about equilibrium: iteration number of cycle start and end, type of equilibrium
    output['cycle_start'] = None
    output['cycle_end'] = None
    output['cycle_type'] = None

    # Initialize variables of the sequential game
    iter = 0
    cycle = False
    data['lb_profit'] = None

    # Main loop
    while (iter < data['max_iter']) and cycle is False:
        
        iter += 1
        print('\n\nITERATION %r' %iter)

        #########################
        # PREPROCESS
        #########################
        
        # Update price bounds
        for i in range(data['I_opt_out'], data['I_tot']):
            if data['operator'][i] != data['optimizer']:
                data['lb_p'][i] = copy.deepcopy(data['p_fixed'][i])
                data['ub_p'][i] = copy.deepcopy(data['p_fixed'][i])
            else:
                data['lb_p'][i] = copy.deepcopy(data['initial_data']['lb_p'][i])
                data['ub_p'][i] = copy.deepcopy(data['initial_data']['ub_p'][i])
        
        # Update utility bounds and preprocess choices
        update_bounds.updateUtilityBounds(data)
        choice_preprocess.choicePreprocess(data)

        # Print fixed prices of non-optimizing suppliers
        print('\nAlt  Supplier   Fixed price')
        for i in range(data['I_opt_out'], data['I_tot']):
            if data['operator'][i] != data['optimizer']:
                print(' {:2d}       {:2d}      {:8.2f}'.format(i, data['operator'][i], data['p_fixed'][i]))

        #########################
        # BEST RESPONSE PROBLEM
        #########################

        ### Run the best response (BR) problem for the current optimizer
        model = supply_opt.getModel(f, data)
        BR_results = supply_opt.solveModel(f, data, model)

        #########################
        # POSTPROCESS
        #########################
        
        # Update the price history
        for i in range(data['I_tot']):
            output['p_history'][iter, i] = BR_results['prices'][i]
        # Update the profit history
        for k in range(data['K'] + 1):
            output['profit'][iter][k] = 0.0
            for i in data['list_alt_supplier'][k]:
                output['profit'][iter][k] += BR_results['demand'][i]*BR_results['prices'][i]    #Revenues
                output['profit'][iter][k] += -data['fixed_cost'][i]                             #Fixed costs
                output['profit'][iter][k] += -data['customer_cost'][i]*BR_results['demand'][i]  #Var costs
        # Update the market share history
        for i in range(data['I_tot']):
            output['market_share'][iter, i] = float(BR_results['demand'][i]) / data['Pop']
        # Update the demand history
        for i in range(data['I_tot']):
            output['demand_history'][iter, i] = BR_results['demand'][i]

        #########################
        # UPDATE INDICATORS
        #########################

        ### Cycle detection: all iterations in which the current optimizer previously optimized
        iterations_to_check = range(iter - data['K'], 0, -data['K'])
        print('\nIterations to check: %r' % list(iterations_to_check))
        print('\nIter     MaxDevPrice')

        for j in iterations_to_check:

            cycle = True
            deviation = np.amax(abs(output['p_history'][j,:] - output['p_history'][iter,:]))
            print(' {:3d}      {:10.2f}'.format(j, deviation))

            if deviation > data['tolerance_cyclic_equilibrium']:
                cycle = False
                    
            else:
                # Cycle detected: save all useful information
                output['cycle_start'] = j
                output['cycle_end'] = iter
                output['p_history'] = output['p_history'][:iter+1,:]
                output['profit'] = output['profit'][:iter+1,:]
                output['market_share'] = output['market_share'][:iter+1,:]
                output['demand_history'] = output['demand_history'][:iter+1,:]

                # Define the type of cycle
                if iter - output['cycle_start'] == data['K'] and deviation < data['tolerance_equilibrium'] and data['DCM'] != 'NestedLogit':
                    print('\nNash equilibrium detected\n')
                    print('\nNash equilibrium detected\n', file=f)
                    output['cycle_type'] = 'NashEquilibrium'
                else:
                    print('\nCycle detected. Length of cycle = %r\n' %(iter - output['cycle_start']))
                    print('\nCycle detected. Length of cycle = %r\n' %(iter - output['cycle_start']), file=f)
                    output['cycle_type'] = 'CyclicEquilibrium'
                break

        ### Update the data for the next iteration
        if cycle is False:

            data['optimizer'] = ((data['optimizer']) % data['K']) + 1
            data['lb_profit'] = output['profit'][iter][data['optimizer']]   #To be checked
            
            data['p_fixed'] = copy.deepcopy(BR_results['prices'])
            
            # If no convergence before max iter is reached, then just save last 5*K iterations
            if iter == data['max_iter']:
                # Save all useful information
                output['cycle_start'] = max(iter - 5 * data['K'], data['K'])
                output['cycle_end'] = iter
                output['p_history'] = output['p_history'][:iter+1,:]
                output['profit'] = output['profit'][:iter+1,:]
                output['market_share'] = output['market_share'][:iter+1,:]
                output['demand_history'] = output['demand_history'][:iter+1,:]
                
                print('\nMax number of iterations reached. Length of cycle = %r\n' %(iter - output['cycle_start']))
                output['cycle_type'] = 'CyclicEquilibrium'
    
    # Print summary of profits and prices for all iterations
    print('\nPRICES : \nIter   Profit1   Profit2   ', end=" ")
    for i in range(data['I_tot']):
        print('   {:2d}    '.format(i), end=" ")
    for it in range(1, iter + 1):
        print('\n{:3d}   {:8.3f}  {:8.3f} '.format(it, output['profit'][it, 1], output['profit'][it, 2]), end=" ")
        for i in range(data['I_tot']):
            print(' {:8.3f}'.format(output['p_history'][it, i]), end=" ")
    print()
    
    return output


def plotGraphs(title, data, results):
    '''
    Plot the values of the prices as a function of the iteration number.
    '''

    if data['K'] == 2:
        color = {1: '0', 2: '0.7'}
    elif data['K'] == 3:
        color = {1: '0.5', 2: '0.8', 3: '0.2'}

    #lines = ["-",":","-.","--"]
    lines = [":","--"]
    linecycler = cycle(lines)


    ### Price graph

    # Get the price history for each operator
    p_hist = [[] for i in range(data['I_tot'])]
    for i in range(data['I_tot']):
        p_hist[i] = [prices for prices in results['p_history'][:,i] if prices != -1]
    # Plot them
    for i in range(data['I_opt_out'], data['I_tot']):
        if data['operator'][i] > 0:
            plt.plot(p_hist[i], label=data['name_mapping'][i],
                        color=color[data['operator'][i]],
                        linestyle=next(linecycler))

    # Plot vertical line to indicate the beginning and end of the cycle
    if results['cycle_start'] is not None:
        plt.axvline(x=results['cycle_start']-1, linestyle=':', color='0.8', linewidth=0.25)
        plt.axvline(x=results['cycle_end']-1, linestyle=':', color='0.8', linewidth=0.25)
    plt.ylabel('Price')
    plt.xlabel('Iteration number')
    if data['Instance'] == 'Parking_MixedLogit':
        plt.ylim(0.2,1.0)
    else:
        plt.ylim(min(data['lb_p'][data['I_opt_out']:]), max(data['ub_p'][data['I_opt_out']:]))
    if results['cycle_end'] is None:
        plt.xticks(range(0, data['max_iter'], int(np.floor(data['max_iter']/10.0))))
    else:
        plt.xticks(range(0, results['cycle_end'], max(int(np.floor(results['cycle_end']/10.0)),1)))
    plt.legend(loc='upper left',fontsize=6.5)
    plt.savefig('OutputGraphs/price_history_%r_%r.png' %(title, data['countFixedPointIter']))
    plt.close()

    ### Market shares graph

    # Get the demand history for each operator
    market_hist = [[] for i in range(data['I_tot'])]
    for i in range(data['I_tot']):
        market_hist[i] = [market for market in results['market_share'][:,i] if market != -1]
    market_sum = []
    for iter in range(len(market_hist[0])):
        market_sum.append(sum([market_hist[i][iter] for i in range(data['I_opt_out'], data['I_tot'])]))
    # Plot them
    for i in range(data['I_opt_out'], data['I_tot']):
        if data['operator'][i] == 1:
            plt.plot(market_hist[i], label=data['name_mapping'][i],
                        color=color[data['operator'][i]],
                        linestyle=next(linecycler),
                        marker='.')
        elif data['operator'][i] == 2:
            plt.plot(market_hist[i], label=data['name_mapping'][i],
                        color=color[data['operator'][i]],
                        linestyle=next(linecycler),
                        marker='*')
        elif data['operator'][i] == 3:
            plt.plot(market_hist[i], label=data['name_mapping'][i],
                    color=color[data['operator'][i]],
                    linestyle=next(linecycler),
                    marker='x')    
    plt.plot(market_sum, ':' ,label='Total', color='black')

    # Plot vertical line to indicate the beginning of the cycle
    if results['cycle_start'] is not None:
        plt.axvline(x=results['cycle_start']-1, linestyle=':', color='0.8', linewidth=0.25)
        plt.axvline(x=results['cycle_end']-1, linestyle=':', color='0.8', linewidth=0.25)
    plt.ylabel('Market share')
    if data['Instance'] == 'Parking_MixedLogit':
        plt.ylim(0.0, 1.1)
    plt.xlabel('Iteration number')
    if results['cycle_end'] is None:
        plt.xticks(range(0, data['max_iter'], int(np.floor(data['max_iter']/10.0))))
    else:
        plt.xticks(range(0, results['cycle_end'], max(int(np.floor(results['cycle_end']/10.0)),1)))
    plt.legend(loc='upper left', fontsize=6.5)
    plt.savefig('OutputGraphs/market_history_%r_%r.png' %(title, data['countFixedPointIter']))
    plt.close()

    ### Profits graph

    # Get the profit history for each operator
    profit_history = [[] for k in range(data['K'] + 1)]
    for k in range(data['K'] + 1):
        profit_history[k] = [profits for profits in results['profit'][1:, k] if profits != -1]
        profit_history[k].insert(0, profit_history[k][0])
    profit_sum = [0]
    for iter in range(len(profit_history[0])):
        profit_sum.append(sum([profit_history[k][iter] for k in range(1, data['K'] + 1)]))
    # Plot them
    for k in range(1, data['K'] + 1):
        if k == 1:
            plt.plot(profit_history[k], label=k, color=color[k], marker='.')
        else:
            plt.plot(profit_history[k], label=k, color=color[k], marker='*')
    plt.plot(profit_sum[1:], ':' ,label='Total', color='black')
    # Plot vertical line to indicate the beginning of the cycle
    if results['cycle_start'] is not None:
        plt.axvline(x=results['cycle_start'],
                    linestyle=':', color='0.8', linewidth=0.25)
        plt.axvline(x=results['cycle_end'], linestyle=':',
                    color='0.8', linewidth=0.25)
    plt.ylabel('Profit (€)')
    if data['Instance'] == 'Parking_MixedLogit':
        plt.ylim(0.0, 15.0)
    plt.xlabel('Iteration number')
    if results['cycle_end'] is None:
        plt.xticks(range(0, data['max_iter'], int(np.floor(data['max_iter']/10.0))))
    else:
        plt.xticks(range(0, results['cycle_end'], max(int(np.floor(results['cycle_end']/10.0)),1)))
    plt.legend(loc='upper left', fontsize=8)
    plt.savefig('OutputGraphs/profit_history_%r_%r.png' %(title, data['countFixedPointIter']))
    plt.close()

    data['countFixedPointIter'] += 1


if __name__ == '__main__':

    t_0 = time.time()
    
    #Read instance
    data = data_file.getData()

    #Precompute exogenous terms
    data_file.preprocessUtilities(data)

    #Calculate utility bounds
    update_bounds.updateUtilityBounds(data)

    #Preprocess captive choices
    choice_preprocess.choicePreprocess(data)
        
    #RUN FIXED POINT ITERATION ALGORITHM
    fixed_point_it_results = fixedPointIterationAlg(data)
        
    print('\n\nTotal runtime fixed-point iteration algorithm: {:8.2f}\n'.format(time.time() - t_0))

    ### Graphs
    plotGraphs('FixedPoint_Iteration_Algorithm', data, fixed_point_it_results)
    