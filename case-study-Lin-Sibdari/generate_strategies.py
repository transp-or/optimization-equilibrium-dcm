# Modification of the strategy sets of the competitors

# General
import sys
import time
import copy
import numpy as np

# Data
import Data_LinSibdari_MNL as data_file


def updateListStrategies(data):

    ### For each operator, create the list of the strategies in its restricted sets
    data['tot_strategies'] = int(np.sum(data['n_strategies']))
    data['list_strategies_opt'] = {}
    for k in range(1, data['K'] + 1):
        data['list_strategies_opt'][k] = [s for s, opt in enumerate(data['strategies']['optimizer']) if opt == k]
        print('N strategies supplier {:1d} : {:2d}'.format(k, len(data['list_strategies_opt'][k])))


def updateStrategySets(data, BRinSet):

    for k in range(1, data['K'] + 1):

        ### Incremental update of the strategy sets, used in the following cases:
        # (a) after finding a best response which is not part of the strategy set
        # (b) if the strategy set has less than the min number of strategies
        if data['n_strategies'][k] < data['max_strategies'] and (BRinSet[k] == False or data['n_strategies'][k] <= data['min_strategies']):
            
            print('Add strategy to restricted set of supplier {:1d}'.format(k))
            data['n_strategies'][k] += 1

            data['strategies']['optimizer'].append(k)
            for i in range(data['I_tot']):
                if data['operator'][i] == k:
                    data['strategies']['prices'][i].append(data['best_response_prices'][i])
                else:
                    data['strategies']['prices'][i].append(-1.0)
    
        ### Removal of strategies which block the algorithm, used in the following cases:
        # (a) after finding a subgame equilibrium which was already shown not to be a game equilibrium
        # (b) if the strategy set has more than the max number of strategies
        elif BRinSet[k] == True and np.random.random_sample() < 0.75:
            print('Remove strategy of supplier {:1d}: current best response was already in the restricted set'.format(k))
            data['n_strategies'][k] -= 1

            index = -1
            for s in data['list_strategies_opt'][k]:
                if all(data['fixed_point_prices'][i] == data['strategies']['prices'][i][s] for i in data['list_alt_supplier'][k]):
                    index = s
            for i in range(data['I_tot']):
                data['strategies']['prices'][i].pop(index)
            data['strategies']['optimizer'].pop(index)
        
        else:
            if data['n_strategies'][k] == data['max_strategies']:
                print('Remove strategy of supplier {:1d}: maximum size of restricted set'.format(k))
            else:
                print('Remove strategy of supplier {:1d}'.format(k))
            data['n_strategies'][k] -= 1

            # Find alternative with largest difference UB-LP (to keep bounds tight)
            max_diff = 0
            for i in data['list_alt_supplier'][k]:
                if data['subgame_ub_p'][i] - data['subgame_lb_p'][i] >= max_diff:
                    alt = i
                    max_diff = data['subgame_ub_p'][i] - data['subgame_lb_p'][i]
            # Find strategy with highest/lowest price for that alternative
            if np.random.random_sample()> 0.5:
                lowest_price = data['subgame_ub_p'][alt]
                for s in data['list_strategies_opt'][k]:
                    if data['strategies']['prices'][alt][s] <= lowest_price:
                        index = s
                        lowest_price = data['strategies']['prices'][alt][s]
            else:
                highest_price = data['subgame_lb_p'][alt]
                for s in data['list_strategies_opt'][k]:
                    if data['strategies']['prices'][alt][s] >= highest_price:
                        index = s
                        highest_price = data['strategies']['prices'][alt][s]
            # Remove strategy from list
            for i in range(data['I_tot']):
                data['strategies']['prices'][i].pop(index)
            data['strategies']['optimizer'].pop(index)

        # Update the list of strategies in the restricted sets
        updateListStrategies(data)


def generateStrategySets(data):

    data['strategies'] = {}

    ### Initial strategy generation, used in the following cases:
    # (a) when solving only a fixed-point MIP model, starting from initial data
    # (b) in the algorithmic framework, after reducing the search space heuristically
    # Required information:
    # 1) lb and ub of prices
    # 2) a number of strategies to be generated for each supplier

    #print('\nSTRATEGY SETS')
    data['strategies']['prices'] = [[] for i in range(data['I_tot'])]
    data['strategies']['optimizer'] = []
    for k in range(1, data['K'] + 1):
        count = 0
        #print('\nSUPPLIER %r' %k)
        for l in range(data['n_strategies'][k]):
            count += 1
            #print('Strategy %r' %count)
            data['strategies']['optimizer'].append(k)
            for i in range(data['I_tot']):
                if data['operator'][i] == k:
                    data['strategies']['prices'][i].append( data['lb_p'][i] + l / (data['n_strategies'][k] - 1) * (data['ub_p'][i] - data['lb_p'][i] ))
                    #print('     Alt {:2d}    price = {:7.2f}'.format(i, data['strategies']['prices'][i][-1]))
                else:
                    data['strategies']['prices'][i].append(-1.0)

    updateListStrategies(data)


if __name__ == '__main__':

    t_0 = time.time()
    
    # Get the data and compute exogenous terms
    data = data_file.getData()
    data_file.preprocessUtilities(data)

    generateStrategySets(data)
