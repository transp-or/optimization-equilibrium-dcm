# Code for the ex-post analysis (segmentation, market shares, etc.)

# General
import sys
import time
import copy
import matplotlib.pyplot as plt
import numpy as np

# Data
import Data_LinSibdari_MNL as data_file


def printOutputAnalysis(data):

    # Print prices, subsidies, taxes, market shares for all alternatives
    print('\nALTERNATIVES:              ', end=" ")
    for i in range(data['I_tot']):
        print('   {:2d}    '.format(i), end=" ")
    print('\n\nPRICES                     ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.4f}'.format(data['p_fixed'][i]), end=" ")
    print('\nTOTAL DEMAND               ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.3f}'.format(data['output']['demand'][i]), end=" ")
    print('\nTOTAL MARKET SHARE         ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.4f}'.format(data['output']['market_share'][i]), end=" ")

    # Print profits for all suppliers
    print('\n\n\nSUPPLIERS:             ', end=" ")
    for k in range(data['K']+1):
        print('   {:2d}    '.format(k), end=" ")
    print('\nPROFITS               ', end =" ")
    for k in range(data['K']+1):
        print(' {:8.4f}'.format(data['output']['profit'][k]), end=" ")

    # Print utilities for all customers
    print('\n\nCUSTOMERS:             ', end=" ")
    for n in range(data['N']):
        print('   {:2d}    '.format(n), end=" ")
    print('\nSIZE SEGMENT           ', end=" ")
    for n in range(data['N']):
        print(' {:4.0f}    '.format(data['popN'][n]), end=" ")
    print('\nUTILITIES             ', end =" ")
    for n in range(data['N']):
        print(' {:8.2f}'.format(data['output']['EMU'][n]), end=" ")
    print('\nPROB 0                ', end=" ")
    for n in range(data['N']):
        print(' {:8.2f}'.format(data['output']['P'][0,n]), end=" ")
    print('\nPROB 1                ', end =" ")
    for n in range(data['N']):
        print(' {:8.2f}'.format(data['output']['P'][1,n]), end=" ")
    print('\nPROB 2                ', end =" ")
    for n in range(data['N']):
        print(' {:8.2f}'.format(data['output']['P'][2,n]), end=" ")
    print()


def calculation(data):

    price = copy.deepcopy(data['p_fixed'])

    data['output'] = {}
    data['output']['U'] = np.zeros([data['I_tot'], data['N'], data['R']])
    data['output']['EMU'] = np.zeros([data['N']])
    data['output']['w'] = np.zeros([data['I_tot'], data['N'], data['R']])
    data['output']['P'] = np.zeros([data['I_tot'], data['N']])
    data['output']['profit'] = np.zeros([data['K']+1])
    data['output']['demand'] = np.zeros([data['I_tot']])
    data['output']['market_share'] = np.zeros([data['I_tot']])

    # Calculate utilities and choices
    for n in range(data['N']):
        for r in range(data['R']):
            if data['DCM'] == 'MixedLogit':
                data['output']['U'][0,n,r] = data['xi'][0, n, r]
                data['output']['U'][1,n,r] = data['endo_coef'][1,n,r] * price[1] + data['exo_utility'][1,n,r] + data['xi'][1, n, r]
                data['output']['U'][2,n,r] = data['endo_coef'][2,n,r] * price[2] + data['exo_utility'][2,n,r] + data['xi'][2, n, r]
            else:
                data['output']['U'][0,n,r] = data['xi'][0, n, r]
                data['output']['U'][1,n,r] = data['beta'] * price[1] + data['exo_utility'][1,n] + data['xi'][1, n, r]
                data['output']['U'][2,n,r] = data['beta'] * price[2] + data['exo_utility'][1,n] + data['xi'][2, n, r]
            i_UMax = np.argmax(data['output']['U'][:,n,r])
            data['output']['EMU'][n] += data['output']['U'][i_UMax, n, r]/data['R']
            data['output']['w'][i_UMax,n,r] = 1
            data['output']['P'][i_UMax, n] += 1/data['R']
    
    # Calculate demand
    for i in range(data['I_tot']):
        for n in range(data['N']):
            data['output']['demand'][i] += data['output']['P'][i,n] * data['popN'][n]
    
    # Calculate market shares
    data['output']['market_share'] = data['output']['demand'] / data['Pop']
    
    # Calculate profits
    for k in range(data['K'] + 1):
        for i in data['list_alt_supplier'][k]:
            data['output']['profit'][k] += data['output']['demand'][i] * price[i]


def linearProfitCurve():

        data = data_file.getData()
        data_file.printCustomers(data)

        price = copy.deepcopy(data['p_fixed'])

        opt = data['optimizer']
        price[opt] = data['lb_p'][opt]

        while price[opt] <= data['ub_p'][opt]:

            data['output'] = {}
            data['output']['U'] = np.zeros([data['I_tot'], data['N'], data['R']])
            data['output']['EMU'] = np.zeros([data['N']])
            data['output']['w'] = np.zeros([data['I_tot'], data['N'], data['R']])
            data['output']['P'] = np.zeros([data['I_tot'], data['N']])
            data['output']['profit'] = np.zeros([data['K']+1])
            data['output']['demand'] = np.zeros([data['I_tot']])
            data['output']['market_share'] = np.zeros([data['I_tot']])

            # Calculate utilities and choices
            for n in range(data['N']):
                for r in range(data['R']):
                    if data['DCM'] == 'MixedLogit':
                        data['output']['U'][0,n,r] = data['xi'][0, n, r]
                        data['output']['U'][1,n,r] = data['endo_coef'][i,n,r] * price[1] + data['exo_utility'][1,n,r] + data['xi'][1, n, r]
                        data['output']['U'][2,n,r] = data['endo_coef'][i,n,r] * price[2] + data['exo_utility'][2,n,r] + data['xi'][2, n, r]
                    else:
                        data['output']['U'][0,n,r] = data['xi'][0, n, r]
                        data['output']['U'][1,n,r] = data['beta'] * price[1] + data['exo_utility'][1,n] + data['xi'][1, n, r]
                        data['output']['U'][2,n,r] = data['beta'] * price[2] + data['exo_utility'][2,n] + data['xi'][2, n, r]
                    i_UMax = np.argmax(data['output']['U'][:,n,r])
                    data['output']['EMU'][n] += data['output']['U'][i_UMax, n, r]/data['R']
                    data['output']['w'][i_UMax,n,r] = 1
                    data['output']['P'][i_UMax, n] += 1/data['R']
            
            # Calculate demand
            for i in range(data['I_tot']):
                for n in range(data['N']):
                    data['output']['demand'][i] += data['output']['P'][i,n] * data['popN'][n]
            
            # Calculate market shares
            data['output']['market_share'] = data['output']['demand'] / data['Pop']
            
            # Calculate profits
            for k in range(data['K'] + 1):
                for i in data['list_alt_supplier'][k]:
                    data['output']['profit'][k] += data['output']['demand'][i] * price[i]

            data['p_fixed'] = price

            printOutputAnalysis(data)

            price[opt] += 1


def mainSimulation():
    
        replications = 100

        demand = np.zeros((replications,3))
        marketshares = np.zeros((replications,3))
        
        for seed in range(replications):
            
            # Set random seed
            np.random.seed(seed)

            # Read instance
            data = data_file.getData()

            data_file.printCustomers(data)

            calculation(data)
            printOutputAnalysis(data)

            demand[seed] = data['output']['demand']
            marketshares[seed] = data['output']['market_share']
        
        for seed in range(replications):
            print('\n{:4d}  '.format(seed+1), end =" ")
            for i in range(3):
                print(' {:6.4f}'.format(marketshares[seed][i]), end=" ")


if __name__ == '__main__':

    #mainSimulation()
    linearProfitCurve()


