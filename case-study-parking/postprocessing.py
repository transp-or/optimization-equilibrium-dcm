# Code for the ex-post analysis (segmentation, market shares, etc.)

# General
import sys
import time
import copy
import matplotlib.pyplot as plt
import warnings
import numpy as np

# Data
import data_parking as data_file


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
    print('\nDEMAND RESIDENTS           ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.3f}'.format(data['output']['demand_res'][i]), end=" ")
    print('\nDEMAND NON-RESIDENTS       ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.3f}'.format(data['output']['demand_nonres'][i]), end=" ")
    print('\nTOTAL MARKET SHARE         ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.4f}'.format(data['output']['market_share'][i]), end=" ")
    print('\nMARKET SHARE RESIDENTS     ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.4f}'.format(data['output']['market_share_res'][i]), end=" ")
    print('\nMARKET SHARE NON-RESIDENTS ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.4f}'.format(data['output']['market_share_nonres'][i]), end=" ")

    # Print profits for all suppliers
    print('\n\n\nSUPPLIERS:             ', end=" ")
    for k in range(data['K']+1):
        print('   {:2d}    '.format(k), end=" ")
    print('\n\nPROFITS               ', end =" ")
    for k in range(data['K']+1):
        print(' {:8.4f}'.format(data['output']['profit'][k]), end=" ")

    # Print utilities for all customers
    print('\n\n\nCUSTOMERS:             ', end=" ")
    for n in range(data['N']):
        print('   {:2d}    '.format(n), end=" ")
    print('\n\nSIZE SEGMENT           ', end=" ")
    for n in range(data['N']):
        print(' {:4.0f}    '.format(data['popN'][n]), end=" ")
    print('\nORIGIN                 ', end=" ")
    for n in range(data['N']):
        print('   {:2.0f}    '.format(data['ORIGIN'][n]), end=" ")
    print('\nAGE VEHICLE            ', end=" ")
    for n in range(data['N']):
        print('   {:2.0f}    '.format(data['AGE_VEH'][n]), end=" ")
    print('\nLOW INCOME             ', end=" ")
    for n in range(data['N']):
        print('   {:2.0f}    '.format(data['LOW_INC'][n]), end=" ")
    print('\nRESIDENT               ', end=" ")
    for n in range(data['N']):
        print('   {:2.0f}    '.format(data['RESIDENT'][n]), end=" ")
    print('\nUTILITIES             ', end =" ")
    for n in range(data['N']):
        print(' {:8.2f}'.format(data['output']['EMU'][n]), end=" ")
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
    data['output']['demand_res'] = np.zeros([data['I_tot']])
    data['output']['demand_nonres'] = np.zeros([data['I_tot']])
    data['output']['market_share'] = np.zeros([data['I_tot']])
    data['output']['market_share_res'] = np.zeros([data['I_tot']])
    data['output']['market_share_nonres'] = np.zeros([data['I_tot']])


    # Calculate utilities and choices
    for n in range(data['N']):
        for r in range(data['R']):
            for i in range(data['I_tot']):
                if data['RESIDENT'][n] == 1 and i == 2:
                    data['output']['U'][i,n,r] = data['endo_coef'][i,n,r] * (1 - data['disc_residents_PUP']) * price[i] +\
                                                 data['exo_utility'][i,n,r] + data['xi'][i,n,r]
                else:
                    data['output']['U'][i,n,r] = data['endo_coef'][i,n,r] * price[i] +\
                                                 data['exo_utility'][i,n,r] + data['xi'][i,n,r]
            i_UMax = np.argmax(data['output']['U'][:,n,r])
            data['output']['EMU'][n] += data['output']['U'][i_UMax, n, r]/data['R']
            data['output']['w'][i_UMax,n,r] = 1
            data['output']['P'][i_UMax, n] += 1/data['R']
    
    # Calculate demand
    for i in range(data['I_tot']):
        for n in range(data['N']):
            data['output']['demand'][i] += data['output']['P'][i,n] * data['popN'][n]
            if data['RESIDENT'][n] == 1:
                data['output']['demand_res'][i] += data['output']['P'][i,n] * data['popN'][n]
            elif data['RESIDENT'][n] == 0:
                data['output']['demand_nonres'][i] += data['output']['P'][i,n] * data['popN'][n]
    
    # Calculate market shares
    popR = int(sum([data['popN'][n] for n in range(data['N']) if data['RESIDENT'][n] == 1]))
    popNR = int(sum([data['popN'][n] for n in range(data['N']) if data['RESIDENT'][n] == 0]))

    data['output']['market_share'] = data['output']['demand'] / data['Pop']
    data['output']['market_share_res'] = data['output']['demand_res'] / popR
    data['output']['market_share_nonres'] = data['output']['demand_nonres'] / popNR
    
    # Calculate profits
    for k in range(data['K'] + 1):
        for i in data['list_alt_supplier'][k]:
            data['output']['profit'][k] += data['output']['demand'][i] * (price[i] - data['customer_cost'][i])


def segments(data):

    # Define number of segments and names
    # Residence (2) * Income (2)
    data['nSegments'] = 8
    data['SEGMENT_NAME'] = ['Resident',
                            'Non-resident',
                            'Low income',
                            'High income',
                            'Resident low income',
                            'Resident high income',
                            'Non-resident low income',
                            'Non-resident high income']
    

    # Initialize population in each segment
    data['SEGMENT'] = np.zeros((data['N'],data['nSegments']))
    data['popSegments'] = np.zeros((data['nSegments']))

    # Assign people to segments
    for n in range(data['N']):
        if data['RESIDENT'][n] == 1:
            data['SEGMENT'][n,0] = 1
        elif data['RESIDENT'][n] == 0:
            data['SEGMENT'][n,1] = 1

        if data['LOW_INC'][n] == 1:
            data['SEGMENT'][n,2] = 1
        elif data['LOW_INC'][n] == 0:
            data['SEGMENT'][n,3] = 1
        
        if data['RESIDENT'][n] == 1 and data['LOW_INC'][n] == 1:
            data['SEGMENT'][n,4] = 1
        if data['RESIDENT'][n] == 1 and data['LOW_INC'][n] == 0:
            data['SEGMENT'][n,5] = 1
        if data['RESIDENT'][n] == 0 and data['LOW_INC'][n] == 1:
            data['SEGMENT'][n,6] = 1
        if data['RESIDENT'][n] == 0 and data['LOW_INC'][n] == 0:
            data['SEGMENT'][n,7] = 1
 
        for s in range(data['nSegments']):
            data['popSegments'][s] += data['SEGMENT'][n][s] * data['popN'][n]

    # Create list of people for each segment
    data['list_customer_segment'] = {}
    for s in range(data['nSegments']):
        data['list_customer_segment'][s] = [i for i, segment in enumerate(data['SEGMENT'][:,s]) if segment == s]

    # Calculate market shares for segments
    data['SEGMENT_MARKET_SHARE'] = np.zeros((data['nSegments'], data['I_tot']))
    for s in range(data['nSegments']):
        for n in range(data['N']):
            if data['SEGMENT'][n,s] == 1:
                for i in range(data['I_tot']):
                    data['SEGMENT_MARKET_SHARE'][s,i] += data['output']['P'][i,n] * data['popN'][n]

    # Print market shares for segments
    print('\n\nMARKET SHARES PER SEGMENT  ', end=' ')
    for i in range(data['I_tot']):
        print('  Alt_{:1d} '.format(i), end=' ')
    for s in range(data['nSegments']):
        print('\n{:25s}  '.format(data['SEGMENT_NAME'][s]), end=' ')
        for i in range(data['I_tot']):
                print('{:7.4f} '.format(data['SEGMENT_MARKET_SHARE'][s,i]/data['popSegments'][s]), end = ' ')


if __name__ == '__main__':

    # Read instance
    data = data_file.getData()
    # Precompute exogenous terms
    data_file.preprocessUtilities(data)

    data_file.printCustomers(data)

    calculation(data)
    printOutputAnalysis(data)
    segments(data)
