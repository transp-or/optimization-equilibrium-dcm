# Code for the ex-post analysis (segmentation, market shares, etc.)

# General
import sys
import time
import copy
import numpy as np

# Functions
import nested_logit

# Data
import data_HSR as data_file


def printOutputAnalysis(data):

    # Print prices, subsidies, taxes, market shares for all alternatives
    print('\nALTERNATIVES:          ', end=" ")
    for i in range(data['I_tot']):
        print('   {:2d}    '.format(i), end=" ")
    print('\n\nPRICES URBAN          ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.2f}'.format(data['p_urban_fixed'][i]), end=" ")
    print('\nPRICES RURAL          ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.2f}'.format(data['p_rural_fixed'][i]), end=" ")
    print('\nTOTAL DEMAND          ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.2f}'.format(data['output']['demand'][i]), end=" ")
    print('\nDEMAND URBAN          ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.2f}'.format(data['output']['demand_urban'][i]), end=" ")
    print('\nDEMAND RURAL          ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.2f}'.format(data['output']['demand_rural'][i]), end=" ")
    print('\nTOTAL MARKET SHARE    ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.4f}'.format(data['output']['market_share'][i]), end=" ")
    print('\nMARKET SHARE URBAN    ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.4f}'.format(data['output']['market_share_urban'][i]), end=" ")
    print('\nMARKET SHARE RURAL    ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.4f}'.format(data['output']['market_share_rural'][i]), end=" ")

    # Print profits for all suppliers
    print('\n\nSUPPLIERS:             ', end=" ")
    for k in range(data['K']+1):
        print('   {:2d}    '.format(k), end=" ")
    print('\nPROFITS               ', end =" ")
    for k in range(data['K']+1):
        print(' {:8.2f}'.format(data['output']['profit'][k]), end=" ")

    # Print utilities for all customers
    print('\n\nCUSTOMERS:             ', end=" ")
    for n in range(data['N']):
        print('   {:2d}    '.format(n), end=" ")
    print('\n\nSIZE SEGMENT           ', end=" ")
    for n in range(data['N']):
        print(' {:4.0f}    '.format(data['popN'][n]), end=" ")
    print('\nINCOME                 ', end=" ")
    for n in range(data['N']):
        print('   {:2.0f}    '.format(data['INCOME'][n]), end=" ")
    print('\nBUSINESS               ', end=" ")
    for n in range(data['N']):
        print('   {:2.0f}    '.format(data['BUSINESS'][n]), end=" ")
    print('\nORIGIN                 ', end=" ")
    for n in range(data['N']):
        print('   {:2.0f}    '.format(data['ORIGIN'][n]), end=" ")
    print('\nUTILITIES             ', end =" ")
    for n in range(data['N']):
        print(' {:8.2f}'.format(data['output']['EMU'][n]), end=" ")
    print()


def calculation(data):

    price_U = copy.deepcopy(data['p_urban_fixed'])
    price_R = copy.deepcopy(data['p_rural_fixed'])

    data['output'] = {}
    data['output']['U'] = np.zeros([data['I_tot'], data['N'], data['R']])
    data['output']['EMU'] = np.zeros([data['N']])
    data['output']['w'] = np.zeros([data['I_tot'], data['N'], data['R']])
    data['output']['P'] = np.zeros([data['I_tot'], data['N']])
    data['output']['profit'] = np.zeros([data['K']+1])
    data['output']['demand'] = np.zeros([data['I_tot']])
    data['output']['demand_urban'] = np.zeros([data['I_tot']])
    data['output']['demand_rural'] = np.zeros([data['I_tot']])
    data['output']['market_share'] = np.zeros([data['I_tot']])
    data['output']['market_share_urban'] = np.zeros([data['I_tot']])
    data['output']['market_share_rural'] = np.zeros([data['I_tot']])


    # Calculate utilities and choices
    for n in range(data['N']):
        for r in range(data['R']):
            for i in range(data['I_tot']):
                if data['ORIGIN'][n] == 1:
                    data['output']['U'][i,n,r] = data['endo_coef'][i,n] * price_U[i] +\
                                                 data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i,n,r]
                elif data['ORIGIN'][n] == 0:
                    data['output']['U'][i,n,r] = data['endo_coef'][i,n] * price_R[i] +\
                                                 data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i,n,r]
            i_UMax = np.argmax(data['output']['U'][:,n,r])
            data['output']['EMU'][n] += data['output']['U'][i_UMax, n, r]/data['R']
            data['output']['w'][i_UMax,n,r] = 1
            data['output']['P'][i_UMax, n] += 1/data['R']
    
    # Calculate demand
    for i in range(data['I_tot']):
        for n in range(data['N']):
            data['output']['demand'][i] += data['output']['P'][i,n] * data['popN'][n]
            if data['ORIGIN'][n] == 1:
                data['output']['demand_urban'][i] += data['output']['P'][i,n] * data['popN'][n]
            elif data['ORIGIN'][n] == 0:
                data['output']['demand_rural'][i] += data['output']['P'][i,n] * data['popN'][n]
    
    # Calculate market shares
    popU = int(sum([data['popN'][n] for n in range(data['N']) if data['ORIGIN'][n] == 1]))
    popR = int(sum([data['popN'][n] for n in range(data['N']) if data['ORIGIN'][n] == 0]))

    data['output']['market_share'] = data['output']['demand'] / data['Pop']
    data['output']['market_share_urban'] = data['output']['demand_urban'] / popU
    data['output']['market_share_rural'] = data['output']['demand_rural'] / popR
    
    # Calculate profits
    for k in range(data['K'] + 1):
        for i in data['list_alt_supplier'][k]:
            data['output']['profit'][k] += (data['output']['demand_urban'][i] * price_U[i] + data['output']['demand_rural'][i] * price_R[i])


def segments(data):

    # Define number of segments and names
    # Income (2) * Business (2) * Origin (2)
    data['nSegments'] = 12
    data['SEGMENT_NAME'] = ['Low income',
                            'High income',
                            'Non-business',
                            'Business',
                            'Rural',
                            'Urban',
                            'Non-business low income',
                            'Non-business rural',
                            'Low income rural',
                            'Business high income',
                            'Business urban',
                            'High income urban']
    

    # Initialize population in each segment
    data['SEGMENT'] = np.zeros((data['N'],data['nSegments']))
    data['popSegments'] = np.zeros((data['nSegments']))

    # Assign people to segments
    for n in range(data['N']):
        if data['INCOME'][n] == 0:
            data['SEGMENT'][n,0] = 1
        elif data['INCOME'][n] == 1:
            data['SEGMENT'][n,1] = 1

        if data['BUSINESS'][n] == 0:
            data['SEGMENT'][n,2] = 1
        elif data['BUSINESS'][n] == 1:
            data['SEGMENT'][n,3] = 1
        
        if data['ORIGIN'][n] == 0:
            data['SEGMENT'][n,4] = 1
        elif data['ORIGIN'][n] == 1:
            data['SEGMENT'][n,5] = 1
        
        if data['BUSINESS'][n] == 0 and data['INCOME'][n] == 0:
            data['SEGMENT'][n,6] = 1
        if data['BUSINESS'][n] == 0 and data['ORIGIN'][n] == 0:
            data['SEGMENT'][n,7] = 1
        if data['INCOME'][n] == 0 and data['ORIGIN'][n] == 0:
            data['SEGMENT'][n,8] = 1
        if data['BUSINESS'][n] == 1 and data['INCOME'][n] == 1:
            data['SEGMENT'][n,9] = 1
        if data['BUSINESS'][n] == 1 and data['ORIGIN'][n] == 1:
            data['SEGMENT'][n,10] = 1
        if data['INCOME'][n] == 1 and data['ORIGIN'][n] == 1:
            data['SEGMENT'][n,11] = 1

        for s in range(data['nSegments']):
            data['popSegments'][s] += data['SEGMENT'][n][s] * data['popN'][n]

    # Create list of people for each segment
    data['list_customer_segment'] = {}
    for s in range(data['nSegments']):
        data['list_customer_segment'][s] = [i for i, segment in enumerate(data['SEGMENT'][:,s]) if segment == s]

    # Calculate market shares for segments
    data['SEGMENT_DEMAND'] = np.zeros((data['nSegments'], data['I_tot']))
    for s in range(data['nSegments']):
        for n in range(data['N']):
            if data['SEGMENT'][n,s] == 1:
                for i in range(data['I_tot']):
                    data['SEGMENT_DEMAND'][s,i] += data['output']['P'][i,n] * data['popN'][n]

    # Print market shares for segments
    print('\nMARKET SHARES PER SEGMENT  ', end=' ')
    for i in range(data['I_tot']):
        print('  Alt_{:1d} '.format(i), end=' ')
    for s in range(data['nSegments']):
        print('\n{:25s}  '.format(data['SEGMENT_NAME'][s]), end=' ')
        for i in range(data['I_tot']):
                print('{:7.4f} '.format(data['SEGMENT_DEMAND'][s,i]/data['popSegments'][s]), end = ' ')


if __name__ == '__main__':

    # Read instance
    data = data_file.getData()
    # Precompute exogenous terms
    data_file.preprocessUtilities(data)

    data_file.printCustomers(data)

    if data['DCM'] == 'NestedLogit':
        #Calculate initial values of logsum terms
        nested_logit.logsumNestedLogit(data)

    calculation(data)
    segments(data)
    printOutputAnalysis(data)

