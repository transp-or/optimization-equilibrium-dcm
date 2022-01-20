# Code for the ex-post analysis (segmentation, market shares, etc.)

# General
import copy
import numpy as np

# Functions
import nested_logit

# Data
import data_intercity as data_file


def printOutputAnalysis(data):
    
    print('\n\n----------------------------------------------')
    print('------------------ SOLUTION ------------------')
    print('----------------------------------------------')

    # Print prices, subsidies, taxes, market shares for all alternatives
    print('\nALTERNATIVES:              ', end=" ")
    for i in range(data['I_tot']):
        print('   {:2d}    '.format(i), end=" ")
    print('\n\nPRICES URBAN              ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.2f}'.format(data['p_urban_fixed'][i]), end=" ")
    print('\nPRICES RURAL              ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.2f}'.format(data['p_rural_fixed'][i]), end=" ")
    print('\nTAX/SUBSIDY HIGH INC      ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.2f}'.format(data['fixed_taxsubsidy_highinc'][i]), end=" ")
    print('\nTAX/SUBSIDY LOW INC       ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.2f}'.format(data['fixed_taxsubsidy_lowinc'][i]), end=" ")
    print('\n\nTOTAL DEMAND              ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.2f}'.format(data['output']['demand'][i]), end=" ")
    print('\nDEMAND HIGH INC           ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.2f}'.format(data['output']['demand_high'][i]), end=" ")
    print('\nDEMAND LOW INC            ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.2f}'.format(data['output']['demand_low'][i]), end=" ")
    print('\nDEMAND URBAN              ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.2f}'.format(data['output']['demand_urban'][i]), end=" ")
    print('\nDEMAND RURAL              ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.2f}'.format(data['output']['demand_rural'][i]), end=" ")
    print('\nDEMAND BUSINESS           ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.2f}'.format(data['output']['demand_business'][i]), end=" ")
    print('\nDEMAND NON-BUSINESS       ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.2f}'.format(data['output']['demand_nonbusiness'][i]), end=" ")
    print('\n\nTOTAL MARKET SHARE        ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.4f}'.format(data['output']['market_share'][i]), end=" ")
    print('\nMARKET SHARE HIGH INC     ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.4f}'.format(data['output']['market_share_high'][i]), end=" ")
    print('\nMARKET SHARE LOW INC      ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.4f}'.format(data['output']['market_share_low'][i]), end=" ")
    print('\nMARKET SHARE URBAN        ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.4f}'.format(data['output']['market_share_urban'][i]), end=" ")
    print('\nMARKET SHARE RURAL        ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.4f}'.format(data['output']['market_share_rural'][i]), end=" ")
    print('\nMARKET SHARE BUSINESS     ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.4f}'.format(data['output']['market_share_business'][i]), end=" ")
    print('\nMARKET SHARE NON-BUSINESS ', end =" ")
    for i in range(data['I_tot']):
        print(' {:8.4f}'.format(data['output']['market_share_nonbusiness'][i]), end=" ")

    # Print profits for all suppliers
    print('\n\n\nSUPPLIERS:             ', end=" ")
    for k in range(data['K']+1):
        print('   {:2d}    '.format(k), end=" ")
    print('\n\nPROFITS               ', end =" ")
    for k in range(data['K']+1):
        print(' {:8.2f}'.format(data['output']['profit'][k]), end=" ")

    # Print utilities for all customers
    print('\n\n\nCUSTOMERS:             ', end=" ")
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

    # Print emissions
    print('\n\nTOTAL EMISSIONS:       {:8.2f} tons '.format(data['output']['emissions']/1000.0))
    print('COST OF EMISSIONS:    {:9.2f}  eur'.format(data['output']['cost_emissions']))


def calculation(data):

    price_urban = copy.deepcopy(data['p_urban_fixed'])
    price_rural = copy.deepcopy(data['p_rural_fixed'])
    taxsubsidy_high = copy.deepcopy(data['fixed_taxsubsidy_highinc'])
    taxsubsidy_low = copy.deepcopy(data['fixed_taxsubsidy_lowinc'])

    data['output'] = {}
    data['output']['U'] = np.zeros([data['I_tot'], data['N'], data['R']])
    data['output']['EMU'] = np.zeros([data['N']])
    data['output']['w'] = np.zeros([data['I_tot'], data['N'], data['R']])
    data['output']['P'] = np.zeros([data['I_tot'], data['N']])
    data['output']['profit'] = np.zeros([data['K']+1])
    data['output']['demand'] = np.zeros([data['I_tot']])
    data['output']['demand_high'] = np.zeros([data['I_tot']])
    data['output']['demand_low'] = np.zeros([data['I_tot']])
    data['output']['demand_urban'] = np.zeros([data['I_tot']])
    data['output']['demand_rural'] = np.zeros([data['I_tot']])
    data['output']['demand_business'] = np.zeros([data['I_tot']])
    data['output']['demand_nonbusiness'] = np.zeros([data['I_tot']])
    data['output']['market_share'] = np.zeros([data['I_tot']])
    data['output']['market_share_high'] = np.zeros([data['I_tot']])
    data['output']['market_share_low'] = np.zeros([data['I_tot']])
    data['output']['market_share_urban'] = np.zeros([data['I_tot']])
    data['output']['market_share_rural'] = np.zeros([data['I_tot']])
    data['output']['market_share_business'] = np.zeros([data['I_tot']])
    data['output']['market_share_nonbusiness'] = np.zeros([data['I_tot']])

    # Calculate utilities and choices
    for n in range(data['N']):
        for r in range(data['R']):
            for i in range(data['I_tot']):
                if data['INCOME'][n] == 1 and data['ORIGIN'][n] == 1:
                    data['output']['U'][i,n,r] = data['endo_coef'][i,n] * (price_urban[i] + taxsubsidy_high[i]) +\
                                                 data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i,n,r]
                elif data['INCOME'][n] == 1 and data['ORIGIN'][n] == 0:
                    data['output']['U'][i,n,r] = data['endo_coef'][i,n] * (price_rural[i] + taxsubsidy_high[i]) +\
                                                 data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i,n,r]
                elif data['INCOME'][n] == 0 and data['ORIGIN'][n] == 1:
                    data['output']['U'][i,n,r] = data['endo_coef'][i,n] * (price_urban[i] + taxsubsidy_low[i]) +\
                                                 data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i,n,r]
                elif data['INCOME'][n] == 0 and data['ORIGIN'][n] == 0:
                    data['output']['U'][i,n,r] = data['endo_coef'][i,n] * (price_rural[i] + taxsubsidy_low[i]) +\
                                                 data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i,n,r]
            i_UMax = np.argmax(data['output']['U'][:,n,r])
            data['output']['EMU'][n] += data['output']['U'][i_UMax, n, r]/data['R']
            data['output']['w'][i_UMax,n,r] = 1
            data['output']['P'][i_UMax, n] += 1/data['R']
    
    # Calculate demand
    for i in range(data['I_tot']):
        for n in range(data['N']):
            data['output']['demand'][i] += data['output']['P'][i,n] * data['popN'][n]
            if data['INCOME'][n] == 1:
                data['output']['demand_high'][i] += data['output']['P'][i,n] * data['popN'][n]
            elif data['INCOME'][n] == 0:
                data['output']['demand_low'][i] += data['output']['P'][i,n] * data['popN'][n]
            if data['ORIGIN'][n] == 1:
                data['output']['demand_urban'][i] += data['output']['P'][i,n] * data['popN'][n]
            elif data['ORIGIN'][n] == 0:
                data['output']['demand_rural'][i] += data['output']['P'][i,n] * data['popN'][n]    
            if data['BUSINESS'][n] == 1:
                data['output']['demand_business'][i] += data['output']['P'][i,n] * data['popN'][n]
            elif data['BUSINESS'][n] == 0:
                data['output']['demand_nonbusiness'][i] += data['output']['P'][i,n] * data['popN'][n] 

    # Calculate market shares
    popHI = int(sum([data['popN'][n] for n in range(data['N']) if data['INCOME'][n] == 1]))
    popLI = int(sum([data['popN'][n] for n in range(data['N']) if data['INCOME'][n] == 0]))
    popU = int(sum([data['popN'][n] for n in range(data['N']) if data['ORIGIN'][n] == 1]))
    popR = int(sum([data['popN'][n] for n in range(data['N']) if data['ORIGIN'][n] == 0]))
    popB = int(sum([data['popN'][n] for n in range(data['N']) if data['BUSINESS'][n] == 1]))
    popN = int(sum([data['popN'][n] for n in range(data['N']) if data['BUSINESS'][n] == 0]))

    data['output']['market_share'] = data['output']['demand'] / data['Pop']
    data['output']['market_share_high'] = data['output']['demand_high'] / popHI
    data['output']['market_share_low'] = data['output']['demand_low'] / popLI
    data['output']['market_share_urban'] = data['output']['demand_urban'] / popU
    data['output']['market_share_rural'] = data['output']['demand_rural'] / popR
    data['output']['market_share_business'] = data['output']['demand_business'] / popB
    data['output']['market_share_nonbusiness'] = data['output']['demand_nonbusiness'] / popN
    
    # Calculate profits
    for k in range(data['K'] + 1):
        for i in data['list_alt_supplier'][k]:
            data['output']['profit'][k] +=\
                (data['output']['demand_urban'][i] * price_urban[i] +
                data['output']['demand_rural'][i] * price_rural[i])

    # Calculate emissions and cost of emissions
    data['output']['emissions'] = 0
    for i in range(data['I_tot']):
        if data['alternatives'][i]['Mode'] == 'Plane':
            data['output']['emissions'] += data['Distance']*data['emissions_per_pass_km_air']*data['output']['demand'][i]
        elif data['alternatives'][i]['Mode'] == 'Car':
            data['output']['emissions'] += data['Distance']*data['emissions_per_pass_km_car']*data['output']['demand'][i]
        elif data['alternatives'][i]['Mode'] == 'Train':
            data['output']['emissions'] += data['Distance']*data['emissions_per_pass_km_train']*data['output']['demand'][i]
    data['output']['cost_emissions'] = data['output']['emissions'] * \
        data['social_cost_of_carbon']


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
    data['SEGMENT_MARKET_SHARE'] = np.zeros((data['nSegments'], data['I_tot']))
    for s in range(data['nSegments']):
        for n in range(data['N']):
            if data['SEGMENT'][n,s] == 1:
                for i in range(data['I_tot']):
                    data['SEGMENT_MARKET_SHARE'][s,i] += data['output']['P'][i,n] * data['popN'][n]

    # Print market shares for segments
    print('\nMARKET SHARES PER SEGMENT  ', end=' ')
    for i in range(data['I_tot']):
        print('  Alt_{:1d} '.format(i), end=' ')
    for s in range(data['nSegments']):
        print('\n{:25s}  '.format(data['SEGMENT_NAME'][s]), end=' ')
        for i in range(data['I_tot']):
                print('{:7.4f} '.format(data['SEGMENT_MARKET_SHARE'][s,i]/data['popSegments'][s]), end = ' ')


if __name__ == '__main__':
    
    data = {}

    # Define parameters of the algorithm
    data_file.setAlgorithmParameters(data)

    #Read instance
    data_file.getData(data)

    #Precompute exogenous terms
    data_file.preprocessUtilities(data)

    if data['DCM'] == 'NestedLogit':
        #Calculate initial values of logsum terms
        nested_logit.logsumNestedLogitRegulator(data)

    data_file.printCustomers(data)

    calculation(data)
    segments(data)
    printOutputAnalysis(data)
