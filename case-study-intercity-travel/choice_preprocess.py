# Modification of the strategy sets of the competitors

# General
import numpy as np

# Project
import nested_logit
import update_bounds

# Data
import data_intercity as data_file


def choicePreprocess(data):

    ##########################################################
    # Pre-computation of customer choices (captive customers)
    ##########################################################
    # Initialize to -1, since 0 and 1 have other meanings
    data['w_pre'] = np.full((data['I_tot'], data['N'], data['R']), -1.0)
    countCaptive = 0

    # General choice preprocess
    for n in range(data['N']):
        for r in range(data['R']):

            # (1) Find the alternative with the greatest utility lower bound (value and index)
            LbMax = np.max(data['lb_U'][:, n, r])
            i_LbMax = np.argmax(data['lb_U'][:, n, r])

            # (2) Find the alternative with the greatest utility upper bound (excluding the alt found before)
            a = np.ma.masked_values(data['ub_U'][:, n, r], data['ub_U'][i_LbMax, n, r])
            sub_UbMax = np.max(a)

            # If (1) is greater than (2), then the customer is captive and the choice can be precomputed
            if LbMax > sub_UbMax:
                countCaptive += 1
                data['w_pre'][:,n,r] = 0.0
                data['w_pre'][i_LbMax,n,r] = 1.0

            # Else, exclude only alternatives for which ub < LbMax
            else:
                for i in range(data['I_tot']):
                    if LbMax > data['ub_U'][i, n, r]:
                        data['w_pre'][i,n,r] = 0.0

    #Captive choices per alternative
    captive = np.zeros((data['I_tot']))
    for i in range(data['I_tot']):
        captive[i] = np.count_nonzero(data['w_pre'][i,:,:] == 1)

    #Count preprocess
    countEliminated = 0
    for n in range(data['N']):
        for r in range(data['R']):
            countEliminated += len([x for x in data['w_pre'][:,n,r] if x == 0.0])

    print('\nCHOICE PREPROCESSING :')
    print('Eliminated alternatives: {:6d} of {:6d}'.format(countEliminated, data['I_tot']*data['N']*data['R']))
    print('Captive choices:         {:6d} of {:6d}'.format(countCaptive, data['N'] * data['R']))


def choicePreprocessStrategies(data):
    
    ##########################################################
    # Pre-computation of customer choices (captive customers)
    ##########################################################
    # Initialize to -1, since 0 and 1 have other meanings
    data['w_pre_strategies'] = np.full((data['I_tot'], data['N'], data['R'], data['tot_strategies']), -1.0)

    # General choice preprocess
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                if data['w_pre'][i, n, r] == 0 or data['w_pre'][i, n, r] == 1:
                    for l in range(data['tot_strategies']):
                        data['w_pre_strategies'][i,n,r,l] = data['w_pre'][i, n, r]

    # Choice preprocess within strategies
    if data['DCM'] == 'NestedLogit':
            
        for k in range(1, data['K'] + 1):
            for l in data['list_strategies_opt'][k]:
                for n in range(data['N']):
                    for r in range(data['R']):
                        if np.max(data['w_pre_strategies'][:, n, r, l]) < 0.5:
                            maxU = -1000
                            i_maxU = -1
                            for i in data['list_alt_supplier'][k]:
                                if data['ORIGIN'][n] == 1:
                                    U = data['endo_coef'][i, n] * data['strategies']['prices_urban'][i][l] +\
                                        data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r]
                                elif data['ORIGIN'][n] == 0:
                                    U = data['endo_coef'][i, n] * data['strategies']['prices_rural'][i][l] +\
                                        data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r]
                                if U > maxU:
                                    maxU = U
                                    i_maxU = i
                            for i in data['list_alt_supplier'][k]:
                                if i != i_maxU:
                                    data['w_pre_strategies'][i,n,r,l] = 0

    #Count preprocess
    countEliminated = 0
    for n in range(data['N']):
        for r in range(data['R']):
            for l in range(data['tot_strategies']):
                countEliminated += len([x for x in data['w_pre_strategies'][:,n,r,l] if x == 0.0])

    print('\nCHOICE PREPROCESSING FOR STRATEGIES :')
    print('Eliminated alternatives: {:6d} of {:6d}'.format(countEliminated, data['I_tot']*data['N']*data['R']*data['tot_strategies']))





if __name__ == '__main__':

    # Get the data and preprocess
    data = data_file.getData()
    
    #Precompute exogenous terms
    data_file.preprocessUtilities(data)
    
    if data['DCM'] == 'NestedLogit':
        #Calculate initial values of logsum terms
        nested_logit.logsumNestedLogitRegulator(data)
            
    #Calculate utility bounds
    update_bounds.updateUtilityBoundsWithRegulator(data)

    #Preprocess captive choices
    choicePreprocess(data)
