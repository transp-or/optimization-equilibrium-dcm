# Calibration of the alternative specific constants
# of a discrete choice model according to target market shares

# General
import time
import numpy as np
from scipy.optimize import fsolve

# Data
import data_HSR as data_file


def calculate(x):
    
    data['iter'] += 1

    #################################
    # CALCULATE UTILITIES
    #################################
    # The calibration is done separately for business and non-business trips
    # Assign group = 1 for business, group = 0 for non-business
    
    # Initialization
    logitUtility = np.empty([data['I_tot'], data['N'], data['R']])
    data['nestedLogitUtility'] = np.empty([data['I_tot'], data['N'], data['R']])
 
    for n in range(data['N']):
        
        if data['BUSINESS'][n] == data['calibration_group']:

            ###################################################
            # PART 1: LOGIT PART OF THE NESTED LOGIT UTILITY
            ###################################################
            for r in range(data['R']):
                for i in range(data['I_tot']):
        
                    if data['alternatives'][i]['Mode'] == 'Car':
                        logitUtility[i,n,r] = x[0] +\
                                        data['exo_utility'][i,n,r] +\
                                        data['endo_coef'][i,n] * data['price'][i]
                    elif data['alternatives'][i]['Mode'] == 'Plane':
                        logitUtility[i,n,r] = x[1] +\
                                        data['exo_utility'][i,n,r] +\
                                        data['endo_coef'][i,n] * data['price'][i]
                    elif data['alternatives'][i]['Operator'] == 'IC':
                        logitUtility[i,n,r] = x[2] +\
                                        data['exo_utility'][i,n,r] +\
                                        data['endo_coef'][i,n] *  data['price'][i]
                    elif data['alternatives'][i]['Operator'] == 'HSR_Supplier1':
                        logitUtility[i,n,r] = x[3] +\
                                        data['exo_utility'][i,n,r] +\
                                        data['endo_coef'][i,n] *  data['price'][i]
                    elif data['alternatives'][i]['Operator'] == 'HSR_Supplier2':
                        logitUtility[i,n,r] = x[4] +\
                                        data['exo_utility'][i,n,r] +\
                                        data['endo_coef'][i,n] *  data['price'][i]

            ###################################################
            # PART 2: LOGSUM TERM OF THE NESTED LOGIT UTILITY
            ###################################################
    
            for nest in range(data['Nests']):
                mu_nest = data['MU'][nest][data['calibration_group']]

                for r in range(data['R']):

                    #Calculate logsum term for the nest
                    sumNest = 0
                    
                    for j in data['list_alt_nest'][nest]:
                        sumNest += np.exp(logitUtility[j,n,r] * mu_nest)
        
                    for i in data['list_alt_nest'][nest]:
                        logsum = np.log( np.exp (logitUtility[i,n,r] * (mu_nest-1)) * np.power(sumNest, (1.0/mu_nest-1)) )
        
                        #Deterministic part of the utility function
                        data['nestedLogitUtility'][i,n,r] = logitUtility[i,n,r] + logsum

    #################################
    # CALCULATE PROBABILITIES
    #################################
    
    data['ProbNested'] = np.zeros([data['I_tot'], data['N']])
    
    #CALCULATE CHOICE PROBABILITIES
    for n in range(data['N']):

        if data['BUSINESS'][n] == data['calibration_group']:

            for r in range(data['R']):
            
                # The denominator is the sum over all alternatives of the exponentials
                DenomNested = 0
                for i in range(data['I_tot']):
                    DenomNested += np.exp(data['nestedLogitUtility'][i,n,r])
                
                # The choice probability formula is the same as in the logit
                for i in range(data['I_tot']):
                    data['ProbNested'][i,n] += 1/data['R'] * (np.exp(data['nestedLogitUtility'][i,n,r]) / DenomNested)

    #################################
    # CALCULATE CHOICES
    #################################
    
    # Sum of probabilistic choices
    data['choiceCount'] = np.zeros([data['I_tot']])
    # Counter of customers
    nCustomers = 0
    
    for n in range(data['N']):
        if data['BUSINESS'][n] == data['calibration_group']:
            nCustomers += data['popN'][n]
            data['choiceCount'][:] += data['ProbNested'][:,n] * data['popN'][n]
    
    # Aggregate market shares by market segments:
    # (1) Car, (2) Air, (3) Trenitalia IC, (4) Trenitalia HSR, (5) NTV HSR
    data['market_share'] = np.zeros([5])
    for i in range(data['I_tot']):
        if data['alternatives'][i]['Mode'] == 'Car':
            data['market_share'][0] += data['choiceCount'][i]
        elif data['alternatives'][i]['Operator'] == 'Airline':
            data['market_share'][1] += data['choiceCount'][i]
        elif data['alternatives'][i]['Operator'] == 'IC':
            data['market_share'][2] += data['choiceCount'][i]
        elif data['alternatives'][i]['Operator'] == 'HSR_Supplier1':
            data['market_share'][3] += data['choiceCount'][i]
        elif data['alternatives'][i]['Operator'] == 'HSR_Supplier2':
            data['market_share'][4] += data['choiceCount'][i]
    
    # Difference between observed and desired market shares
    # Desired market shares should reflect current estimates 
    gap = (data['market_share']/nCustomers) - data['target_market_share']
    
    # Print results
    print('\nIter {:3d}\nNest  Market share   Target'.format(data['iter']))
    for m in range(5):
        print('  {:2d}        {:6.4f}   {:6.4f}'.format(m, data['market_share'][m]/nCustomers, data['target_market_share'][m]))
    
    return gap
        


if __name__ == '__main__':

    t_0 = time.time()
    
    # Read instance
    data = data_file.getData()
    data_file.preprocessUtilities(data)

    # The calibration is done separately for business and non-business trips
    # Assign group = 1 for business, group = 0 for non-business
    data['calibration_group'] = 0
    
    # Market segments: (1) Car, (2) Air, (3) IC, (4) HSR1, (5) HSR2
    segments = 5
    
    # Initial values of the ASC
    ASC_initial = [data['ASC_CAR'][data['calibration_group']],
                   data['ASC_PLANE'][data['calibration_group']],
                   data['ASC_IC'][data['calibration_group']],
                   data['ASC_AV'][data['calibration_group']],
                   data['ASC_NTV'][data['calibration_group']]]
    
    # Desired market shares (= current observed market split)       #Modify here for testing
    if data['calibration_group'] == 1:
        data['target_market_share'] = np.array([0.025, 0.170, 0.005, 0.500, 0.300]) 
    elif data['calibration_group'] == 0:
        data['target_market_share'] = np.array([0.050, 0.300, 0.050, 0.350, 0.250]) 
    
    print('\n ASC_initial %r \n' %ASC_initial)
    
    # The starting estimate for the roots of func(x) = 0.
    x0 = np.zeros(segments)
    data['iter'] = 0

    # Find the roots of the non-linear system of equations    
    x = fsolve(calculate, x0, xtol = 1e-5)
    print('\nSolution of the system of equations:\n%r' %x)
    
    # Fix ASC_CAR to be equal to 0
    x = x - x[0]
    print('\nSolution of the system of equations after fixing ASC_CAR to 0:\n%r' % x)

    # Final ASC
    ASC_initial = ASC_initial + x
    print('\nFinal values of the ASC:\n%r' % ASC_initial)

    print('\nTime : {:7.3f}'.format(time.time()-t_0))
