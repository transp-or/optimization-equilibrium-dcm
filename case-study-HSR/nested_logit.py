# General
import time
import numpy as np
import matplotlib.pyplot as plt

# Data
import data_HSR as data_file


def logsumNestedLogit(data):

    data['LogitUtility'] = np.empty([data['I_tot'],data['N'],data['R']])
    data['Logsum'] = np.empty([data['I_tot'],data['N'],data['R']])

    #PART 1: LOGIT
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                if data['ORIGIN'][n] == 1:
                    data['LogitUtility'][i,n,r] = data['endo_coef'][i,n] * data['p_urban_fixed'][i] + data['exo_utility'][i,n,r]
                elif data['ORIGIN'][n] == 0:
                    data['LogitUtility'][i,n,r] = data['endo_coef'][i,n] * data['p_rural_fixed'][i] + data['exo_utility'][i,n,r]


    #PART 2: NESTED LOGIT
    for n in range(data['N']):
        #The model estimation is different for business and non-business customers
        # Assign group = 1 for business, group = 0 for non-business
        group = int(data['BUSINESS'][n])

        for nest in range(data['Nests']):
            mu_nest = data['MU'][nest][group]

            #If the nest has only one alternative or mu_nest = 1, then logsum = 0
            if len(data['list_alt_nest'][nest]) == 1 or mu_nest <= 1.01:
                for r in range(data['R']):
                    for i in data['list_alt_nest'][nest]:
                        data['Logsum'][i,n,r] = 0.0
            #Else, calculate logsum term for the nest
            else:
                for r in range(data['R']):
                    sumNest = 0.0
                    for j in data['list_alt_nest'][nest]:
                        sumNest += np.exp(data['LogitUtility'][j,n,r] * mu_nest)
        
                    for i in data['list_alt_nest'][nest]:
                        data['Logsum'][i,n,r] = np.log( np.exp( data['LogitUtility'][i,n,r]*(mu_nest-1) ) * np.power( sumNest, (1.0/mu_nest-1) ) )
                        if data['Logsum'][i,n,r] > 0.01:
                            print('\n\nLogsum[{:2d}][{:2d}] = {:7.4f}'.format(i, n, data['Logsum'][i,n,r]))
                            assert(data['Logsum'][i,n,r] <= 0.05)


def calculateNestedLogitUtilities(data):
    '''
    This function is used to look at the effect of nesting on the utilities.
    The logsum term determines how much the utility is affected
    by the similarity with other alternatives (substitution patterns) 
    '''
    data['LogitUtility'] = np.empty([data['I_tot'],data['N'],data['R']])
    data['Simulated_LogitUtility'] = np.zeros([data['I_tot'], data['N'], data['R']])
    data['Probabilistic_NestedUtility'] = np.empty([data['I_tot'],data['N'],data['R']])
    data['Simulated_NestedUtility'] = np.empty([data['I_tot'],data['N'],data['R']])

    #PART 1: LOGIT
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                #Deterministic part of the utility function
                if data['ORIGIN'][n] == 1:
                    data['LogitUtility'][i,n,r] = data['endo_coef'][i,n] * data['p_urban_fixed'][i] + data['exo_utility'][i,n,r]
                elif data['ORIGIN'][n] == 0:
                    data['LogitUtility'][i,n,r] = data['endo_coef'][i,n] * data['p_rural_fixed'][i] + data['exo_utility'][i,n,r]
                #Simulated utility with error term
                data['Simulated_LogitUtility'][i,n,r] = data['LogitUtility'][i,n,r] + data['xi'][i,n,r]

    #PART 2: NESTED LOGIT
    for n in range(data['N']):
        group = int(data['BUSINESS'][n])
        for nest in range(data['Nests']):
            mu_nest = data['MU'][nest][group]

            for r in range(data['R']):

                #Calculate logsum term for the nest
                sumNest = 0.0
                for j in data['list_alt_nest'][nest]:
                    sumNest += np.exp(data['LogitUtility'][j,n,r]*mu_nest)

                for i in data['list_alt_nest'][nest]:
                    logsum = np.log(np.exp(data['LogitUtility'][i,n,r]*(mu_nest-1)) * np.power(sumNest, (1.0/mu_nest-1)))

                    #Deterministic part of the utility function
                    data['Probabilistic_NestedUtility'][i,n,r] = data['LogitUtility'][i,n,r] + logsum
                    #Simulated utility with error term
                    data['Simulated_NestedUtility'][i,n,r] = data['Probabilistic_NestedUtility'][i,n,r] + data['xi'][i,n,r]


def calculateNestedLogitProbabilities(data):

    # INITIALIZE ALTERNATIVE PROBABILITIES
    data['Simulated_ProbLogit'] = np.zeros([data['I_tot'], data['N']])
    data['Simulated_ProbNested'] = np.zeros([data['I_tot'], data['N']])
    data['Probabilistic_ProbLogit'] = np.zeros([data['I_tot'], data['N']])
    data['Probabilistic_ProbNested'] = np.zeros([data['I_tot'], data['N']])
    # INITIALIZE NEST PROBABILITIES
    data['Simulated_NestProbLogit'] = np.zeros([data['Nests'], data['N']])
    data['Simulated_NestProbNested'] = np.zeros([data['Nests'], data['N']])
    data['Probabilistic_NestProbLogit'] = np.zeros([data['Nests'], data['N']])
    data['Probabilistic_NestProbNested'] = np.zeros([data['Nests'], data['N']])

    #CALCULATE CHOICE PROBABILITIES
    for n in range(data['N']):
        for r in range(data['R']):
            DenomLogit = 0
            DenomNested = 0
            for i in range(data['I_tot']):
                DenomLogit += np.exp(data['LogitUtility'][i,n,r])
                DenomNested += np.exp(data['Probabilistic_NestedUtility'][i,n,r])
            for i in range(data['I_tot']):
                data['Probabilistic_ProbLogit'][i,n] += 1/data['R'] * (np.exp(data['LogitUtility'][i,n,r]) / DenomLogit)
                data['Probabilistic_ProbNested'][i,n] += 1/data['R'] * (np.exp(data['Probabilistic_NestedUtility'][i,n,r]) / DenomNested)

    #CALCULATE SIMULATED CHOICE PROBABILITIES (ALT WITH MAX UTILITY = 1)
    for n in range(data['N']):
        for r in range(data['R']):
            i_choiceLogit = np.argmax(data['Simulated_LogitUtility'][:, n, r])
            data['Simulated_ProbLogit'][i_choiceLogit][n] += 1/data['R']
            i_choiceNested = np.argmax(data['Simulated_NestedUtility'][:, n, r])
            data['Simulated_ProbNested'][i_choiceNested][n] += 1/data['R']

    #CALCULATE NEST PROBABILITIES
    for i in range(data['I_tot']):
        data['Simulated_NestProbLogit'][data['nest'][i],:] += data['Simulated_ProbLogit'][i,:]
        data['Simulated_NestProbNested'][data['nest'][i],:] += data['Simulated_ProbNested'][i,:]
        data['Probabilistic_NestProbLogit'][data['nest'][i],:] += data['Probabilistic_ProbLogit'][i,:]
        data['Probabilistic_NestProbNested'][data['nest'][i],:] += data['Probabilistic_ProbNested'][i,:]

    #CALCULATE TOTAL MARKET SHARES
    data['Simulated_DemandLogit'] = np.sum(data['Simulated_ProbLogit'],axis=1)
    data['Simulated_DemandNested'] = np.sum(data['Simulated_ProbNested'],axis=1)
    data['Simulated_NestDemandLogit'] = np.sum(data['Simulated_NestProbLogit'],axis=1)
    data['Simulated_NestDemandNested'] = np.sum(data['Simulated_NestProbNested'],axis=1)
    data['Probabilistic_DemandLogit'] = np.sum(data['Probabilistic_ProbLogit'],axis=1)
    data['Probabilistic_DemandNested'] = np.sum(data['Probabilistic_ProbNested'],axis=1)
    data['Probabilistic_NestDemandLogit'] = np.sum(data['Probabilistic_NestProbLogit'],axis=1)
    data['Probabilistic_NestDemandNested'] = np.sum(data['Probabilistic_NestProbNested'],axis=1)

    #PRINT RESULTS
    print('\n\nCHOICE PROBABILITIES     (N ={:5d})'.format(data['N']))
    print('               Simulation (R={:5d})     Closed-form probabilities'.format(data['R']))
    print('Alt  Nest        Logit      Nested              Logit      Nested')
    for i in range(data['I_tot']):
        print('{:2d}   {:2d}       {:8.5f}    {:8.5f}           {:8.5f}    {:8.5f}'\
              .format(i, data['nest'][i], data['Simulated_DemandLogit'][i]/data['N'],\
                      data['Simulated_DemandNested'][i]/data['N'],\
                      data['Probabilistic_DemandLogit'][i]/data['N'],\
                      data['Probabilistic_DemandNested'][i]/data['N']))


if __name__ == '__main__':

    t_0 = time.time()

    # Get the data and preprocess
    data = data_file.getData()
    data_file.preprocessUtilities(data)
    
    calculateNestedLogitUtilities(data)
    calculateNestedLogitProbabilities(data)

    print('\nTime : {:7.3f}'.format(time.time()-t_0))
