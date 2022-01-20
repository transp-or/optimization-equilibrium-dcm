# General
import numpy as np

# Data
import data_intercity as data_file

def logsumNestedLogitRegulator(data):
    
    data['LogitUtility'] = np.empty([data['I_tot'],data['N'],data['R']])
    data['Logsum'] = np.empty([data['I_tot'],data['N'],data['R']])

    #PART 1: LOGIT
    for i in range(data['I_tot']):
        for n in range(data['N']):
           for r in range(data['R']):
                if data['INCOME'][n] == 1 and data['ORIGIN'][n] == 1:      # High income urban
                    data['LogitUtility'][i,n,r] = data['endo_coef'][i,n] * (data['p_urban_fixed'][i] + data['fixed_taxsubsidy_highinc'][i]) + data['exo_utility'][i,n,r]
                elif data['INCOME'][n] == 1 and data['ORIGIN'][n] == 0:    # High income rural
                    data['LogitUtility'][i,n,r] = data['endo_coef'][i,n] * (data['p_rural_fixed'][i] + data['fixed_taxsubsidy_highinc'][i]) + data['exo_utility'][i,n,r]
                elif data['INCOME'][n] == 0 and data['ORIGIN'][n] == 1:    # Low income urban
                    data['LogitUtility'][i,n,r] = data['endo_coef'][i,n] * (data['p_urban_fixed'][i] + data['fixed_taxsubsidy_lowinc'][i]) + data['exo_utility'][i,n,r]
                elif data['INCOME'][n] == 0 and data['ORIGIN'][n] == 0:    # Low income rural
                    data['LogitUtility'][i,n,r] = data['endo_coef'][i,n] * (data['p_rural_fixed'][i] + data['fixed_taxsubsidy_lowinc'][i]) + data['exo_utility'][i,n,r]

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
                        data['Logsum'][i,n,r] = np.log(np.exp(data['LogitUtility'][i,n,r]*(mu_nest-1)) * np.power(sumNest, (1.0/mu_nest-1)))
                        if data['Logsum'][i,n,r] > 0.01:
                            print('\n\nLogsum[{:2d}][{:2d}] = {:7.4f}'.format(i, n, data['Logsum'][i,n,r]))
                            assert(data['Logsum'][i,n,r] <= 0.05)

if __name__ == '__main__':

    data = {}

    data_file.setAlgorithmParameters(data)

    data['Seed'] = 0

    # Get the data and preprocess
    data_file.getData(data)
    data_file.preprocessUtilities(data)
    
    logsumNestedLogitRegulator(data)