# Calculate bounds on the utilities of the customers

# General
import numpy as np

# Project
import nested_logit

# Data
import data_intercity as data_file


def updateUtilityBoundsWithRegulator(data):

    # Utility bounds
    lb_U = np.empty([data['I_tot'], data['N'], data['R']])
    ub_U = np.empty([data['I_tot'], data['N'], data['R']])
    lb_Umin = np.full((data['N'], data['R']), np.inf)
    ub_Umax = np.full((data['N'], data['R']), -np.inf)

    M = np.empty([data['N'], data['R']])

    for n in range(data['N']):
        for r in range(data['R']):
            for i in range(data['I_tot']):
                if data['DCM'] == 'NestedLogit':
                    if data['endo_coef'][i, n] <= 0: #This is always the case when prices are the only endogenous variables
                        if data['INCOME'][n] == 1:      #High income
                            if data['ORIGIN'][n] == 1: 
                                lb_U[i, n, r] = (data['endo_coef'][i, n] * (data['ub_p_urban'][i] + data['ub_tax_highincome'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                                ub_U[i, n, r] = (data['endo_coef'][i, n] * (data['lb_p_urban'][i] - data['ub_subsidy_highincome'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                            elif data['ORIGIN'][n] == 0:
                                lb_U[i, n, r] = (data['endo_coef'][i, n] * (data['ub_p_rural'][i] + data['ub_tax_highincome'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                                ub_U[i, n, r] = (data['endo_coef'][i, n] * (data['lb_p_rural'][i] - data['ub_subsidy_highincome'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                        elif data['INCOME'][n] == 0:    #Low income
                            if data['ORIGIN'][n] == 1:
                                lb_U[i, n, r] = (data['endo_coef'][i, n] * (data['ub_p_urban'][i] + data['ub_tax_lowincome'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                                ub_U[i, n, r] = (data['endo_coef'][i, n] * (data['lb_p_urban'][i] - data['ub_subsidy_lowincome'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                            elif data['ORIGIN'][n] == 0:
                                lb_U[i, n, r] = (data['endo_coef'][i, n] * (data['ub_p_rural'][i] + data['ub_tax_lowincome'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                                ub_U[i, n, r] = (data['endo_coef'][i, n] * (data['lb_p_rural'][i] - data['ub_subsidy_lowincome'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
            # Bounds for each customer, for each draw
            lb_Umin[n, r] = np.min(lb_U[:, n, r])
            ub_Umax[n, r] = np.max(ub_U[:, n, r])

            # Calcule the big-M values
            M[n, r] = ub_Umax[n, r] - lb_Umin[n, r]

    data['lb_U'] = lb_U
    data['ub_U'] = ub_U
    data['lb_Umin'] = lb_Umin
    data['ub_Umax'] = ub_Umax
    data['M_U'] = M


def updateUtilityBoundsFixedPrice(data):
    
    # Utility bounds
    lb_U = np.empty([data['I_tot'], data['N'], data['R']])
    ub_U = np.empty([data['I_tot'], data['N'], data['R']])
    lb_Umin = np.full((data['N'], data['R']), np.inf)
    ub_Umax = np.full((data['N'], data['R']), -np.inf)

    M = np.empty([data['N'], data['R']])

    for n in range(data['N']):
        for r in range(data['R']):
            for i in range(data['I_tot']):
                if data['DCM'] == 'NestedLogit':
                    if data['endo_coef'][i, n] <= 0: #This is always the case when prices are the only endogenous variables
                        if data['INCOME'][n] == 1:      #High income
                            if data['ORIGIN'][n] == 1:
                                lb_U[i, n, r] = (data['endo_coef'][i, n] * (data['p_urban_fixed'][i] + data['ub_tax_highincome'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                                ub_U[i, n, r] = (data['endo_coef'][i, n] * (data['p_urban_fixed'][i] - data['ub_subsidy_highincome'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                            elif data['ORIGIN'][n] == 0:
                                lb_U[i, n, r] = (data['endo_coef'][i, n] * (data['p_rural_fixed'][i] + data['ub_tax_highincome'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                                ub_U[i, n, r] = (data['endo_coef'][i, n] * (data['p_rural_fixed'][i] - data['ub_subsidy_highincome'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                        elif data['INCOME'][n] == 0:    #Low income
                            if data['ORIGIN'][n] == 1:
                                lb_U[i, n, r] = (data['endo_coef'][i, n] * (data['p_urban_fixed'][i] + data['ub_tax_lowincome'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                                ub_U[i, n, r] = (data['endo_coef'][i, n] * (data['p_urban_fixed'][i] - data['ub_subsidy_lowincome'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                            elif data['ORIGIN'][n] == 0:
                                lb_U[i, n, r] = (data['endo_coef'][i, n] * (data['p_rural_fixed'][i] + data['ub_tax_lowincome'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                                ub_U[i, n, r] = (data['endo_coef'][i, n] * (data['p_rural_fixed'][i] - data['ub_subsidy_lowincome'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])

            # Bounds for each customer, for each draw
            lb_Umin[n, r] = np.min(lb_U[:, n, r])
            ub_Umax[n, r] = np.max(ub_U[:, n, r])

            # Calcule the big-M values
            M[n, r] = ub_Umax[n, r] - lb_Umin[n, r]

    data['lb_U'] = lb_U
    data['ub_U'] = ub_U
    data['lb_Umin'] = lb_Umin
    data['ub_Umax'] = ub_Umax
    data['M_U'] = M


def updateUtilityBoundsFixedRegulator(data):
    
    # Utility bounds
    lb_U = np.empty([data['I_tot'], data['N'], data['R']])
    ub_U = np.empty([data['I_tot'], data['N'], data['R']])
    lb_Umin = np.full((data['N'], data['R']), np.inf)
    ub_Umax = np.full((data['N'], data['R']), -np.inf)

    M = np.empty([data['N'], data['R']])

    for n in range(data['N']):
        for r in range(data['R']):
            for i in range(data['I_tot']):
                if data['DCM'] == 'NestedLogit':
                    if data['endo_coef'][i, n] <= 0: #This is always the case when prices are the only endogenous variables
                        if data['INCOME'][n] == 1:  # High income
                            if data['ORIGIN'][n] == 1:
                                lb_U[i, n, r] = (data['endo_coef'][i, n] * (data['ub_p_urban'][i] + data['fixed_taxsubsidy_highinc'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                                ub_U[i, n, r] = (data['endo_coef'][i, n] * (data['lb_p_urban'][i] + data['fixed_taxsubsidy_highinc'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                            elif data['ORIGIN'][n] == 0:
                                lb_U[i, n, r] = (data['endo_coef'][i, n] * (data['ub_p_rural'][i] + data['fixed_taxsubsidy_highinc'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                                ub_U[i, n, r] = (data['endo_coef'][i, n] * (data['lb_p_rural'][i] + data['fixed_taxsubsidy_highinc'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                        elif data['INCOME'][n] == 0:    #Low income
                            if data['ORIGIN'][n] == 1:
                                lb_U[i, n, r] = (data['endo_coef'][i, n] * (data['ub_p_urban'][i] + data['fixed_taxsubsidy_lowinc'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                                ub_U[i, n, r] = (data['endo_coef'][i, n] * (data['lb_p_urban'][i] + data['fixed_taxsubsidy_lowinc'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                            elif data['ORIGIN'][n] == 0:
                                lb_U[i, n, r] = (data['endo_coef'][i, n] * (data['ub_p_rural'][i] + data['fixed_taxsubsidy_lowinc'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                                ub_U[i, n, r] = (data['endo_coef'][i, n] * (data['lb_p_rural'][i] + data['fixed_taxsubsidy_lowinc'][i]) +
                                                data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])

            # Bounds for each customer, for each draw
            lb_Umin[n, r] = np.min(lb_U[:, n, r])
            ub_Umax[n, r] = np.max(ub_U[:, n, r])

            # Calcule the big-M values
            M[n, r] = ub_Umax[n, r] - lb_Umin[n, r]

    data['lb_U'] = lb_U
    data['ub_U'] = ub_U
    data['lb_Umin'] = lb_Umin
    data['ub_Umax'] = ub_Umax
    data['M_U'] = M


if __name__ == '__main__':

    data = {}

    data_file.setAlgorithmParameters(data)

    data['Seed'] = 0

    data_file.getData(data)
    
    #Precompute exogenous terms
    data_file.preprocessUtilities(data)
    
    #Calculate initial values of logsum terms
    nested_logit.logsumNestedLogitRegulator(data)
            
    #Calculate utility bounds
    updateUtilityBoundsFixedRegulator(data)
