# Rank fixed utilities in facility location problem 

# General
import time
import copy
import numpy as np
import math

import update_bounds

# CPLEX
import cplex

# Data
#import data_N80_I14 as data_file
import data_N08_I10 as data_file


def discretePriceAlternativeDuplication(data):

    # Define number of "expanded" alternatives: 1 per price level
    index = 0
    index_opt_out = 0
    index_endogenous = 0
    for i in range(data['I_tot']):
        for p in data['discr_price'][i]:
            if i < data['I_opt_out']:
                index_opt_out += 1
            else:
                index_endogenous += 1
            index += 1
    data['I_tot_exp'] = index
    data['I_end_exp'] = index_endogenous
    data['I_out_exp'] = index_opt_out

    data['alt'] = np.empty(data['I_tot_exp'], dtype=int)
    data['p'] = np.empty(data['I_tot_exp'])
    data['y'] = np.empty(data['I_tot_exp'])
    data['list_open'] = np.empty(data['I_tot_exp'])
    
    # Map original alternatives to "expanded" alternatives
    index = 0
    for i in range(data['I_tot']):
        for p in data['discr_price'][i]:
            data['alt'][index] = i
            data['p'][index] = p
            # Assign only opt-out alternatives to be open
            if i < data['I_opt_out']:
                data['y'][index] = 1.0
                data['list_open'][index] = 1.0
            else:
                data['y'][index] = 0.0
                data['list_open'][index] = 0.0
            index += 1

def nonNegativeUtilities(data):
    
    # For each n and r, translate all utilities so that the smallest utility is equal to 1 
    for n in range(data['N']):
        for r in range(data['R']):
            minU = np.min(data['endo_coef'][:,n,r]*data['ub_p'][:]+data['exo_utility'][:,n,r]+data['xi'][:,n,r])
            if minU < 0:
                for i in range(data['I_tot']):
                    data['exo_utility'][i,n,r] = data['exo_utility'][i,n,r] - minU + 1.0

def calcUtilitiesContinuous(data):

    data['U'] = np.empty([data['I_tot'], data['N'], data['R']])

    # Calculate all utilities
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                data['U'][i,n,r] = data['endo_coef'][i,n] * data['p'][i] + data['exo_utility'][i,n,r] + data['xi'][i,n,r]

def calcUtilities(data):

    data['U'] = np.empty([data['I_tot'], data['N'], data['R']])

    # Calculate all utilities
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                data['U'][i,n,r] = data['endo_coef'][i,n] * data['price'][i] + data['exo_utility'][i,n,r] + data['xi'][i,n,r]
    
    # For each n and r, translate all utilities so that the smallest utility is equal to 1 
    for n in range(data['N']):
        for r in range(data['R']):
            minU = np.min(data['U'][:,n,r])
            if minU < 0:
                for i in range(data['I_tot']):
                    data['U'][i,n,r] = data['U'][i,n,r] - minU + 1
    
def calcDuplicatedUtilities(data):

    data['U'] = np.empty([data['I_tot_exp'], data['N'], data['R']])

    # Calculate all utilities
    for i in range(data['I_tot_exp']):
        for n in range(data['N']):
            for r in range(data['R']):
                data['U'][i,n,r] = data['endo_coef'][data['alt'][i],n,r] * data['p'][i] +\
                                   data['exo_utility'][data['alt'][i],n,r] + data['xi'][data['alt'][i],n,r]
                data['U'][i,n,r] = round(data['U'][i,n,r],data['round'])
    
    # For each n and r, translate all utilities in a way that the smallest utility is equal to 1 
    for n in range(data['N']):
        for r in range(data['R']):
            minU = np.min(data['U'][:,n,r])
            if minU < 0:
                for i in range(data['I_tot_exp']):
                    data['U'][i,n,r] = data['U'][i,n,r] - minU + 1.0

def calcUtilitiesOne(data, i):

    for n in range(data['N']):
        for r in range(data['R']):
            data['U'][i,n,r] = data['endo_coef'][i,n] * data['price'][i] + data['exo_utility'][i,n,r] + data['xi'][i,n,r]

def rankUtilitiesAll(data):
    '''
    Pairwise comparison of all utilities
    '''
    t_rankUtil_start = time.time()

    data['a'] = np.empty([data['I_tot'], data['I_tot'], data['N'], data['R']])

    for n in range(data['N']):
        for r in range(data['R']):
            for i in range(data['I_tot']):
                for j in range(i, data['I_tot']):
                    if i == j:
                        data['a'][i,i,n,r] = 0
                    if data['U'][i,n,r] < data['U'][j,n,r]:
                        data['a'][i,j,n,r] = 0
                        data['a'][j,i,n,r] = 1
                    else:
                        data['a'][i,j,n,r] = 1
                        data['a'][j,i,n,r] = 0

    t_rankUtil_end = time.time()
    print('Time to run rankUtilitiesAll    : {:10.4f}'.format(t_rankUtil_end-t_rankUtil_start))

def rankUtilitiesOne(data, i):
    '''
    Pairwise comparison of one utility with all others
    '''    

    for n in range(data['N']):
        for r in range(data['R']):
            for j in range(data['I_tot']):
                if i != j:
                    if data['U'][i,n,r] < data['U'][j,n,r]:
                        data['a'][i,j,n,r] = 0
                        data['a'][j,i,n,r] = 1
                    else:
                        data['a'][i,j,n,r] = 1
                        data['a'][j,i,n,r] = 0

def rankAlternativesAll(data):
    '''
    Sorting all alternatives by utilities
    '''
    t_rankAlt_start = time.time()

    data['rankAlt'] = []
    for n in range(data['N']):
        data['rankAlt'].append([])
        for r in range(data['R']):
            data['rankAlt'][n].append([b[0] for b in sorted(enumerate(data['U'][:,n,r]),key=lambda i:i[1])])
            data['rankAlt'][n][r].reverse()

    t_rankAlt_end = time.time()
    print('Time to run rankAlternativesAll : {:10.4f}'.format(t_rankAlt_end-t_rankAlt_start))

def calculateMarkup(data):

    data['markup'] = np.empty([data['I_tot_exp']])
    # Markup = selling price - production cost
    for i in range(data['I_tot_exp']):
        data['markup'][i] = data['p'][i] - data['customer_cost'][data['alt'][i]]

def calculateProb(data):

    availableAlt = [i for i in range(data['I_tot']) if data['y'][i] == 1]
    choice = np.zeros((data['N'], data['R']), dtype=int)

    # Derive choices
    for n in range(data['N']):
        for r in range(data['R']):
            for i in availableAlt:
                sum = 0
                for j in availableAlt:
                    sum = sum + data['a'][i,j,n,r]
                if sum == len(availableAlt) - 1:
                    choice[n,r] = i
                    break

    # Compute choice probabilities
    data['P'] = np.zeros((data['I_tot'], data['N']))
    for n in range(data['N']):
        for i in availableAlt:
            data['P'][i,n] = (choice[n,:] == i).sum() / data['R']

def calculateProbDirect(data):

    data['P'] = np.zeros((data['I_tot'], data['N']))

    # Derive choices
    for n in range(data['N']):
        for r in range(data['R']):
            #Find maximum U[:,n,r]
            i = np.argmax(data['U'][:,n,r] - data['M_nr'][n,r] * (1 - data['y']))
            data['P'][i,n] = data['P'][i,n] + 1.0 / data['R']

def objSolution(data, y):

    # RETRIEVE MAX UTILITIES OF CUSTOMERS
    choice = np.zeros([data['N'], data['R']], dtype=int)
    #x = np.zeros([data['I_tot_exp'], data['N'], data['R']])
    for n in range(data['N']):
        for r in range(data['R']):
            choice[n,r] = np.argmax(data['U'][:,n,r] * y[:])
            #x[choice[n,r],n,r] = 1.0
            #markup_i = data['p'][choice[n,r]] - data['customer_cost'][data['alt'][choice[n,r]]]

    # RETRIEVE DEMAND
    demand = np.zeros([data['I_tot_exp']])
    for n in range(data['N']):
        for r in range(data['R']):
            demand[choice[n,r]] += (data['popN'][n]/data['R'])
    
    # RETRIEVE OBJECTIVE FUNCTION VALUE
    obj = 0.0
    for i in range(data['I_tot_exp']):
        # Fixed costs
        obj = obj - data['fixed_cost'][data['alt'][i]] * y[i]
        # Profits
        obj = obj + (data['p'][i] - data['customer_cost'][data['alt'][i]]) * demand[i]

    return obj

def objSolutionPrice(data, p):

    # RETRIEVE MAX UTILITIES OF CUSTOMERS
    choice = np.zeros([data['N'], data['R']], dtype=int)
    #x = np.zeros([data['I_tot_exp'], data['N'], data['R']])
    for n in range(data['N']):
        for r in range(data['R']):
            choice[n,r] = np.argmax(data['endo_coef'][:,n]*p[:]+data['exo_utility'][:,n,r]+data['xi'][:,n,r])
            #x[choice[n,r],n,r] = 1.0
            #markup_i = data['p'][choice[n,r]] - data['customer_cost'][data['alt'][choice[n,r]]]

    # RETRIEVE DEMAND
    demand = np.zeros([data['I_tot']])
    for n in range(data['N']):
        for r in range(data['R']):
            demand[choice[n,r]] += (data['popN'][n]/data['R'])
    
    # RETRIEVE OBJECTIVE FUNCTION VALUE
    obj = 0.0
    for i in range(data['I_tot']):
        # Profits
        obj = obj + (p[i] - data['customer_cost'][i]) * demand[i]

    return obj

def objScenario(data, y, r):

    # RETRIEVE MAX UTILITIES OF CUSTOMERS
    choice = np.zeros([data['N']], dtype=int)
    for n in range(data['N']):
        choice[n] = np.argmax(data['U'][:,n,r] * y[:])

    # RETRIEVE DEMAND
    demand = np.zeros([data['I_tot_exp']])
    for n in range(data['N']):
        demand[choice[n]] += data['popN'][n]
    
    # RETRIEVE OBJECTIVE FUNCTION VALUE
    obj = 0.0
    for i in range(data['I_tot_exp']):
        # Fixed costs
        obj = obj - data['fixed_cost'][data['alt'][i]] * y[i]
        # Profits
        obj = obj + (data['p'][i] - data['customer_cost'][data['alt'][i]]) * demand[i]

    return obj

def objCustomer(data, y, n):

    # RETRIEVE MAX UTILITIES OF CUSTOMERS
    choice = np.zeros([data['R']], dtype=int)
    for r in range(data['R']):
        choice[r] = np.argmax(data['U'][:,n,r] * y[:])

    # RETRIEVE DEMAND
    demand = np.zeros([data['I_tot_exp']])
    for r in range(data['R']):
        demand[choice[r]] += data['popN'][n] / data['R']
    
    # RETRIEVE OBJECTIVE FUNCTION VALUE
    obj = 0.0
    for i in range(data['I_tot_exp']):
        # Fixed costs
        obj = obj - data['fixed_cost'][data['alt'][i]] * y[i]
        # Profits
        obj = obj + (data['p'][i] - data['customer_cost'][data['alt'][i]]) * demand[i]

    return obj

def allSolutionsOneFacility(data):

    data['OF1Facility'] = np.zeros([data['I_tot_exp']])
    y = data['y']
    data['bestOF1'] = 0.0
    for i in range(data['I_out_exp'], data['I_tot_exp']):
        for f in range(data['I_out_exp'], data['I_tot_exp']):
            y[f] = 0.0
        y[i] = 1.0
        data['OF1Facility'][i] = objSolution(data, y)
        if data['OF1Facility'][i] > data['bestOF1']:
            data['bestOF1'] = data['OF1Facility'][i]
            data['bestSol1'] = y
    print('\nBest OF with one open facility     : {:10.3f}'.format(data['bestOF1']))
    for i in range(data['I_tot_exp']):
        print('{:2d}'.format(i), end='')
    print()
    for i in range(data['I_tot_exp']):
        print('{:2.0f}'.format(data['bestSol1'][i]), end='')
    print()

def allSolutionsTwoFacilities(data):

    data['OF2Facilities'] = np.zeros([data['I_tot_exp'], data['I_tot_exp']])

    y = data['y']
    data['bestOF2'] = 0.0
    for i in range(data['I_out_exp'], data['I_tot_exp']):
        for j in range(i+1, data['I_tot_exp']):
            if data['alt'][i] != data['alt'][j]:
                for f in range(data['I_out_exp'], data['I_tot_exp']):
                    y[f] = 0.0
                y[i] = 1.0
                y[j] = 1.0
                data['OF2Facilities'][i,j] = objSolution(data, y)
                data['OF2Facilities'][j,i] = data['OF2Facilities'][i,j]
                if data['OF2Facilities'][i,j] > data['bestOF2']:
                    data['bestOF2'] = data['OF2Facilities'][i,j]
                    data['bestSol2'] = y
    print('\nBest OF with two open facilities   : {:10.3f}'.format(data['bestOF2']))
    for i in range(data['I_tot_exp']):
        print('{:2d}'.format(i), end='')
    print()
    for i in range(data['I_tot_exp']):
        print('{:2.0f}'.format(data['bestSol2'][i]), end='')
    print('\n')

def allSolutionsThreeFacilities(data):

    data['OF3Facilities'] = np.zeros([data['I_tot_exp'], data['I_tot_exp'], data['I_tot_exp']])

    y = data['y']
    for i in range(data['I_out_exp'], data['I_tot_exp']):
        for j in range(i+1, data['I_tot_exp']):
            for k in range(j+1, data['I_tot_exp']):
                if data['alt'][i] != data['alt'][j] and data['alt'][j] != data['alt'][k] and data['alt'][i] != data['alt'][k]:
                    for f in range(data['I_out_exp'], data['I_tot_exp']):
                        y[f] = 0.0
                    y[i] = 1.0
                    y[j] = 1.0
                    y[k] = 1.0
                    data['OF3Facilities'][i,j,k] = objSolution(data, y)
                    data['OF3Facilities'][i,k,j] = data['OF3Facilities'][i,j,k]
                    data['OF3Facilities'][k,i,j] = data['OF3Facilities'][i,j,k]
                    data['OF3Facilities'][j,i,k] = data['OF3Facilities'][i,j,k]
                    data['OF3Facilities'][j,k,i] = data['OF3Facilities'][i,j,k]
                    data['OF3Facilities'][k,j,i] = data['OF3Facilities'][i,j,k]
                    if data['OF3Facilities'][i,j,k] > data['bestOF3']:
                        data['bestOF3'] = data['OF3Facilities'][i,j,k]
                        data['bestSol3'] = y
    print('\nBest OF with three open facilities : {:10.3f}'.format(data['bestOF3']))
    for i in range(data['I_tot_exp']):
        print('{:2d}'.format(i), end='')
    print()
    for i in range(data['I_tot_exp']):
        print('{:2.0f}'.format(data['bestSol3'][i]), end='')
    print('\n')

def incompatibilityCuts(data, y, master):
    
    data['incompatibility_LHS_cut'] = []
    data['incompatibility_senses_cut'] = []
    data['incompatibility_RHS_cut'] = []
    for i in range(data['I_out_exp'], data['I_tot_exp']):
        for j in range(i+1, data['I_tot_exp']):
            if data['OF1Facility'][i] > data['OF2Facilities'][i,j] and data['alt'][i] != data['alt'][j]:
                data['incompatibility_LHS_cut'].append(cplex.SparsePair(ind=[y[i], y[j]], val=[1.0, 1.0]))
                data['incompatibility_senses_cut'].append('L')
                data['incompatibility_RHS_cut'].append(1.0)
            if data['bestOF3'] > 0.0:
                for k in range(j+1, data['I_tot_exp']):
                    if data['OF2Facilities'][i,j] > data['OF3Facilities'][i,j,k] and data['alt'][k] != data['alt'][j] and data['alt'][k] != data['alt'][i]:
                        data['incompatibility_LHS_cut'].append(cplex.SparsePair(ind=[y[i], y[j], y[k]], val=[1.0, 1.0, 1.0]))
                        data['incompatibility_senses_cut'].append('L')
                        data['incompatibility_RHS_cut'].append(2.0)
    print('nIncompatibilityCuts = {:4d}\n'.format(len(data['incompatibility_LHS_cut'])))

    #master.linear_constraints.advanced.add_user_cuts(
    #master.linear_constraints.advanced.add_lazy_constraints(
    master.linear_constraints.add(
        lin_expr = [data['incompatibility_LHS_cut'][i] for i in range(len(data['incompatibility_LHS_cut']))],
        senses = [data['incompatibility_senses_cut'][i] for i in range(len(data['incompatibility_senses_cut']))],
        rhs = [data['incompatibility_RHS_cut'][i] for i in range(len(data['incompatibility_RHS_cut']))])

def createDualFollower(data):

    dualFollower = []
    for n in range(data['N']):
        dualFollower.append([])
        for r in range(data['R']):
            
            if data['PB_RetainedInMaster'][r] == 1: #if r is in the Master problem, it's not necessary to generate the dual subproblem
                dualFollower[n].append(-1)
            
            elif data['PB_RetainedInMaster'][r] == 0:
            
                if data['Clustering'] == 'Yes' and (data['presolveCutCustomer'][n] == 0 or data['presolveCutScenario'][r] == 0):
                    dualFollower[n].append(-1)
            
                else:
                    dualFollower[n].append(cplex.Cplex())

                    ##########################################
                    ##### ----- OBJECTIVE FUNCTION ----- #####
                    ##########################################
                    dualFollower[n][r].objective.set_sense(dualFollower[n][r].objective.sense.maximize)

                    ##########################################
                    ##### ----- DECISION VARIABLES ----- #####
                    ##########################################

                    objVar = []
                    typeVar = []
                    nameVar = []
                    lbVar = []
                    ubVar = []

                    objVar.append(1.0)
                    typeVar.append(dualFollower[n][r].variables.type.continuous)
                    nameVar.append('alpha1[' + str(n) + ']' + '[' + str(r) + ']')
                    lbVar.append(-cplex.infinity)
                    ubVar.append(0.0)

                    for i in range(data['I_tot_exp']):
                        objVar.append(data['y'][i])
                        typeVar.append(dualFollower[n][r].variables.type.continuous)
                        nameVar.append('alpha2[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                        lbVar.append(-cplex.infinity)
                        ubVar.append(0.0)

                    objVar.append(0.0)
                    typeVar.append(dualFollower[n][r].variables.type.continuous)
                    nameVar.append('gamma2[' + str(n) + ']' + '[' + str(r) + ']')
                    lbVar.append(-cplex.infinity)
                    ubVar.append(0.0)

                    for i in range(data['I_tot_exp']):
                        objVar.append(-data['U'][i,n,r])
                        typeVar.append(dualFollower[n][r].variables.type.continuous)
                        nameVar.append('gamma1[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                        lbVar.append(-cplex.infinity)
                        ubVar.append(0.0)

                    for i in range(data['I_tot_exp']):
                        objVar.append(0.0)
                        typeVar.append(dualFollower[n][r].variables.type.continuous)
                        nameVar.append('delta1[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                        lbVar.append(-cplex.infinity)
                        ubVar.append(0.0)

                    for i in range(data['I_tot_exp']):
                        objVar.append(data['M'][i,n,r] * (1 - data['y'][i]))
                        typeVar.append(dualFollower[n][r].variables.type.continuous)
                        nameVar.append('delta2[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                        lbVar.append(-cplex.infinity)
                        ubVar.append(0.0)

                    for i in range(data['I_tot_exp']):
                        objVar.append(data['M'][i,n,r] * data['y'][i])
                        typeVar.append(dualFollower[n][r].variables.type.continuous)
                        nameVar.append('delta3[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                        lbVar.append(-cplex.infinity)
                        ubVar.append(0.0)

                    dualFollower[n][r].variables.add(obj = [objVar[i] for i in range(len(objVar))],
                                                    types = [typeVar[i] for i in range(len(typeVar))],
                                                    lb = [lbVar[i] for i in range(len(typeVar))],
                                                    ub = [ubVar[i] for i in range(len(typeVar))],
                                                    names = [nameVar[i] for i in range(len(nameVar))])
                    
                    #print('CPLEX model: all decision variables added. N variables: %r.' %(dualFollower[n][r].variables.get_num()))
                    
                    # Creating a dictionary that maps variable names to indices, to speed up constraints creation
                    nameToIndex = { n : j for j, n in enumerate(dualFollower[n][r].variables.get_names()) }

                    #########################################
                    ##### -------- CONSTRAINTS -------- #####
                    #########################################

                    indicesConstr = []
                    coefsConstr = []
                    sensesConstr = []
                    rhsConstr = []
                    
                    # (DUAL: x_inr)
                    for i in range(data['I_tot_exp']):
                        indicesConstr.append([nameToIndex['alpha1[' + str(n) + ']' + '[' + str(r) + ']'],
                                            nameToIndex['alpha2[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                            nameToIndex['gamma2[' + str(n) + ']' + '[' + str(r) + ']']])
                        coefsConstr.append([1.0, 1.0, -data['U'][i,n,r]])
                        sensesConstr.append('L')
                        rhsConstr.append(-(data['p'][i] - data['customer_cost'][data['alt'][i]]) * data['popN'][n]/data['R'])
                    
                    # (DUAL: beta_nr):
                    ind = []
                    co = []
                    ind.append(nameToIndex['gamma2[' + str(n) + ']' + '[' + str(r) + ']'])
                    co.append(1.0)
                    for i in range(data['I_tot_exp']):
                        ind.append(nameToIndex['gamma1[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                        co.append(-1.0)
                    indicesConstr.append(ind)
                    coefsConstr.append(co)
                    sensesConstr.append('L')
                    rhsConstr.append(0.0)
                    
                    # (DUAL: lambda_inr)
                    for i in range(data['I_tot_exp']):
                        indicesConstr.append([nameToIndex['gamma1[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                            nameToIndex['delta1[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                            nameToIndex['delta2[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                        coefsConstr.append([-1.0, -1.0, 1.0])
                        sensesConstr.append('L')
                        rhsConstr.append(0.0)

                    # (DUAL: mu_inr)
                    for i in range(data['I_tot_exp']):
                        indicesConstr.append([nameToIndex['gamma2[' + str(n) + ']' + '[' + str(r) + ']'],
                                            nameToIndex['delta1[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                            nameToIndex['delta2[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                            nameToIndex['delta3[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                        coefsConstr.append([1.0, 1.0, -1.0, 1.0])
                        sensesConstr.append('L')
                        rhsConstr.append(0.0)

                    dualFollower[n][r].linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                                              senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                                              rhs = [rhsConstr[i] for i in range(len(rhsConstr))])

                    #print('CPLEX model: all constraints added. N constraints: %r\n' %(dualFollower[n][r].linear_constraints.get_num()))

                    # Set up Cplex instance to solve the worker LP
                    dualFollower[n][r].set_results_stream(None)
                    dualFollower[n][r].set_log_stream(None)

                    # Turn off the presolve reductions and set the CPLEX optimizer
                    # to solve the worker LP with primal simplex method.
                    dualFollower[n][r].parameters.preprocessing.reduce.set(0)
                    dualFollower[n][r].parameters.lpmethod.set(dualFollower[n][r].parameters.lpmethod.values.primal)

    return dualFollower


def createDualFollowerAggregate(data):

    dualFollower = cplex.Cplex()

    ##########################################
    ##### ----- OBJECTIVE FUNCTION ----- #####
    ##########################################
    dualFollower.objective.set_sense(dualFollower.objective.sense.maximize)

    ##########################################
    ##### ----- DECISION VARIABLES ----- #####
    ##########################################

    objVar = []
    typeVar = []
    nameVar = []
    lbVar = []
    ubVar = []

    for n in range(data['N']):
        for r in range(data['R']):

            objVar.append(1.0)
            typeVar.append(dualFollower.variables.type.continuous)
            nameVar.append('alpha1[' + str(n) + ']' + '[' + str(r) + ']')
            lbVar.append(-cplex.infinity)
            ubVar.append(0.0)

            for i in range(data['I_tot_exp']):
                objVar.append(data['y'][i])
                typeVar.append(dualFollower.variables.type.continuous)
                nameVar.append('alpha2[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                lbVar.append(-cplex.infinity)
                ubVar.append(0.0)

            objVar.append(0.0)
            typeVar.append(dualFollower.variables.type.continuous)
            nameVar.append('gamma2[' + str(n) + ']' + '[' + str(r) + ']')
            lbVar.append(-cplex.infinity)
            ubVar.append(0.0)

            for i in range(data['I_tot_exp']):
                objVar.append(-data['U'][i,n,r])
                typeVar.append(dualFollower.variables.type.continuous)
                nameVar.append('gamma1[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                lbVar.append(-cplex.infinity)
                ubVar.append(0.0)

            for i in range(data['I_tot_exp']):
                objVar.append(0.0)
                typeVar.append(dualFollower.variables.type.continuous)
                nameVar.append('delta1[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                lbVar.append(-cplex.infinity)
                ubVar.append(0.0)

            for i in range(data['I_tot_exp']):
                objVar.append(data['M'][i,n,r] * (1 - data['y'][i]))
                typeVar.append(dualFollower.variables.type.continuous)
                nameVar.append('delta2[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                lbVar.append(-cplex.infinity)
                ubVar.append(0.0)

            for i in range(data['I_tot_exp']):
                objVar.append(data['M'][i,n,r] * data['y'][i])
                typeVar.append(dualFollower.variables.type.continuous)
                nameVar.append('delta3[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                lbVar.append(-cplex.infinity)
                ubVar.append(0.0)

    dualFollower.variables.add(obj = [objVar[i] for i in range(len(objVar))],
                               types = [typeVar[i] for i in range(len(typeVar))],
                               lb = [lbVar[i] for i in range(len(typeVar))],
                               ub = [ubVar[i] for i in range(len(typeVar))],
                               names = [nameVar[i] for i in range(len(nameVar))])
    
    #print('CPLEX model: all decision variables added. N variables: %r.' %(dualFollower.variables.get_num()))
    
    # Creating a dictionary that maps variable names to indices, to speed up constraints creation
    nameToIndex = { n : j for j, n in enumerate(dualFollower.variables.get_names()) }

    #########################################
    ##### -------- CONSTRAINTS -------- #####
    #########################################

    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []

    for n in range(data['N']):
        for r in range(data['R']):

            # (DUAL: x_inr)
            for i in range(data['I_tot_exp']):
                indicesConstr.append([nameToIndex['alpha1[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['alpha2[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['gamma2[' + str(n) + ']' + '[' + str(r) + ']']])
                coefsConstr.append([1.0, 1.0, -data['U'][i,n,r]])
                sensesConstr.append('L')
                rhsConstr.append(-(data['p'][i] - data['customer_cost'][data['alt'][i]]) * data['popN'][n]/data['R'])
            
            # (DUAL: beta_nr):
            ind = []
            co = []
            ind.append(nameToIndex['gamma2[' + str(n) + ']' + '[' + str(r) + ']'])
            co.append(1.0)
            for i in range(data['I_tot_exp']):
                ind.append(nameToIndex['gamma1[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                co.append(-1.0)
            indicesConstr.append(ind)
            coefsConstr.append(co)
            sensesConstr.append('L')
            rhsConstr.append(0.0)
            
            # (DUAL: lambda_inr)
            for i in range(data['I_tot_exp']):
                indicesConstr.append([nameToIndex['gamma1[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['delta1[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['delta2[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                coefsConstr.append([-1.0, -1.0, 1.0])
                sensesConstr.append('L')
                rhsConstr.append(0.0)

            # (DUAL: mu_inr)
            for i in range(data['I_tot_exp']):
                indicesConstr.append([nameToIndex['gamma2[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['delta1[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['delta2[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['delta3[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                coefsConstr.append([1.0, 1.0, -1.0, 1.0])
                sensesConstr.append('L')
                rhsConstr.append(0.0)

    dualFollower.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                        senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                        rhs = [rhsConstr[i] for i in range(len(rhsConstr))])

    #print('CPLEX model: all constraints added. N constraints: %r\n' %(dualFollower.linear_constraints.get_num()))

    # Set up Cplex instance to solve the worker LP
    dualFollower.set_results_stream(None)
    dualFollower.set_log_stream(None)

    # Turn off the presolve reductions and set the CPLEX optimizer
    # to solve the worker LP with primal simplex method.
    dualFollower.parameters.preprocessing.reduce.set(0)
    dualFollower.parameters.lpmethod.set(dualFollower.parameters.lpmethod.values.primal)

    return dualFollower


def dualWorker(data, solution, dualFollower):

    print('DUAL OF SOLUTION {:4d}'.format(solution))

    data['x_dual'] = np.zeros([data['I_tot_exp'], data['N'], data['R']])
    data['alpha1_dual'] = np.zeros([data['N'], data['R']])
    data['alpha2_dual'] = np.zeros([data['I_tot_exp'], data['N'], data['R']])
    data['gamma2_dual'] = np.zeros([data['N'], data['R']])
    data['gamma1_dual'] = np.zeros([data['I_tot_exp'], data['N'], data['R']])
    data['delta1_dual'] = np.zeros([data['I_tot_exp'], data['N'], data['R']])
    data['delta2_dual'] = np.zeros([data['I_tot_exp'], data['N'], data['R']])
    data['delta3_dual'] = np.zeros([data['I_tot_exp'], data['N'], data['R']])

    for n in range(data['N']):
        for r in range(data['R']):

            if data['PB_RetainedInMaster'][r] == 1 or (data['Clustering'] == 'Yes' and (data['presolveCutCustomer'][n] == 0 or data['presolveCutScenario'][r] == 0)):
                continue
            
            else:
                # Update the objective function coefficients in the worker LP (index, newValue)
                sparsePair = []
                # Alpha1 not needed
                count = 1
                # Alpha2
                for i in range(data['I_tot_exp']):
                    sparsePair.append((count, data['all_y'][solution][i]))
                    count += 1
                # Gamma1, Gamma2, Delta1 not needed
                count += (1 + 2*data['I_tot_exp']) 
                # Delta2
                for i in range(data['I_tot_exp']):
                    sparsePair.append((count, data['M'][i,n,r] * (1 - data['all_y'][solution][i])))
                    count += 1
                # Delta3
                for i in range(data['I_tot_exp']):
                    sparsePair.append((count, data['M'][i,n,r] * data['all_y'][solution][i]))
                    count += 1
                    
                dualFollower[n][r].objective.set_linear(sparsePair)

                # Solve the worker LP
                dualFollower[n][r].set_problem_type(dualFollower[n][r].problem_type.LP)
                dualFollower[n][r].solve()

                if dualFollower[n][r].solution.get_status() == dualFollower[n][r].solution.status.optimal:
                    objDual = dualFollower[n][r].solution.get_objective_value()

                    # dual: alpha1[n][r]
                    data['alpha1_dual'][n,r] = dualFollower[n][r].solution.get_values('alpha1[' + str(n) + ']' + '[' + str(r) + ']')
                    # dual: alpha2[i][n][r]
                    for i in range(data['I_tot_exp']):
                        data['alpha2_dual'][i,n,r] = dualFollower[n][r].solution.get_values('alpha2[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                    # dual: gamma2[n][r]
                    data['gamma2_dual'][n,r] = dualFollower[n][r].solution.get_values('gamma2[' + str(n) + ']' + '[' + str(r) + ']')
                    # dual: gamma1[i][n][r]
                    for i in range(data['I_tot_exp']):
                        data['gamma1_dual'][i,n,r] = -dualFollower[n][r].solution.get_values('gamma1[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                    # dual: delta1[i][n][r]
                    for i in range(data['I_tot_exp']):
                        data['delta1_dual'][i,n,r] = -dualFollower[n][r].solution.get_values('delta1[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                    # dual: delta2[i][n][r]
                    for i in range(data['I_tot_exp']):
                        data['delta2_dual'][i,n,r] = -dualFollower[n][r].solution.get_values('delta2[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                    # dual: delta3[i][n][r]
                    for i in range(data['I_tot_exp']):
                        data['delta3_dual'][i,n,r] = -dualFollower[n][r].solution.get_values('delta3[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')


def dualWorkerAggregate(data, solution, dualFollower):

    print('DUAL OF SOLUTION {:4d}'.format(solution))

    data['x_dual'] = np.zeros([data['I_tot_exp'], data['N'], data['R']])
    data['alpha1_dual'] = np.zeros([data['N'], data['R']])
    data['alpha2_dual'] = np.zeros([data['I_tot_exp'], data['N'], data['R']])
    data['gamma2_dual'] = np.zeros([data['N'], data['R']])
    data['gamma1_dual'] = np.zeros([data['I_tot_exp'], data['N'], data['R']])
    data['delta1_dual'] = np.zeros([data['I_tot_exp'], data['N'], data['R']])
    data['delta2_dual'] = np.zeros([data['I_tot_exp'], data['N'], data['R']])
    data['delta3_dual'] = np.zeros([data['I_tot_exp'], data['N'], data['R']])

    # Update the objective function coefficients in the worker LP (index, newValue)
    sparsePair = []
    count = 0

    for n in range(data['N']):
        for r in range(data['R']):

            # Alpha1 not needed
            count += 1
            # Alpha2
            for i in range(data['I_tot_exp']):
                sparsePair.append((count, data['all_y'][solution][i]))
                count += 1
            # Gamma1, Gamma2, Delta1 not needed
            count += (1 + 2*data['I_tot_exp']) 
            # Delta2
            for i in range(data['I_tot_exp']):
                sparsePair.append((count, data['M'][i,n,r] * (1 - data['all_y'][solution][i])))
                count += 1
            # Delta3
            for i in range(data['I_tot_exp']):
                sparsePair.append((count, data['M'][i,n,r] * data['all_y'][solution][i]))
                count += 1
                
    dualFollower.objective.set_linear(sparsePair)

    # Solve the worker LP
    dualFollower.set_problem_type(dualFollower.problem_type.LP)
    dualFollower.solve()

    if dualFollower.solution.get_status() == dualFollower.solution.status.optimal:
        objDual = dualFollower.solution.get_objective_value()

        for n in range(data['N']):
            for r in range(data['R']):
                # dual: alpha1[n][r]
                data['alpha1_dual'][n,r] = dualFollower.solution.get_values('alpha1[' + str(n) + ']' + '[' + str(r) + ']')
                # dual: alpha2[i][n][r]
                for i in range(data['I_tot_exp']):
                    data['alpha2_dual'][i,n,r] = dualFollower.solution.get_values('alpha2[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                # dual: gamma2[n][r]
                data['gamma2_dual'][n,r] = dualFollower.solution.get_values('gamma2[' + str(n) + ']' + '[' + str(r) + ']')
                # dual: gamma1[i][n][r]
                for i in range(data['I_tot_exp']):
                    data['gamma1_dual'][i,n,r] = -dualFollower.solution.get_values('gamma1[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                # dual: delta1[i][n][r]
                for i in range(data['I_tot_exp']):
                    data['delta1_dual'][i,n,r] = -dualFollower.solution.get_values('delta1[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                # dual: delta2[i][n][r]
                for i in range(data['I_tot_exp']):
                    data['delta2_dual'][i,n,r] = -dualFollower.solution.get_values('delta2[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                # dual: delta3[i][n][r]
                for i in range(data['I_tot_exp']):
                    data['delta3_dual'][i,n,r] = -dualFollower.solution.get_values('delta3[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')


def disaggregateCuts(data, scenario, y, z):

    LHS_cuts = []
    senses_cuts = []
    RHS_cuts = []

    for n in range(data['N']):
        for r in range(data['R']):

            if data['PB_RetainedInMaster'][r] == 1 or (data['Clustering'] == 'Yes' and (data['presolveCutCustomer'][n] == 0 or data['presolveCutScenario'][r] == 0)):
                continue
            
            else:
                # Components of the dual objective function
                objA1_nr = data['alpha1_dual'][n,r]
                objA2_nr = 0.0
                for i in range(data['I_tot_exp']):
                    objA2_nr = objA2_nr + data['alpha2_dual'][i,n,r] * data['all_y'][scenario][i]
                objG1_nr = 0.0
                for i in range(data['I_tot_exp']):
                    objG1_nr = objG1_nr - data['U'][i,n,r] * data['gamma1_dual'][i,n,r]
                objD2rhs_nr = 0.0
                objD2y_nr = 0.0
                for i in range(data['I_tot_exp']):
                    objD2rhs_nr = objD2rhs_nr + data['M'][i,n,r] * data['delta2_dual'][i,n,r]
                    objD2y_nr = objD2y_nr - data['M'][i,n,r] * data['all_y'][scenario][i] * data['delta2_dual'][i,n,r]
                objD3_nr = 0.0
                for i in range(data['I_tot_exp']):
                    objD3_nr = objD3_nr + data['M'][i,n,r] * data['all_y'][scenario][i] * data['delta3_dual'][i,n,r]

                # RHS
                rhs_nr = -(objA1_nr + objG1_nr + objD2rhs_nr)
                # LHS
                data['alpha2_i'] = np.zeros([data['I_tot_exp']])
                data['delta2_i'] = np.zeros([data['I_tot_exp']])
                data['delta3_i'] = np.zeros([data['I_tot_exp']])
                ind = []
                co = []
                for i in range(data['I_tot_exp']):
                    data['alpha2_i'][i] = data['alpha2_i'][i] + data['alpha2_dual'][i,n,r]
                    data['delta2_i'][i] = data['delta2_i'][i] + data['M'][i,n,r] * data['delta2_dual'][i,n,r]
                    data['delta3_i'][i] = data['delta3_i'][i] - data['M'][i,n,r] * data['delta3_dual'][i,n,r]
                    ind.append(y[i])
                    co.append(data['alpha2_i'][i] + data['delta2_i'][i] + data['delta3_i'][i])
                ind.append(z[n][r])
                co.append(-1.0)

                LHS_cuts.append(cplex.SparsePair(ind=ind, val=co))
                senses_cuts.append('L')
                RHS_cuts.append(rhs_nr)
    
    return LHS_cuts, senses_cuts, RHS_cuts

def aggregateCut(data, scenario, y, z_agg):

    # Components of the dual objective function
    objA1 = 0.0
    for n in range(data['N']):
        for r in range(data['R']):
            if data['PB_RetainedInMaster'][r] == 0:
                objA1 = objA1 + data['alpha1_dual'][n,r]
    objA2 = 0.0
    for n in range(data['N']):
        for r in range(data['R']):
            if data['PB_RetainedInMaster'][r] == 0:
                for i in range(data['I_tot_exp']):
                    objA2 = objA2 + data['alpha2_dual'][i,n,r] * data['all_y'][scenario][i]
    objG1 = 0.0
    for n in range(data['N']):
        for r in range(data['R']):
            if data['PB_RetainedInMaster'][r] == 0:
                for i in range(data['I_tot_exp']):
                    objG1 = objG1 - data['U'][i,n,r] * data['gamma1_dual'][i,n,r]
    objD2rhs = 0.0
    objD2y = 0.0
    for n in range(data['N']):
        for r in range(data['R']):
            if data['PB_RetainedInMaster'][r] == 0:
                for i in range(data['I_tot_exp']):
                    objD2rhs = objD2rhs + data['M'][i,n,r] * data['delta2_dual'][i,n,r]
                    objD2y = objD2y - data['M'][i,n,r] * data['all_y'][scenario][i] * data['delta2_dual'][i,n,r]
    objD3 = 0.0
    for n in range(data['N']):
        for r in range(data['R']):
            if data['PB_RetainedInMaster'][r] == 0:
                for i in range(data['I_tot_exp']):
                    objD3 = objD3 + data['M'][i,n,r] * data['all_y'][scenario][i] * data['delta3_dual'][i,n,r]

    rhs = -(objA1 + objG1 + objD2rhs)
    data['alpha2_i'] = np.zeros([data['I_tot_exp']])
    data['delta2_i'] = np.zeros([data['I_tot_exp']])
    data['delta3_i'] = np.zeros([data['I_tot_exp']])
    ind = []
    co = []
    for i in range(data['I_tot_exp']):
        for n in range(data['N']):
            for r in range(data['R']):
                if data['PB_RetainedInMaster'][r] == 0:
                    data['alpha2_i'][i] = data['alpha2_i'][i] + data['alpha2_dual'][i,n,r]
                    data['delta2_i'][i] = data['delta2_i'][i] + data['M'][i,n,r] * data['delta2_dual'][i,n,r]
                    data['delta3_i'][i] = data['delta3_i'][i] - data['M'][i,n,r] * data['delta3_dual'][i,n,r]
        ind.append(y[i])
        co.append(data['alpha2_i'][i] + data['delta2_i'][i] + data['delta3_i'][i])
    ind.append(z_agg)
    co.append(-1.0)
        
    LHS_cut = [cplex.SparsePair(ind=ind, val=co)]
    sense_cut = ['L']
    RHS_cut = [rhs]

    return LHS_cut, sense_cut, RHS_cut

def subsetCut(data, sol_y, y, lhs_cuts, sense_cuts, rhs_cuts):

    t_start_subsetCut = time.time()

    initialOF = objSolution(data, sol_y)
    bestOF = initialOF
    list_open_y = [i for i in range(data['I_out_exp'], data['I_tot_exp']) if sol_y[i] == 1]
    #random.shuffle(list_open_y)
    
    for i in list_open_y:
        sol = copy.deepcopy(sol_y)
        sol[i] = 0.0
        newOF = objSolution(data, sol)
        if newOF > bestOF:
            bestOF = newOF
            bestSol = sol
            closed = i
    
    if bestOF > initialOF:
        # Recursive check to find minimal infeasible subset
        subsetCut(data, bestSol, y, lhs_cuts, sense_cuts, rhs_cuts)
        # Generate subset cut
        count = 0.0
        #if len(list_open_y) <= 3:
        ind = []
        co = []
        for j in list_open_y:
            ind.append(y[j])
            co.append(1.0)
            count += 1
        # Verify other price levels from "worst" alternative in the subset
        for l in range(data['I_out_exp'], data['I_tot_exp']):
            if data['alt'][l] == data['alt'][closed] and l != closed:
                sol = bestSol
                sol[l] = 1.0
                newOF = objSolution(data, sol)
                if newOF < bestOF:
                    ind.append(y[l])
                    co.append(1.0)
        lhs_cuts.append(cplex.SparsePair(ind=ind, val=co))
        sense_cuts.append('L')
        rhs_cuts.append(count-1)
            
        # Check print
        '''
        print('\n               ', end='')
        for i in range(data['I_tot_exp']):
            print('{:2d}'.format(i), end='')
        print('\n Orig  {:8.3f}'.format(OF), end='')
        for i in range(data['I_tot_exp']):
            print('{:2.0f}'.format(sol_y[i]), end='')
        print('\n  Sub  {:8.3f}'.format(newOF), end='')
        for i in range(data['I_tot_exp']):
            print('{:2.0f}'.format(sol[i]), end='')
        print()
        '''
        # Stopping criterion to avoid exploring all exponentially many subsets
        #if len(lhs_cuts) > data['I_tot_exp']:
        #if len(lhs_cuts) > 10:
    
    data['timeSubsetCuts'] += time.time() - t_start_subsetCut

def subsetCutFrac(data, sol_y, y, lhs_cuts, sense_cuts, rhs_cuts):

    t_start_subsetCut = time.time()

    initialOF = objSolution(data, sol_y)
    bestOF = initialOF
    list_open_y = [i for i in range(data['I_out_exp'], data['I_tot_exp']) if sol_y[i] > 0]
    
    for i in list_open_y:
        sol = copy.deepcopy(sol_y)
        sol[i] = 0.0
        newOF = objSolution(data, sol)
        if newOF > bestOF:
            bestOF = newOF
            bestSol = sol
    
    if bestOF > initialOF:
        # Recursive check to find minimal infeasible subset
        subsetCut(data, bestSol, y, lhs_cuts, sense_cuts, rhs_cuts)
        # Generate subset cut
        count = 0.0
        #if len(list_open_y) <= 3:
        ind = []
        co = []
        for j in list_open_y:
            ind.append(y[j])
            co.append(sol_y[j])
            count += sol_y[j]
        lhs_cuts.append(cplex.SparsePair(ind=ind, val=co))
        sense_cuts.append('L')
        rhs_cuts.append(math.floor(count))
            
        # Check print
        '''
        print('\n               ', end='')
        for i in range(data['I_tot_exp']):
            print('{:2d}'.format(i), end='')
        print('\n Orig  {:8.3f}'.format(OF), end='')
        for i in range(data['I_tot_exp']):
            print('{:2.0f}'.format(sol_y[i]), end='')
        print('\n  Sub  {:8.3f}'.format(newOF), end='')
        for i in range(data['I_tot_exp']):
            print('{:2.0f}'.format(sol[i]), end='')
        print()
        '''
        # Stopping criterion to avoid exploring all exponentially many subsets
        #if len(lhs_cuts) > data['I_tot_exp']:
        #if len(lhs_cuts) > 10:
    
    data['timeSubsetCuts'] += time.time() - t_start_subsetCut


if __name__ == '__main__':
    
    t_total_start = time.time()
    
    # Read instance and print aggregate customer data
    data = data_file.getData()
    data_file.printCustomers(data)

    #Generate initial solution
    #                     Walk PT IW1 IW2 IW3 IE1 IE2 IE3  IN  IS  UW  UE  UN  US
    data['y'] = np.array([  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])

    #                         Walk    PT   IW1   IW2   IW3   IE1   IE2   IE3    IN    IS    UW    UE    UN    US
    data['price'] = np.array([ 0.0,  0.0,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  1.0,  1.0,  1.0,  1.0])
            
    #Calculate utilities 
    data_file.preprocessUtilities(data)
    calcUtilities(data)
    '''
    #Rank utilities pairwise and calculate choice probabilities
    rankUtilitiesAll(data)
    calculateProb(data)
    '''
    #Calculate choice probabilities directly
    update_bounds.updateUtilityBounds(data)
    calculateProbDirect(data)

    #Calculate objective function
    calculateObj(data)

    t_total_end = time.time()
    
    print('\n\nTotal time algorithm: {:8.4f}\n'.format(t_total_end - t_total_start))
