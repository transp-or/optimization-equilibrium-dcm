# Modelisation of the choice-based optimization problem with continuous prices
#   Supplier = continuous prices
#   Other suppliers = fixed prices
#   Customers = followers

# General
import time
import copy
import numpy as np

# CPLEX
import cplex
from cplex.exceptions import CplexSolverError

# Project
import update_bounds

# Data
import Data_LinSibdari_MNL as data_file


def getModel(data):
    '''
    CPLEX model for the choice-based optimization problem
    (1 supplier optimizing, all the rest fixed)
    '''

    print('\nOPTIMIZER {:2d}'.format(data['optimizer']))

    t_in = time.time()
    
    # Initialize the model
    model = cplex.Cplex()


    ##########################################
    ##### ----- OBJECTIVE FUNCTION ----- #####
    ##########################################

    # Set the objective function sense
    model.objective.set_sense(model.objective.sense.maximize)


    ##########################################
    ##### ----- DECISION VARIABLES ----- #####
    ##########################################

    # BATCH OF CHOICE VARIABLES
    typeVar = []
    nameVar = []

    # Customer choice variables
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                typeVar.append(model.variables.type.binary)
                nameVar.append('w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')

    model.variables.add(types = [typeVar[i] for i in range(len(typeVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])


    # BATCH OF UTILITY AND PRICE VARIABLES
    typeVar = []
    nameVar = []
    lbVar = []
    ubVar = []

    # Utility variables
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                typeVar.append(model.variables.type.continuous)
                nameVar.append('U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                lbVar.append(data['lb_U'][i,n,r])
                ubVar.append(data['ub_U'][i,n,r])

    # Maximum utility for each customer and draw
    for n in range(data['N']):
        for r in range(data['R']):
            typeVar.append(model.variables.type.continuous)
            nameVar.append('Umax[' + str(n) + ']' + '[' + str(r) + ']')
            lbVar.append(-cplex.infinity)
            ubVar.append(cplex.infinity)

    # Price variables
    for i in range(data['I_tot']):
        typeVar.append(model.variables.type.continuous)
        nameVar.append('p[' + str(i) + ']')
        lbVar.append(data['lb_p'][i])
        ubVar.append(data['ub_p'][i])


    model.variables.add(types = [typeVar[i] for i in range(len(typeVar))],
                        lb = [lbVar[i] for i in range(len(typeVar))],
                        ub = [ubVar[i] for i in range(len(typeVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])


    # BATCH OF CHOICE-PRICE AND DEMAND VARIABLES
    objVar = []
    typeVar = []
    nameVar = []
    lbVar = []
    ubVar = []

    # Linearized choice-price variables
    for i in range(data['I_opt_out'], data['I_tot']):
        if (data['optimizer'] is None) or (data['operator'][i] == data['optimizer']):
            for n in range(data['N']):
                for r in range(data['R']):
                    objVar.append(data['popN'][n]/data['R'])
                    typeVar.append(model.variables.type.continuous)
                    nameVar.append('alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                    lbVar.append(0.0)
                    ubVar.append(data['ub_p'][i])

    # Auxiliary variables to calculate the demand
    for i in range(data['I_tot']):
        objVar.append(0.0)
        typeVar.append(model.variables.type.continuous)
        nameVar.append('d[' + str(i) + ']')
        lbVar.append(0.0)
        ubVar.append(data['Pop'])

    model.variables.add(obj = [objVar[i] for i in range(len(objVar))],
                        types = [typeVar[i] for i in range(len(typeVar))],
                        lb = [lbVar[i] for i in range(len(typeVar))],
                        ub = [ubVar[i] for i in range(len(typeVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])

    print('CPLEX model: all decision variables added. N variables: %r. Time: %r'\
          %(model.variables.get_num(), round(time.time()-t_in,2)))

    # Creating a dictionary that maps variable names to indices, to speed up constraints creation
    # https://www.ibm.com/developerworks/community/forums/html/topic?id=2349f613-26b1-4c29-aa4d-b52c9505bf96
    nameToIndex = { n : j for j, n in enumerate(model.variables.get_names()) }


    #########################################
    ##### -------- CONSTRAINTS -------- #####
    #########################################

    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []

    ###################################################
    ### ------ Instance-specific constraints ------ ###
    ###################################################
        
    ##### Fixed price constraints
    if data['p_fixed'] is not None:
        for i in range(data['I_tot']):
            if (i >= data['I_opt_out']) and (data['operator'][i] != data['optimizer']):
                indicesConstr.append([nameToIndex['p[' + str(i) + ']']])
                coefsConstr.append([1.0])
                sensesConstr.append('E')
                rhsConstr.append(data['p_fixed'][i])

    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])

    ###################################################
    ### ------------ Choice constraints ----------- ###
    ###################################################

    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []

    # Each customer chooses one alternative
    for n in range(data['N']):
        for r in range(data['R']):
            ind = []
            co = []
            for i in range(data['I_tot']):
                ind.append(nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                co.append(1.0)
            indicesConstr.append(ind)
            coefsConstr.append(co)
            sensesConstr.append('E')
            rhsConstr.append(1.0)

    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])

    #######################################
    ##### ----- Price constraints ---- ####
    #######################################

    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []

    # Linearized price
    for i in range(data['I_opt_out'], data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                if (data['optimizer'] is None) or (data['operator'][i] == data['optimizer']):

                    # Alpha is equal to 0 if alternative is not chosen
    
                    # Lower bound constraint
                    indicesConstr.append([nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                          nameToIndex['alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                    coefsConstr.append([data['lb_p'][i], -1.0])
                    sensesConstr.append('L')
                    rhsConstr.append(0.0)
    
                    # Upper bound constraint
                    indicesConstr.append([nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                          nameToIndex['alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                    coefsConstr.append([data['ub_p'][i], -1.0])
                    sensesConstr.append('G')
                    rhsConstr.append(0.0)
    
                    # Alpha is equal to the price if alternative is chosen
    
                    # Alpha is greater than the price for the chosen alternative
                    indicesConstr.append([nameToIndex['p[' + str(i) + ']'],
                                          nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                          nameToIndex['alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                    coefsConstr.append([1.0, data['ub_p'][i], -1.0])
                    sensesConstr.append('L')
                    rhsConstr.append(data['ub_p'][i])

                    # Alpha is smaller than the price
                    indicesConstr.append([nameToIndex['alpha[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                          nameToIndex['p[' + str(i) + ']']])
                    coefsConstr.append([1.0, -1.0])
                    sensesConstr.append('L')
                    rhsConstr.append(0.0)

    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])

    #######################################
    #### ----- Utility constraints ---- ###
    #######################################

    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []

    for i in range(data['I_tot']):
        
        #### Utility constraints
        for n in range(data['N']):
            for r in range(data['R']):
                    indicesConstr.append([nameToIndex['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                          nameToIndex['p[' + str(i) + ']']])
                    if data['DCM'] == 'MixedLogit':
                        coefsConstr.append([1.0, -data['endo_coef'][i,n,r]])
                        rhsConstr.append(data['exo_utility'][i,n,r] + data['xi'][i, n, r])
                    else:
                        coefsConstr.append([1.0, -data['beta']])
                        rhsConstr.append(data['exo_utility'][i,n] + data['xi'][i, n, r])
                    sensesConstr.append('E')

        #### Utility maximization: the selected alternative is the one with the highest utility
        for n in range(data['N']):
            for r in range(data['R']):
                indicesConstr.append([nameToIndex['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['Umax[' + str(n) + ']' + '[' + str(r) + ']']])
                coefsConstr.append([1.0, -1.0])
                sensesConstr.append('L')
                rhsConstr.append(0.0)

                indicesConstr.append([nameToIndex['Umax[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                coefsConstr.append([1.0, -1.0, data['M_U'][n, r]])
                sensesConstr.append('L')
                rhsConstr.append(data['M_U'][n, r])

    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])

    #######################################
    #### ---- Auxiliary constraints --- ###
    #######################################
        
    ### Calculating demands (not part of the model)
    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []

    for i in range(data['I_tot']):
        ind = []
        co = []
        for n in range(data['N']):
            for r in range(data['R']):
                ind.append(nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                co.append(-data['popN'][n]/data['R'])
        ind.append(nameToIndex['d[' + str(i) + ']'])
        co.append(1.0)
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(0.0)

    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])
    
    print('CPLEX model: all constraints added. N constraints: %r. Time: %r\n'\
          %(model.linear_constraints.get_num(), round(time.time()-t_in,2)))

    return model


def solveModel(data, model):

    try:
        model.set_results_stream(None)
        #model.set_warning_stream(None)

        model.solve()

        ### PRINT OBJ FUNCTION
        if data['lb_profit'] is not None:
            print('Lower bound of optimizer profit              : {:10.4f}'.format(data['lb_profit']))
        print('Objective function value of optimizer profit : {:10.4f}'.format(model.solution.get_objective_value()))

        ### INITIALIZE DICTIONARY OF RESULTS
        results = {}
        results['prices'] = copy.deepcopy(data['p_fixed'])
        results['demand'] = np.empty(data['I_tot'])
        results['profits'] = np.full([data['K']+1], 0.0)

        ### SAVE RESULTS
        for i in range(data['I_tot']):
            results['prices'][i] = model.solution.get_values('p[' + str(i) + ']')
            results['demand'][i] = model.solution.get_values('d[' + str(i) + ']')
            results['profits'][data['operator'][i]] += results['demand'][i]*results['prices'][i]

        ### PRINT PRICES, DEMANDS, PROFITS
        print('\nAlt  Supplier     Price    Demand  Market share      Profit')
        for i in range(data['I_tot']):
                print(' {:2d}       {:2d}     {:6.3f}    {:6.3f}        {:6.3f}     {:7.3f}'
                        .format(i, data['operator'][i], results['prices'][i], results['demand'][i], results['demand'][i] / data['Pop'], results['profits'][i]))

        return results
    
    except CplexSolverError:
        raise Exception('Exception raised during solve')

    #return results


if __name__ == '__main__':

    t_0 = time.time()
    
    #Read instance
    data = data_file.getData()

    #Calculate utility bounds
    update_bounds.updateUtilityBounds(data)
    
    t_1 = time.time()

    #Solve choice-based optimization problem
    model = getModel(data)
    results = solveModel(data, model)

    t_2 = time.time()
    
    print('\n -- TIMING -- ')
    print('Get data + Preprocess: %r sec' %(t_1 - t_0))
    print('Run the game: %r sec' %(t_2 - t_1))
    print('\n ------------ ')
