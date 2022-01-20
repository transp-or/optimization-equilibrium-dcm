# Choice-based optimization problem with continuous prices
#   Supplier = continuous prices and assortment
#   Customers = followers

# General
import time
import copy
import numpy as np

# CPLEX
import cplex
from cplex.exceptions import CplexSolverError

# Project
import functions
import update_bounds
import model_discrete_bigM_assortment

# Data
#import Data_Parking_N80_I14 as data_file
import Data_Parking_N08_I10 as data_file


def getModel(data):
    
    # Initialize the model
    t_in = time.time()
    model = cplex.Cplex()

    ##########################################
    ##### ----- OBJECTIVE FUNCTION ----- #####
    ##########################################
    model.objective.set_sense(model.objective.sense.maximize)

    ##########################################
    ##### ----- DECISION VARIABLES ----- #####
    ##########################################

    typeVar = []
    nameVar = []

    # Customer choice variables
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                typeVar.append(model.variables.type.binary)
                nameVar.append('w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')

    # Assortment variables
    for i in range(data['I_tot']):
        typeVar.append(model.variables.type.binary)
        nameVar.append('y[' + str(i) + ']')

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

    # Utility-assortment variables
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                typeVar.append(model.variables.type.continuous)
                nameVar.append('U_a[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                lbVar.append(data['lb_U'][i,n,r]-data['ub_U'][i,n,r])
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
        if data['operator'][i] == 1:
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
    # The price of the alternatives not managed by the current optimizer are fixed
    for i in range(data['I_tot']):
        if data['operator'][i] == 0:
            indicesConstr.append([nameToIndex['p[' + str(i) + ']']])
            coefsConstr.append([1.0])
            sensesConstr.append('E')
            rhsConstr.append(data['price'][i])

    ##### Fixed assortment constraints
    # Opt-out alternatives must be offered
    for i in range(data['I_tot']):
        if data['operator'][i] == 0:
            indicesConstr.append([nameToIndex['y[' + str(i) + ']']])
            coefsConstr.append([1.0])
            sensesConstr.append('E')
            rhsConstr.append(1.0)

    ###################################################
    ### ------------ Choice constraints ----------- ###
    ###################################################

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

    #######################################
    ##### ----- Price constraints ---- ####
    #######################################

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

    #######################################
    #### ----- Utility constraints ---- ###
    #######################################

    for i in range(data['I_tot']):
        
        #### Utility constraints
        for n in range(data['N']):
            for r in range(data['R']):
                indicesConstr.append([nameToIndex['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['p[' + str(i) + ']']])
                coefsConstr.append([1.0, -data['endo_coef'][i,n,r]])
                rhsConstr.append(data['exo_utility'][i,n,r] + data['xi'][i,n,r])
                sensesConstr.append('E')
        
        #### Utility-assortment constraints
        for n in range(data['N']):
            for r in range(data['R']):
                indicesConstr.append([nameToIndex['U_a[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['y[' + str(i) + ']']])
                coefsConstr.append([1.0, -1.0, -data['ub_U'][i,n,r]])
                rhsConstr.append(-data['ub_U'][i,n,r])
                sensesConstr.append('E')

        #### Utility maximization: the selected alternative is the one with the highest utility
        for n in range(data['N']):
            for r in range(data['R']):

                indicesConstr.append([nameToIndex['U_a[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['Umax[' + str(n) + ']' + '[' + str(r) + ']']])
                coefsConstr.append([1.0, -1.0])
                sensesConstr.append('L')
                rhsConstr.append(0.0)

                indicesConstr.append([nameToIndex['Umax[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['U_a[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                coefsConstr.append([1.0, -1.0, data['M_U'][n, r]])
                sensesConstr.append('L')
                rhsConstr.append(data['M_U'][n,r])

    #######################################
    #### ---- Auxiliary constraints --- ###
    #######################################
        
    ### Calculating demands (not part of the model)
    for i in range(data['I_tot']):
        # Total demand
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
        #model.set_results_stream(None)
        #model.set_warning_stream(None)

        model.parameters.timelimit.set(172000.0)
        model.solve()

        ### PRINT OBJ FUNCTION
        print('Objective function value of optimizer profit : {:10.2f}'
            .format(model.solution.get_objective_value()))

        ### INITIALIZE DICTIONARY OF RESULTS
        results = {}
        results['assortment'] = np.empty(data['I_tot'])
        results['prices'] = copy.deepcopy(data['price'])
        results['demand'] = np.empty(data['I_tot'])
        results['profits'] = np.full([data['K']+1], 0.0)

        ### SAVE RESULTS
        for i in range(data['I_tot']):
            results['assortment'][i] = model.solution.get_values('y[' + str(i) + ']')
            results['prices'][i] = model.solution.get_values('p[' + str(i) + ']')
            results['demand'][i] = model.solution.get_values('d[' + str(i) + ']')
            results['profits'][data['operator'][i]] += (results['demand'][i]*results['prices'][i])

        ### PRINT PRICES, DEMANDS, PROFITS
        print('\nAlt  Name            Supplier  Assortment     Price    Demand   Market share')
        for i in range(data['I_tot']):
                print(' {:2d}  {:14s}        {:2d}          {:2.0f}    {:6.2f}  {:8.2f}       {:8.4f}'
                        .format(i, data['name_mapping'][i], data['operator'][i], results['assortment'][i],
                        results['prices'][i], results['demand'][i], results['demand'][i]/data['Pop']))

        return results
    
    except CplexSolverError:
        raise Exception('Exception raised during solve')


if __name__ == '__main__':

    nSimulations = 1

    for seed in range(1,nSimulations+1):

        print('\n\n\n\n\n---------\nSEED ={:3d}\n---------\n\n'.format(seed))

        t_preproc_start = time.time()
        
        data = data_file.getData(seed)
        data_file.preprocessUtilities(data)
        functions.nonNegativeUtilities(data)

        t_preproc_end = time.time()

        #####################################
        # CONTINUOUS
        #####################################
        print('\n -- CONTINUOUS PRICES -- ')
        t_cont_start = time.time()

        #Calculate utility bounds
        update_bounds.updateUtilityBounds(data)

        #Solve choice-based optimization problem
        model = getModel(data)
        results = solveModel(data, model)

        t_cont_end = time.time()

        #####################################
        # DISCRETE
        #####################################
        print('\n -- DISCRETIZED PRICES -- ')
        t_discr_start = time.time()

        # Calculate utilities for all alternatives (1 per discrete price)
        functions.discretePriceAlternativeDuplication(data)
        functions.calcDuplicatedUtilities(data)

        #Solve choice-based optimization problem
        model = model_discrete_bigM_assortment.getModel(data)
        results = model_discrete_bigM_assortment.solveModel(data, model)

        t_discr_end = time.time()
        
        print('\n ------ TIMING ------ ')
        print('Preprocessing time :                   {:10.5f} sec'
            .format(t_preproc_end - t_preproc_start))
        print('Model with continuous prices :         {:10.5f} sec'
            .format(t_cont_end - t_cont_start))
        print('Model with discretized prices :        {:10.5f} sec'
            .format(t_discr_end - t_discr_start))
        print('\n -------------------- ')