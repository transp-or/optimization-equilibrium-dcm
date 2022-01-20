# CPLEX model for the choice-based facility location
# and pricing problem with discrete prices (compact formulation)
# Alternatives are duplicated to account for different possible price levels.

# General
import time
import numpy as np

# CPLEX
import cplex
from cplex.exceptions import CplexSolverError

# Project
import functions

# Data
#import Data_Parking_N80_I14 as data_file
#import Data_Parking_N08_I10 as data_file
import Data_Parking_N08_I10_FC as data_file


def getModel(data):

    # Initialize the model
    t_in = time.time()
    model = cplex.Cplex()

    # Set number of threads
    #model.parameters.threads.set(model.get_num_cores())
    model.parameters.threads.set(1)
    print('\n############\nTHREADS = ', end='')
    print(model.parameters.threads.get())
    print('############\n')

    ##########################################
    ##### ----- OBJECTIVE FUNCTION ----- #####
    ##########################################
    model.objective.set_sense(model.objective.sense.maximize)

    ##########################################
    ##### ----- DECISION VARIABLES ----- #####
    ##########################################

    # Customer choice variables
    objVar = []
    typeVar = []
    nameVar = []
    lbVar = []
    ubVar = []
    
    for i in range(data['I_tot_exp']):
        for n in range(data['N']):
            for r in range(data['R']):
                if data['operator'][data['alt'][i]] == 1:
                    objVar.append((data['p'][i] - data['customer_cost'][data['alt'][i]]) * data['popN'][n]/data['R'])
                else:
                    objVar.append(0.0)
                typeVar.append(model.variables.type.binary)
                nameVar.append('x[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                lbVar.append(0.0)
                ubVar.append(1.0)

    model.variables.add(obj = [objVar[i] for i in range(len(objVar))],
                        types = [typeVar[i] for i in range(len(typeVar))],
                        lb = [lbVar[i] for i in range(len(typeVar))],
                        ub = [ubVar[i] for i in range(len(typeVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])

    # Facility location variables
    objVar = []
    typeVar = []
    nameVar = []
    
    for i in range(data['I_tot_exp']):
        if data['operator'][data['alt'][i]] == 1:
            objVar.append(-data['fixed_cost'][data['alt'][i]])
        else:
            objVar.append(0.0)
        typeVar.append(model.variables.type.binary)
        nameVar.append('y[' + str(i) + ']')

    model.variables.add(obj = [objVar[i] for i in range(len(objVar))],
                        types = [typeVar[i] for i in range(len(typeVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])

    # Auxiliary demand variables
    typeVar = []
    nameVar = []
    lbVar = []
    ubVar = []
    
    for i in range(data['I_tot_exp']):
        typeVar.append(model.variables.type.continuous)
        nameVar.append('d[' + str(i) + ']')
        lbVar.append(0.0)
        ubVar.append(data['Pop'])
    
    model.variables.add(types = [typeVar[i] for i in range(len(typeVar))],
                        lb = [lbVar[i] for i in range(len(typeVar))],
                        ub = [ubVar[i] for i in range(len(typeVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])

    print('\nCPLEX model: all decision variables added. N variables: %r. Time: %r'\
          %(model.variables.get_num(), round(time.time()-t_in,2)))

    # Creating a dictionary that maps variable names to indices, to speed up constraints creation
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

    ### --- Instance-specific constraints on the binary variables --- ###
    for i in range(data['I_out_exp']):
        indicesConstr.append([nameToIndex['y[' + str(i) + ']']])
        coefsConstr.append([1.0])
        sensesConstr.append('E')
        rhsConstr.append(1.0)
    
    ### --- Choose at most one price level per alternative --- ###
    for alt in range(data['I_opt_out'], data['I_tot']):
        ind = []
        co = []
        for i in range(data['I_out_exp'], data['I_tot_exp']):
            if data['alt'][i] == alt:
                ind.append(nameToIndex['y[' + str(i) + ']'])
                co.append(1.0)
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('L')
        rhsConstr.append(1.0)        

    ###################################################
    ### ------------ Choice constraints ----------- ###
    ###################################################

    # Each customer chooses one alternative
    for n in range(data['N']):
        for r in range(data['R']):
            ind = []
            co = []
            for i in range(data['I_tot_exp']):
                ind.append(nameToIndex['x[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                co.append(1.0)
            indicesConstr.append(ind)
            coefsConstr.append(co)
            sensesConstr.append('E')
            rhsConstr.append(1.0)

    # A customer cannot choose an alternative that is not offered
    for i in range(data['I_tot_exp']):
        for n in range(data['N']):
            for r in range(data['R']):
                indicesConstr.append([nameToIndex['x[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['y[' + str(i) + ']']])
                coefsConstr.append([1.0, -1.0])
                sensesConstr.append('L')
                rhsConstr.append(0.0)
                    
    # A customer chooses the alternative with the highest utility
    for i in range(data['I_tot_exp']):
        for n in range(data['N']):
            for r in range(data['R']):
                ind = []
                co = []
                for j in range(data['I_tot_exp']):
                    ind.append(nameToIndex['x[' + str(j) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                    co.append(data['U'][j,n,r])
                ind.append(nameToIndex['y[' + str(i) + ']'])
                co.append(-data['U'][i,n,r])
                indicesConstr.append(ind)
                coefsConstr.append(co)
                sensesConstr.append('G')
                rhsConstr.append(0.0)

    #######################################
    #### ---- Auxiliary constraints --- ###
    #######################################
    
    ### Calculating demands (not part of the model)
    for i in range(data['I_tot_exp']):
        ind = []
        co = []
        for n in range(data['N']):
            for r in range(data['R']):
                ind.append(nameToIndex['x[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
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

        model.solve()

        ### PRINT OBJ FUNCTION
        print('Objective function value (maximum profit): {:10.4f}'.format(model.solution.get_objective_value()))

        ### INITIALIZE DICTIONARY OF RESULTS AND SAVE RESULTS
        results = {}
        results['facilities'] = np.empty(data['I_tot_exp'])
        results['demand'] = np.empty(data['I_tot_exp'])
        for i in range(data['I_tot_exp']):
            results['facilities'][i] = model.solution.get_values('y[' + str(i) + ']')
            results['demand'][i] = model.solution.get_values('d[' + str(i) + ']')

        ### PRINT PRICES, DEMANDS, PROFITS
        print('\nAlt  Name    Supplier   Facility     Price      Demand  Market share')
        for i in range(data['I_tot_exp']):
            print('{:3d}  {:6s}       {:2d}        {:4.0f}     {:6.4f}    {:8.3f}       {:7.4f}'
                .format(i, data['name_mapping'][data['alt'][i]], data['operator'][data['alt'][i]],
                results['facilities'][i], data['p'][i],
                results['demand'][i], results['demand'][i] / data['Pop']))

        return results
    
    except CplexSolverError:
        raise Exception('Exception raised during solve')


def modelOneScenario(data, r):

    print('SCENARIO {:4d}    '.format(r), end='')

    model = cplex.Cplex()

    ##########################################
    ##### ----- OBJECTIVE FUNCTION ----- #####
    ##########################################
    model.objective.set_sense(model.objective.sense.maximize)

    ##########################################
    ##### ----- DECISION VARIABLES ----- #####
    ##########################################

    # Customer choice variables
    objVar = []
    typeVar = []
    nameVar = []
    lbVar = []
    ubVar = []
    
    for i in range(data['I_tot_exp']):
        for n in range(data['N']):
            if data['operator'][data['alt'][i]] == 1:
                objVar.append((data['p'][i] - data['customer_cost'][data['alt'][i]]) * data['popN'][n])
            else:
                objVar.append(0.0)
            typeVar.append(model.variables.type.binary)
            nameVar.append('x[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
            lbVar.append(0.0)
            ubVar.append(1.0)

    model.variables.add(obj = [objVar[i] for i in range(len(objVar))],
                        types = [typeVar[i] for i in range(len(typeVar))],
                        lb = [lbVar[i] for i in range(len(typeVar))],
                        ub = [ubVar[i] for i in range(len(typeVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])

    # Facility location variables
    objVar = []
    typeVar = []
    nameVar = []
    
    for i in range(data['I_tot_exp']):
        if data['operator'][data['alt'][i]] == 1:
            objVar.append(-data['fixed_cost'][data['alt'][i]])
        else:
            objVar.append(0.0)
        typeVar.append(model.variables.type.binary)
        nameVar.append('y[' + str(i) + ']')

    model.variables.add(obj = [objVar[i] for i in range(len(objVar))],
                        types = [typeVar[i] for i in range(len(typeVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])

    # Creating a dictionary that maps variable names to indices, to speed up constraints creation
    nameToIndex = { n : j for j, n in enumerate(model.variables.get_names()) }

    #########################################
    ##### -------- CONSTRAINTS -------- #####
    #########################################

    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []
    
    ### --- Instance-specific constraints on the binary variables --- ###
    for i in range(data['I_out_exp']):
        indicesConstr.append([nameToIndex['y[' + str(i) + ']']])
        coefsConstr.append([1.0])
        sensesConstr.append('E')
        rhsConstr.append(1.0)
    
    for i in range(data['I_out_exp'], data['I_tot_exp']):
        if data['list_open'][i] == 1:
            indicesConstr.append([nameToIndex['y[' + str(i) + ']']])
            coefsConstr.append([1.0])
            sensesConstr.append('E')
            rhsConstr.append(1.0)

    ### --- Choose at most one price level per alternative --- ###
    for alt in range(data['I_opt_out'], data['I_tot']):
        ind = []
        co = []
        for i in range(data['I_out_exp'], data['I_tot_exp']):
            if data['alt'][i] == alt:
                ind.append(nameToIndex['y[' + str(i) + ']'])
                co.append(1.0)
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('L')
        rhsConstr.append(1.0)        

    # Each customer chooses one alternative
    for n in range(data['N']):
        ind = []
        co = []
        for i in range(data['I_tot_exp']):
            ind.append(nameToIndex['x[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
            co.append(1.0)
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(1.0)

    # A customer cannot choose an alternative that is not offered
    for i in range(data['I_tot_exp']):
        for n in range(data['N']):
            indicesConstr.append([nameToIndex['x[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                  nameToIndex['y[' + str(i) + ']']])
            coefsConstr.append([1.0, -1.0])
            sensesConstr.append('L')
            rhsConstr.append(0.0)
                    
    # A customer chooses the alternative with the highest utility
    for i in range(data['I_tot_exp']):
        for n in range(data['N']):
            ind = []
            co = []
            for j in range(data['I_tot_exp']):
                ind.append(nameToIndex['x[' + str(j) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                co.append(data['U'][j,n,r])
            ind.append(nameToIndex['y[' + str(i) + ']'])
            co.append(-data['U'][i,n,r])
            indicesConstr.append(ind)
            coefsConstr.append(co)
            sensesConstr.append('G')
            rhsConstr.append(0.0)
    
    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])

    #########################################
    ##### ----------- SOLVE ----------- #####
    #########################################
    
    try:
        model.set_results_stream(None)
        #model.set_warning_stream(None)

        model.parameters.timelimit.set(172000.0)
        model.solve()
        
        OF = model.solution.get_objective_value()
        y = np.empty(data['I_tot_exp'])
        for i in range(data['I_tot_exp']):
            y[i] = model.solution.get_values('y[' + str(i) + ']')
        
        print('    OF {:10.4f}'.format(OF))
    
    except CplexSolverError:
        raise Exception('Exception raised during solve')

    return OF, y


def modelOneCustomer(data, n):

    print('CUSTOMER {:4d}    '.format(n), end='')

    model = cplex.Cplex()

    ##########################################
    ##### ----- OBJECTIVE FUNCTION ----- #####
    ##########################################
    model.objective.set_sense(model.objective.sense.maximize)

    ##########################################
    ##### ----- DECISION VARIABLES ----- #####
    ##########################################

    # Customer choice variables
    objVar = []
    typeVar = []
    nameVar = []
    lbVar = []
    ubVar = []
    
    for i in range(data['I_tot_exp']):
        for r in range(data['R']):
            if data['operator'][data['alt'][i]] == 1:
                objVar.append((data['p'][i] - data['customer_cost'][data['alt'][i]]) * data['popN'][n] / data['R'])
            else:
                objVar.append(0.0)
            typeVar.append(model.variables.type.binary)
            nameVar.append('x[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
            lbVar.append(0.0)
            ubVar.append(1.0)

    model.variables.add(obj = [objVar[i] for i in range(len(objVar))],
                        types = [typeVar[i] for i in range(len(typeVar))],
                        lb = [lbVar[i] for i in range(len(typeVar))],
                        ub = [ubVar[i] for i in range(len(typeVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])

    # Facility location variables (fixed cost is adapted to F_i / |N|)
    objVar = []
    typeVar = []
    nameVar = []
    
    for i in range(data['I_tot_exp']):
        if data['operator'][data['alt'][i]] == 1:
            objVar.append(-data['fixed_cost'][data['alt'][i]] * data['popN'][n] / data['Pop'])
        else:
            objVar.append(0.0)
        typeVar.append(model.variables.type.binary)
        nameVar.append('y[' + str(i) + ']')

    model.variables.add(obj = [objVar[i] for i in range(len(objVar))],
                        types = [typeVar[i] for i in range(len(typeVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])

    # Creating a dictionary that maps variable names to indices, to speed up constraints creation
    nameToIndex = { n : j for j, n in enumerate(model.variables.get_names()) }

    #########################################
    ##### -------- CONSTRAINTS -------- #####
    #########################################

    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []
    
    ### --- Instance-specific constraints on the binary variables --- ###
    for i in range(data['I_out_exp']):
        indicesConstr.append([nameToIndex['y[' + str(i) + ']']])
        coefsConstr.append([1.0])
        sensesConstr.append('E')
        rhsConstr.append(1.0)
    
    for i in range(data['I_out_exp'], data['I_tot_exp']):
        if data['list_open'][i] == 1:
            indicesConstr.append([nameToIndex['y[' + str(i) + ']']])
            coefsConstr.append([1.0])
            sensesConstr.append('E')
            rhsConstr.append(1.0)

    ### --- Choose at most one price level per alternative --- ###
    for alt in range(data['I_opt_out'], data['I_tot']):
        ind = []
        co = []
        for i in range(data['I_out_exp'], data['I_tot_exp']):
            if data['alt'][i] == alt:
                ind.append(nameToIndex['y[' + str(i) + ']'])
                co.append(1.0)
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('L')
        rhsConstr.append(1.0)        

    # The customer chooses one alternative
    for r in range(data['R']):
        ind = []
        co = []
        for i in range(data['I_tot_exp']):
            ind.append(nameToIndex['x[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
            co.append(1.0)
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(1.0)

    # The customer cannot choose an alternative that is not offered
    for i in range(data['I_tot_exp']):
        for r in range(data['R']):
            indicesConstr.append([nameToIndex['x[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                  nameToIndex['y[' + str(i) + ']']])
            coefsConstr.append([1.0, -1.0])
            sensesConstr.append('L')
            rhsConstr.append(0.0)
                    
    # The customer chooses the alternative with the highest utility
    for i in range(data['I_tot_exp']):
        for r in range(data['R']):
            ind = []
            co = []
            for j in range(data['I_tot_exp']):
                ind.append(nameToIndex['x[' + str(j) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                co.append(data['U'][j,n,r])
            ind.append(nameToIndex['y[' + str(i) + ']'])
            co.append(-data['U'][i,n,r])
            indicesConstr.append(ind)
            coefsConstr.append(co)
            sensesConstr.append('G')
            rhsConstr.append(0.0)
    
    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])

    #########################################
    ##### ----------- SOLVE ----------- #####
    #########################################
    
    try:
        model.set_results_stream(None)
        #model.set_warning_stream(None)

        model.parameters.timelimit.set(172000.0)
        model.solve()
        
        OF = model.solution.get_objective_value()
        y = np.empty(data['I_tot_exp'])
        for i in range(data['I_tot_exp']):
            y[i] = model.solution.get_values('y[' + str(i) + ']')
        
        print('    OF {:10.4f}'.format(OF))
    
    except CplexSolverError:
        raise Exception('Exception raised during solve')

    return OF, y


if __name__ == '__main__':

    nSimulations = 1

    for seed in range(1,nSimulations+1):
        
        if nSimulations > 1:
            print('\n\n\n\n\n---------\nSEED ={:3d}\n---------\n\n'.format(seed))
        
        t_0 = time.time()
        
        # Read instance and print aggregate customer data
        data = data_file.getData(seed)

        data_file.printCustomers(data)

        # Calculate utilities for all alternatives (1 per discrete price)
        functions.discretePriceAlternativeDuplication(data)
        data_file.preprocessUtilities(data)
        functions.calcDuplicatedUtilities(data)
        
        t_1 = time.time()

        #Solve choice-based optimization problem
        model = getModel(data)
        #model.parameters.preprocessing.presolve.set(0)
        results = solveModel(data, model)

        t_2 = time.time()
        
        print('\n -- TIMING -- ')
        print('Get data + Preprocess: {:10.5f} sec'.format(t_1 - t_0))
        print('Run the model:         {:10.5f} sec'.format(t_2 - t_1))
        print('\n ------------ ')
