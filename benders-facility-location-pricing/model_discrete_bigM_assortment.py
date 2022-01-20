# CPLEX model for the choice-based facility location and
# pricing problem with discrete prices (big M formulation).
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
import Data_Parking_N08_I10 as data_file


def getModel(data):
    
    # Initialize the model
    t_in = time.time()
    model = cplex.Cplex()

    ##########################################
    ##### ----- OBJECTIVE FUNCTION ----- #####
    ##########################################

    # Set the objective function sense
    model.objective.set_sense(model.objective.sense.maximize)

    ##########################################
    ##### ----- DECISION VARIABLES ----- #####
    ##########################################

    # BATCH OF CHOICE AND FACILITY LOCATION VARIABLES
    objVar = []
    typeVar = []
    nameVar = []
    
    # Customer choice variables
    for i in range(data['I_tot_exp']):
        for n in range(data['N']):
            for r in range(data['R']):
                if data['operator'][data['alt'][i]] == 1:
                    objVar.append(data['p'][i] * data['popN'][n]/data['R'])
                else:
                    objVar.append(0.0)
                typeVar.append(model.variables.type.binary)
                nameVar.append('x[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')

    # Facility location variables
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

    # BATCH OF UTILITY VARIABLES
    typeVar = []
    nameVar = []
    lbVar = []
    ubVar = []

    # Utility variables
    for i in range(data['I_tot_exp']):
        for n in range(data['N']):
            for r in range(data['R']):
                typeVar.append(model.variables.type.continuous)
                nameVar.append('U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                lbVar.append(-cplex.infinity)
                ubVar.append(cplex.infinity)

    # Maximum utility for each customer and draw
    for n in range(data['N']):
        for r in range(data['R']):
            typeVar.append(model.variables.type.continuous)
            nameVar.append('Umax[' + str(n) + ']' + '[' + str(r) + ']')
            lbVar.append(-cplex.infinity)
            ubVar.append(cplex.infinity)
    
    # BATCH OF DEMAND VARIABLES
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
    
    #######################################
    #### ----- Utility constraints ---- ###
    #######################################

    #### Utility constraints
    for i in range(data['I_tot_exp']):
        for n in range(data['N']):
            for r in range(data['R']):
                indicesConstr.append([nameToIndex['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['y[' + str(i) + ']']])
                coefsConstr.append([1.0, -data['U'][i,n,r]])
                sensesConstr.append('E')
                rhsConstr.append(0.0)                    
    
    #### Utility maximization: the selected alternative is the one with the highest utility
    for n in range(data['N']):
        for r in range(data['R']):
            data['UMax_n_r'] = np.max(data['U'][:,n,r])
            for i in range(data['I_tot_exp']):

                indicesConstr.append([nameToIndex['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['Umax[' + str(n) + ']' + '[' + str(r) + ']']])
                coefsConstr.append([1.0, -1.0])
                sensesConstr.append('L')
                rhsConstr.append(0.0)

                indicesConstr.append([nameToIndex['Umax[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['x[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                coefsConstr.append([1.0, -1.0, data['UMax_n_r']])
                sensesConstr.append('L')
                rhsConstr.append(data['UMax_n_r'])
    
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

        model.parameters.timelimit.set(172000.0)
        model.solve()

        ### PRINT OBJ FUNCTION
        print('Objective function value (maximum profit): {:10.4f}'
            .format(model.solution.get_objective_value()))

        ### INITIALIZE DICTIONARY OF RESULTS
        results = {}
        results['facilities'] = np.empty(data['I_tot_exp'])
        results['demand'] = np.empty(data['I_tot_exp'])
        results['profits'] = np.full([data['K']+1], 0.0)

        ### SAVE RESULTS
        for i in range(data['I_tot_exp']):
            results['facilities'][i] = model.solution.get_values('y[' + str(i) + ']')
            results['demand'][i] = model.solution.get_values('d[' + str(i) + ']')
            results['profits'][data['operator'][data['alt'][i]]] += results['demand'][i]*(data['p'][i] - data['customer_cost'][data['alt'][i]])

        ### PRINT PRICES, DEMANDS, PROFITS
        print('\nAlt  Name            Supplier    Facility      Price      Demand  Market share')
        for i in range(data['I_tot_exp']):
            if results['facilities'][i] > 0.5:
                print('{:3d}  {:14s}        {:2d}        {:4.0f}     {:6.2f}    {:8.3f}       {:7.4f}'
                    .format(i, data['name_mapping'][data['alt'][i]], data['operator'][data['alt'][i]],
                    results['facilities'][i], data['p'][i],
                    results['demand'][i], results['demand'][i] / data['Pop']))

        return results
    
    except CplexSolverError:
        raise Exception('Exception raised during solve')


if __name__ == '__main__':

    nSimulations = 1

    for seed in range(1,nSimulations+1):
        
        if nSimulations > 1:
            print('\n\n\n\n\n---------\nSEED ={:3d}\n---------\n\n'.format(seed))

        t_0 = time.time()
        
        # Read instance and print aggregate customer data
        data = data_file.getData(seed)

        # Calculate utilities for all alternatives (1 per discrete price)
        functions.discretePriceAlternativeDuplication(data)
        data_file.preprocessUtilities(data)
        functions.calcDuplicatedUtilities(data)
        
        t_1 = time.time()

        #Solve choice-based optimization problem
        model = getModel(data)
        results = solveModel(data, model)

        t_2 = time.time()
        
        print('\n -- TIMING -- ')
        print('Get data + Preprocess: {:10.5f} sec'.format(t_1 - t_0))
        print('Run the model:         {:10.5f} sec'.format(t_2 - t_1))
        print('\n ------------ ')
