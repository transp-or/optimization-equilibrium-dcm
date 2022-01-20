'''
Fixed-point MIP model (14)-(30) from the following article:
S. Bortolomiol, V. Lurkin, M. Bierlaire (2021).
A Simulation-Based Heuristic to Find Approximate Equilibria with
Disaggregate Demand Models. Transportation Science 55(5):1025-1045.
https://doi.org/10.1287/trsc.2021.1071
'''

# General
import sys
import time
import copy
import numpy as np

# CPLEX
import cplex
from cplex.exceptions import CplexSolverError

# Project
import generate_strategies
import update_bounds
import choice_preprocess
import nested_logit

# Data
import data_HSR as data_file


def getModel(data):
    '''
    CPLEX model for the fixed-point MIP model
    (multiple suppliers, finite strategy sets)
    '''
    
    print('\n\nFIXED-POINT MIP MODEL:')
    t_in = time.time()
    
    model = cplex.Cplex() # Initialize the model

    # Set time limit
    model.parameters.timelimit.set(600.0)
    # Emphasize feasibility over proof of optimality
    model.parameters.emphasis.mip.set(1)
    
    data['scenarios'] = {}
    data['scenarios'] = copy.deepcopy(data['strategies'])
    
    
    ##########################################
    ##### ----- OBJECTIVE FUNCTION ----- #####
    ##########################################
    model.objective.set_sense(model.objective.sense.minimize)


    ##########################################
    ##### ---------- VARIABLES --------- #####
    ##########################################

    # ------------------------------------------------- #
    # ----- LEVEL 1 : SUPPLIER DECISION VARIABLES ----- #
    # ------------------------------------------------- #

    ### --- Strategy picked by each operator (strategy = a bundle of decisions) --- ###
    typeVar = []
    nameVar = []

    # INITIAL
    for k in range(1, data['K'] + 1):
        for l in data['list_strategies_opt'][k]:
            typeVar.append(model.variables.type.binary)
            nameVar.append('x_In[' + str(k) + ']' + '[' + str(l) + ']')
    # FINAL
    for k in range(1, data['K'] + 1):
        for l in data['list_strategies_opt'][k]:
            typeVar.append(model.variables.type.binary)
            nameVar.append('x_Fin[' + str(k) + ']' + '[' + str(l) + ']')

    model.variables.add(types = [typeVar[i] for i in range(len(typeVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])


    objVar = []
    typeVar = []
    nameVar = []
    lbVar = []
    ubVar = []

    ### --- Maximum revenue for each operator --- ###

    # INITIAL
    # Not needed

    # FINAL
    for k in range(1, data['K'] + 1):
        objVar.append(1.0)
        typeVar.append(model.variables.type.continuous)
        nameVar.append('revenueMax_Fin[' + str(k) + ']')
        lbVar.append(-cplex.infinity)
        ubVar.append(data['M_Rev'][k])

    ### --- Revenue for each operator for each strategy  --- ###

    # INITIAL: strategies are implicit = 1 index
    for k in range(1, data['K'] + 1):
        objVar.append(-1.0)
        typeVar.append(model.variables.type.continuous)
        nameVar.append('revenue_In[' + str(k) + ']')
        lbVar.append(-cplex.infinity)
        ubVar.append(data['M_Rev'][k])
    # FINAL: strategies are explicit = 2 indexes
    for k in range(1, data['K'] + 1):
        for l in range(data['tot_strategies']):
            objVar.append(0.0)
            typeVar.append(model.variables.type.continuous)
            nameVar.append('revenue_Fin[' + str(k) + ']' + '[' + str(l) + ']')
            lbVar.append(-cplex.infinity)
            ubVar.append(data['M_Rev'][k])

    model.variables.add(obj = [objVar[i] for i in range(len(objVar))],
                        types = [typeVar[i] for i in range(len(typeVar))],
                        lb = [lbVar[i] for i in range(len(lbVar))],
                        ub = [ubVar[i] for i in range(len(ubVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])

    typeVar = []
    nameVar = []
    lbVar = []
    ubVar = []

    ### --- Prices --- ###

    # INITIAL
    for i in range(data['I_tot']):
        typeVar.append(model.variables.type.continuous)
        nameVar.append('price_urban_In[' + str(i) + ']')
        lbVar.append(data['lb_p_urban'][i])
        ubVar.append(data['ub_p_urban'][i])
        typeVar.append(model.variables.type.continuous)
        nameVar.append('price_rural_In[' + str(i) + ']')
        lbVar.append(data['lb_p_rural'][i])
        ubVar.append(data['ub_p_rural'][i])
    # FINAL
    for i in range(data['I_tot']):
        for l in range(data['tot_strategies']):
            typeVar.append(model.variables.type.continuous)
            nameVar.append('price_urban_Fin[' + str(i) + ']' + '[' + str(l) + ']')
            lbVar.append(data['lb_p_urban'][i])
            ubVar.append(data['ub_p_urban'][i])
            typeVar.append(model.variables.type.continuous)
            nameVar.append('price_rural_Fin[' + str(i) + ']' + '[' + str(l) + ']')
            lbVar.append(data['lb_p_rural'][i])
            ubVar.append(data['ub_p_rural'][i])

    ### --- Linearized choice-price variables ---###

    # INITIAL
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                if data['w_pre'][i,n,r] != 0:

                    typeVar.append(model.variables.type.continuous)
                    nameVar.append('alpha_In[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                    lbVar.append(0.0)
                    if data['ORIGIN'][n] == 1:
                        ubVar.append(data['ub_p_urban'][i])
                    elif data['ORIGIN'][n] == 0:
                        ubVar.append(data['ub_p_rural'][i])
    
                    # FINAL
                    for l in range(data['tot_strategies']):
                        if data['w_pre_strategies'][i,n,r,l] != 0:
                            typeVar.append(model.variables.type.continuous)
                            nameVar.append('alpha_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')
                            lbVar.append(0.0)
                            if data['ORIGIN'][n] == 1:
                                ubVar.append(data['ub_p_urban'][i])
                            elif data['ORIGIN'][n] == 0:
                                ubVar.append(data['ub_p_rural'][i])

    model.variables.add(types = [typeVar[i] for i in range(len(typeVar))],
                        lb = [lbVar[i] for i in range(len(lbVar))],
                        ub = [ubVar[i] for i in range(len(ubVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])


    # ------------------------------------------------- #
    # ----- LEVEL 2 : CUSTOMER DECISION VARIABLES ----- #
    # ------------------------------------------------- #

    # CHOICE VARIABLES
    typeVar = []
    nameVar = []

    # INITIAL
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                if data['w_pre'][i,n,r] != 0:
                    typeVar.append(model.variables.type.binary)
                    nameVar.append('w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
    # FINAL
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                for l in range(data['tot_strategies']):
                    if data['w_pre_strategies'][i,n,r,l] != 0:
                        typeVar.append(model.variables.type.binary)
                        nameVar.append('w_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')

    model.variables.add(types = [typeVar[i] for i in range(len(typeVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])

    # BATCH OF UTILITY VARIABLES
    typeVar = []
    nameVar = []
    lbVar = []
    ubVar = []

    ## INITIAL ITERATION

    # Customer utility
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                if data['w_pre'][i,n,r] != 0 and data['w_pre'][i,n,r] != 1:
                    typeVar.append(model.variables.type.continuous)
                    nameVar.append('U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                    lbVar.append(data['lb_U'][i,n,r])
                    ubVar.append(data['ub_U'][i,n,r])

    # Maximum utility
    for n in range(data['N']):
        for r in range(data['R']):
            if np.max(data['w_pre'][:,n,r]) < 0.5:
                typeVar.append(model.variables.type.continuous)
                nameVar.append('UMax[' + str(n) + ']' + '[' + str(r) + ']')
                lbVar.append(-cplex.infinity)
                ubVar.append(cplex.infinity)

    ## FINAL ITERATION

    # Customer utility
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                for l in range(data['tot_strategies']):
                    if data['w_pre_strategies'][i,n,r,l] != 0 and data['w_pre'][i,n,r] != 1:
                        typeVar.append(model.variables.type.continuous)
                        nameVar.append('U_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')
                        lbVar.append(data['lb_U'][i,n,r])
                        ubVar.append(data['ub_U'][i,n,r])

    # Maximum utility
    for n in range(data['N']):
        for r in range(data['R']):
            for l in range(data['tot_strategies']):
                if np.max(data['w_pre'][:,n,r]) < 0.5:
                    typeVar.append(model.variables.type.continuous)
                    nameVar.append('UMax_Fin[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']')
                    lbVar.append(-cplex.infinity)
                    ubVar.append(cplex.infinity)

    model.variables.add(types = [typeVar[i] for i in range(len(typeVar))],
                        lb = [lbVar[i] for i in range(len(lbVar))],
                        ub = [ubVar[i] for i in range(len(ubVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])

    typeVar = []
    nameVar = []
    lbVar = []
    ubVar = []

    ## AUXILIARY VARIABLES
    # Demand (initial iteration)
    for i in range(data['I_tot']):
        typeVar.append(model.variables.type.continuous)
        nameVar.append('demand[' + str(i) + ']')
        lbVar.append(0.0)
        ubVar.append(data['Pop'])
        typeVar.append(model.variables.type.continuous)
        nameVar.append('demand_urban[' + str(i) + ']')
        lbVar.append(0.0)
        ubVar.append(data['Pop'])
        typeVar.append(model.variables.type.continuous)
        nameVar.append('demand_rural[' + str(i) + ']')
        lbVar.append(0.0)
        ubVar.append(data['Pop'])
        # Demand (final iteration)
        for l in range(data['tot_strategies']):
            typeVar.append(model.variables.type.continuous)
            nameVar.append('demand_Fin[' + str(i) + ']' + '[' + str(l) + ']')
            lbVar.append(0.0)
            ubVar.append(data['Pop'])

    model.variables.add(types = [typeVar[i] for i in range(len(typeVar))],
                        lb = [lbVar[i] for i in range(len(lbVar))],
                        ub = [ubVar[i] for i in range(len(ubVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])
    print('\nCPLEX model: all decision variables added. N variables: %r. Time: %r'\
          %(model.variables.get_num(), round(time.time()-t_in,2)))

    # Creating a dictionary that maps variable names to indices, to speed up constraints creation
    # https://www.ibm.com/developerworks/community/forums/html/topic?id=2349f613-26b1-4c29-aa4d-b52c9505bf96
    nameToIndex = { n : j for j, n in enumerate(model.variables.get_names()) }


    # --------------------------------------------------#
    # ------------------ CONSTRAINTS -------------------#
    # --------------------------------------------------#

    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []

    # ------------------------------------------------- #
    # ---- LEVEL 1 : SUPPLIER UTILITY MAXIMIZATION ---- #
    # ------------------------------------------------- #

    ### Prices

    # INITIAL
    # Endogenous alternatives: prices are derived from x_In
    for i in range(data['I_opt_out'], data['I_tot']):
        k = data['operator'][i]
        ind = [nameToIndex['price_urban_In[' + str(i) + ']']]
        co = [1.0]
        for l in data['list_strategies_opt'][k]:
            ind.append(nameToIndex['x_In[' + str(k) + ']' + '[' + str(l) + ']'])
            co.append(-data['scenarios']['prices_urban'][i][l])
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(0.0)
        ind = [nameToIndex['price_rural_In[' + str(i) + ']']]
        co = [1.0]
        for l in data['list_strategies_opt'][k]:
            ind.append(nameToIndex['x_In[' + str(k) + ']' + '[' + str(l) + ']'])
            co.append(-data['scenarios']['prices_rural'][i][l])
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(0.0)
    # For opt-out options the price is fixed
    for i in range(data['I_opt_out']):
        indicesConstr.append([nameToIndex['price_urban_In[' + str(i) + ']']])
        coefsConstr.append([1.0])
        sensesConstr.append('E')
        rhsConstr.append(data['p_urban_fixed'][i])
        indicesConstr.append([nameToIndex['price_rural_In[' + str(i) + ']']])
        coefsConstr.append([1.0])
        sensesConstr.append('E')
        rhsConstr.append(data['p_rural_fixed'][i])

    #FINAL
    # We need to retrieve the price of each alternative i in each scenario l
    for l in range(data['tot_strategies']):
        for i in range(data['I_opt_out'], data['I_tot']):
            # URBAN
            ind = [nameToIndex['price_urban_Fin[' + str(i) + ']' + '[' + str(l) + ']']]
            co = [1.0]
            sensesConstr.append('E')
            # If the optimizer in scenario l is the owner of alternative i,
            # then the price is given as parameter "data['scenarios']['prices'][i][l]"
            # and the constraint becomes a trivial equality
            if data['scenarios']['optimizer'][l] == data['operator'][i]:
                rhsConstr.append(data['scenarios']['prices_urban'][i][l])
            # Else, we sum over all scenarios where supplier
            # "k = data['operator'][i]" is the optimizer
            # to find the scenario chosen in the initial solution
            else:
                for s in data['list_strategies_opt'][data['operator'][i]]:
                    ind.append(nameToIndex['x_In[' + str(data['operator'][i]) + ']' + '[' + str(s) + ']'])
                    co.append(-data['scenarios']['prices_urban'][i][s])
                rhsConstr.append(0.0)
            indicesConstr.append(ind)
            coefsConstr.append(co)
            # RURAL
            ind = [nameToIndex['price_rural_Fin[' + str(i) + ']' + '[' + str(l) + ']']]
            co = [1.0]
            sensesConstr.append('E')
            # If the optimizer in scenario l is the owner of alternative i,
            # then the price is given as parameter "data['scenarios']['prices'][i][l]"
            # and the constraint becomes a trivial equality
            if data['scenarios']['optimizer'][l] == data['operator'][i]:
                rhsConstr.append(data['scenarios']['prices_rural'][i][l])
            # Else, we sum over all scenarios where supplier
            # "k = data['operator'][i]" is the optimizer
            # to find the scenario chosen in the initial solution
            else:
                for s in data['list_strategies_opt'][data['operator'][i]]:
                    ind.append(nameToIndex['x_In[' + str(data['operator'][i]) + ']' + '[' + str(s) + ']'])
                    co.append(-data['scenarios']['prices_rural'][i][s])
                rhsConstr.append(0.0)
            indicesConstr.append(ind)
            coefsConstr.append(co)
        # For opt-out options the price is fixed
        for i in range(data['I_opt_out']):
            indicesConstr.append([nameToIndex['price_urban_Fin[' + str(i) + ']' + '[' + str(l) + ']']])
            coefsConstr.append([1.0])
            sensesConstr.append('E')
            rhsConstr.append(data['p_urban_fixed'][i])
            indicesConstr.append([nameToIndex['price_rural_Fin[' + str(i) + ']' + '[' + str(l) + ']']])
            coefsConstr.append([1.0])
            sensesConstr.append('E')
            rhsConstr.append(data['p_rural_fixed'][i])


    ### Price-choice constraints

    # INITIAL
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                if data['w_pre'][i,n,r] != 0 and data['w_pre'][i,n,r] != 1:
                    # Linearized price-choice: variable equal to 0 if alternative is not chosen
                    indicesConstr.append([nameToIndex['alpha_In[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                          nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                    if data['ORIGIN'][n] == 1:                      
                        coefsConstr.append([1.0, -data['ub_p_urban'][i]])
                    elif data['ORIGIN'][n] == 0:                      
                        coefsConstr.append([1.0, -data['ub_p_rural'][i]])
                    sensesConstr.append('L')
                    rhsConstr.append(0.0)
                    # Linearized price-choice: variable smaller than price
                    if data['ORIGIN'][n] == 1:
                        indicesConstr.append([nameToIndex['alpha_In[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                              nameToIndex['price_urban_In[' + str(i) + ']']])
                    elif data['ORIGIN'][n] == 0:
                        indicesConstr.append([nameToIndex['alpha_In[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                              nameToIndex['price_rural_In[' + str(i) + ']']])
                    coefsConstr.append([1.0, -1.0])
                    sensesConstr.append('L')
                    rhsConstr.append(0.0)
                    # Linearized price-choice
                    #       forall (i in Alternatives, n in Customers, r in Draws)
                    #           p[i] - (1 - w[i][n][r]) * ub_p[i] <= alpha[i][n][r];
                    if data['ORIGIN'][n] == 1:
                        indicesConstr.append([nameToIndex['alpha_In[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                              nameToIndex['price_urban_In[' + str(i) + ']'],
                                              nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                        coefsConstr.append([-1.0, 1.0, data['ub_p_urban'][i]])
                        rhsConstr.append(data['ub_p_urban'][i])
                    elif data['ORIGIN'][n] == 0:
                        indicesConstr.append([nameToIndex['alpha_In[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                              nameToIndex['price_rural_In[' + str(i) + ']'],
                                              nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                        coefsConstr.append([-1.0, 1.0, data['ub_p_rural'][i]])
                        rhsConstr.append(data['ub_p_rural'][i])
                    sensesConstr.append('L')
                    
                elif data['w_pre'][i,n,r] == 1:
                    if data['ORIGIN'][n] == 1:
                        indicesConstr.append([nameToIndex['alpha_In[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                              nameToIndex['price_urban_In[' + str(i) + ']']])
                    elif data['ORIGIN'][n] == 0:
                        indicesConstr.append([nameToIndex['alpha_In[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                              nameToIndex['price_rural_In[' + str(i) + ']']])
                    coefsConstr.append([1.0, -1.0])
                    sensesConstr.append('E')
                    rhsConstr.append(0.0)
    #FINAL
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                for l in range(data['tot_strategies']):
                    if data['w_pre_strategies'][i,n,r,l] != 0 and data['w_pre'][i,n,r] != 1:
                        # Linearized price-choice: variable equal to 0 if alternative is not chosen
                        indicesConstr.append([nameToIndex['alpha_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'],
                                              nameToIndex['w_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']])
                        if data['ORIGIN'][n] == 1:
                            coefsConstr.append([1.0, -data['ub_p_urban'][i]])
                        elif data['ORIGIN'][n] == 0:
                            coefsConstr.append([1.0, -data['ub_p_rural'][i]])
                        sensesConstr.append('L')
                        rhsConstr.append(0.0)
                        # Linearized price-choice: variable smaller than price
                        if data['ORIGIN'][n] == 1:
                            indicesConstr.append([nameToIndex['alpha_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'],
                                                  nameToIndex['price_urban_Fin[' + str(i) + ']' + '[' + str(l) + ']']])
                        elif data['ORIGIN'][n] == 0:
                            indicesConstr.append([nameToIndex['alpha_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'],
                                                  nameToIndex['price_rural_Fin[' + str(i) + ']' + '[' + str(l) + ']']])
                        coefsConstr.append([1.0, -1.0])
                        sensesConstr.append('L')
                        rhsConstr.append(0.0)
                        # Linearized price-choice
                        if data['ORIGIN'][n] == 1:
                            indicesConstr.append([nameToIndex['alpha_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'],
                                                  nameToIndex['price_urban_Fin[' + str(i) + ']' + '[' + str(l) + ']'],
                                                  nameToIndex['w_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']])
                            coefsConstr.append([-1.0, 1.0, data['ub_p_urban'][i]])
                            rhsConstr.append(data['ub_p_urban'][i])
                        elif data['ORIGIN'][n] == 0:
                            indicesConstr.append([nameToIndex['alpha_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'],
                                                  nameToIndex['price_rural_Fin[' + str(i) + ']' + '[' + str(l) + ']'],
                                                  nameToIndex['w_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']])
                            coefsConstr.append([-1.0, 1.0, data['ub_p_rural'][i]])
                            rhsConstr.append(data['ub_p_rural'][i])
                        sensesConstr.append('L')
                    elif data['w_pre'][i,n,r] == 1:
                        if data['ORIGIN'][n] == 1:
                            indicesConstr.append([nameToIndex['alpha_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'],
                                                  nameToIndex['price_urban_Fin[' + str(i) + ']' + '[' + str(l) + ']']])
                        elif data['ORIGIN'][n] == 0:
                            indicesConstr.append([nameToIndex['alpha_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'],
                                                  nameToIndex['price_rural_Fin[' + str(i) + ']' + '[' + str(l) + ']']])
                        coefsConstr.append([1.0, -1.0])
                        sensesConstr.append('E')
                        rhsConstr.append(0.0)


    ### Profit function of the operators

    # INITIAL
    for k in range(1, data['K'] + 1):
        ind = [nameToIndex['revenue_In[' + str(k) + ']']]
        co = [1.0]
        fixedcosts = 0.0
        # Sum over all alternatives controlled by supplier k
        for i in data['list_alt_supplier'][k]:
            fixedcosts += -data['fixed_cost'][i]
            for n in range(data['N']):
                for r in range(data['R']):
                    if data['w_pre'][i,n,r] != 0:
                        ind.append(nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                        co.append(data['customer_cost'][i] * data['popN'][n] / data['R'])
                        ind.append(nameToIndex['alpha_In[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                        co.append(-data['popN'][n] / data['R'])
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(fixedcosts)
    # FINAL
    for k in range(1, data['K'] + 1):
        for l in range(data['tot_strategies']):
            ind = [nameToIndex['revenue_Fin[' + str(k) + ']' + '[' + str(l) + ']']]
            co = [1.0]
            fixedcosts = 0.0
            # Sum over all alternatives controlled by supplier k
            for i in data['list_alt_supplier'][k]:
                fixedcosts += -data['fixed_cost'][i]
                for n in range(data['N']):
                    for r in range(data['R']):
                        if data['w_pre_strategies'][i,n,r,l] != 0:
                            ind.append(nameToIndex['w_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'])
                            co.append(data['customer_cost'][i] * data['popN'][n] / data['R'])
                            ind.append(nameToIndex['alpha_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'])
                            co.append(-data['popN'][n] / data['R'])
            indicesConstr.append(ind)
            coefsConstr.append(co)
            sensesConstr.append('E')
            rhsConstr.append(fixedcosts)


    ### Each supplier picks exactly one strategy.

    for k in range(1, data['K'] + 1):
        # INITIAL
        ind = []
        co = []
        for l in data['list_strategies_opt'][k]:
            ind.append(nameToIndex['x_In[' + str(k) + ']' + '[' + str(l) + ']'])
            co.append(1.0)
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(1.0)
        # FINAL
        ind = []
        co = []
        for l in data['list_strategies_opt'][k]:
            ind.append(nameToIndex['x_Fin[' + str(k) + ']' + '[' + str(l) + ']'])
            co.append(1.0)
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(1.0)


    ### The selected final strategy is the one maximizing profits (best response)

    # INITIAL
    # No constraints needed

    # FINAL
    for k in range(1, data['K'] + 1):
        for l in data['list_strategies_opt'][k]:
            # Lower bound on revenueMax_Fin
            ind = [nameToIndex['revenue_Fin[' + str(k) + ']' + '[' + str(l) + ']']]
            co = [1.0]
            ind.append(nameToIndex['revenueMax_Fin[' + str(k) + ']'])
            co.append(-1.0)
            indicesConstr.append(ind)
            coefsConstr.append(co)
            sensesConstr.append('L')
            rhsConstr.append(0.0)
            # Upper bound on revenueMax_Fin if strategy l is chosen
            ind = [nameToIndex['revenueMax_Fin[' + str(k) + ']']]
            co = [1.0]
            ind.append(nameToIndex['revenue_Fin[' + str(k) + ']' + '[' + str(l) + ']'])
            co.append(-1.0)
            ind.append(nameToIndex['x_Fin[' + str(k) + ']' + '[' + str(l) + ']'])
            co.append(data['M_Rev'][k])
            indicesConstr.append(ind)
            coefsConstr.append(co)
            sensesConstr.append('L')
            rhsConstr.append(data['M_Rev'][k])

    ### The final revenue is always greater than or equal to the initial revenue
    # (constraints not needed, but they improve the linear relaxation)
    for k in range(1, data['K'] + 1):
        indicesConstr.append([nameToIndex['revenueMax_Fin[' + str(k) + ']'],
                              nameToIndex['revenue_In[' + str(k) + ']']])
        coefsConstr.append([1.0, -1.0])
        sensesConstr.append('G')
        rhsConstr.append(0.0)


    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])

    print('CPLEX model: supplier constraints added. N constraints: %r. Time: %r'\
          %(model.linear_constraints.get_num(), round(time.time()-t_in,2)))


    ###################################################################
    ###### ------ Level 2 : CUSTOMER UTILITY MAXIMIZATION ------ ######
    ###################################################################

    #######################################
    ##### ---- Choice constraints ---- ####
    #######################################

    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []

    ### All customers choose one alternative

    for n in range(data['N']):
        for r in range(data['R']):

            # INITIAL
            ind = []
            co = []
            for i in range(data['I_tot']):
                if data['w_pre'][i,n,r] != 0:
                    ind.append(nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                    co.append(1.0)
            indicesConstr.append(ind)
            coefsConstr.append(co)
            sensesConstr.append('E')
            rhsConstr.append(1.0)

            # FINAL
            for l in range(data['tot_strategies']):
                ind = []
                co = []
                for i in range(data['I_tot']):
                    if data['w_pre_strategies'][i,n,r,l] != 0:
                        ind.append(nameToIndex['w_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'])
                        co.append(1.0)
                indicesConstr.append(ind)
                coefsConstr.append(co)
                sensesConstr.append('E')
                rhsConstr.append(1.0)

    ### All captive customers are assigned

    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                if data['w_pre'][i, n, r] == 1:

                    # INITIAL
                    indicesConstr.append([nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                    coefsConstr.append([1.0])
                    sensesConstr.append('E')
                    rhsConstr.append(data['w_pre'][i,n,r])
                    # FINAL
                    for l in range(data['tot_strategies']):
                        indicesConstr.append([nameToIndex['w_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']])
                        coefsConstr.append([1.0])
                        sensesConstr.append('E')
                        rhsConstr.append(data['w_pre'][i,n,r])

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
    
    ### Utility function constraints

    # INITIAL
    for k in range(data['K'] + 1):
        for i in data['list_alt_supplier'][k]:
            for n in range(data['N']):
                for r in range(data['R']):
                    if data['w_pre'][i,n,r] != 0 and data['w_pre'][i,n,r] != 1:
                        ind = [nameToIndex['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']]
                        co = [1.0]
                        # Add endogenous variables for endogenous alternatives
                        if k > 0:
                            for l in data['list_strategies_opt'][k]:
                                ind.append(nameToIndex['x_In[' + str(k) + ']' + '[' + str(l) + ']'])
                                if data['DCM'] == 'NestedLogit':
                                    if data['ORIGIN'][n] == 1:
                                        co.append(-data['endo_coef'][i, n] * data['scenarios']['prices_urban'][i][l])
                                    elif data['ORIGIN'][n] == 0:
                                        co.append(-data['endo_coef'][i, n] * data['scenarios']['prices_rural'][i][l])
                            # Add right-hand side according to the used DCM
                            if data['DCM'] == 'NestedLogit':
                                rhsConstr.append(data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i,n,r])
                        elif k == 0:
                            # Add right-hand side according to the used DCM
                            if data['DCM'] == 'NestedLogit':
                                rhsConstr.append(data['endo_coef'][i,n] * data['price'][i] + data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i,n,r])
                        indicesConstr.append(ind)
                        coefsConstr.append(co)
                        sensesConstr.append('E')

    # FINAL
    for l in range(data['tot_strategies']):
        for i in range(data['I_tot']):
            if data['operator'][i] == data['scenarios']['optimizer'][l]:
                for n in range(data['N']):
                    for r in range(data['R']):
                        if data['w_pre_strategies'][i,n,r,l] != 0 and data['w_pre'][i,n,r] != 1:
                            if data['ORIGIN'][n] == 1:
                                indicesConstr.append([nameToIndex['U_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'],
                                                      nameToIndex['price_urban_Fin[' + str(i) + ']' + '[' + str(l) + ']']])
                            elif data['ORIGIN'][n] == 0:
                                indicesConstr.append([nameToIndex['U_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'],
                                                      nameToIndex['price_rural_Fin[' + str(i) + ']' + '[' + str(l) + ']']])
                            sensesConstr.append('E')
                            if data['DCM'] == 'NestedLogit':
                                coefsConstr.append([1.0, -data['endo_coef'][i, n]])
                                rhsConstr.append(data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i,n,r])
            else:
                for n in range(data['N']):
                    for r in range(data['R']):
                        if data['w_pre_strategies'][i,n,r,l] != 0 and data['w_pre'][i,n,r] != 1:
                            indicesConstr.append([nameToIndex['U_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'],
                                                  nameToIndex['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                            coefsConstr.append([1.0, -1.0])
                            sensesConstr.append('E')
                            rhsConstr.append(0.0)
    

    ### Utility-choice constraints: the selected alternative is the one with maximum utility

    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                if data['w_pre'][i,n,r] != 0 and data['w_pre'][i,n,r] != 1:

                    # INITIAL
                    indicesConstr.append([nameToIndex['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                          nameToIndex['UMax[' + str(n) + ']' + '[' + str(r) + ']']])
                    coefsConstr.append([1.0, -1.0])
                    sensesConstr.append('L')
                    rhsConstr.append(0.0)
                    indicesConstr.append([nameToIndex['UMax[' + str(n) + ']' + '[' + str(r) + ']'],
                                          nameToIndex['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                          nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                    coefsConstr.append([1.0, -1.0, data['M_U'][n, r]])
                    sensesConstr.append('L')
                    rhsConstr.append(data['M_U'][n, r])

                    # FINAL
                    for l in range(data['tot_strategies']):
                        if data['w_pre_strategies'][i,n,r,l] != 0:
                            indicesConstr.append([nameToIndex['U_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'],
                                                nameToIndex['UMax_Fin[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']])
                            coefsConstr.append([1.0, -1.0])
                            sensesConstr.append('L')
                            rhsConstr.append(0.0)
                            indicesConstr.append([nameToIndex['UMax_Fin[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'],
                                                nameToIndex['U_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'],
                                                nameToIndex['w_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']']])
                            coefsConstr.append([1.0, -1.0, data['M_U'][n, r]])
                            sensesConstr.append('L')
                            rhsConstr.append(data['M_U'][n, r])


    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])
    print('CPLEX model: customer utility constraints added. N constraints: %r. Time: %r'\
          %(model.linear_constraints.get_num(), round(time.time()-t_in,2)))
    

    #######################################
    #### ---- Auxiliary constraints --- ###
    #######################################

    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []

    ### Calculating demands

    # INITIAL
    for i in range(data['I_tot']):
        ind = [nameToIndex['demand[' + str(i) + ']']]
        co = [1.0]
        for n in range(data['N']):
            for r in range(data['R']):
                if data['w_pre'][i, n, r] != 0:
                    ind.append(nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                    co.append(-data['popN'][n]/data['R'])
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(0.0)
        # URBAN
        ind = [nameToIndex['demand_urban[' + str(i) + ']']]
        co = [1.0]
        for n in range(data['N']):
            if data['ORIGIN'][n] == 1: 
                for r in range(data['R']):
                    if data['w_pre'][i, n, r] != 0:
                        ind.append(nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                        co.append(-data['popN'][n]/data['R'])
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(0.0)
        # RURAL
        ind = [nameToIndex['demand_rural[' + str(i) + ']']]
        co = [1.0]
        for n in range(data['N']):
            if data['ORIGIN'][n] == 0:
                for r in range(data['R']):
                    if data['w_pre'][i, n, r] != 0:
                        ind.append(nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                        co.append(-data['popN'][n]/data['R'])
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(0.0)

    # FINAL
    for i in range(data['I_tot']):
        for l in range(data['tot_strategies']):
            ind = [nameToIndex['demand_Fin[' + str(i) + ']' + '[' + str(l) + ']']]
            co = [1.0]
            for n in range(data['N']):
                for r in range(data['R']):
                    if data['w_pre_strategies'][i,n,r,l] != 0:
                        ind.append(nameToIndex['w_Fin[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']' + '[' + str(l) + ']'])
                        co.append(-data['popN'][n]/data['R'])
            indicesConstr.append(ind)
            coefsConstr.append(co)
            sensesConstr.append('E')
            rhsConstr.append(0.0)

    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])
    print('CPLEX model: all constraints added. N constraints: %r. Time: %r\n'\
          %(model.linear_constraints.get_num(), round(time.time()-t_in,2)))

    return model, nameToIndex


#######################################################
######### SOLVE MODEL, SAVE RESULTS AND PRINT #########
#######################################################

def solveModel(model, nameToIndex, data):

    t_in = time.time()

    #model.set_results_stream(None)
    model.set_warning_stream(None)
    
    model.solve()

    print('Constraints: {:8.0f}\nVariables  : {:8.0f}'.format(model.linear_constraints.get_num(), model.variables.get_num()))

    ### SAVE THE RESULTS OBTAINED FROM THE MODEL IN A DICTIONARY OF RESULTS
    results = {}
    results['prices_urban_in'] = copy.deepcopy(data['price'])
    results['prices_urban_fin'] = copy.deepcopy(data['price'])
    results['prices_rural_in'] = copy.deepcopy(data['price'])
    results['prices_rural_fin'] = copy.deepcopy(data['price'])
    results['prices_scenarios'] = np.full([data['I_tot'], data['tot_strategies']], -1.0)
    results['demand'] = np.full([data['I_tot']], -1.0)
    results['demand_urban'] = np.full([data['I_tot']], -1.0)
    results['demand_rural'] = np.full([data['I_tot']], -1.0)
    results['demand_scenarios'] = np.full([data['I_tot'], data['tot_strategies']], -1.0)
    results['max_profit_in'] = np.full(data['K']+1, -1.0)
    results['max_profit_fin'] = np.full(data['K']+1, -1.0)
    results['profits_scenarios'] = np.full([data['K']+1, data['tot_strategies']], -1.0)
    results['supply_choice_in'] = np.full([data['K']+1, data['tot_strategies']], -1.0)
    results['supply_choice_fin'] = np.full([data['K']+1, data['tot_strategies']], -1.0)
    results['modeshare'] = np.full([4], 0.0)

    # Obj value
    results['obj'] = model.solution.get_objective_value()

    # Supply choices
    for k in range(1, data['K'] + 1):
        for l in data['list_strategies_opt'][k]:
            results['supply_choice_in'][k][l] = model.solution.get_values(nameToIndex['x_In[' + str(k) + ']' + '[' + str(l) + ']'])
            results['supply_choice_fin'][k][l] = model.solution.get_values(nameToIndex['x_Fin[' + str(k) + ']' + '[' + str(l) + ']'])
            # Prices
            if results['supply_choice_in'][k][l] == 1:
                for i in data['list_alt_supplier'][k]:
                    results['prices_urban_in'][i] = data['scenarios']['prices_urban'][i][l]
                    results['prices_rural_in'][i] = data['scenarios']['prices_rural'][i][l]
            if results['supply_choice_fin'][k][l] == 1:
                for i in data['list_alt_supplier'][k]:
                    results['prices_urban_fin'][i] = data['scenarios']['prices_urban'][i][l]
                    results['prices_rural_fin'][i] = data['scenarios']['prices_rural'][i][l]
    
    for i in range(data['I_tot']):
        # Market shares
        results['demand'][i] = model.solution.get_values(nameToIndex['demand[' + str(i) + ']'])
        results['demand_urban'][i] = model.solution.get_values(nameToIndex['demand_urban[' + str(i) + ']'])
        results['demand_rural'][i] = model.solution.get_values(nameToIndex['demand_rural[' + str(i) + ']'])
        
        if data['DCM'] == 'NestedLogit':
            # Modal shares
            if data['alternatives'][i]['Mode'] == 'Train':
                results['modeshare'][0] += results['demand'][i] / data['Pop']
    
    # Profits
    for k in range(1, data['K'] + 1):
        results['max_profit_in'][k] = model.solution.get_values(nameToIndex['revenue_In[' + str(k) + ']'])
        results['max_profit_fin'][k] = model.solution.get_values(nameToIndex['revenueMax_Fin[' + str(k) + ']'])


    ### PRINT THE RESULTS OBTAINED FROM THE MODEL
    print('\nRuntime    : {:8.3f} sec'.format(time.time() - t_in))

    # Obj value
    print('\nSOLUTION:\n\nObj value: {:10.2f}'.format(results['obj']))
           
    # Supply strategies
    print('\nSupp  Alt     PriceU In  PriceU Fin      PriceR In  PriceR Fin     DemandU In  DemandR In')
    for i in range(data['I_opt_out'], data['I_tot']):
        print('  {:2d}   {:2d}      {:8.2f}    {:8.2f}       {:8.2f}    {:8.2f}       {:8.2f}    {:8.2f}'
            .format(data['operator'][i], i, results['prices_urban_in'][i], results['prices_urban_fin'][i],
            results['prices_rural_in'][i], results['prices_rural_fin'][i], results['demand_urban'][i], results['demand_rural'][i]))

    # Profits
    print('\nSupplier  Profit In   Profit Fin')
    for k in range(1, data['K'] + 1):
        print('   {:2d}     {:9.2f}    {:9.2f}'.format(k, results['max_profit_in'][k], results['max_profit_fin'][k]))

    return results



if __name__ == '__main__':

    t_0 = time.time()
    
    # Get the data and compute exogenous terms
    data = data_file.getData()
    data_file.preprocessUtilities(data)

    generate_strategies.generateStrategySets(data)

    if data['DCM'] == 'NestedLogit':
        #Calculate initial values of logsum terms
        nested_logit.logsumNestedLogit(data)

    update_bounds.updateSubgamePriceBounds(data)
    update_bounds.updateUtilityBounds(data)

    choice_preprocess.choicePreprocess(data)
    choice_preprocess.choicePreprocessStrategies(data)

    t_1 = time.time()

    # Build the model
    model, nameToIndex = getModel(data)
    t_2 = time.time()

    # Solve the model
    results = solveModel(model, nameToIndex, data)
    t_3 = time.time()

    print('\n ---- TIMING ---- ')
    print('Read data + preprocess : {:8.3f} sec'.format(t_1 - t_0))
    print('Build the model        : {:8.3f} sec'.format(t_2 - t_1))
    print('Solve the model        : {:8.3f} sec'.format(t_3 - t_2))
    print('--------------')
    print('Total running time     : {:8.3f} sec'.format(t_3 - t_0))
