#   Regulator = leader
#   Suppliers = fixed prices
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
import choice_preprocess
import nested_logit

# Data
import data_intercity as data_file


def getModel(data):
    '''
    Leader = regulator
    Followers = suppliers (and customers)
    '''
    
    print('\nOPTIMIZATION MODEL - REGULATOR')

    t_in = time.time()
    
    # Initialize the model
    model = cplex.Cplex()

    # Fix prices of the alternatives
    priceU = data['p_urban_fixed']
    priceR = data['p_rural_fixed']

    ##############################################################
    ########## ---------- OBJECTIVE FUNCTION ---------- ##########
    ##############################################################
    # Set the objective function to maximization 
    # (of SWF or of modal share)
    model.objective.set_sense(model.objective.sense.maximize)

    ##############################################################
    ########## --------------- VARIABLES -------------- ##########
    ##############################################################

    # ------------------------------------------------- #
    # ----- LEVEL 0 : REGULATOR DECISION VARIABLES ---- #
    # ------------------------------------------------- #
    '''
    LIST OF VARIABLES:

    taxsubsidy_highincome[i]        continuous>0      level of tax-subsidy (transformation)
    taxsubsidy_lowincome[i]         continuous>0      level of tax-subsidy (transformation)
    delta_highincome_U[i][n][r]     continuous>0      choice-tax-subsidy   (transformation)
    delta_highincome_R[i][n][r]     continuous>0      choice-tax-subsidy   (transformation)
    delta_lowincome_U[i][n][r]      continuous>0      choice-tax-subsidy   (transformation)    
    delta_lowincome_R[i][n][r]      continuous>0      choice-tax-subsidy   (transformation)    
    delta[i][n][r]                  continuous        choice-tax-subsidy

    AUXILIARY VARIABLES:

    SWF_Budget                      continuous
    SWF_CostPublicFunds             continuous
    SWF_Emissions                   continuous
    SWF_Utilities                   continuous
    SWF_Profits                     continuous    
    '''

    objVar = []
    typeVar = []
    nameVar = []
    lbVar = []
    ubVar = []
    
    ### --- Objective function variables --- ###

    objVar.append(1.0)
    typeVar.append(model.variables.type.continuous)
    nameVar.append('SWF_Budget')
    lbVar.append(-cplex.infinity)
    ubVar.append(cplex.infinity)

    objVar.append(1.0)
    typeVar.append(model.variables.type.continuous)
    nameVar.append('SWF_CostPublicFunds')
    lbVar.append(-cplex.infinity)
    ubVar.append(cplex.infinity)

    objVar.append(1.0)
    typeVar.append(model.variables.type.continuous)
    nameVar.append('SWF_Emissions')
    lbVar.append(-cplex.infinity)
    ubVar.append(cplex.infinity)

    objVar.append(0.0)
    typeVar.append(model.variables.type.continuous)
    nameVar.append('SWF_Utilities')
    lbVar.append(-cplex.infinity)
    ubVar.append(cplex.infinity)

    objVar.append(1.0)
    typeVar.append(model.variables.type.continuous)
    nameVar.append('SWF_Utilities_high')
    lbVar.append(-cplex.infinity)
    ubVar.append(cplex.infinity)
    
    objVar.append(1.0)
    typeVar.append(model.variables.type.continuous)
    nameVar.append('SWF_Utilities_low')
    lbVar.append(-cplex.infinity)
    ubVar.append(cplex.infinity)

    objVar.append(1.0)
    typeVar.append(model.variables.type.continuous)
    nameVar.append('SWF_Profits')
    lbVar.append(-cplex.infinity)
    ubVar.append(cplex.infinity)

    model.variables.add(obj = [objVar[i] for i in range(len(objVar))],
                        types=[typeVar[i] for i in range(len(typeVar))],
                        lb=[lbVar[i] for i in range(len(typeVar))],
                        ub=[ubVar[i] for i in range(len(typeVar))],
                        names=[nameVar[i] for i in range(len(nameVar))])

    typeVar = []
    nameVar = []
    lbVar = []
    ubVar = []

    ### --- Tax-subsidy decision variables for high income and low income --- ###

    for i in range(data['I_tot']):
        
        typeVar.append(model.variables.type.continuous)
        nameVar.append('taxsubsidy_highincome[' + str(i) + ']')
        lbVar.append(0.0)
        ubVar.append(data['ub_tax_highincome'][i]+data['ub_subsidy_highincome'][i])
        
        typeVar.append(model.variables.type.continuous)
        nameVar.append('taxsubsidy_lowincome[' + str(i) + ']')
        lbVar.append(0.0)
        ubVar.append(data['ub_tax_lowincome'][i]+data['ub_subsidy_lowincome'][i])
    
    ### --- Linearized choice-tax-subsidy variables --- ###

    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                if data['INCOME'][n] == 1:
                    typeVar.append(model.variables.type.continuous)
                    if data['ORIGIN'][n] == 1:
                        nameVar.append('delta_highincome_U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                    elif data['ORIGIN'][n] == 0:
                        nameVar.append('delta_highincome_R[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                    lbVar.append(0.0)
                    ubVar.append(data['ub_tax_highincome'][i]+data['ub_subsidy_highincome'][i])
                elif data['INCOME'][n] == 0:
                    typeVar.append(model.variables.type.continuous)
                    if data['ORIGIN'][n] == 1:
                        nameVar.append('delta_lowincome_U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                    elif data['ORIGIN'][n] == 0:
                        nameVar.append('delta_lowincome_R[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                    lbVar.append(0.0)
                    ubVar.append(data['ub_tax_lowincome'][i]+data['ub_subsidy_lowincome'][i])
    
    ### --- Transformed linearized choice-tax-subsidy variables --- ###

    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                typeVar.append(model.variables.type.continuous)
                nameVar.append('delta[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                lbVar.append(-cplex.infinity)
                ubVar.append(cplex.infinity)

    ### --- Absolute value of choice-tax-subsidy variables (for marginal cost of public funds) --- ###

    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                typeVar.append(model.variables.type.continuous)
                nameVar.append('delta_pos[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                lbVar.append(0.0)
                ubVar.append(cplex.infinity)

    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                typeVar.append(model.variables.type.continuous)
                nameVar.append('delta_neg[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                lbVar.append(0.0)
                ubVar.append(cplex.infinity)

    model.variables.add(types = [typeVar[i] for i in range(len(typeVar))],
                        lb = [lbVar[i] for i in range(len(typeVar))],
                        ub = [ubVar[i] for i in range(len(typeVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])

    # ------------------------------------------------- #
    # ----- LEVEL 1 : OPERATOR DECISION VARIABLES ----- #
    # ------------------------------------------------- #
    '''
    No variables
    '''

    # ------------------------------------------------- #
    # ----- LEVEL 2 : CUSTOMER DECISION VARIABLES ----- #
    # ------------------------------------------------- #
    '''
    LIST OF VARIABLES:

    w[i][n][r]              binary          customer choice 
    U[i][n][r]              continuous      utility 
    UMax[n][r]              continuous      max utility

    demand[i]               continuous      sum of choices over n and r
    '''

    # CHOICE VARIABLES
    typeVar = []
    nameVar = []

    # Customer choice variables
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                typeVar.append(model.variables.type.binary)
                nameVar.append('w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')

    model.variables.add(types=[typeVar[i] for i in range(len(typeVar))],
                        names=[nameVar[i] for i in range(len(nameVar))])

    # BATCH OF UTILITY VARIABLES
    typeVar = []
    nameVar = []
    lbVar = []
    ubVar = []

    # Utility variables
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                if data['w_pre'][i, n, r] != 0:
                    typeVar.append(model.variables.type.continuous)
                    nameVar.append('U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                    lbVar.append(data['lb_U'][i, n, r])
                    ubVar.append(data['ub_U'][i, n, r])

    # Maximum utility for each customer and draw
    for n in range(data['N']):
        for r in range(data['R']):
            typeVar.append(model.variables.type.continuous)
            nameVar.append('UMax[' + str(n) + ']' + '[' + str(r) + ']')
            lbVar.append(-cplex.infinity)
            ubVar.append(cplex.infinity)

    ## TOTAL DEMAND FOR ALTERNATIVES (AUXILIARY VARIABLES)

    for i in range(data['I_tot']):
        typeVar.append(model.variables.type.continuous)
        nameVar.append('demand[' + str(i) + ']')
        lbVar.append(0.0)
        ubVar.append(data['Pop'])
        # Urban demand
        typeVar.append(model.variables.type.continuous)
        nameVar.append('demand_urban[' + str(i) + ']')
        lbVar.append(0.0)
        ubVar.append(data['Pop'])
        # Rural demand
        typeVar.append(model.variables.type.continuous)
        nameVar.append('demand_rural[' + str(i) + ']')
        lbVar.append(0.0)
        ubVar.append(data['Pop'])

    model.variables.add(types=[typeVar[i] for i in range(len(typeVar))],
                        lb=[lbVar[i] for i in range(len(typeVar))],
                        ub=[ubVar[i] for i in range(len(typeVar))],
                        names=[nameVar[i] for i in range(len(nameVar))])

    print('CPLEX model: all decision variables added. N variables: %r. Time: %r'
          % (model.variables.get_num(), round(time.time()-t_in, 2)))

    # Creating a dictionary that maps variable names to indices, to speed up constraints creation
    # https://www.ibm.com/developerworks/community/forums/html/topic?id=2349f613-26b1-4c29-aa4d-b52c9505bf96
    nameToIndex = {n: j for j, n in enumerate(model.variables.get_names())}


    ##############################################################
    ########## -------------- CONSTRAINTS ------------- ##########
    ##############################################################
    
    indicesConstr = []
    coefsConstr = []
    sensesConstr = []
    rhsConstr = []

    ###################################################################
    ###### --- Level 0 : REGULATOR / EQUILIBRIUM CONSTRAINTS --- ######
    ###################################################################

    ##### Subsidy/taxation constraints (problem-specific)
    for i in range(data['I_tot']):
        
        # Subsidies cannot be higher than prices
        indicesConstr.append([nameToIndex['taxsubsidy_highincome[' + str(i) + ']']])
        coefsConstr.append([1.0])
        sensesConstr.append('G')
        rhsConstr.append(data['ub_subsidy_highincome'][i]-priceU[i])
        indicesConstr.append([nameToIndex['taxsubsidy_highincome[' + str(i) + ']']])
        coefsConstr.append([1.0])
        sensesConstr.append('G')
        rhsConstr.append(data['ub_subsidy_highincome'][i]-priceR[i])

        indicesConstr.append([nameToIndex['taxsubsidy_lowincome[' + str(i) + ']']])
        coefsConstr.append([1.0])
        sensesConstr.append('G')
        rhsConstr.append(data['ub_subsidy_lowincome'][i]-priceU[i])
        indicesConstr.append([nameToIndex['taxsubsidy_lowincome[' + str(i) + ']']])
        coefsConstr.append([1.0])
        sensesConstr.append('G')
        rhsConstr.append(data['ub_subsidy_lowincome'][i]-priceR[i])
        
        # Same subsidy to all customers
        indicesConstr.append([nameToIndex['taxsubsidy_highincome[' + str(i) + ']'],
                              nameToIndex['taxsubsidy_lowincome[' + str(i) + ']']])
        coefsConstr.append([1.0, -1.0])
        sensesConstr.append('E')
        rhsConstr.append(0.0)
        
        # Subsidy to HSR trains should be the same for all travellers and departure times
        #if data['alternatives'][i]['Mode'] == 'Train' and data['alternatives'][i]['Endogenous'] == 1:
        # Subsidy to all trains should be the same for all travellers and departure times
        if data['alternatives'][i]['Mode'] == 'Train':
            for j in range(i+1, data['I_tot']):
                if data['alternatives'][j]['Mode'] == 'Train':
                    indicesConstr.append([nameToIndex['taxsubsidy_highincome[' + str(i) + ']'],
                                          nameToIndex['taxsubsidy_highincome[' + str(j) + ']']])
                    coefsConstr.append([1.0, -1.0])
                    sensesConstr.append('E')
                    rhsConstr.append(0.0)
                    indicesConstr.append([nameToIndex['taxsubsidy_lowincome[' + str(i) + ']'],
                                          nameToIndex['taxsubsidy_lowincome[' + str(j) + ']']])
                    coefsConstr.append([1.0, -1.0])
                    sensesConstr.append('E')
                    rhsConstr.append(0.0)
        
        # Flight tax should be the same for all travellers and departure times
        if data['alternatives'][i]['Mode'] == 'Plane':
            for j in range(i+1, data['I_tot']):
                if data['alternatives'][j]['Mode'] == 'Plane':
                    indicesConstr.append([nameToIndex['taxsubsidy_highincome[' + str(i) + ']'],
                                          nameToIndex['taxsubsidy_highincome[' + str(j) + ']']])
                    coefsConstr.append([1.0, -1.0])
                    sensesConstr.append('E')
                    rhsConstr.append(0.0)
                    indicesConstr.append([nameToIndex['taxsubsidy_lowincome[' + str(i) + ']'],
                                          nameToIndex['taxsubsidy_lowincome[' + str(j) + ']']])
                    coefsConstr.append([1.0, -1.0])
                    sensesConstr.append('E')
                    rhsConstr.append(0.0)
        
        # No tax or subsidy to cars (opt-out option)
        if data['alternatives'][i]['Mode'] == 'Car':
            indicesConstr.append([nameToIndex['taxsubsidy_highincome[' + str(i) + ']']])
            coefsConstr.append([1.0])
            sensesConstr.append('E')
            rhsConstr.append(data['ub_subsidy_highincome'][i])
            indicesConstr.append([nameToIndex['taxsubsidy_lowincome[' + str(i) + ']']])
            coefsConstr.append([1.0])
            sensesConstr.append('E')
            rhsConstr.append(data['ub_subsidy_lowincome'][i])
        
    ##### Choice-tax-subsidy constraints
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                if data['INCOME'][n] == 1:
                    
                    if data['w_pre'][i, n, r] != 0 and data['w_pre'][i, n, r] != 1:

                        # Linearized subsidy-choice: delta is equal to 0 if alternative is not chosen
                        if data['ORIGIN'][n] == 1:
                            indicesConstr.append([nameToIndex['delta_highincome_U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                                  nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                        elif data['ORIGIN'][n] == 0:
                            indicesConstr.append([nameToIndex['delta_highincome_R[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                                  nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                        coefsConstr.append([1.0, -(data['ub_tax_highincome'][i]+data['ub_subsidy_highincome'][i])])
                        sensesConstr.append('L')
                        rhsConstr.append(0.0)

                        # Linearized subsidy-choice: delta is equal to the subsidy if alternative is chosen
                        # Delta is greater than or equal to the subsidy for the chosen alternative
                        if data['ORIGIN'][n] == 1:
                            indicesConstr.append([nameToIndex['taxsubsidy_highincome[' + str(i) + ']'],
                                                  nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                                  nameToIndex['delta_highincome_U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                        elif data['ORIGIN'][n] == 0:
                            indicesConstr.append([nameToIndex['taxsubsidy_highincome[' + str(i) + ']'],
                                                  nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                                  nameToIndex['delta_highincome_R[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                        coefsConstr.append([1.0, data['ub_tax_highincome'][i]+data['ub_subsidy_highincome'][i], -1.0])
                        sensesConstr.append('L')
                        rhsConstr.append(data['ub_tax_highincome'][i]+data['ub_subsidy_highincome'][i])
                        # Delta is smaller than or equal to the subsidy
                        if data['ORIGIN'][n] == 1:
                            indicesConstr.append([nameToIndex['delta_highincome_U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                                  nameToIndex['taxsubsidy_highincome[' + str(i) + ']']])
                        elif data['ORIGIN'][n] == 0:
                            indicesConstr.append([nameToIndex['delta_highincome_R[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                                  nameToIndex['taxsubsidy_highincome[' + str(i) + ']']])
                        coefsConstr.append([1.0, -1.0])
                        sensesConstr.append('L')
                        rhsConstr.append(0.0)

                    # Case of captive customer - automatic relation between subsidy and delta through data['w_pre']
                    elif data['w_pre'][i, n, r] == 1:
                        if data['ORIGIN'][n] == 1:
                            indicesConstr.append([nameToIndex['taxsubsidy_highincome[' + str(i) + ']'],
                                                  nameToIndex['delta_highincome_U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                        elif data['ORIGIN'][n] == 0:
                            indicesConstr.append([nameToIndex['taxsubsidy_highincome[' + str(i) + ']'],
                                                  nameToIndex['delta_highincome_R[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                        coefsConstr.append([data['w_pre'][i, n, r], -1.0])
                        sensesConstr.append('E')
                        rhsConstr.append(0.0)
                
                elif data['INCOME'][n] == 0:

                    if data['w_pre'][i, n, r] != 0 and data['w_pre'][i, n, r] != 1:
    
                        # Linearized subsidy-choice: delta is equal to 0 if alternative is not chosen
                        if data['ORIGIN'][n] == 1:
                            indicesConstr.append([nameToIndex['delta_lowincome_U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                                  nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                        elif data['ORIGIN'][n] == 0:
                            indicesConstr.append([nameToIndex['delta_lowincome_R[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                                  nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                        coefsConstr.append([1.0, -(data['ub_tax_lowincome'][i]+data['ub_subsidy_lowincome'][i])])
                        sensesConstr.append('L')
                        rhsConstr.append(0.0)

                        # Linearized subsidy-choice: delta is equal to the subsidy if alternative is chosen
                        # Delta is greater than or equal to the subsidy for the chosen alternative
                        if data['ORIGIN'][n] == 1:
                            indicesConstr.append([nameToIndex['taxsubsidy_lowincome[' + str(i) + ']'],
                                                  nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                                  nameToIndex['delta_lowincome_U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                        elif data['ORIGIN'][n] == 0:
                            indicesConstr.append([nameToIndex['taxsubsidy_lowincome[' + str(i) + ']'],
                                                  nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                                  nameToIndex['delta_lowincome_R[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                        coefsConstr.append([1.0, data['ub_tax_lowincome'][i]+data['ub_subsidy_lowincome'][i], -1.0])
                        sensesConstr.append('L')
                        rhsConstr.append(data['ub_tax_lowincome'][i]+data['ub_subsidy_lowincome'][i])
                        # Delta is smaller than or equal to the subsidy
                        if data['ORIGIN'][n] == 1:
                            indicesConstr.append([nameToIndex['delta_lowincome_U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                                  nameToIndex['taxsubsidy_lowincome[' + str(i) + ']']])
                        elif data['ORIGIN'][n] == 0:
                            indicesConstr.append([nameToIndex['delta_lowincome_R[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                                  nameToIndex['taxsubsidy_lowincome[' + str(i) + ']']])
                        coefsConstr.append([1.0, -1.0])
                        sensesConstr.append('L')
                        rhsConstr.append(0.0)

                    # Case of captive customer - automatic relation between subsidy and delta through data['w_pre']
                    elif data['w_pre'][i, n, r] == 1:
                        if data['ORIGIN'][n] == 1:
                            indicesConstr.append([nameToIndex['taxsubsidy_lowincome[' + str(i) + ']'],
                                                  nameToIndex['delta_lowincome_U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                        elif data['ORIGIN'][n] == 0:
                            indicesConstr.append([nameToIndex['taxsubsidy_lowincome[' + str(i) + ']'],
                                                  nameToIndex['delta_lowincome_R[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                        coefsConstr.append([data['w_pre'][i, n, r], -1.0])
                        sensesConstr.append('E')
                        rhsConstr.append(0.0)
    
    ##### Transformation of choice-tax-subsidy variables
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                if data['INCOME'][n] == 1:
                    if data['ORIGIN'][n] == 1:
                        indicesConstr.append([nameToIndex['delta_highincome_U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                              nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                              nameToIndex['delta[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                    elif data['ORIGIN'][n] == 0:
                        indicesConstr.append([nameToIndex['delta_highincome_R[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                              nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                              nameToIndex['delta[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                    coefsConstr.append([1.0, -data['ub_subsidy_highincome'][i], -1.0])
                    sensesConstr.append('E')
                    rhsConstr.append(0.0)
                elif data['INCOME'][n] == 0:
                    if data['ORIGIN'][n] == 1:
                        indicesConstr.append([nameToIndex['delta_lowincome_U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                              nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                              nameToIndex['delta[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                    elif data['ORIGIN'][n] == 0:
                        indicesConstr.append([nameToIndex['delta_lowincome_R[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                              nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                              nameToIndex['delta[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                    coefsConstr.append([1.0, -data['ub_subsidy_lowincome'][i], -1.0])
                    sensesConstr.append('E')
                    rhsConstr.append(0.0)
    
    ##### Derive absolute value of delta (delta = delta^+ - delta^- , with delta^+ and delta^- non-negative)
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                indicesConstr.append([nameToIndex['delta[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['delta_pos[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                      nameToIndex['delta_neg[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                coefsConstr.append([1.0, -1.0, 1.0])
                sensesConstr.append('E')
                rhsConstr.append(0.0)
    
    ##### Budget constraints
    indicesConstr.append([nameToIndex['SWF_Budget']])
    coefsConstr.append([-1.0])
    sensesConstr.append('L')
    rhsConstr.append(data['Budget'])
    
    ##### Cost of policy
    ind = [nameToIndex['SWF_Budget']]
    co = [1.0]
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                if data['w_pre'][i, n, r] != 0:
                    ind.append(nameToIndex['delta[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                    co.append(-data['popN'][n]/data['R'])
    indicesConstr.append(ind)
    coefsConstr.append(co)
    sensesConstr.append('E')
    rhsConstr.append(0.0)

    ##### Marginal cost of public funds
    ind = [nameToIndex['SWF_CostPublicFunds']]
    co = [1.0]
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                if data['w_pre'][i, n, r] != 0:
                    ind.append(nameToIndex['delta_pos[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                    co.append(data['MarginalCostPublicFunds']*data['popN'][n]/data['R'])
                    ind.append(nameToIndex['delta_neg[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                    co.append(data['MarginalCostPublicFunds']*data['popN'][n]/data['R'])
    indicesConstr.append(ind)
    coefsConstr.append(co)
    sensesConstr.append('E')
    rhsConstr.append(0.0)
    
    ##### Cost of emissions
    ind = [nameToIndex['SWF_Emissions']]
    co = [-1.0]
    for i in range(data['I_tot']):
        ind.append(nameToIndex['demand[' + str(i) + ']'])
        if data['alternatives'][i]['Mode'] == 'Plane':
            co.append(-data['Distance']*data['emissions_per_pass_km_air']*data['social_cost_of_carbon'])
        elif data['alternatives'][i]['Mode'] == 'Car':
            co.append(-data['Distance']*data['emissions_per_pass_km_car']*data['social_cost_of_carbon'])
        elif data['alternatives'][i]['Mode'] == 'Train':
            co.append(-data['Distance']*data['emissions_per_pass_km_train']*data['social_cost_of_carbon'])
    indicesConstr.append(ind)
    coefsConstr.append(co)
    sensesConstr.append('E')
    rhsConstr.append(0.0)
    
    ##### Cost of consumer utility
    
    ind = [nameToIndex['SWF_Utilities']]
    co = [1.0]
    for n in range(data['N']):
        for r in range(data['R']):            
            ind.append(nameToIndex['UMax[' + str(n) + ']' + '[' + str(r) + ']'])
            co.append(data['popN'][n]/(data['R']*data['MARGINAL_UTILITY_INCOME']))
    indicesConstr.append(ind)
    coefsConstr.append(co)
    sensesConstr.append('E')
    rhsConstr.append(0.0)
    # Only high income customers 
    ind = [nameToIndex['SWF_Utilities_high']]
    co = [1.0]
    for n in range(data['N']):
        if data['INCOME'][n] == 1:
            for r in range(data['R']):            
                ind.append(nameToIndex['UMax[' + str(n) + ']' + '[' + str(r) + ']'])
                co.append(data['popN'][n]/(data['R']*data['MARGINAL_UTILITY_INCOME']))
    indicesConstr.append(ind)
    coefsConstr.append(co)
    sensesConstr.append('E')
    rhsConstr.append(0.0)
    # Only low income customers 
    ind = [nameToIndex['SWF_Utilities_low']]
    co = [1.0]
    for n in range(data['N']):
        if data['INCOME'][n] == 0:
            for r in range(data['R']):            
                ind.append(nameToIndex['UMax[' + str(n) + ']' + '[' + str(r) + ']'])
                co.append(data['popN'][n]/(data['R']*data['MARGINAL_UTILITY_INCOME']))
    indicesConstr.append(ind)
    coefsConstr.append(co)
    sensesConstr.append('E')
    rhsConstr.append(0.0)
    
    ##### Cost of supplier utility
    ind = [nameToIndex['SWF_Profits']]
    co = [1.0]
    for i in range(data['I_tot']):
        ind.append(nameToIndex['demand_urban[' + str(i) + ']'])
        co.append(-data['p_urban_fixed'][i])
        ind.append(nameToIndex['demand_rural[' + str(i) + ']'])
        co.append(-data['p_rural_fixed'][i])
    indicesConstr.append(ind)
    coefsConstr.append(co)
    sensesConstr.append('E')
    rhsConstr.append(0.0)
    
    print('CPLEX model: regulator constraints added. Time: %r' %(round(time.time()-t_in, 2)))


    ###################################################################
    ###### ------ Level 1 : SUPPLIER UTILITY MAXIMIZATION ------ ######
    ###################################################################
    '''
    No constraints
    '''

    ###################################################################
    ###### ------ Level 2 : CUSTOMER UTILITY MAXIMIZATION ------ ######
    ###################################################################

    ##### Choice constraints

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
    
    # All captive customers are assigned (uncapacitated problem only)
    for i in range(data['I_tot']):
        for n in range(data['N']):
            for r in range(data['R']):
                if data['w_pre'][i, n, r] == 0 or data['w_pre'][i, n, r] == 1:
                    indicesConstr.append([nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']']])
                    coefsConstr.append([1.0])
                    sensesConstr.append('E')
                    rhsConstr.append(data['w_pre'][i, n, r])
    
    ##### Utility constraints

    for i in range(data['I_tot']):
        
        for n in range(data['N']):
            for r in range(data['R']):
                if data['w_pre'][i, n, r] != 0:
                    
                    # Utility function constraints
                    if data['INCOME'][n] == 1:
                        indicesConstr.append([nameToIndex['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                              nameToIndex['taxsubsidy_highincome[' + str(i) + ']']])
                        coefsConstr.append([1.0, -data['endo_coef'][i,n]])
                        if data['ORIGIN'][n] == 1:
                            rhsConstr.append(data['endo_coef'][i,n] * (priceU[i] - data['ub_subsidy_highincome'][i]) + data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                        elif data['ORIGIN'][n] == 0:
                            rhsConstr.append(data['endo_coef'][i,n] * (priceR[i] - data['ub_subsidy_highincome'][i]) + data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])

                    
                    elif data['INCOME'][n] == 0:
                        indicesConstr.append([nameToIndex['U[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'],
                                              nameToIndex['taxsubsidy_lowincome[' + str(i) + ']']])
                        coefsConstr.append([1.0, -data['endo_coef'][i,n]])
                        if data['ORIGIN'][n] == 1:
                            rhsConstr.append(data['endo_coef'][i,n] * (priceU[i] - data['ub_subsidy_lowincome'][i]) + data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                        elif data['ORIGIN'][n] == 0:
                            rhsConstr.append(data['endo_coef'][i,n] * (priceR[i] - data['ub_subsidy_lowincome'][i]) + data['exo_utility'][i,n,r] + data['Logsum'][i,n,r] + data['xi'][i, n, r])
                    sensesConstr.append('E')
                    
                    # Utility maximization constraints
                    # The selected alternative is the one with the highest utility
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
                    
    ##### Auxiliary constraints

    # Calculating demands (not part of the model)
    for i in range(data['I_tot']):
        ind = []
        co = []
        for n in range(data['N']):
            for r in range(data['R']):
                ind.append(nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                co.append(-data['popN'][n]/data['R'])
        ind.append(nameToIndex['demand[' + str(i) + ']'])
        co.append(1.0)
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(0.0)
        # Urban demand
        ind = []
        co = []
        for n in range(data['N']):
            if data['ORIGIN'][n] == 1:    
                for r in range(data['R']):
                    ind.append(nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                    co.append(-data['popN'][n]/data['R'])
        ind.append(nameToIndex['demand_urban[' + str(i) + ']'])
        co.append(1.0)
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(0.0)
        # Rural demand
        ind = []
        co = []
        for n in range(data['N']):
            if data['ORIGIN'][n] == 0:    
                for r in range(data['R']):
                    ind.append(nameToIndex['w[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
                    co.append(-data['popN'][n]/data['R'])
        ind.append(nameToIndex['demand_rural[' + str(i) + ']'])
        co.append(1.0)
        indicesConstr.append(ind)
        coefsConstr.append(co)
        sensesConstr.append('E')
        rhsConstr.append(0.0)

    model.linear_constraints.add(lin_expr = [[indicesConstr[i], coefsConstr[i]] for i in range(len(indicesConstr))],
                                 senses = [sensesConstr[i] for i in range(len(sensesConstr))],
                                 rhs = [rhsConstr[i] for i in range(len(rhsConstr))])

    print('CPLEX model: all constraints added. N constraints: %r. Time: %r\n'
          % (model.linear_constraints.get_num(), round(time.time()-t_in, 2)))

    return model


def solveModel(data, model):

    try:
        model.set_results_stream(None)
        #model.set_warning_stream(None)

        model.solve()

        ### INITIALIZE DICTIONARY OF RESULTS
        results = {}
        results['prices_urban'] = copy.deepcopy(data['p_urban_fixed'])
        results['prices_rural'] = copy.deepcopy(data['p_rural_fixed'])
        results['cust_prices_urban_high'] = np.full([data['I_tot']], -1.0)
        results['cust_prices_urban_low'] = np.full([data['I_tot']], -1.0)
        results['cust_prices_rural_high'] = np.full([data['I_tot']], -1.0)
        results['cust_prices_rural_low'] = np.full([data['I_tot']], -1.0)
        results['taxsubsidy_highinc'] = np.full([data['I_tot']], -1.0)
        results['taxsubsidy_lowinc'] = np.full([data['I_tot']], -1.0)
        results['demand'] = np.empty(data['I_tot'])
        results['demand_urban'] = np.empty(data['I_tot'])
        results['demand_rural'] = np.empty(data['I_tot'])
        results['profit'] = np.full([data['K']+1], 0.0)
        results['modeshare'] = np.full([4], 0.0)
        results['delta'] = np.zeros((data['I_tot'], data['N'],data['R']))
        results['delta_pos'] = np.zeros((data['I_tot'], data['N'],data['R']))
        results['delta_neg'] = np.zeros((data['I_tot'], data['N'],data['R']))
        results['SWF_Budget'] = model.solution.get_values('SWF_Budget')
        results['SWF_CostPublicFunds'] = model.solution.get_values('SWF_CostPublicFunds')
        results['SWF_Emissions'] = model.solution.get_values('SWF_Emissions')
        results['SWF_Utilities'] = model.solution.get_values('SWF_Utilities')
        results['SWF_Utilities_high'] = model.solution.get_values('SWF_Utilities_high')
        results['SWF_Utilities_low'] = model.solution.get_values('SWF_Utilities_low')
        results['SWF_Profits'] = model.solution.get_values('SWF_Profits')

        ### SAVE RESULTS
        
        for i in range(data['I_tot']):
            # Subsidies / taxes
            results['taxsubsidy_highinc'][i] = model.solution.get_values('taxsubsidy_highincome[' + str(i) + ']') - data['ub_subsidy_highincome'][i]
            results['taxsubsidy_lowinc'][i] = model.solution.get_values('taxsubsidy_lowincome[' + str(i) + ']') - data['ub_subsidy_lowincome'][i]
            # Customer price
            results['cust_prices_urban_high'][i] = results['prices_urban'][i] + results['taxsubsidy_highinc'][i]
            results['cust_prices_urban_low'][i] = results['prices_urban'][i] + results['taxsubsidy_lowinc'][i]
            results['cust_prices_rural_high'][i] = results['prices_rural'][i] + results['taxsubsidy_highinc'][i]
            results['cust_prices_rural_low'][i] = results['prices_rural'][i] + results['taxsubsidy_lowinc'][i]
            # Demands
            results['demand'][i] = model.solution.get_values('demand[' + str(i) + ']')
            results['demand_urban'][i] = model.solution.get_values('demand_urban[' + str(i) + ']')
            results['demand_rural'][i] = model.solution.get_values('demand_rural[' + str(i) + ']')

        for i in range(data['I_tot']):
            # Profits
            results['profit'][data['operator'][i]] +=\
                (results['demand_urban'][i]*results['prices_urban'][i] + results['demand_rural'][i]*results['prices_rural'][i])            
            # Modal shares
            if data['alternatives'][i]['Mode'] == 'Train':
                results['modeshare'][0] += results['demand'][i] / data['Pop']
            elif data['alternatives'][i]['Mode'] == 'Plane':
                results['modeshare'][1] += results['demand'][i] / data['Pop']
            elif data['alternatives'][i]['Mode'] == 'Car':
                results['modeshare'][2] += results['demand'][i] / data['Pop']
        
        # Government expenses
        for i in range(data['I_tot']):
            for n in range(data['N']):
                for r in range(data['R']):
                    results['delta'][i,n,r] = model.solution.get_values('delta[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                    results['delta_pos'][i,n,r] = model.solution.get_values('delta_pos[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                    results['delta_neg'][i,n,r] = model.solution.get_values('delta_neg[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')

        ### PRINT OBJ FUNCTION
        print('\nSOLUTION:\n\nPopulation                         : {:10.0f}'.format(data['Pop']))
        print('Budget                             : {:10.2f}'.format(data['Budget']))
        print('Obj fun value                      : {:10.2f}'.format(model.solution.get_objective_value()))
        print('Consumer utility                   : {:10.2f}'.format(results['SWF_Utilities']))
        print('Consumer utility (high inc)        : {:10.2f}'.format(results['SWF_Utilities_high']))
        print('Consumer utility (low inc)         : {:10.2f}'.format(results['SWF_Utilities_low']))
        print('Supplier utility                   : {:10.2f}'.format(results['SWF_Profits']))
        print('Cost of policy                     : {:10.2f}'.format(-results['SWF_Budget']))
        print('Cost of public intervention        : {:10.2f}'.format(-results['SWF_CostPublicFunds']))
        print('Cost of emissions                  : {:10.2f}'.format(-results['SWF_Emissions']))
        print('Emissions (tons of CO2)            : {:10.2f}'.format(-results['SWF_Emissions']/(data['social_cost_of_carbon']*1000)))
        print('Train modal share                  : {:10.4f}'.format(results['modeshare'][0]))
        print('Air   modal share                  : {:10.4f}'.format(results['modeshare'][1]))
        print('Car   modal share                  : {:10.4f}'.format(results['modeshare'][2]))

        # PRINT PRICES/SUBSIDIES/TAXES/DEMANDS
        if data['DCM'] == 'NestedLogit':
            print('\n                       SUPPLIER  PRICE           TAX/SUBSIDY                         C U S T O M E R   P R I C E                                    ')
            print('Alt  Operator          Urban     Rural       High inc   Low inc       HighInc Urban  HighInc Rural   LowInc Urban   LowInc Rural        Market share')
            for i in range(data['I_tot']):
                print(' {:2d}        {:2d}       {:8.2f}  {:8.2f}       {:8.2f}  {:8.2f}            {:8.2f}       {:8.2f}       {:8.2f}       {:8.2f}              {:6.4f}'
                        .format(i, data['operator'][i], results['prices_urban'][i], results['prices_rural'][i],
                        results['taxsubsidy_highinc'][i], results['taxsubsidy_lowinc'][i],
                        results['cust_prices_urban_high'][i], results['cust_prices_rural_high'][i],
                        results['cust_prices_urban_low'][i], results['cust_prices_rural_low'][i],
                        results['demand'][i] / data['Pop']))

        # PRINT PROFITS
        print('\nSupplier    Profit')
        for k in range(data['K'] + 1):
            print('   {:2d}     {:8.2f}'.format(k, results['profit'][k]))

        return results

    # This is run if there is an error while running the model (e.g. infeasibility)
    except CplexSolverError:

        #Solution status codes:
        #https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.10.0/ilog.odms.cplex.help/refpythoncplex/html/cplex._internal._subinterfaces.SolutionStatus-class.html
        print(model.solution.get_status())
        raise Exception('Exception raised during solve')


if __name__ == '__main__':

    t_0 = time.time()

    # Initialize the dictionary 'dict' containing all the input/output data
    data = {}

    # Define parameters of the algorithm
    data_file.setAlgorithmParameters(data)

    # Get the data and preprocess
    data_file.getData(data)
    data_file.preprocessUtilities(data)

    # Calculate initial values of logsum terms
    nested_logit.logsumNestedLogitRegulator(data)
    
    data_file.printCustomers(data)

    # Calculate utility bounds
    update_bounds.updateUtilityBoundsWithRegulator(data)

    # Preprocess captive choices
    choice_preprocess.choicePreprocess(data)
    t_1 = time.time()

    # SOLVE STACKELBERG GAME
    model = getModel(data)
    results = solveModel(data, model)
    t_2 = time.time()

    print('\n\n-- TIMING -- ')
    print('Get data + Preprocess: {:8.2f}'.format(t_1 - t_0))
    print('Solve the problem:     {:8.2f}'.format(t_2 - t_1))
    print('------------ ')
