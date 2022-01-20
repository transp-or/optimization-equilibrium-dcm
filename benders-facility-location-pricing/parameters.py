# Define parameters of the algorithms
import numpy as np
import math

def BendersParameters(data):

    ################ Number of threads
    
    #cores = cplex.Cplex()
    #data['num_threads'] = cores.get_num_cores()
    data['num_threads'] = 1


    ################ Separation procedure:
    # 1 = Integer and fractional infeasible solutions
    # 0 = Only integer infeasible solutions
    
    data['sep_frac_sols'] = 0


    ################ Initial cuts: aggregate or disaggregate

    #data['cuts'] = 'Aggregate'
    data['cuts'] = 'Disaggregate'
    #data['cuts'] = 'None'


    ################ Initial cuts: generate additional cuts
    
    #data['additionalSolutions'] = 'Yes'
    data['additionalSolutions'] = 'No'
    #data['minSolutionsWithI'] = data['R']/100.0
    data['minSolutionsWithI'] = 1


    ################ Warm start: yes or no
    
    data['WarmStart'] = 'Yes'
    #data['WarmStart'] = 'No'


    ################ Benders presolve: yes or no
    
    data['BendersPresolve'] = 'Yes'
    #data['BendersPresolve'] = 'No'


    ################ Benders generate presolve cuts: yes or no
    
    #data['BendersPresolveCuts'] = 'Yes'
    data['BendersPresolveCuts'] = 'No'
    
    data['AddPresolveCutsAs'] = 'Constraints'
    #data['AddPresolveCutsAs'] = 'LazyConstraints'

    data['AddedPresolveCuts'] = 1


    ################ Benders enumerate small solutions: yes or no
    
    #data['BendersEnumerateSmall'] = 'Yes'
    data['BendersEnumerateSmall'] = 'No'


    ################ Benders solve single scenarios: yes or no
    
    data['BendersSingleScenarios'] = 'Yes'
    #data['BendersSingleScenarios'] = 'No'


    ################ Benders solve single customers: yes or no
    
    #data['BendersSingleCustomers'] = 'Yes'
    data['BendersSingleCustomers'] = 'No'


    ################ Subset cuts: yes or no
    
    #data['subsetCuts'] = 'Yes'
    data['subsetCuts'] = 'No'


    ################ Partial Benders: yes (and which scenarios) or no
    
    data['PartialBenders'] = 'Yes'
    #data['PartialBenders'] = 'No'
    
    # If a partial Benders decomposition is used, list the scenarios retained in the master problem
    data['PB_RetainedInMaster'] = np.zeros([data['R']])
    #data['PB_RetainedInMaster'] = np.ones([data['R']])


    ################ Scenario selection and clustering
    
    data['Clustering'] = 'Yes'
    #data['Clustering'] = 'No'

    data['nClustersR'] = int(min(max(5.0,math.floor(data['R']/5.0)),10.0)) #number of clusters to generate (R<25 -> 5, R>50 -> 10)
    data['minRClustering'] = 10
    data['nClustersN'] = int(math.floor(data['N']/4.0)) #number of clusters to generate (R<25 -> 5, R>50 -> 10)
    data['minNClustering'] = 10
    