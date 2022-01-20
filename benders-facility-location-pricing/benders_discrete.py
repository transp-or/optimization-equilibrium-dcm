##########################################
# BRANCH-AND-BENDERS-CUT ALGORITHM
# FOR CHOICE-BASED OPTIMIZATION
# WITH DISCRETE SUPPLY VARIABLES
##########################################

# General
import sys
import traceback
import time
import numpy as np
import random
import itertools

# CPLEX
import cplex

# Project
import parameters
import prints
import functions
import scenario_clustering
import model_discrete_compact_assortment

# Data
#import Data_Parking_N80_I14 as data_file
#import Data_Parking_N08_I10 as data_file
import Data_Parking_N08_I10_FC as data_file


class WorkerLP():
    """
    This class builds the worker LP for each n and r and
    allows to separate violated Benders' cuts.
    """
        
    def __init__(self, data):
        
        t_Duals_start = time.time()
        
        # Initialize a CPLEX model instance for each subproblem
        dualFollower = []
        for n in range(data['N']):
            dualFollower.append([])
            for r in range(data['R']):
                # Special case: partial Benders decomposition
                if data['PartialBenders'] == 'Yes' and data['PB_RetainedInMaster'][r] == 1:
                    dualFollower[n].append(-1)
                else:
                    dualFollower[n].append(cplex.Cplex())

        # Construct CPLEX model for each subproblem
        for n in range(data['N']):
            for r in range(data['R']):
                if dualFollower[n][r] != -1:

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

                    # Set up Cplex instance to solve the worker LP
                    dualFollower[n][r].set_results_stream(None)
                    dualFollower[n][r].set_log_stream(None)

                    # Turn off the presolve reductions and set the CPLEX optimizer
                    # to solve the worker LP with primal simplex method.
                    dualFollower[n][r].parameters.preprocessing.reduce.set(0)
                    dualFollower[n][r].parameters.lpmethod.set(dualFollower[n][r].parameters.lpmethod.values.primal)

        self.dualFollower = dualFollower
        self.data = data
        self.cut_lhs = None
        self.senses = None
        self.cut_rhs = None

        data['timeDuals'] += (time.time() - t_Duals_start)

    def presolveCuts(self,y,z):
        
        data = self.data
        self.cut_lhs = []
        self.senses = []
        self.cut_rhs = []

        # Create a sorted list of solutions by objective function value
        sort_all_y = np.argsort(data['OF_all'])
        sort_all_y = list(reversed(sort_all_y))
        pos = np.zeros((len(data['OF_all'])))
        for sol in range(len(data['OF_all'])):
            for p in range(len(data['OF_all'])):
                if sort_all_y[p] == sol:
                    pos[sol] = p
        for n in range(len(sort_all_y)):
            print('{:3.0f}'.format(sort_all_y[n]),end='')
        print()
        for n in range(len(pos)):
            print('{:3.0f}'.format(pos[n]),end='')
        print()
        
        # Disaggregate cuts (nCuts = N*R*solutions)
        if data['cuts'] == 'Disaggregate':
            count = 0
            for solution in range(data['R']):
                if pos[solution] < data['R'] / 10.0:
                    WorkerLP.separateDual(self, data['all_y'][solution], y, z)
                    count += 1
                    print('{:3.0f} out of {:4.0f}'.format(count, data['R'] / 10.0))
                
    def initializeDualVariables(data):

        data['x_dual'] = np.zeros([data['I_tot_exp'], data['N'], data['R']])
        data['alpha1_dual'] = np.zeros([data['N'], data['R']])
        data['alpha2_dual'] = np.zeros([data['I_tot_exp'], data['N'], data['R']])
        data['gamma2_dual'] = np.zeros([data['N'], data['R']])
        data['gamma1_dual'] = np.zeros([data['I_tot_exp'], data['N'], data['R']])
        data['delta1_dual'] = np.zeros([data['I_tot_exp'], data['N'], data['R']])
        data['delta2_dual'] = np.zeros([data['I_tot_exp'], data['N'], data['R']])
        data['delta3_dual'] = np.zeros([data['I_tot_exp'], data['N'], data['R']])

    def separateDual(self, y_sol, y, z):
        '''
        This method separates Benders' cuts violated by the current y - z solution.
        Violated cuts are found by solving the worker LP.
        '''

        t_Duals_start = time.time()

        dualFollower = self.dualFollower
        data = self.data
        self.cut_lhs = []
        self.senses = []
        self.cut_rhs = []

        y_dual = y_sol

        WorkerLP.initializeDualVariables(data)

        for n in range(data['N']):
            for r in range(data['R']):
                if dualFollower[n][r] != -1 and data['generateCut'][n,r] == 1:

                    #################################################################################
                    # Update the objective function coefficients in the worker LP (index, newValue)
                    #################################################################################
                    sparsePair = []
                    # Alpha1 not needed
                    count = 1
                    # Alpha2
                    for i in range(data['I_tot_exp']):
                        sparsePair.append((count, y_dual[i]))
                        count += 1
                    # Gamma1, Gamma2, Delta1 not needed
                    count += (1 + 2*data['I_tot_exp']) 
                    # Delta2
                    for i in range(data['I_tot_exp']):
                        sparsePair.append((count, data['M'][i,n,r] * (1 - y_dual[i])))
                        count += 1
                    # Delta3
                    for i in range(data['I_tot_exp']):
                        sparsePair.append((count, data['M'][i,n,r] * y_dual[i]))
                        count += 1
                    
                    dualFollower[n][r].objective.set_linear(sparsePair)

                    #################################################################################
                    # Solve the worker LP
                    #################################################################################
                    dualFollower[n][r].set_problem_type(dualFollower[n][r].problem_type.LP)
                    dualFollower[n][r].solve()

                    #################################################################################
                    # Derive optimality cut (worker LP is ALWAYS feasible, returns optimal solution, status = 1)
                    #################################################################################
                    '''
                    # A feasibility cut is available iff the solution status is unbounded (status = 2)
                    print(dualFollower[n][r].solution.get_status())
                    if dualFollower[n][r].solution.get_status() == dualFollower[n][r].solution.status.unbounded:
                        ray = dualFollower[n][r].solution.advanced.get_ray()
                        #print(ray)
                    '''
                    if dualFollower[n][r].solution.get_status() == dualFollower[n][r].solution.status.optimal:
                        objDual = dualFollower[n][r].solution.get_objective_value()
                        #print('OF dual:{:8.2f}  '.format(objDual),end='')
                        
                        # Retrieve optimal dual variables
                        data['alpha1_dual'][n,r] = dualFollower[n][r].solution.get_values('alpha1[' + str(n) + ']' + '[' + str(r) + ']')
                        for i in range(data['I_tot_exp']):
                            data['alpha2_dual'][i,n,r] = dualFollower[n][r].solution.get_values('alpha2[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                        data['gamma2_dual'][n,r] = dualFollower[n][r].solution.get_values('gamma2[' + str(n) + ']' + '[' + str(r) + ']')
                        for i in range(data['I_tot_exp']):
                            data['gamma1_dual'][i,n,r] = -dualFollower[n][r].solution.get_values('gamma1[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                        for i in range(data['I_tot_exp']):
                            data['delta1_dual'][i,n,r] = -dualFollower[n][r].solution.get_values('delta1[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                            data['delta2_dual'][i,n,r] = -dualFollower[n][r].solution.get_values('delta2[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                            data['delta3_dual'][i,n,r] = -dualFollower[n][r].solution.get_values('delta3[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')

                        # Components of the dual objective function
                        objA1_nr = data['alpha1_dual'][n,r]
                        objA2_nr = 0.0
                        for i in range(data['I_tot_exp']):
                            objA2_nr = objA2_nr + data['alpha2_dual'][i,n,r] * y_sol[i]
                        objG1_nr = 0.0
                        for i in range(data['I_tot_exp']):
                            objG1_nr = objG1_nr - data['U'][i,n,r] * data['gamma1_dual'][i,n,r]
                        objD2rhs_nr = 0.0
                        objD2y_nr = 0.0
                        for i in range(data['I_tot_exp']):
                            objD2rhs_nr = objD2rhs_nr + data['M'][i,n,r] * data['delta2_dual'][i,n,r]
                            objD2y_nr = objD2y_nr - data['M'][i,n,r] * y_sol[i] * data['delta2_dual'][i,n,r]
                        objD3_nr = 0.0
                        for i in range(data['I_tot_exp']):
                            objD3_nr = objD3_nr + data['M'][i,n,r] * y_sol[i] * data['delta3_dual'][i,n,r]

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

                        self.cut_lhs.append(cplex.SparsePair(ind=ind, val=co))
                        self.senses.append('L')
                        self.cut_rhs.append(rhs_nr)
                        '''
                        print('n={:1d} r={:1d}: '.format(n,r),end='')
                        print(' z[{:1d},{:1d}] >='.format(n,r),end='') 
                        for i in range(data['I_tot_exp']):
                            print('{:7.2f}y{:1d}'.format(co[i],i),end='')
                        print('{:7.2f} (RHS)'.format(-rhs_nr))
                        '''
                        data['nDualSubproblems'] += 1
    
        data['timeDuals'] += (time.time() - t_Duals_start)
        data['nDualIterations'] += 1


class Callback():
    """Callback function for the problem.

    This callback can do two different things:
       - Separate Benders cuts at fractional solutions as user cuts
       - Separate Benders cuts at integer solutions as lazy constraints

    Everything is setup in the invoke function that is called by CPLEX.
    """

    def __init__(self, data, y, z):
        self.num_threads = data['num_threads']
        self.cutlhs = None
        self.cutrhs = None
        self.data = data
        self.y = y
        self.z = z
        # Create workerLP for Benders' cuts separation
        self.workers = [None] * data['num_threads']
        
    def separate_lazy_constraints(self, context, worker):
        '''Separate Benders cuts at integer solutions as lazy constraints.'''

        # Initialize lists of cuts
        cutlhs = []
        senses = []
        cutrhs = []

        # We only work with bounded models
        if not context.is_candidate_point():
            raise Exception('Unbounded solution')
        
        # Print number of current Benders iteration
        print('     {:4d}'.format(len(self.data['all_y'])))
        
        # Get the current y solution
        sol_y = []
        for i in range(self.data['I_tot_exp']):
            sol_y.append(context.get_candidate_point(self.y[i]))
        self.data['y'] = sol_y
        self.data['all_y'].append(sol_y)

        # Get the current z and z[n][r] solution
        sum_z = context.get_candidate_point(1)
        sol_z = []
        for n in range(self.data['N']):
            sol_z.append([])
            for r in range(self.data['R']):
                if self.data['PB_RetainedInMaster'][r] == 0:
                    sol_z[n].append(context.get_candidate_point(self.z[n][r]))
                else:
                    sol_z[n].append(0)

        # Retrieve choices of customers at current solution and subproblem objectives
        self.data['UMax'] = np.zeros([self.data['N'], self.data['R']])
        self.data['choice'] = np.zeros([self.data['N'], self.data['R']], dtype=int)
        self.data['x'] = np.zeros([self.data['I_tot_exp'], self.data['N'], self.data['R']])
        self.data['objPrimalSub'] = np.zeros([self.data['N'], self.data['R']])
        for n in range(self.data['N']):
            for r in range(self.data['R']):
                self.data['UMax'][n,r] = np.max(self.data['U'][:,n,r] * sol_y[:])
                self.data['choice'][n,r] = np.argmax(self.data['U'][:,n,r] * sol_y[:])
                self.data['x'][self.data['choice'][n,r],n,r] = 1.0
                self.data['objPrimalSub'][n,r] = self.data['markup'][self.data['choice'][n,r]] * self.data['popN'][n]/self.data['R']

        # Retrieve objective function value
        self.data['obj'] = 0.0
        for i in range(self.data['I_tot_exp']):
            self.data['obj'] = self.data['obj'] + self.data['fixed_cost'][self.data['alt'][i]] * sol_y[i]
            for n in range(self.data['N']):
                for r in range(self.data['R']):
                    self.data['obj'] = self.data['obj'] - (self.data['p'][i] - self.data['customer_cost'][self.data['alt'][i]]) * self.data['popN'][n]/self.data['R'] * self.data['x'][i,n,r]
        print('\nPrimal objective    = {:12.3f}'.format(self.data['obj']))
        if self.data['obj'] <= self.data['UB']:
            self.data['UB'] = self.data['obj']

        print('Obj Master          = {:12.3f}'.format(context.get_candidate_objective()))
        print('UB Master           = {:12.3f}'.format(self.data['UB']))

        # Decide which cuts to add (i.e. which subproblems to solve)
        cutGeneration = 1

        # Option 1: only the subproblems where the current solution is better than the corresponding z in the master
        if cutGeneration == 1:
            self.data['generateCut'] = np.zeros([self.data['N'], self.data['R']])
            countGenCut = 0
            totalSub = 0
            #print('  n     Subproblem            z')
            for n in range(self.data['N']):
                for r in range(self.data['R']):
                    if self.data['PB_RetainedInMaster'][r] == 0:
                        if -self.data['objPrimalSub'][n,r] > sol_z[n][r] + self.data['eps_slack']:
                            self.data['generateCut'][n,r] = 1
                            countGenCut += 1
                        totalSub += 1
                    #print('{:3d}   {:12.5f} {:12.5f}'.format(n, -self.data['objPrimalSub'][n,r], sol_z[n][r]))

        # Option 2: all subproblems
        elif cutGeneration == 2:
            self.data['generateCut'] = np.ones([self.data['N'], self.data['R']])
            countGenCut = self.data['N']*self.data['R']
            totalSub = self.data['N']*self.data['R']

        #######################################
        # Add presolve cuts in first iteration
        #######################################
        if self.data['AddedPresolveCuts'] == 0:
            print('\nAdding presolve cuts:')

            self.data['generateCut'] = np.ones([self.data['N'], self.data['R']])
            
            cutlhs = []
            senses = []
            cutrhs = []
            
            # Create a sorted list of solutions by objective function value
            sort_all_y = list(reversed(np.argsort(self.data['OF_all'])))
            pos = np.zeros((len(self.data['OF_all'])))
            for sol in range(len(self.data['OF_all'])):
                for p in range(len(self.data['OF_all'])):
                    if sort_all_y[p] == sol:
                        pos[sol] = p
            for n in range(len(sort_all_y)):
                print('{:3.0f}'.format(sort_all_y[n]),end='')
            print()
            for n in range(len(pos)):
                print('{:3.0f}'.format(pos[n]),end='')
            print()
            
            # Disaggregate cuts (nCuts = N*R*solutions)
            count = 0
            for solution in range(self.data['R']):
                if pos[solution] < self.data['R'] / 10.0:
                    
                    worker.separateDual(self.data['all_y'][solution], self.y, self.z)

                    for c in range(len(worker.cut_lhs)):
                        cutlhs.append(worker.cut_lhs[c])
                        senses.append(worker.senses[c])
                        cutrhs.append(worker.cut_rhs[c])

                    count += 1
                    print('{:3.0f} out of {:4.0f}'.format(count, self.data['R'] / 10.0))
                    
            context.reject_candidate(constraints=cutlhs, senses=senses, rhs=cutrhs)

            for c in range(len(cutlhs)):
                self.data['listCuts']['LHS_cut'].append(cutlhs[c])
                self.data['listCuts']['senses_cut'].append(senses[c])
                self.data['listCuts']['RHS_cut'].append(cutrhs[c])
            print('Length list cuts:{:6d}'.format(len(self.data['listCuts']['LHS_cut'])))

            self.data['AddedPresolveCuts'] = 1

        #######################################
        # Normal Benders' cut separation: solve dual subproblems for current solution
        #######################################
        else:
            worker.separateDual(sol_y, self.y, self.z)

            candidate_cutlhs = worker.cut_lhs
            candidate_senses = worker.senses
            candidate_cutrhs = worker.cut_rhs
            
            # Verify if proposed cut for n and r improves the subproblem bound for current solution
            reject = False
            violCut = 0
            evaluateLhs = [0.0 for i in range(len(candidate_cutlhs))]
            #print('  n   r           LHS  >=      RHS   Verified    RejectSol')
            for cut in range(len(candidate_cutlhs)):
                evaluateLhs[cut] = 0.0
                for i in range(len(candidate_cutlhs[cut].ind)):
                    evaluateLhs[cut] += (candidate_cutlhs[cut].val[i] * context.get_candidate_point(candidate_cutlhs[cut].ind[i]))
                if evaluateLhs[cut] > candidate_cutrhs[cut] + self.data['eps_slack']:
                    active = False
                    reject = True
                    violCut += 1
                    cutlhs.append(candidate_cutlhs[cut])
                    senses.append('L')
                    cutrhs.append(candidate_cutrhs[cut])
                else:
                    active = True
                    if cutGeneration == 2:
                        cutlhs.append(candidate_cutlhs[cut])
                        senses.append('L')
                        cutrhs.append(candidate_cutrhs[cut])                    
                '''
                n, r = divmod(candidate_cutlhs[cut].ind[len(candidate_cutlhs[cut].ind)-1] - 2, self.data['R']) 
                print('{:3d} {:3d}  {:12.5f} {:12.5f} '.format(n, r, -evaluateLhs[cut], -candidate_cutrhs[cut]), end='     ')
                print(active, end='         ')
                print(reject)
                '''

            print('countGenCut     : {:5d} out of {:5d}'.format(countGenCut, totalSub))
            print('Violated cuts   : {:5d} out of {:5d}'.format(violCut, len(candidate_cutlhs)))

            # Add subset cuts
            if self.data['subsetCuts'] == 'Yes' and countGenCut > 0:
                lhs_cuts = []
                sense_cuts = []
                rhs_cuts = []
                functions.subsetCut(self.data, sol_y, self.y, lhs_cuts, sense_cuts, rhs_cuts)

                if len(lhs_cuts) > 0:
                    cutlhs.append(lhs_cuts[-1])
                    senses.append(sense_cuts[-1])
                    cutrhs.append(rhs_cuts[-1])
            
            # Sanity check to reject solutions
            if self.data['obj'] > context.get_candidate_objective() + self.data['eps_slack'] and reject == False:
                print('\nERROR?\n')
                reject = True

            if reject:
                context.reject_candidate(constraints=cutlhs, senses=senses, rhs=cutrhs)
                for c in range(len(cutlhs)):
                    self.data['listCuts']['LHS_cut'].append(cutlhs[c])
                    self.data['listCuts']['senses_cut'].append(senses[c])
                    self.data['listCuts']['RHS_cut'].append(cutrhs[c])
            print('Length list cuts:{:6d}'.format(len(self.data['listCuts']['LHS_cut'])))


    def invoke(self, context):
        """Whenever CPLEX needs to invoke the callback it calls this
        method with exactly one argument: an instance of
        cplex.callbacks.Context.
        """
        try:
            thread_id = context.get_int_info(cplex.callbacks.Context.info.thread_id)
            print('\nthread_id = ' + str(thread_id), end='    ')
            print('Context ={:3d} = '.format(context.get_id()), end='')

            if context.get_id() == cplex.callbacks.Context.id.thread_up:
                print('thread_up')
                self.workers[thread_id] = WorkerLP(self.data)
            
            elif context.get_id() == cplex.callbacks.Context.id.thread_down:
                print('thread_down')
                self.workers[thread_id] = None
            
            elif context.get_id() == cplex.callbacks.Context.id.relaxation:
                print('Relaxation        Separate_user_cuts')
                self.separate_user_cuts(context, self.workers[thread_id])
            
            elif context.get_id() == cplex.callbacks.Context.id.candidate:
                print('Candidate         Separate_lazy_constraints', end='')
                self.separate_lazy_constraints(context, self.workers[thread_id])
            
            else:
                print('Callback called in an unexpected context {}'.format(context.get_id()))
        
        except:
            info = sys.exc_info()
            print('#### Exception in callback: ', info[0])
            print('####                        ', info[1])
            print('####                        ', info[2])
            traceback.print_tb(info[2], file=sys.stdout)
            raise


def create_master_ilp(model, data, y, z):
    '''
    This function creates the master ILP
    Optimize over the binary variables determining location and price.
    '''
    print('\nMASTER PROBLEM (BENDERS)')
    
    ##########################################
    ##### ----- OBJECTIVE FUNCTION ----- #####
    ##########################################
    model.objective.set_sense(model.objective.sense.minimize)

    ##########################################
    ##### ----- DECISION VARIABLES ----- #####
    ##########################################

    # OBJECTIVE FUNCTION
    objVar = []
    typeVar = []
    nameVar = []
    lbVar = []
    ubVar = []

    objVar.append(1.0)
    typeVar.append(model.variables.type.continuous)
    nameVar.append('obj')
    lbVar.append(-cplex.infinity)
    ubVar.append(cplex.infinity)

    objVar.append(0.0)
    typeVar.append(model.variables.type.continuous)
    nameVar.append('z')
    lbVar.append(-cplex.infinity)
    ubVar.append(cplex.infinity)

    count = 2
    for n in range(data['N']):
        z.append([])
        for r in range(data['R']):
            if data['PB_RetainedInMaster'][r] == 0:
                z[n].append(count)
                count += 1
                objVar.append(0.0)
                typeVar.append(model.variables.type.continuous)
                nameVar.append('z[' + str(n) + ']' + '[' + str(r) + ']')
                lbVar.append(-cplex.infinity)
                ubVar.append(cplex.infinity)
            elif data['PB_RetainedInMaster'][r] == 1:
                z[n].append(-1)
    
    if data['PartialBenders'] == 'Yes':
        for r in range(data['R']):
            if data['PB_RetainedInMaster'][r] == 1:
                for n in range(data['N']):
                    for i in range(data['I_tot_exp']):
                        objVar.append(0.0)
                        typeVar.append(model.variables.type.continuous)
                        nameVar.append('x[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                        lbVar.append(0.0)
                        ubVar.append(1.0)

    model.variables.add(obj = [objVar[i] for i in range(len(objVar))],
                        types = [typeVar[i] for i in range(len(typeVar))],
                        lb = [lbVar[i] for i in range(len(typeVar))],
                        ub = [ubVar[i] for i in range(len(typeVar))],
                        names = [nameVar[i] for i in range(len(nameVar))])

    # FACILITY LOCATION VARIABLES
    objVar = []
    typeVar = []
    nameVar = []
    count = 0
    
    # Facility location variables
    for i in range(data['I_tot_exp']):
        objVar.append(0.0)
        typeVar.append(model.variables.type.binary)
        nameVar.append('y[' + str(i) + ']')
        y.append(model.variables.get_num() + count)
        count += 1

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

    ### --- Constraint (0): value of the objective function --- ###
    ind = []
    co = []
    for i in range(data['I_tot_exp']):
        ind.append(nameToIndex['y[' + str(i) + ']'])
        co.append(data['fixed_cost'][data['alt'][i]])
    for r in range(data['R']):
        if data['PB_RetainedInMaster'][r] == 1:
            for n in range(data['N']):
                for i in range(data['I_tot_exp']):
                    if data['operator'][data['alt'][i]] == data['optimizer']:
                        co.append(-(data['p'][i] - data['customer_cost'][data['alt'][i]]) * data['popN'][n] / data['R'])
                        ind.append(nameToIndex['x[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']'])
    ind.append(nameToIndex['z'])
    co.append(1.0)
    ind.append(nameToIndex['obj'])
    co.append(-1.0)
    indicesConstr.append(ind)
    coefsConstr.append(co)
    sensesConstr.append('E')
    rhsConstr.append(0.0)
    
    ### --- Constraint (1): master objective function must be lower than current upper bound --- ###
    indicesConstr.append([nameToIndex['obj']])
    coefsConstr.append([1.0])
    sensesConstr.append('L')
    rhsConstr.append(data['UB'])

    ### --- Constraint (2): master objective function must be higher than current lower bound --- ###
    indicesConstr.append([nameToIndex['obj']])
    coefsConstr.append([1.0])
    sensesConstr.append('G')
    rhsConstr.append(data['LB'])

    ### --- Value z --- ###
    ind = []
    co = []
    for r in range(data['R']):
        if data['PB_RetainedInMaster'][r] == 0:
            for n in range(data['N']):
                ind.append(nameToIndex['z[' + str(n) + ']' + '[' + str(r) + ']'])
                co.append(1.0)
    ind.append(nameToIndex['z'])
    co.append(-1.0)
    indicesConstr.append(ind)
    coefsConstr.append(co)
    sensesConstr.append('E')
    rhsConstr.append(0.0)

    ### --- Lower bound objective function subproblems --- ###
    for r in range(data['R']):
        if data['PB_RetainedInMaster'][r] == 0:
            for n in range(data['N']):
                indicesConstr.append([nameToIndex['z[' + str(n) + ']' + '[' + str(r) + ']']])
                coefsConstr.append([1.0])
                sensesConstr.append('G')
                # (maximum contribution of given n,r to objective function)
                rhsConstr.append(np.min(-(data['p'][:] - data['customer_cost'][data['alt'][:]]) * data['popN'][n]/data['R']))

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
    ### ------- Partial Benders constraints ------- ###
    ###################################################

    if data['PartialBenders'] == 'Yes':
        for r in range(data['R']):
            if data['PB_RetainedInMaster'][r] == 1:

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
    
    print('N variables in Master problem   : {:4d}'.format(model.variables.get_num()))
    print('N constraints in Master problem : {:4d}'.format(model.linear_constraints.get_num()))


def branch_and_Benders_cut(master, data, y, z):

    print('\nBenders\' cuts separated to cut off: ', end=' ')
    if data['sep_frac_sols'] == 1:
        print('Integer and fractional infeasible solutions.')
    elif data['sep_frac_sols'] == 0:
        print('Only integer infeasible solutions.')
    else:
        raise ValueError('sep_frac_sols must be either "0" or "1"')

    # Set number of threads
    master.parameters.threads.set(data['num_threads'])
    print('\n################\nTHREADS = ', end='')
    print(master.parameters.threads.get())
    print('################\n')

    # Define structure of the callback
    problem_callback = Callback(data, y, z)
    contextmask = cplex.callbacks.Context.id.thread_up
    contextmask |= cplex.callbacks.Context.id.thread_down
    contextmask |= cplex.callbacks.Context.id.candidate
    if data['sep_frac_sols'] == 1:
        contextmask |= cplex.callbacks.Context.id.relaxation
    master.set_callback(problem_callback, contextmask)

    # Solve the model
    master.solve()


def singleCustomers(data):

    data['all_y_customer'] = []

    # Optimize for each customer independently
    for n in range(data['N']):
        OF_customer, y_customer = model_discrete_compact_assortment.modelOneCustomer(data,n)
        OF_all = functions.objSolution(data, y_customer)
        data['all_y'].append(list(y_customer))
        data['all_y_customer'].append(list(y_customer))
        data['OF_customer'].append(OF_customer)
        data['OF_all'].append(OF_all)
    data['best_y_scenario'] = data['all_y'][np.argmax(data['OF_all'])]

    # Print optimal solutions for each scenario
    print('\n  n    LowInc  OrigCity      OF   OFall',end='')
    for i in range(data['I_tot_exp']):
        print('{:2d}'.format(i),end='')
    for n in range(data['N']):
        print('\n{:3d}      {:4.0f}      {:4.0f}{:8.2f}{:8.2f}'
            .format(n, data['LOW_INC'][n], data['ORIGIN'][n], data['OF_customer'][n], data['OF_all'][n]),end='')
        for i in range(data['I_tot_exp']):
            print('{:2.0f}'.format(data['all_y'][n][i]),end='')
    print('\n\nSum of optimal solutions for single customers (LB)              : {:8.2f}'.format(-np.sum(data['OF_customer'])))
    print('Best optimal solution for single customer on all customers (UB) : {:8.2f}\n'.format(-np.max(data['OF_all'])))

def singleScenarios(data):

    data['all_y_scenario'] = []

    count = len(data['OF_all'])
    # Solve each scenario independently
    for s in range(data['R']):
        OF_scenario, y_scenario = model_discrete_compact_assortment.modelOneScenario(data,s)
        OF_all = functions.objSolution(data, y_scenario)
        data['all_y'].append(list(y_scenario))
        data['all_y_scenario'].append(list(y_scenario))
        data['OF_scenario'].append(OF_scenario)
        data['OF_all'].append(OF_all)
    data['best_y_scenario'] = data['all_y'][np.argmax(data['OF_all'])]

    # Print optimal solutions for each scenario
    print('\n  r      OF   OFall',end='')
    for i in range(data['I_tot_exp']):
        print('{:2d}'.format(i),end='')
    for r in range(data['R']):
        print('\n{:3d}{:8.2f}{:8.2f}'.format(r, data['OF_scenario'][r], data['OF_all'][count+r]),end='')
        for i in range(data['I_tot_exp']):
            print('{:2.0f}'.format(data['all_y'][count+r][i]),end='')
    print('\n\nAverage of optimal solutions of single scenarios (LB)          : {:8.2f}'.format(-np.average(data['OF_scenario'])))
    print('Best optimal solution of single scenario on all scenarios (UB) : {:8.2f}\n'.format(-np.max(data['OF_all'])))

def presolveCuts(data,master,y,z,z_agg):
    
    # Create a sorted list of solutions by objective function value
    sort_all_y = list(reversed(np.argsort(data['OF_all'])))
    pos = np.zeros((len(data['OF_all'])))
    for sol in range(len(data['OF_all'])):
        for p in range(len(data['OF_all'])):
            if sort_all_y[p] == sol:
                pos[sol] = p

    # Disaggregate cuts (nCuts = N*R*solutions)
    if data['cuts'] == 'Disaggregate':
        dualFollower = functions.createDualFollower(data)
        for solution in range(len(data['all_y'])):
            if pos[solution] <= data['R'] / 10.0:
                functions.dualWorker(data, solution, dualFollower)
                LHS_cuts, senses_cuts, RHS_cuts = functions.disaggregateCuts(data, solution, y, z)
                for c in range(len(LHS_cuts)):
                    data['listCuts']['LHS_cut'].append(LHS_cuts[c])
                    data['listCuts']['senses_cut'].append(senses_cuts[c])
                    data['listCuts']['RHS_cut'].append(RHS_cuts[c])

    # Aggregate cuts (nCuts = solutions)
    elif data['cuts'] == 'Aggregate':
        dualFollower = functions.createDualFollowerAggregate(data)
        for solution in range(len(data['all_y'])):
            if pos[solution] <= data['R'] / 10.0:
                functions.dualWorkerAggregate(data, solution, dualFollower)
                LHS_cuts, senses_cuts, RHS_cuts = functions.aggregateCut(data, solution, y, z_agg)
                for c in range(len(LHS_cuts)):
                    data['listCuts']['LHS_cut'].append(LHS_cuts[c])
                    data['listCuts']['senses_cut'].append(senses_cuts[c])
                    data['listCuts']['RHS_cut'].append(RHS_cuts[c])
    
    # Subset cuts (R)
    if data['subsetCuts'] == 'Yes':
        for solution in range(len(data['all_y'])):
            functions.subsetCut(data, data['all_y'][solution], y,
                                data['listCuts']['LHS_cut'], data['listCuts']['senses_cut'], data['listCuts']['RHS_cut'])

    # Add initial cuts as constraints / lazy constraints
    print('Number of initial cuts :{:5d}'.format(len(data['listCuts']['LHS_cut'])))
    if data['AddPresolveCutsAs'] == 'Constraints':
        master.linear_constraints.add(
            lin_expr = [data['listCuts']['LHS_cut'][c] for c in range(len(data['listCuts']['LHS_cut']))],
            senses = [data['listCuts']['senses_cut'][c] for c in range(len(data['listCuts']['senses_cut']))],
            rhs = [data['listCuts']['RHS_cut'][c] for c in range(len(data['listCuts']['RHS_cut']))])
    elif data['AddPresolveCutsAs'] == 'LazyConstraints':
        master.linear_constraints.advanced.add_lazy_constraints(
            lin_expr = [data['listCuts']['LHS_cut'][c] for c in range(len(data['listCuts']['LHS_cut']))],
            senses = [data['listCuts']['senses_cut'][c] for c in range(len(data['listCuts']['senses_cut']))],
            rhs = [data['listCuts']['RHS_cut'][c] for c in range(len(data['listCuts']['RHS_cut']))])
            
def additionalSolutions(data):
    
    print('\nGenerate additional solutions to cover all master variables:')
    count_tot = 0
    usedScenario = np.zeros(([data['I_tot_exp'], data['R']]))
    for i in range(data['I_tot_exp']):
        for r in range(data['R']):
            usedScenario[i,r] = data['all_y'][r][i]
    for i in range(data['I_tot_exp']):
        print('{:3.0f}'.format(np.sum(usedScenario[i,:])), end='')
    print()
    for i in range(data['I_tot_exp']):
        # Start with a minimum number of solutions that include each i as open facility
        while round(np.sum(usedScenario[i,:]),data['round']) < data['minSolutionsWithI']:
            scen = random.randint(0,data['R']-1)
            if usedScenario[i,scen] == 0.0:
                print('Alt {:3d}    '.format(i), end='')
                usedScenario[i,scen] = 1.0
                for j in range(data['I_tot_exp']):
                    if j != i:
                        data['list_open'][j] = 0.0
                    elif j == i:
                        data['list_open'][i] = 1.0
                OF_opt, y_opt = model_discrete_compact_assortment.modelOneScenario(data,scen)
                OF_all = functions.objSolution(data, y_opt)
                data['all_y'].append(list(y_opt))
                data['OF_scenario'].append(OF_opt)
                data['OF_all'].append(OF_all)
                for j in range(data['I_tot_exp']):
                    if usedScenario[j,scen] == 0.0 and y_opt[j] == 1:
                        usedScenario[j,scen] = 1.0
                count_tot += 1
    print('Additional solutions: {:4d} \nTotal solutions: {:4d}'.format(count_tot, len(data['all_y'])))

def removeDuplicates(data):
    if len(data['all_y']) > 0:
        print('\nN solutions        : {:4d}'.format(len(data['all_y'])))
        data['all_y'] = list(data['all_y'] for data['all_y'],_ in itertools.groupby(data['all_y']))
        print('N unique solutions : {:4d}'.format(len(data['all_y'])))

def MIPWarmStart(master, data, y):
    # Starting from a solution: MIP starts: https://www.ibm.com/docs/en/icos/20.1.0?topic=mip-starting-from-solution-starts
    #master.MIP_starts.effort_level.auto = 0
    #master.MIP_starts.effort_level.check_feasibility = 1
    #master.MIP_starts.effort_level.solve_fixed = 2
    #master.MIP_starts.effort_level.solve_MIP = 3
    #master.MIP_starts.effort_level.repair = 4
    
    for s in range(len(data['all_y'])):
        master.MIP_starts.add(cplex.SparsePair(ind = y, val = data['all_y'][s]), 2)


def preprocessing(data):

    data_file.printCustomers(data)

    # Calculate utilities for all alternatives (1 per discrete price)
    functions.discretePriceAlternativeDuplication(data)
    data_file.preprocessUtilities(data)
    functions.calcDuplicatedUtilities(data)
    
    # Calculate markups to be used to generate subproblem cuts
    functions.calculateMarkup(data)

    # Rank alternatives for all n and r
    functions.rankAlternativesAll(data)

    # Big M
    data['M'] = data['U']

    # Set initial bounds
    data['UB'] = 1000000.0
    data['LB'] = -1000000.0
    data['eps_slack'] = 10**(-5)


def benders(data):

    # Initialize count and time variables
    data['timeSubsetCuts'] = 0.0
    data['timeDuals'] = 0.0
    data['nDualIterations'] = 0
    data['nDualSubproblems'] = 0

    # Initialize vector of indexes of master variables
    y = []
    z = []
    z_agg = 1

    # Initialize vector of all y solutions found during the branch-and-cut
    data['all_y'] = []
    data['OF_all'] = []

    # Initialize list of cuts
    data['listCuts'] = {}
    data['listCuts']['LHS_cut'] = []
    data['listCuts']['senses_cut'] = []
    data['listCuts']['RHS_cut'] = []

    if data['BendersPresolve'] == 'Yes':
        t_start_presolve1 = time.time()

        ###################################################
        # Enumerate all solutions with 1, 2 facilities
        ###################################################
        if data['BendersEnumerateSmall'] == 'Yes':
            functions.allSolutionsOneFacility(data)
            functions.allSolutionsTwoFacilities(data)
            #data['UB'] = -max(data['bestOF1'], data['bestOF2'])
        
        if data['N'] >= data['minNClustering']:
            ###################################################
            # Solve for individual (groups of) customers
            ###################################################        
            if data['BendersSingleCustomers'] == 'Yes':
                data['OF_customer'] = []
                singleCustomers(data)
                data['UB'] = -max(-data['UB'], np.max(data['OF_all']))
                #data['LB'] = -np.sum(data['OF_customer'])

        if data['R'] >= data['minRClustering']:
            ###################################################
            # Solve for individual scenarios
            ###################################################
            if data['BendersSingleScenarios'] == 'Yes':
                data['OF_scenario'] = []
                singleScenarios(data)
                #data['UB'] = -max(-data['UB'], np.max(data['OF_all']))
                #data['LB'] = -np.average(data['OF_scenario'])

            #########################################################
            # Scenario clustering
            #########################################################
            if data['Clustering'] == 'Yes':
                if data['N'] >= data['minNClustering']:
                    # Calculate opportunity cost distance function
                    scenario_clustering.opportunityCostCustomers(data)
                    # Apply k-medoids clustering
                    scenario_clustering.kMedoidsCustomer(data)
                
                if data['R'] >= data['minRClustering']:
                    # Calculate opportunity cost distance function
                    scenario_clustering.opportunityCostScenarios(data)
                    # Apply k-medoids clustering
                    scenario_clustering.kMedoidsScenario(data)

                if data['BendersPresolveCuts'] == 'Yes':
                    data['presolveCutCustomer'] = np.zeros([data['N']])
                    data['presolveCutScenario'] = np.zeros([data['R']])
                    for n in data['list_medoids_N']:
                        data['presolveCutCustomer'][n] = 1
                    for r in data['list_medoids_R']:
                        data['presolveCutScenario'][r] = 1

                if data['PartialBenders'] == 'Yes':
                    data['PB_RetainedInMaster'] = np.zeros([data['R']])
                    for s in data['list_medoids_R']:
                        data['PB_RetainedInMaster'][s] = 1
            else:
                data['presolveCutCustomer'] = np.ones([data['N']])
                data['presolveCutScenario'] = np.ones([data['R']])

        #############################################################
        # Generate additional solutions to cover all master variables
        #############################################################
        if data['additionalSolutions'] == 'Yes':
            additionalSolutions(data)

        data['timePresolve'] = time.time() - t_start_presolve1
        print('\n\nTime presolve 1     : {:10.3f}'.format(data['timePresolve']))

    #########################################################
    # Create master ILP
    #########################################################
    master = cplex.Cplex()
    create_master_ilp(master, data, y, z)
    master.parameters.preprocessing.presolve.set(0)
    master.parameters.timelimit.set(172000.0)

    if data['BendersPresolve'] == 'Yes':
        t_start_presolve2 = time.time()

        # Remove duplicate solutions
        #removeDuplicates(data)

        ##############################################################
        # Generate initial cuts from solutions of scenario subproblems
        ##############################################################
        if data['BendersPresolveCuts'] == 'Yes':
            presolveCuts(data,master,y,z,z_agg)

        ###################################################
        # Generate incompatibility cuts
        ###################################################
        if data['BendersEnumerateSmall'] == 'Yes':
            functions.incompatibilityCuts(data,y,master)

        #####################
        # MIP warm start
        #####################
        if data['WarmStart'] == 'Yes':
            MIPWarmStart(master, data, y)
        
        data['timePresolve'] += (time.time() - t_start_presolve2)
        print('\n\nTime presolve 2     : {:10.3f}'.format(time.time() - t_start_presolve2))
        print('\nTotal time presolve : {:10.3f}'.format(data['timePresolve']))

    #########################################################
    # Branch-and-Benders-cut
    #########################################################
    branch_and_Benders_cut(master, data, y, z)

    # Save and print solution
    solution = master.solution
    prints.printBest(data, solution)   

    if solution.get_status() == solution.status.MIP_optimal:
        print('\nOptimal solution found')
        print('Solution status:{:4.0f}'.format(solution.get_status()))
        print('Objective value: {:10.3f}'.format(solution.get_objective_value()))
    else:
        print('\nSolution status:{:4.0f}'.format(solution.get_status()))

    # Print computational times
    if data['BendersPresolve'] == 'Yes':
        print('\nTime presolve      : {:10.4f} sec'.format(data['timePresolve']))
    print('Time duals         : {:10.4f} sec'.format(data['timeDuals']))
    print('nDualIterations    :{:7d}'.format(data['nDualIterations']))
    print('nDualSubproblems   :{:7d}'.format(data['nDualSubproblems']))


def main():

    nSimulations = 1

    for seed in range(1,nSimulations+1):

        if nSimulations > 1:
            print('\n\n\n\n---------\nSEED ={:3d}\n---------\n\n'.format(seed))
        
        t_0 = time.time()
        
        # Read instance and print aggregate customer data
        data = data_file.getData(seed)

        # Preprocessing alternatives and utilities
        preprocessing(data)

        # Set initial solution
        for i in range(data['I_tot_exp']):
            data['y'][i] = 0.0
            data['list_open'][i] = 0.0
        for i in range(data['I_out_exp']):
            data['y'][i] = 1.0

        parameters.BendersParameters(data)

        #################################### 
        # Main algorithm
        ####################################
        t_1 = time.time()
        benders(data)
        t_2 = time.time()
        
        print('\nTotal computational time: {:8.2f} sec'.format(t_2 - t_1))
        print('Length list of cuts: {:6d}'.format(len(data['listCuts']['LHS_cut'])))

if __name__ == "__main__":
    main()
