# Print solutions
import numpy as np

def printBest(data, solution):

    print()

    # SAVE OPTIMAL SOLUTION
    data['best_obj'] = solution.get_objective_value()
    data['best_facilities'] = data['y']
    for i in range(data['I_tot_exp']):
        data['best_facilities'][i] = solution.get_values('y[' + str(i) + ']')

    # RETRIEVE MAX UTILITIES OF CUSTOMERS
    data['UMax'] = np.zeros([data['N'], data['R']])
    data['choice'] = np.zeros([data['N'], data['R']], dtype=int)
    data['x'] = np.zeros([data['I_tot_exp'], data['N'], data['R']])
    for n in range(data['N']):
        for r in range(data['R']):
            data['UMax'][n,r] = np.max(data['U'][:,n,r] * data['best_facilities'][:])
            data['choice'][n,r] = np.argmax(data['U'][:,n,r] * data['best_facilities'][:])
            data['x'][data['choice'][n,r],n,r] = 1.0

    # RETRIEVE DEMAND
    data['best_demand'] = np.zeros([data['I_tot_exp']])
    for i in range(data['I_tot_exp']):
        for n in range(data['N']):
            for r in range(data['R']):
                if data['choice'][n,r] == i:
                    data['best_demand'][i] += (data['popN'][n]/data['R'])

    ### PRINT FACILITIES, PRICES, DEMANDS, PROFITS
    print('\nObjective function value : {:12.4f}'.format(data['best_obj']))
    print('\nAlt  Name    Supplier   Facility      Price      Demand  Market share')
    for i in range(data['I_tot_exp']):
        print('{:3d}  {:6s}       {:2d}        {:4.0f}    {:7.4f}    {:8.3f}       {:7.4f}'
            .format(i, data['name_mapping'][data['alt'][i]], data['operator'][data['alt'][i]],
            data['best_facilities'][i], data['p'][i],
            data['best_demand'][i], data['best_demand'][i] / data['Pop']))

    ### PRINT SUBPROBLEM CONTRIBUTIONS
    data['z_opt'] = np.zeros([data['N'], data['R']])
    for r in range(data['R']):
        for n in range(data['N']):
            if data['PB_RetainedInMaster'][r] == 0:
                data['z_opt'][n,r] = solution.get_values('z[' + str(n) + ']' + '[' + str(r) + ']')
            else:
                data['z_opt'][n,r] = -1
    
    ### PRINT CHOICES IN RETAINED PROBLEMS
    data['x_opt'] = np.zeros([data['I_tot_exp'], data['N'], data['R']])
    for r in range(data['R']):
        for n in range(data['N']):
            for i in range(data['I_tot_exp']):
                if data['PB_RetainedInMaster'][r] == 1:
                    data['x_opt'][i,n,r] = solution.get_values('x[' + str(i) + ']' + '[' + str(n) + ']' + '[' + str(r) + ']')
                else:
                    data['x_opt'][i,n,r] = -1
    print()

def printAllSolutions(data):

    print('\n\nALL INTEGER SOLUTIONS FOUND IN BRANCH-AND-BOUND TREE')
    print('  Sol ', end='')
    for i in range(data['I_tot_exp']):
        print('{:2d}'.format(i), end='')
    for sol in range(len(data['all_y'])):
        print('\n{:5d} '.format(sol), end='')
        for i in range(data['I_tot_exp']):
            print('{:2.0f}'.format(data['all_y'][sol][i]), end='')
    print()