# Calculate bounds on the utilities of the customers

# General
import numpy as np

# Data
#import Data_Parking_N80_I14 as data_file
import Data_Parking_N08_I10 as data_file


def updateUtilityBounds(data):
    
    # Utility bounds
    lb_U = np.empty([data['I_tot'], data['N'], data['R']])
    ub_U = np.empty([data['I_tot'], data['N'], data['R']])
    lb_Umin = np.full((data['N'], data['R']), np.inf)
    ub_Umax = np.full((data['N'], data['R']), -np.inf)

    M = np.empty([data['N'], data['R']])

    for n in range(data['N']):
        for r in range(data['R']):
            for i in range(data['I_tot']):
                if data['endo_coef'][i,n,r] <= 0:
                    lb_U[i,n,r] = data['endo_coef'][i,n,r] * data['ub_p'][i] + data['exo_utility'][i,n,r] + data['xi'][i,n,r]
                    ub_U[i,n,r] = data['endo_coef'][i,n,r] * data['lb_p'][i] + data['exo_utility'][i,n,r] + data['xi'][i,n,r]
                else:
                    lb_U[i,n,r] = data['endo_coef'][i,n,r] * data['lb_p'][i] + data['exo_utility'][i,n,r] + data['xi'][i,n,r]
                    ub_U[i,n,r] = data['endo_coef'][i,n,r] * data['ub_p'][i] + data['exo_utility'][i,n,r] + data['xi'][i,n,r]

            # Bounds for each customer, for each draw
            lb_Umin[n,r] = np.min(lb_U[:,n,r])
            ub_Umax[n,r] = np.max(ub_U[:,n,r])

            # Calcule the big-M values
            M[n, r] = ub_Umax[n, r] - lb_Umin[n, r]

    data['lb_U'] = lb_U
    data['ub_U'] = ub_U
    data['lb_Umin'] = lb_Umin
    data['ub_Umax'] = ub_Umax
    data['M_U'] = M


def updateUtilityBoundsFixed(data):
    
    # Utility bounds
    lb_Umin = np.empty([data['N'], data['R']])
    ub_Umax = np.empty([data['N'], data['R']])
    M = np.empty([data['N'], data['R']])

    for n in range(data['N']):
        for r in range(data['R']):

            # Bounds for each customer, for each draw
            lb_Umin[n, r] = np.min(data['U'][:, n, r])
            ub_Umax[n, r] = np.max(data['U'][:, n, r])

            # Calcule the big-M values
            M[n, r] = ub_Umax[n, r] - lb_Umin[n, r]

    data['M_nr'] = M

    
if __name__ == '__main__':

    nSimulations = 1

    for seed in range(1,nSimulations+1):

        print('\n\n\n\n\n---------\nSEED ={:3d}\n---------\n\n'.format(seed))

        # Get the data and preprocess
        data = data_file.getData(seed)
        
        #Precompute exogenous terms
        data_file.preprocessUtilities(data)
                
        #Calculate utility bounds
        updateUtilityBounds(data)
