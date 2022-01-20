'''
MAIN.PY: run this to test the heuristic algorithm to find
approximate equilibria for regulated markets,
described in the following article:
S. Bortolomiol, V. Lurkin, M. Bierlaire (2021).
Price‑based regulation of oligopolistic markets under
discrete choice models of demand. Transportation.
https://link.springer.com/article/10.1007/s11116-021-10217-0

CASE STUDY: INTERCITY TRAVEl REGULATED COMPETITION
Case study derived from the following article:
Cascetta E, Coppola P (2012) An elastic demand schedule-based
multimodal assignment model for the simulation of high speed
rail (HSR) systems. EURO J. Transportation Logist. 1(1-2):3-27.
'''

# Code for the fixed-point iteration algorithm with regulator
# (simultaneous game solved in a sequential manner)

import time

# Models
import algorithm_regulation
import nested_logit

# Data
import data_intercity as data_file


if __name__ == '__main__':

    data = {}

    # Define parameters of the algorithm
    data_file.setAlgorithmParameters(data)

    data['Seed'] = 0

    for experiment in range(2):         # Change here to perform sensitivity analyses
    #for experiment in range(21):       # on SCC or budget
        data['Seed'] = data['Seed'] + 1

        #Read instance
        data_file.getData(data)

        t_0 = time.time()

        #Set SCC
        data['social_cost_of_carbon'] = 0.100 * experiment          #Modify/comment here for testing
        
        #Set budget
        #data['Budget'] = 2.0*experiment*data['Pop']                 #Modify/comment here for testing

        #for it in range(26):
        for it in range(2):
            #with open('Regulation_MultiObj_SCC{:03.0f}.txt'.format(data['social_cost_of_carbon']*1000), "w") as f:
            #with open('Regulation_AllMultiObj_SCC{:03.0f}.txt'.format(data['social_cost_of_carbon']*1000), "w") as f:
            with open('Regulation_Emissions_B{:05.0f}N_{:02.0f}.txt'.format(data['Budget'],it), "w") as f:

                #Precompute exogenous terms
                data_file.preprocessUtilities(data)

                #Calculate initial values of logsum terms
                nested_logit.logsumNestedLogitRegulator(data)
                
                #Print demand data
                data_file.printCustomers(data)

                #RUN SIMULTANEOUS FIXED POINT ITERATION ALGORITHM
                results = algorithm_regulation.heuristic_algorithm_regulation(data)

                print('\n\nTotal runtime: {:8.2f}\n'.format(time.time() - t_0))
