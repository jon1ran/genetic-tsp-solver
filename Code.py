import Reporter
import numpy as np
import random
import matplotlib.pyplot as plt
from numba import float64, int32, njit


# Modify the class name to match your student number .

class r0 :
    
    def __init__ ( self ):
        self.reporter = Reporter . Reporter ( self.__class__.__name__)
        
        self.lambdaa = 100
        self.mu = self.lambdaa

        #Pop inicialization
        self.rand_prop = 0.9#The closer to 1, the higher proportion of random individuals
        
        #For k-opt
        self.hm = 500
        
        #For fitness sharing
        self.sigma = 0.9 #Initial distance
        self.sigma_threshold = 0.5 #Distance lower bound
        self.alphaa = 1/2
        
        #Don't change
        self.iterations = 1
    
    
    #The fitness function
    @staticmethod
    @njit(float64(int32[:], float64[:,:]))
    def fitness(x, distanceMatrix):
        val = distanceMatrix[x[distanceMatrix.shape[0]-1]][x[0]]
        if np.isinf(val):
            return np.inf
        for i in range(distanceMatrix.shape[0]-1):
            if np.isinf(distanceMatrix[x[i]][x[i+1]]):
                return np.inf
            val += distanceMatrix[x[i]][x[i+1]]
        return val

    #Function that given a population orders it according to fitness, and original indexes ordered
    def order_fitness(self, population, distanceMatrix):
        fit = np.apply_along_axis(self.fitness, axis=1, arr=population, distanceMatrix=distanceMatrix)
        index_order = np.argsort(fit)
        ranked = population[index_order]
        return ranked, index_order, fit[index_order]
    
    #Population initialitation
    def init_population(self, distanceMatrix: np.ndarray):
        population = np.empty((1, distanceMatrix.shape[0]), dtype=np.int32)
        
        while population.shape[0]-1 != self.lambdaa: 
            
            x = np.empty(distanceMatrix.shape[0], dtype=np.int32)
            visited = np.zeros(distanceMatrix.shape[0], dtype=bool)
            current_city = 0 #The path is always going to start from 0
            x[0] = current_city
            visited[current_city] = True
            is_valid = True
            
            for i in range(1, distanceMatrix.shape[0]):
                
                finite_indexes = np.isfinite(distanceMatrix[current_city])
                finite_and_no_visited = np.logical_and(finite_indexes, ~visited)
                index_ordered = np.argsort(distanceMatrix[current_city][finite_and_no_visited])
                finite_indexes_ordered = np.where(finite_and_no_visited)[0][index_ordered]
                
                if finite_indexes_ordered.size == 0:
                    is_valid = False
                    self.e *= 0.999
                    print(self.e)
                    break
                
                linear_space = np.linspace(1, 10e-10, finite_indexes_ordered.size)
                
                if population.shape[0]-1 < int(self.lambdaa * (1 - self.rand_prop)):
                    weights = np.power(linear_space, self.e)
                else:
                    weights = np.power(linear_space, 0)
                    
                sum_weights = np.sum(weights)
                
                if sum_weights > 0:
                    probs = weights / sum_weights
                else:
                    probs = np.ones(finite_indexes_ordered.size) / finite_indexes_ordered.size
                
                current_city = np.random.choice(finite_indexes_ordered, p=probs)
                x[i] = current_city
                visited[current_city] = True
            
            if is_valid and np.isfinite(distanceMatrix[x[distanceMatrix.shape[0]-1]][x[0]]):
                population = np.vstack([population, x])
                
                
        population = population[1:]
        
        linear_space = np.linspace(1, 10e-10, population.shape[0])
        probs = np.power(linear_space, 0)
        probs = probs/np.sum(probs)
        
        #K
        population = np.column_stack((population, np.random.choice(np.round(np.linspace(int(population.shape[0] * 0.4), int(population.shape[0] * 0.8), population.shape[0])).astype(int), size=population.shape[0], p=probs)))
        #K-opt
        population = np.column_stack((population, np.random.randint(2, 4, population.shape[0], dtype=np.int32)))
        #Mutation
        population = np.column_stack((population, np.random.choice(np.round(np.linspace(1,10, population.shape[0])).astype(int), size=population.shape[0], p=probs)))
        return population[1:]
    
    #Top - k
    def k_tournament_selection(self, population):
        n = population.shape[0]
        selected_indices = []

        # Perform k-tournament selection
        for _ in range(2*self.mu):
            # Randomly select k individuals for the tournament
            tournament_indices = np.random.choice(n, self.k, replace=False)
            selected_indices.append(np.min(tournament_indices)) #Since population is already fit sorted
            
        # Select the individuals from the population using the selected indices
        selected_population = population[selected_indices, :]
    
        return selected_population
    
    @staticmethod
    @njit(int32[:](int32[:], int32[:], int32))
    def edge_crossover(parent1, parent2, n):
        
        offspring = np.empty(n+3, dtype=np.int32)
        left = np.arange(n)
        
        #Construct edge table
        edge_tabl = np.zeros((n, n))
        edge_tabl[parent1[0]][parent1[1]] += 1
        edge_tabl[parent1[0]][parent1[n-1]] += 1
        edge_tabl[parent1[-1]][parent1[n-2]] += 1
        edge_tabl[parent1[-1]][parent1[0]] += 1
        edge_tabl[parent2[0]][parent2[1]] += 1
        edge_tabl[parent2[0]][parent2[n-1]] += 1
        edge_tabl[parent2[n-1]][parent2[n-2]] += 1
        edge_tabl[parent2[n-1]][parent2[0]] += 1
        for i in range(1, n-1):
            edge_tabl[parent1[i]][parent1[i+1]] += 1
            edge_tabl[parent1[i]][parent1[i-1]] += 1
            edge_tabl[parent2[i]][parent2[i+1]] += 1
            edge_tabl[parent2[i]][parent2[i-1]] += 1

        for i in range(n-1):
            
            #First city always 0
            if i == 0:
                current_edge = 0
                
            left = left[left != current_edge]
            offspring[i] = current_edge
            
            #Remove all references to current element from the table
            edge_tabl[:, current_edge] = 0 

            #Examine list for current element
            positions = np.where(edge_tabl[current_edge] == 2)[0]
            
            #If there is a common edge, pick that to be the next element
            if positions.shape[0] >  0:
                current_edge = positions[random.randint(0, positions.shape[0]-1)]
            
            #Otherwise pick the entry in the list which itself has the shortest list
            else:
                positions = np.where(edge_tabl[current_edge] == 1)[0]
                if len(positions) == 0:
                    current_edge = left[random.randint(0, left.shape[0]-1)] #np.random.choice(left)
                else:
                    zeros_number = np.count_nonzero(edge_tabl[positions, :] == 0, axis=1)
                    shortest = np.max(zeros_number)
                    a = np.where(zeros_number == shortest)[0]
                    shortest_list_edges = [positions[idx] for idx in a]
                    current_edge = shortest_list_edges[random.randint(0, len(shortest_list_edges)-1)] #np.random.choice(shortest_list_edges)
        offspring[n-1] = left[0]
        
        #Self-adapt
        rand = random.random()
        #K
        offspring[-3] = parent1[-3] * rand + parent2[-3] * (1-rand)
        #K-opt
        offspring[-2] = [parent1[-2], parent2[-2]][random.randint(0,1)]
        #Mutation
        offspring[-1] = parent1[-1] * rand + parent2[-1] * (1-rand)
        
        return offspring
                
    #Function that applies edge-3 crossover to the given population
    def EC(self, population, n):
        
        offspring = np.empty((1, n+3), dtype = np.int32)
        
        for i in range(population.shape[0]//2):
            i1 = population[i]
            i2 = population[population.shape[0]//2+i]
            son = self.edge_crossover(i1, i2, n)
            offspring = np.vstack([offspring, son])
        return offspring[1:]
    
    #Swap mutation
    def swap_mutation(self, population: np.ndarray, n):
        
        for i in range(population.shape[0]):
                if random.uniform(0, 1)<self.alpha:
                    
                    for _ in range(random.randint(1,5)):
                        j = random.randint(1, n-2)
                        z = random.randint(j+1, n-1)

                        aux = population[i][j]
                        population[i][j] = population[i][z]
                        population[i][z] = aux
                    
        return population
    
    #Inversion Mutation
    def inversion_mutation(self, population, n):
        for i in range(population.shape[0]):  
            
            if random.random() < population[i][-1]/100:
                
                #for _ in range(random.randint(1,5)):
                    # Select start and end points for the inversion
                    start_point = np.random.randint(0, n)
                    end_point = np.random.randint(start_point, n)
                    # Perform the inversion mutation only on the first n positions
                    population[i][:n][start_point:end_point + 1] = np.flip(population[i][:n][start_point:end_point + 1])
    
        return population
    
    #mu + lamda
    def elimination(self, population, mutation, fit_population, distanceMatrix):
        
        ranked_mutation, _, fit_mutation = self.order_fitness(mutation, distanceMatrix)
        
        combined_population = np.vstack((population, ranked_mutation))
        combined_fitness = np.concatenate((fit_population, fit_mutation))
        sorted_indices = np.argsort(combined_fitness)

        ranked= combined_population[sorted_indices]
        
        return ranked[:self.lambdaa]
    
    
    #Search in the k_opt neighbourhood of x and returns the best neighbour, and if not x
    def k_opt(self, x, distanceMatrix):
        #K
        kk = x[-3]
        #K-opt
        kkk = x[-2]
        #Mutation
        m = x[-1]
        
        x = x[:distanceMatrix.shape[0]]
        best_route = x
        best_fitness = self.fitness(best_route, distanceMatrix)
        
        if kkk == 2:
            
            for _ in range(int(self.hm * 7)): #2-opt to be in equals conditions to 3-opt, can do self.hm * 7/2 searches
                
                i = random.randint(0, distanceMatrix.shape[0]-1)
                k = random.randint(i+1, distanceMatrix.shape[0]+1)
                
                neighbour = np.concatenate((x[:i], x[i:k+1][::-1], x[k+1:]))
                neighbour_fitness = self.fitness(neighbour, distanceMatrix)
                
                if neighbour_fitness<best_fitness:
                    best_route = neighbour
                    best_fitness = neighbour_fitness
        
        if kkk == 3:
            
            for _ in range(self.hm):
                i = random.randint(0, distanceMatrix.shape[0]-3)
                j = random.randint(i+1, distanceMatrix.shape[0]-2)
                k = random.randint(j+1, distanceMatrix.shape[0]-1)
                
                if -1*(distanceMatrix.shape[0]-k-1) == 0:
                    a = x[:i+1]
                else:
                    a = np.concatenate(([x[-1*(distanceMatrix.shape[0]-k-1):]][0], x[:i+1]))
                b = x[i+1:j+1]
                c = x[j+1:k+1]
                
                neighbours = [
                    np.concatenate((a[::-1], b, c)),  
                    np.concatenate((a, b[::-1], c)),               
                    np.concatenate((a, b, c[::-1])),         
                    np.concatenate((a, b[::-1], c[::-1])),         
                    np.concatenate((a[::-1], b[::-1], c)),   
                    np.concatenate((a[::-1], b, c[::-1])),
                    np.concatenate((a[::-1], b[::-1], c[::-1]))         
                ]
            
                for n in neighbours:
                    neighbour_fitness = self.fitness(n, distanceMatrix)
                
                    if neighbour_fitness<best_fitness:
                        best_route = n
                        best_fitness = neighbour_fitness
        
        i = np.where(best_route == 0)[0][0]
        
        return np.concatenate((best_route[i:], best_route[:i], [kk,kkk,m]))
    
    #K-opt local search applied to a population
    def k_opt_ls(self, population, distanceMatrix):
        return np.apply_along_axis(self.k_opt, axis=1, arr=population, distanceMatrix=distanceMatrix)
    
    @staticmethod
    @njit(float64(int32[:], int32[:], int32))
    def mid_distance(v1, v2, n):
  
        # Add type hints to function parameters and return type
        pairs_v1 = list(zip(v1[:n-1], v1[1:])) + [(v1[n-1], v1[0])]
        pairs_v2 = list(zip(v2[:n-1], v2[1:])) + [(v2[n-1], v2[0])]
        pairs_v1 = set(pairs_v1)
        pairs_v2 = set(pairs_v2)

        return 1.0 - len(pairs_v1.intersection(pairs_v2)) / n
         
    
    def fitness_share(self, x, x_fit, x_shared_fit, last_selected, n):
       
        d = self.mid_distance(last_selected, x, n)
        
        if d > self.sigma or np.isinf(x_shared_fit): 
            return x_shared_fit
        
        p = 1 - (d/self.sigma)**self.alphaa

        return x_shared_fit + p * x_fit
        
    def fitness_sharing_elimination(self, population, mutation, fit_population, distanceMatrix):
        selected = np.empty((1, population.shape[1]), dtype=np.int32)
        
        ranked_mutation, _, fit_mutation = self.order_fitness(mutation, distanceMatrix)
        
        combined_population = np.vstack((population, ranked_mutation))
        combined_fitness = np.concatenate((fit_population, fit_mutation))
        sorted_indices = np.argsort(combined_fitness)

        ranked= combined_population[sorted_indices]
        ranked_fit = combined_fitness[sorted_indices]
        
        prev_fit = ranked_fit
       
        selected = np.vstack([selected, ranked[0]])
    
        for _ in range(self.lambdaa-1):
            
            new_fit = []
            for i in range(len(ranked)):
                new_fit.append(self.fitness_share(ranked[i], ranked_fit[i], prev_fit[i], selected[-1], distanceMatrix.shape[0]))
            
            prev_fit = np.array(new_fit)
            index_order = np.argsort(prev_fit)
            selection = ranked[index_order[0]]
            
            selected = np.vstack([selected, selection])
          
        return selected[1:]

    #PMX
    def PMX(self, population: np.ndarray, n):
        offspring = np.empty((1, n), dtype=np.int32)

        for i in range(population.shape[0]//2):

            i1 = population[i]
            i2 = population[population.shape[0]//2+i]

            r1 = random.randint(1, n - 1)
            r2 = random.randint(r1 + 1, n)

            son = np.full(n,-1, dtype=np.int32)
            son[r1:r2] = i1[r1:r2]

            for j in range(r1, r2):
                if i2[j] not in son:
                    ww = np.where(i2 == i1[j])[0][0]
                    while (r1 <= ww and ww < r2):
                        index = np.where(i2 == i1[ww])[0][0]
                        ww = index

                    son[ww] = i2[j]

            for j in range(n):
                if i2[j] not in son:
                    son[j] = i2[j]

            offspring = np.vstack([offspring, son])

        return offspring[1:]
    
    
    # The evolutionary algorithm ’s main loop
    def optimize (self, filename):
        # Read distance matrix from file .
        file = open(filename)
        distanceMatrix = np.loadtxt(file , delimiter =",")
        file.close ()
        
        n = distanceMatrix.shape[0]
        meanValues = []
        bestValues = []
        medianKopt = []
        medianK = []
        meanMut = []
        
        self.e =  3 * n 
        
        population = self.init_population(distanceMatrix)
        
        print("Initial population")
        population = self.k_opt_ls(population, distanceMatrix)
        
        self.pop_distance(population, n)
        print("Mean mr: {}, Mean k-opt: {}, Median k: {}".format(np.round(np.mean(population[:, -1])/100,4), np.round(np.mean(population[:, -2]),4), int(np.median(population[:, -3]))))
        
        population, _, fit = self.order_fitness(population, distanceMatrix)
        meanObjective, bestObjective, bestSolution = np.mean(fit), fit[0], population[0][:n]
        print(meanObjective, bestObjective)
        print("-------")
        
        meanValues.append(meanObjective)
        bestValues.append(bestObjective)
        medianKopt.append(np.mean(population[:, -2]))
        meanMut.append(np.mean(population[:, -1])/100)
        medianK.append(int(np.median(population[:, -3])))
        yourConvergenceTestsHere = True
        
        ind = 0
        while ( yourConvergenceTestsHere ):
            
            #print("\nIteration: {}".format(self.iterations))
            
            # Your code here .
            self.k = int(np.median(population[:, -3]))
            
            selected = self.k_tournament_selection(population)
            
            offspring = self.EC(selected, n)
            
            mutated = self.inversion_mutation(offspring, n)
            
            mutated = self.k_opt_ls(mutated, distanceMatrix)
            
            if self.iterations % 1 == 0:
                population = self.fitness_sharing_elimination(population, mutated, fit, distanceMatrix)
            else:
                population = self.elimination(population, mutated, fit, distanceMatrix)

                     
            self.pop_distance(population, n)
            print("Mean mr: {}, Mean k-opt: {}, Median k: {}".format(np.round(np.mean(population[:, -1])/100,4), np.round(np.mean(population[:, -2]),4), self.k))
            population, _, fit = self.order_fitness(population, distanceMatrix)
            meanObjective, bestObjective, bestSolution = np.mean(fit), fit[0], population[0][:n]
            print(meanObjective, bestObjective)
            
            meanValues.append(meanObjective)
            bestValues.append(bestObjective)
            medianK.append(self.k)
            medianKopt.append(np.mean(population[:, -2]))
            meanMut.append(np.mean(population[:, -1])/100)
            
            if len(bestValues) >= ind + 10:
                if len(set(bestValues[-10:])) == 1:
                    self.sigma = max(self.sigma_threshold, self.sigma * 0.95)
                    self.hm = int(self.hm * 1.1)
                    ind = len(bestValues)
                    print(self.sigma)
                    
            self.iterations += 1
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution )
            
            if timeLeft < 0:
                break
            

        return meanObjective, bestObjective
    
if __name__ == '__main__ ':
    # Write any testing code here .
    #return 0


