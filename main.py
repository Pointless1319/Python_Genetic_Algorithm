import numpy as np
import random
import matplotlib.pyplot as plt


def worthiness(weight, value, weight_limit, chromo):  # chromo is the bit array of a given backpack
    weight_sum = 0
    value_sum = 0
    for i in range(len(weight)):
        weight_sum = weight_sum + np.sum(weight[i]*chromo[i])
    if weight_sum > weight_limit:
        temp = 0
    else:
        for i in range(len(weight)):
            value_sum = value_sum + np.sum(value[i]*chromo[i])
        temp = value_sum

    #print(value_sum)  # just some interline debugging
    return value_sum, weight_sum


def generate_chromo(chromo_size, weight, weight_limit, probability):
    sum = 0
    temp_chromo = np.zeros(chromo_size)
    for i in range(len(temp_chromo)):
        prob = random.uniform(0, 1)
        if prob < probability:
            temp_chromo[i] = 1
        else:
            temp_chromo[i] = 0
    for c in range(chromo_size):
        sum = sum + np.sum(weight[c]*temp_chromo[c])
    if sum <= weight_limit:
        pass
    return temp_chromo


def initialize_population(population_size, chromo_size, weight, weight_limit, probability_of_ones):  # probability of ones mesures the probability of ones in new chromosomes
    pop1 = np.zeros((population_size, len(weight)))
    for i in range(population_size):
        chromo = generate_chromo(chromo_size, weight, weight_limit, probability_of_ones)
        pop1[i] = chromo
    return pop1


def evaluate_wothiness(population, weight, value, weight_limit):
    f = np.zeros(len(population[:, 0]))
    #print(f.shape)
    wg = np.zeros(len(population[:, 0]))
    #print(wg.shape)
    for i in range(0, len(population[:, 0])):
        f[i], wg[i] = worthiness(weight, value, weight_limit, population[i])
    return f, wg


def roulette_selection(population, worthiness_score, p):  # p = amout of parents being chosen
    worthiness = worthiness_score
    total_worth = sum(worthiness)
    relative_worthiness = [f/total_worth for f in worthiness]  # percentage of worth
    cum_probs = np.cumsum(relative_worthiness)
    matrix = np.zeros((p, len(population[0])))

    for i in range(p):
        r = random.uniform(0, 1)
        for ind in range(len(population[:, 0])):

            if cum_probs[ind] > r:
                # print(ind)
                matrix[i] = population[ind]
                break
    return matrix  # selected parents


def tournament_selection(current_gen, current_values, amount_of_parents):  # I will come back here for sure
    # current_gen = (100,64) current_values = (100,) (not to forget the shapes)

    temporary_holder = np.array(current_values)
    indicator = temporary_holder.argsort()
    working_set = current_gen[indicator[::-1]]
    new_set = np.zeros((amount_of_parents, len(current_gen[0])))

    for i in range(0, amount_of_parents):
        new_set[i] = working_set[i]
    # print(temporary_holder.shape)  # used for debugging
    # print(working_set.shape)
    # print(len(current_values))

    return new_set


def crossover(a, b, p):  # a=chromosome_1, b=chromosome_2, p = probability of crossover
    ind = np.random.randint(0, 64)  # crossover happens at random points of the bit array
    r = random.uniform(0, 1)
    if r < p:
        c1 = list(b[:ind]) + list(a[ind:])  # converting from array to list for easy usage
        c1 = np.array(c1)
        c2 = list(a[:ind]) + list(b[ind:])
        c2 = np.array(c1)
    else:
        c1 = a
        c2 = b
    return c1, c2  # returning two crossover children


def cross_comparison(pr, population, probability):
    new_family = np.zeros((len(pr), 64))
    for i in range(0, len(pr), 2):
        new_family[i], new_family[i+1] = crossover(pr[i], pr[i+1], probability)
    return new_family


def mutation(g, p):  # g = chromosome, p = probability of mutation
    N = len(g)
    m = np.zeros(len(g))
    for i in range(N):
        d = g[i]
        r = random.uniform(0, 1)
        if g[i] == 1.0 and r < p:
            m[i] = 0
        elif g[i] == 0.0 and r < p:
            m[i] = 1
        else:
            m[i] = d
    return m


def defined_mutation(g, probability, mutation_table):  # g = children, probability of mutation, a defined mutation table
    N = len(g)
    m = np.zeros(len(g))
    for i in range(N):
        temp = g[i]
        r = random.uniform(0, 1)
        if mutation_table[i] == 1:
            if g[i] == 1.0 and r < probability:
                m[i] = 0
            elif g[i] == 0.0 and r < probability:
                m[i] = 1
            else:
                m[i] = temp
    return m


def mutation_comparison(h, probability):  # h = children, probability of mutation
    new_family = np.zeros((len(h), 64))
    for i in range(0, len(h)):
        new_family[i] = defined_mutation(h[i], probability, defined_mutation_table)
    return new_family


def sparta_children_selection(p, weight, value,  weight_limit):  # p = parents, weight of items, value of items, knapsack weight limit
    good_children = []  # using list instead of array for easy append
    temp_value1 = 0
    temp_value2 = 0

    for i in range(0, len(p)):
        temp_value1, temp_value2 = worthiness(weight, value, weight_limit, p[i])
        if temp_value2 <= weight_limit:
            good_children.append(p[i])
    fin = np.array(good_children)  # converting back to array
    return fin


def debugging_values(w, v):
    best_value = max(v)
    min_value = min(v)
    max_weight = max(w)
    min_weight = min(w)
    return best_value, min_value, max_weight, min_weight


def name_items_in_the_backpack():
    return 0


weights_of_items = np.array([
    2., 8., 6., 4., 1., 2., 7., 7.,
    3., 5., 3., 5., 3., 5., 8., 6.,
    9., 4., 5., 7., 4., 7., 9., 8.,
    2., 3., 9., 7., 5., 4., 4., 5.,
    4., 2., 1., 2., 5., 7., 7., 7.,
    7., 7., 6., 1., 6., 4., 4., 4.,
    5., 6., 5., 1., 9., 1., 2., 1.,
    6., 2., 7., 6., 8., 9., 1., 7.])

#  64 costs corresponding to weights
value_of_items = [
           6.,  17.,  10., 26.,  19.,  81.,  67., 36.,
          21.,  33.,  13.,  5., 172., 138., 185., 27.,
           4.,   3.,  11., 19.,  95.,  90.,  24., 20.,
          28.,  19.,   7., 28.,  14.,  43.,  40., 12.,
          25.,  37.,  25., 16.,  85.,  20.,  15., 59.,
          72., 168.,  30., 57.,  49.,  66.,  75., 23.,
          79.,  20., 104.,  9.,  32.,  46.,  47., 55.,
          21.,  18.,  23., 44.,  61.,   8.,  42.,  1.]

defined_mutation_table = [0, 0, 0, 1, 0, 0, 0, 0,
                          0, 0, 0, 1, 0, 0, 0, 0,
                          0, 0, 0, 1, 0, 0, 0, 0,
                          0, 0, 0, 1, 0, 0, 0, 0,
                          0, 0, 0, 1, 0, 0, 0, 0,
                          0, 0, 0, 1, 0, 0, 0, 0,
                          0, 0, 0, 1, 0, 0, 0, 0,
                          0, 0, 0, 1, 0, 0, 0, 0]

# Knapsack weight limit
knapsack_weight_limit = 100

# Amount of Generations - 1 (because counting from 0)
amount_of_generations = 99

#Mutation Probability
mutation_probability = 0.5
#Crossover Probabilty
crossover_probability = 0.1




pop = initialize_population(population_size=100, chromo_size=64, weight=weights_of_items, weight_limit=50, probability_of_ones=0.1) #initializing population
temp_pop = np.zeros((100, 64)) # temporary population holder
used_pop = pop # just for safety so i don't work on the generated population
values, weights = evaluate_wothiness(used_pop, weights_of_items, value_of_items, knapsack_weight_limit) # creates two arrays holding all the backpack values and weights
bc, mc, mw, minw = debugging_values(weights, values) # values we extract to ANALise later

#arrays holding the values for each generation
generation   = np.zeros(amount_of_generations+1)
best_cost    = np.zeros(amount_of_generations+1)
min_cost     = np.zeros(amount_of_generations+1)
best_weight  = np.zeros(amount_of_generations+1)
max_weight   = np.zeros(amount_of_generations+1)
min_weight   = np.zeros(amount_of_generations+1)
weight_of_best_cost = np.zeros(amount_of_generations+1)
best_knapsack = np.zeros((amount_of_generations+1, 64))

# initializing the arrays with the first generation
generation[0]  = 0
best_cost[0]   = bc
min_cost[0]    = mc
max_weight[0]  = mw
min_weight[0]  = minw
temp_index = np.where(values == bc)
weight_of_best_cost[0] = weights[temp_index]
best_knapsack[0] = used_pop[temp_index]
#dupa chuj



for gen in range(0, amount_of_generations):

    # ---------------------------- if roulette_selection is used ---------------------------------------------------------------------------
   # parents = roulette_selection(used_pop, values, 60)  # amount of parents must be even (in this case its 60)
    # --------------------------------------------------------------------------------------------------------------------------------------

    # ---------------------------- if tournament_selection is used --------------------------------- ---------------------------------------
    parents = tournament_selection(used_pop, values, 60)
    # ---------------------------------------------------------------------------------------------------------------------------------------

    cross = cross_comparison(parents, used_pop, crossover_probability) # function crossing the a pair of parents and generating 2 children
    mut = mutation_comparison(cross, mutation_probability)  # giving the children their mutations
    fixed_children = sparta_children_selection(mut, weights_of_items, value_of_items, knapsack_weight_limit)  # selecting only the fit children. The fat ones are discarded (weight above backpack limit)
    rest_of_family = roulette_selection(used_pop, values, 100 - len(fixed_children))  # alabama function (fill the population with the parents from previous generation)
    new_pop = np.vstack((fixed_children, rest_of_family)) #join the children and selected parents into a new generation
    used_pop = new_pop #replace old population with new one



    #-------------------------debugging section---------------------------------------------

    #print(len(fixed_children)) # this is to check how many children survive
    #print(fixed_children.shape)
    #print(new_pop.shape)

    # temp_worth = np.zeros((100, 1))
    # temp_worth2 = np.zeros((100, 1))
    # temp_value = 0
    # for i in range(0, 100):
    #     temp_worth[i], temp_worth2[i] = worthiness(weights_of_items, value_of_items, knapsack_weight_limit, new_pop[i])
    # # temp_worth = worthiness(weights_of_items, value_of_items, knapsack_weight_limit, new_pop[1])
    # print(max(temp_worth))
    #----------------------------------------------------------------------------------------

    #ASSigning new values to be plotted
    new_values, new_weights = evaluate_wothiness(new_pop, weights_of_items, value_of_items, knapsack_weight_limit)
    bc, mc, mw, minw = debugging_values(new_weights, new_values)
    index = np.where(new_values == bc)
    #print(new_weights[index])
    weight_of_best_cost[gen+1] = new_weights[index][0]
    #print(used_pop[index])
    best_knapsack[gen+1] = used_pop[index][0]
    #print(index)
    generation[gen+1] = gen+1
    best_cost[gen+1] = bc
    min_cost[gen+1] = mc
    max_weight[gen+1] = mw
    min_weight[gen+1] = minw

#------------------------------more debugging---------------------------------------------------
#print(generation)
#print(best_cost)
#print(min_cost)
#print(max_weight)
#-----------------------------------------------------------------------------------------------

#draw selected fuctions (in this case best cost for every generation)
print(max(best_cost))

#print(np.where(best_cost == max(best_cost)))
print(best_knapsack[np.where(best_cost == max(best_cost))])
print(weight_of_best_cost[np.where(best_cost == max(best_cost))])

plt.figure(figsize=(10, 6))
plt.plot(generation, best_cost)
plt.show()
