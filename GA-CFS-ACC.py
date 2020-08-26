from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from math import ceil, floor
from random import randint
from sklearn.preprocessing import StandardScaler
from random import uniform
from random import seed
from time import time
import sys
import argparse
from warnings import filterwarnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from skfeature.utility.mutual_information import su_calculation


filterwarnings('ignore')


class individual:
	def __init__(self, chrom):
		self.chrom = chrom
		self.bool_index = [bool(i) for i in self.chrom]
		self.atts = sum(self.bool_index)
		self.acc = self.set_acc()
		self.fitness = self.set_fitness()
	
	
	def set_acc(self):
		ind_X_train = X_train.iloc[:, self.bool_index]
		cv_score = list()
		cv = 10
		folds = StratifiedKFold(n_splits = cv, shuffle = True, random_state = 10)
		for train_index, val_index in folds.split(ind_X_train, y_train):
			X_Train, X_Val = ind_X_train.loc[train_index,:], ind_X_train.loc[val_index,:]
			y_Train, y_Val = y_train[train_index], y_train[val_index]
			clf.fit(X_Train, y_Train)
			score = accuracy_score(y_Val, clf.predict(X_Val))
			cv_score.append(score)
		mean_acc = sum(cv_score)/cv
		return mean_acc
	
	
	def set_fitness(self):
		ind_X_train = X_train.iloc[:, self.bool_index]
		ind_X_train = np.array(ind_X_train)
		fitness = abs(merit_calculation(ind_X_train, y_train) - (1 - self.acc))
		return fitness


def merit_calculation(X, y):
	n_samples, n_features = X.shape
	rff = 0
	rcf = 0
	for i in range(n_features):
		fi = X[:, i]
		rcf += su_calculation(fi, y)
		for j in range(n_features):
			if j > i:
				fj = X[:, j]
				rff += su_calculation(fi, fj)
	rff *= 2
	merits = rcf / np.sqrt(n_features + rff)
	return merits


def sort_solutions(solutions, n_solutions):
	fit_index = [[solutions[i].fitness, i] for i in range(n_solutions)]
	fit_index.sort(reverse = True)
	indexes = [fit_index[i][1] for i in range(n_solutions)]
	sorted_pop = [solutions[i] for i in indexes]
	return sorted_pop


def initial_population(n_individuals, chrom_size):
	pop = []
	for i in range(n_individuals):
		while(1):
			chrom = [randint(0, 1) for i in range(chrom_size)]
			if sum(chrom) != 0:
				break
		pop.append(individual(chrom))
	return pop


def tournament_selection(n_selected, pop_size, pop):
	selected_ind_fitness = {}
	for i in range(n_selected):
		n = randint(0, pop_size - 1)
		selected_ind_fitness[pop[n].fitness] = n
	index_min_fitness = selected_ind_fitness[max(selected_ind_fitness.keys())]
	return pop[index_min_fitness]


def intermediate_population(int_pop_size, n_selected, pop_size, pop):
	int_pop = [tournament_selection(n_selected, pop_size, pop) for i in range(int_pop_size)]
	return int_pop


def mutation(child_chrom, chrom_size, mutation_rate):
	mutated_chrom = child_chrom
	while(1):
		for i in range(chrom_size):
			mutation_prob = uniform(0, 1)
			if mutation_prob < mutation_rate:
				if mutated_chrom[i] == 1:
					mutated_chrom[i] = 0
				else:
					mutated_chrom[i] = 1
		if sum(mutated_chrom) != 0:
			break
		else:
			mutated_chrom = child_chrom
	return individual(mutated_chrom)


def crossover(parent1, parent2, chrom_size):
	n = randint(0, chrom_size - 1)
	child_chrom = parent1.chrom[:n] + parent2.chrom[n:]
	return child_chrom


def new_generation(pop, pop_size, int_pop, int_pop_size, chrom_size, mutation_rate):
	new_pop = []
	i = 0
	while len(new_pop) != pop_size:
		parent1 = int_pop[i]
		parent2 = int_pop[i+1]
		child_chrom = crossover(parent1, parent2, chrom_size)
		new_pop.append(mutation(child_chrom, chrom_size, mutation_rate))
		i += 2
		if i == int_pop_size:
			i = 0
	new_pop = sort_solutions(new_pop, pop_size)
	new_pop.pop()
	return [pop[0]] + new_pop


def verify_heuristic_stagnation(best_ind, possible_best_ind, stagnant_gen):
	if best_ind.chrom != possible_best_ind.chrom:
		stagnant_gen = 0
		return possible_best_ind, stagnant_gen
	else:
		stagnant_gen += 1
		return best_ind, stagnant_gen


def calculate_int_pop_size(pop_size):
	int_pop_size = int(pop_size*0.7)
	if int_pop_size%2 != 0:
		int_pop_size += 1
	return int_pop_size


def AG(chrom_size, pop_size, mutation_rate, n_selected):
	seed()
	stagnant_gen = 0
	int_pop_size = calculate_int_pop_size(pop_size)
	pop = initial_population(pop_size, chrom_size)
	pop = sort_solutions(pop, pop_size)
	best_ind = pop[0]

	while(stagnant_gen <= max_stagnant_gen):
		int_pop = intermediate_population(int_pop_size, n_selected, pop_size, pop)
		pop = new_generation(pop, pop_size, int_pop, int_pop_size, chrom_size, mutation_rate)
		pop = sort_solutions(pop, pop_size)
		best_ind, stagnant_gen = verify_heuristic_stagnation(best_ind, pop[0], stagnant_gen)

	return best_ind


def Print(solutions, n_solutions, type_of_solution = None):
	if type_of_solution in ['Heuristic', 'Voted']:
		print(f'{type_of_solution} solutions:')
	else:
		print('Results whithout reduction:')

	print(' Atts    Acc      Fit')

	for i in range(n_solutions):
		fitness = solutions[i].fitness
		print(f'{solutions[i].atts:4}   {solutions[i].acc:.4f}   {fitness:.4f}')
	print()


def set_test_acc(columns):
	ind_X_train = original_X_train.loc[:, columns]
	ind_X_test = X_test.loc[:, columns]
	pred_test = clf.fit(ind_X_train, y_train).predict(ind_X_test)
	test_acc = accuracy_score(y_test, pred_test)
	return test_acc


def round_off(n, decimal_places = 4):
	m = 10 ** (decimal_places + 1)
	n = floor(n * m)/10
	f = floor(n)
	r = n - f
	if r >= 0.5:
		n = ceil(n)
	else:
		n = f
	m /= 10
	n /= m

	return n


def attributes_frequency():
	att_freqs = []
	for i in range(n_attributes):
		att_freq = 0
		for j in range(n_heuristic_solutions):
			att_freq += heuristic_solutions[j].chrom[i]
		att_freqs.append(att_freq)
	return att_freqs


def selected_attributes(att_rate, att_freqs):
	att_number = ceil(att_rate * n_heuristic_solutions)
	sel_attributes = []
	for i in range(n_attributes):
		if att_freqs[i] >= att_number:
			sel_attributes.append(1)
		else:
			sel_attributes.append(0)
	return sel_attributes


def generate_voted_solutions():
	voted_solutions = []
	att_freqs = attributes_frequency()
	for att_rate in att_rates:
		sel_attributes = selected_attributes(att_rate, att_freqs)
		if sum(sel_attributes) == 0:
			sel_attributes = [1] * n_attributes
		voted_solution = individual(sel_attributes)
		voted_solutions.append(voted_solution)
	voted_solutions = sort_solutions(voted_solutions, n_voted_solutions)
	return voted_solutions


def classifier(clf_option):
	if clf_option == 0:
		clf = RandomForestClassifier(random_state = 10)
	elif clf_option == 1:
		clf = DecisionTreeClassifier(criterion = 'gini', max_depth = 2, max_leaf_nodes = None, random_state = 10)
	elif clf_option == 2:
		clf = SVC(max_iter = 50, gamma = 'auto', random_state = 10)
	elif clf_option == 3:
		clf = KNeighborsClassifier()
	elif clf_option == 4:
		clf = GaussianNB()
	elif clf_option == 5:
		clf = GradientBoostingClassifier(n_estimators=400, learning_rate=3.0, max_depth=1, random_state=10)
	elif clf_option == 6:
		clf = BaggingClassifier(random_state = 10)
	elif clf_option == 7:
		clf = AdaBoostClassifier(random_state = 10)
	elif clf_option == 8:
		clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 2), learning_rate_init=0.001, random_state=10)
	return clf


def split_X_y(dataset):
	X = dataset.iloc[:, :-1]
	y = dataset.iloc[:, -1]
	column_names = X.columns
	return X, y, column_names


def update_data():
	global X_train
	global column_names
	global n_attributes
	reduced_attributes = best_solution.bool_index
	X_train = X_train.iloc[:, reduced_attributes]
	column_names = column_names[reduced_attributes]
	n_attributes = sum(reduced_attributes)


def replace_dot(results):
	n_values = len(results['Train accuracy'])
	for k in results.keys():
		for n in range(n_values):
			results[k][n] = str(results[k][n]).replace('.', ',')
	return results


def verify_round_stagnation():
	global stagnant_round
	global Round
	global best_round

	last_fit = rounds_results['Fitness'][Round - 1]
	current_fit = rounds_results['Fitness'][Round]

	if current_fit > last_fit:
		stagnant_round = 0
		best_round = Round
		return False
	elif current_fit < last_fit:
		best_round = Round - 1
		return True
	else:
		stagnant_round += 1
		if stagnant_round == max_stagnant_round:
			best_round = Round
			return True


def save_results(results, round_1 = False):
	results = replace_dot(results)
	results_df = pd.DataFrame(results)
	if round_1 == False:
		file_name = f'alg{alg_option}_clf{clf_option}_{dataset_name}_AG_results.csv'
	else:
		file_name = f'alg{alg_option}_clf{clf_option}_{dataset_name}_round1_AG_results.csv'
	results_df.to_csv(file_name, index = False, sep = ';')


def verify_typing_error(option, Dict, type_of_option):
	n_options = len(Dict)
	while option not in Dict.keys():
		print()
		print(f'Invalid {type_of_option} option. Please type again:\n', end = '')
		for i in range(n_options):
			print(f' {Dict[i]}: {i}')
		option = int(input())
	return option


def read_option(Dict, type_of_option):
	print(f'\nChoose an {type_of_option}:')

	n_options = len(Dict)

	for i in range(n_options):
		print(f' {Dict[i]}: {i}')
	option = int(input())

	return option


def set_options():
	parser = argparse.ArgumentParser()
	parser.add_argument('-train', '--train', help='csv format file (training set), e.g., train.csv')
	parser.add_argument('-test', '--test', help='csv format file (testing set), e.g., test.csv')
	parser.add_argument('-classifier', '--classifier', help='e.g., 0 = RandomForestClassifier, 1 = DecisionTreeClassifier, 2 = SVM, 3 = KNN, 4 = GaussianNB, 5 = GradientBoosting, 6 = Bagging, 7 = AdaBoost, 8 = MLP')
	args = parser.parse_args()
	train = str(args.train)
	test = str(args.test)
	classifier = int(args.classifier)
	
	alg_option = 3
	clf_option = classifier

	# file_name = sys.argv[3]

	alg_option = verify_typing_error(alg_option, alg_option_dict, 'algorithm')
	clf_option = verify_typing_error(clf_option, clf_option_dict, 'classifier')

	print(f'\nAlgorithm option: {alg_option_dict[alg_option]}')
	print(f'Classifier option: {clf_option_dict[clf_option]}')
	print(f'File name - Train: {train}')

	dataset_name = train.split('/')[-1].split('(')[0].split('.')[0]
	
	dataset_train = pd.read_csv(train)
	dataset_test = pd.read_csv(test)

	return alg_option, clf_option, dataset_train, dataset_test, dataset_name


def update_results(dict_results, best_solution, type_of_results):
	if type_of_results == 'round':
		ind = best_solution
		dict_results['Round'].append(Round)
		dict_results['Nº of attributes'].append(ind.atts)
		dict_results['Train accuracy'].append(ind.acc)
		dict_results['Time(min)'].append(Time)
		dict_results['Attributes'].append(column_names[ind.bool_index])
		dict_results['Fitness'].append(ind.fitness)
	else:
		columns = rounds_results['Attributes'][best_round]
		fitness = rounds_results['Fitness'][best_round]
		dict_results['s'].append(s)
		dict_results['Last round'].append(Round)
		dict_results['Best round'].append(best_round)
		dict_results['Nº of attributes'].append(rounds_results['Nº of attributes'][best_round])
		dict_results['Train accuracy'].append(round_off(rounds_results['Train accuracy'][best_round]))
		dict_results['Test accuracy'].append(round_off(set_test_acc(columns)))
		dict_results['Time(min)'].append(round_off(total_time))
		dict_results['Attributes'].append([col for col in columns])
		dict_results['Fitness'].append(round_off(fitness))
	return dict_results


def generate_dict(keys):
	d = {keys[i]: list() for i in range(len(keys))}
	return d


def execution_0(dict_results, type_of_results):
	if type_of_results == 'alg':
		start = time()
	ind = individual([1] * n_attributes)
	if type_of_results == 'alg':
		Time = time() - start

	if type_of_results == 'alg':
		fitness = ind.fitness
		test_acc = set_test_acc(column_names)
		dict_results['s'].append(0)
		dict_results['Last round'].append('-')
		dict_results['Best round'].append('-')
		dict_results['Nº of attributes'].append(ind.atts)
		dict_results['Train accuracy'].append(round_off(ind.acc))
		dict_results['Test accuracy'].append(round_off(test_acc))
		dict_results['Time(min)'].append(round_off(Time))
		dict_results['Attributes'].append([col for col in column_names])
		dict_results['Fitness'].append(round_off(fitness))
		save_results(dict_results)
	else:
		dict_results['Round'].append('-')
		dict_results['Nº of attributes'].append(ind.atts)
		dict_results['Train accuracy'].append(ind.acc)
		dict_results['Time(min)'].append('-')
		dict_results['Attributes'].append([col for col in column_names])
		dict_results['Fitness'].append(ind.fitness)

	Print([ind], 1)
	return dict_results


def get_best_solution():
	if alg_option in voted_options:
		best_solution = voted_solutions[0]
	else:
		best_solution = heuristic_solutions[0]
	return best_solution


alg_option_dict = {
	1: 'Best(function)',
	3: 'Final(function)'
}


clf_option_dict = {
	0: 'RandomForest',
	1: 'DecisionTree',
	2: 'SVM',
	3: 'KNN',
	4: 'GaussianNB',
	5: 'GradientBoosting',
	6: 'Bagging',
	7: 'AdaBoost',
	8: 'MLP'
}


alg_option, clf_option, dataset_train, dataset_test, dataset_name = set_options()
voted_options = [3]


clf = classifier(clf_option)


pop_size = 12
n_selected = 3
mutation_rate = 0.05
max_stagnant_gen = 5
n_heuristic_solutions = 10
max_round = 15
att_rates = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
n_voted_solutions = len(att_rates)
max_stagnant_round = 2
best_round = int()
sampling = 1
a = 0.99
b = 0.01


rounds_results_columns = [
	'Round',
	'Nº of attributes',
	'Train accuracy',
	'Fitness',
	'Time(min)',
	'Attributes'
]

alg_results_columns = [
	's',
	'Last round',
	'Best round',
	'Nº of attributes',
	'Train accuracy',
	'Fitness',
	'Test accuracy',
	'Time(min)',
	'Attributes'
]


alg_results = generate_dict(alg_results_columns)
round_1_alg_results = generate_dict(alg_results_columns)

for s in range(sampling + 1):

	Times = list()
	rounds_results = generate_dict(rounds_results_columns)

	X_train, y_train, column_names = split_X_y(dataset_train)
	X_test, y_test, column_names = split_X_y(dataset_test)

	# print(X_test)
	# print(y_test)
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

	scaler = StandardScaler()
	X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=column_names)
	X_test = pd.DataFrame(scaler.transform(X_test), columns=column_names)

	X_train = X_train.reset_index(drop = True)
	X_test = X_test.reset_index(drop = True)
	y_train = y_train.reset_index(drop = True)
	y_test = y_test.reset_index(drop = True)

	original_X_train = X_train
	n_attributes = len(column_names)

	print(f'\ns: {s}')
	print(f'Dataset: {dataset_name}\n')

	if s == 0:
		alg_results = execution_0(alg_results, 'alg')
		continue

	stagnant_round = 0

	for Round in range(max_round + 1):
		print(f'--------------- Round {Round} ---------------')

		if Round == 0:
			rounds_results = execution_0(rounds_results, 'round')
			continue

		start = time()
		heuristic_solutions = []

		for i in range(n_heuristic_solutions):
			heuristic_solution = AG(n_attributes, pop_size, mutation_rate, n_selected)
			heuristic_solutions.append(heuristic_solution)

		heuristic_solutions = sort_solutions(heuristic_solutions, n_heuristic_solutions)
		Print(heuristic_solutions, n_heuristic_solutions, 'Heuristic')

		if alg_option in voted_options:
			voted_solutions = generate_voted_solutions()
			voted_solutions = sort_solutions(voted_solutions, n_voted_solutions)
			Print(voted_solutions, n_voted_solutions, 'Voted')

		best_solution = get_best_solution()
		
		Time = (time() - start) / 60
		print(f'Time(minute): {round_off(Time):.4f}\n')
		Times.append(Time)

		rounds_results = update_results(rounds_results, best_solution, 'round')

		if Round == 1:
			best_round = Round
			total_time = sum(Times)
			round_1_alg_results = update_results(round_1_alg_results, best_solution, 'alg')
			save_results(round_1_alg_results, round_1 = False)

		if verify_round_stagnation():
			break

		update_data()

	total_time = sum(Times)
	print(f'Total time(minute): {total_time:.4f}\n')

	alg_results = update_results(alg_results, best_solution, 'alg')
	save_results(alg_results)
