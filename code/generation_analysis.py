import utils as ut
from generate import generate_recipe
from itertools import combinations


def main():
	embeddings = ut.load_embeddings()
	# Graph for OCN
	X = range(1, 31)
	Y = []
	for _ in X:
	    recipe = generate_recipe(seeds=None, cuisine=None, network='ocn_fph', minimum=7, maximum=7, accent=0, avoids=None)
	    # Calculate pairwise distance
	    Total = []
	    for iid_1, iid_2 in combinations(recipe, 2):
	        Total.append(ut.euclidean_distance(embeddings['ocn'][iid_1], embeddings['ocn'][iid_2]))
	    Y.append(sum(Total) / float(len(Total)))
	plt.plot(X, sorted(Y, reverse=True), color='blue', label='OCN')

	# Graph for FPH
	X = range(1, 31)
	Y = []
	for _ in X:
	    recipe = generate_recipe(seeds=None, cuisine=None, network='ocn_fph', minimum=7, maximum=7, accent=7, avoids=None)
	    # Calculate pairwise distance
	    Total = []
	    for iid_1, iid_2 in combinations(recipe, 2):
	        Total.append(ut.euclidean_distance(embeddings['fph'][iid_1], embeddings['fph'][iid_2]))
	    Y.append(sum(Total) / float(len(Total)))
	plt.plot(X, sorted(Y, reverse=True), color='purple', label='FPH')

	# Graph for OCN_FPH
	X = range(1, 31)
	Y = []
	for _ in X:
	    recipe = generate_recipe(seeds=None, cuisine=None, network='ocn_fph', minimum=7, maximum=7, accent=3, avoids=None)
	    # Calculate pairwise distance
	    Total = []
	    for iid_1, iid_2 in combinations(recipe, 2):
	        Total.append(ut.euclidean_distance(embeddings['ocn'][iid_1], embeddings['ocn'][iid_2]))
	    Y.append(sum(Total) / float(len(Total)))
	plt.plot(X, sorted(Y, reverse=True), color='red', label='OCN_FPH')

	# Graph for UCN
	X = range(1, 31)
	Y = []
	for _ in X:
	    recipe = generate_recipe(seeds=None, cuisine=None, network='ucn', minimum=7, maximum=7, accent=0, avoids=None)
	    # Calculate pairwise distance
	    Total = []
	    for iid_1, iid_2 in combinations(recipe, 2):
	        Total.append(ut.euclidean_distance(embeddings['ucn'][iid_1], embeddings['ucn'][iid_2]))
	    Y.append(sum(Total) / float(len(Total)))
	plt.plot(X, sorted(Y, reverse=True), color='green', label='UCN')
	plt.legend()
	plt.title('Average Pairwise Distance Within Recipe')
	plt.show()

if __name__ == '__main__':
	main()