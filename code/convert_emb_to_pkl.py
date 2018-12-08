import utils as ut
import numpy as np
import pickle

# def euclidean_distance(X, Y):
# 	return np.sqrt(((X - Y) ** 2).sum())

# def print_random_top(Embeddings, iid_to_ingedient, K=5):
# 	ingredient_to_iid = {ingredient:iid for iid, ingredient in iid_to_ingedient.iteritems()}
# 	# Index = np.random.choice([i for i, E in enumerate(Embeddings)])
# 	# Compare = Embeddings[Index][0]
# 	iid_to_emb = {NId:i for i, (E, NId) in enumerate(Embeddings)}
# 	Compare = Embeddings[iid_to_emb[ingredient_to_iid['caviar']]][0]
# 	Ranked = sorted([(euclidean_distance(Compare, E), iid_to_ingedient[NId]) for E, NId in Embeddings])[:21]
# 	for i, (d, Ingredient) in enumerate(Ranked):
# 		print '{}. {} ({})'.format(i, Ingredient, d)

def main():
	# Get mappings
	# iid_to_ingredient = ut.load_mappings()['IID_to_Ingredient_Mapping']

	Names = ['ocn', 'fph', 'ucn', 'sn']
	for name in Names:	
		Embeddings = []
		with open('../node2vec/embeddings/{}.emb'.format(name), 'r') as f:
			header = f.readline().strip().split()
			for line in f:
				line = line.strip().split()
				NId, Embedding = int(line[0]), [float(n) for n in line[1:]]
				Embeddings.append((np.array(Embedding), NId))
		# print_random_top(Embeddings, iid_to_ingredient)
		Map = {NId:E for E, NId in Embeddings}
		filename = '../data/mappings/{}_emb_map.pkl'.format(name)
		with open(filename, 'wb') as f:
			pickle.dump(Map, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
	main()