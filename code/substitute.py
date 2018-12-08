import argparse
import utils as ut
import numpy as np

parser = argparse.ArgumentParser(description="Run node2vec.")
parser.add_argument('-i', '--ingredients',   action='append',     default=None,    help="Ingredients to substitute")
parser.add_argument('--cuisine',             type=str,            default=None,    help="Which cuisine to mimic (can be random)")
parser.add_argument('--k',                   type=int,            default=5,       help="Number of substitutable ingredients to show")
args = parser.parse_args()

if args.k < 1:
	raise Exception('k must be greater than 0')

def main():
	# Load mappings and embeddings for specified network(s)
	mappings = ut.load_mappings()
	SN, SW = ut.load_sn()

	iid_to_ingredient = mappings['IID_to_Ingredient_Mapping']
	ingredient_to_iid = {ingredient:iid for iid, ingredient in iid_to_ingredient.iteritems()}
	for ingredient in args.ingredients:
		print '#' * 80
		print 'Substitutes for ingredient: {}'.format(ingredient)
		print '#' * 80
		substitutes = []
		for Edge, Weight in SW.iteritems():
			if ingredient_to_iid[ingredient] == Edge[0]:
				substitutes.append((Weight, Edge[1]))
			elif ingredient_to_iid[ingredient] == Edge[1]:
				substitutes.append((Weight, Edge[0]))
		if len(substitutes) > 0:
			Ranked = sorted(substitutes, reverse=True)[:args.k]
			for i, (Weight, IId) in enumerate(Ranked):
				print '{}. {} ({})'.format(i+1, iid_to_ingredient[IId], Weight)
		else:
			print 'NO SUBSTITUTES (sorry...)'





if __name__ == '__main__':
	main()