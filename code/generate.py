import argparse
import utils as ut
import numpy as np

parser = argparse.ArgumentParser(description="Run node2vec.")
parser.add_argument('-s', '--seed_ingredients',   action='append',   default=None,    help="Seed ingredients for recipe generation")
parser.add_argument('--cuisine',                  type=str,          default=None,    help="Which cuisine to mimic (can be random)")
parser.add_argument('--network',                  type=str,          default='ucn',   help="Which network to use for recipe generation (ocn_fph, ucn)")
parser.add_argument('--min',                      type=int,          default=7,       help="Minimum number of ingredients")
parser.add_argument('--max',                      type=int,          default=7,       help="Maximum number of ingredients in a recipe")
parser.add_argument('--accent',                   type=int,          default=0,       help="Number of accent ingredients to use")
parser.add_argument('-a', '--avoids',             action='append',   default=None,    help="Which ingredients to avoid using in the recipe")
args = parser.parse_args()

# Choose k random items from a dictionary
def choose_random_seed(d, k=1):
	iids = d.keys()
	return np.random.choice(iids, size=k, replace=False)


def get_ranked_from_seeds(seeds, embeddings):
	Centroid = np.array([embeddings[iid] for iid in seeds]).mean(0)
	Ranked = sorted([(ut.euclidean_distance(Centroid, E), IId) for IId, E in embeddings.iteritems() if not IId in seeds])
	return Ranked

def choose_new_ingredient(Ranked):
	probs = np.array([1.0 / (d+1) for i, (d, IId) in enumerate(Ranked)])
	probs = probs / probs.sum()
	return np.random.choice([IId for d, IId in Ranked], p=probs)

# Generates a recipe following the base-accent hypothesis
def base_accent_generate(ocn_embeddings, fph_embeddings, seeds, num_ingredients, accent):
	base_ingredients = num_ingredients - accent
	# Sample Base Ingredients First
	base_recipe = generate(ocn_embeddings, seeds, base_ingredients)
	# Sample Accent Ingredients using Base Recipe
	recipe = generate(fph_embeddings, base_recipe, num_ingredients)
	return recipe

# Generates a recipe from the provided network
def generate(embeddings, seeds, num_ingredients):
	if seeds is None:
		seeds = choose_random_seed(embeddings, 2)

	while len(seeds) < num_ingredients:
		# Calculate ranked list of ingredients to current seed set
		Ranked = get_ranked_from_seeds(seeds, embeddings)
		NewIngredientIId = choose_new_ingredient(Ranked)
		seeds = np.append(seeds, [NewIngredientIId], axis=0)
	return seeds

def get_embeddings(embeddings, network, mappings, cuisine=None):
	if cuisine is None:
		return embeddings[network]

	if cuisine == 'random':
		cuisine = np.random.choice(mappings['Cuisine_to_List_of_Ingredients_Mapping'].keys())

	ingredient_to_iid = {Ingredient:IId for IId, Ingredient in mappings['IID_to_Ingredient_Mapping'].iteritems()}
	Ingredients = [ingredient_to_iid[Ingredient] for Ingredient in mappings['Cuisine_to_List_of_Ingredients_Mapping'][cuisine]]

	return {IId:embeddings[network][IId] for IId in Ingredients if IId in embeddings[network].keys()}

def get_avg_dist(embeddings, ing_emb):
	return np.array([ut.euclidean_distance(ing_emb, e) for e in embeddings]).mean()


def substitute_avoids(SW, ocn_emb, avoids, recipe):
	NoSubs = False
	NewRecipe = []
	for ingredient in recipe:
		if not ingredient in avoids:
			NewRecipe.append(ingredient)
		else:
			Ranked = []
			for Edge, Weight in SW.iteritems():
				if ingredient == Edge[0]:
					Ranked.append((Weight, Edge[1]))
				elif ingredient == Edge[1]:
					Ranked.append((Weight, Edge[0]))
			Ranked = sorted(Ranked, reverse=True)[:min(10, len(Ranked))]

			if len(Ranked) == 0:
				NoSubs = True
				NewRecipe.append(ingredient)
			else:
				Scores = [0] * len(Ranked)
				for i, (d, SId) in enumerate(Ranked):
					compare = [ocn_emb[IId] for IId in recipe if IId in ocn_emb and not IId == SId]
					if SId in ocn_emb:
						Scores[i] = get_avg_dist(compare, ocn_emb[SId])
					else:
						Scores[i] = float('inf')
				if all([s == float('inf') for s in Scores]):
					Scores = [0] * len(Scores)
				else:
					MaxScore = max([s for s in Scores if not s == float('inf')])
					Scores = [min(MaxScore, Score) for Score in Scores]
				Ranked = sorted([((1.0 * w) - (0.2 * Scores[i]), SId) for i, (w, SId) in enumerate(Ranked)], reverse=True)
				Min = min([w for w, SId in Ranked])
				Ranked = [(w - Min, SId) for w, SId in Ranked]
				SubstituteIngredientId = choose_new_ingredient(Ranked)
				NewRecipe.append(SubstituteIngredientId)
	if NoSubs == True:
		print 'This recipe may contain avoid items because of a lack of substitutable ingredients'
	return NewRecipe

def main():
	# Load mappings and embeddings for specified network(s)
	mappings = ut.load_mappings()
	embeddings = ut.load_embeddings()
	# Extract IIDs for specified seed ingredients
	ingredient_to_iid = {ingredient:iid for iid, ingredient in mappings['IID_to_Ingredient_Mapping'].iteritems()}
	if args.seed_ingredients is not None:
		args.seed_ingredients = [ingredient_to_iid[ingredient] for ingredient in args.seed_ingredients]

	if args.accent > 0 and not args.network == 'ocn_fph':
		raise Exception('You set accent > 1 but did not use network \'ocn_fph\'.')

	num_ingredients = np.random.randint(args.min, args.max+1)
	if args.accent > args.min:
		raise Exception('Number of accent ingredients cannot be greater than the minimum number of ingredients.')

	if args.network == 'ocn_fph':
		recipe = base_accent_generate(get_embeddings(embeddings, 'ocn', mappings, args.cuisine), \
										get_embeddings(embeddings, 'fph', mappings, args.cuisine), args.seed_ingredients, num_ingredients, args.accent)
	elif args.network == 'ucn':
		recipe = generate(get_embeddings(embeddings, 'ucn', mappings, args.cuisine), args.seed_ingredients, num_ingredients)
	else:
		raise NotImplementedError

	if args.avoids is not None:
		avoid_iids = [ingredient_to_iid[a] for a in args.avoids]
		SN, SW = ut.load_sn()
		recipe = substitute_avoids(SW, \
										get_embeddings(embeddings, 'ocn', mappings, args.cuisine), avoid_iids, recipe)

	base_ingredients = num_ingredients - args.accent
	for i, iid in enumerate(recipe):
		if args.network == 'ocn_fph' and i >= base_ingredients:
			print mappings['IID_to_Ingredient_Mapping'][iid], '(accent)'
		else:
			print mappings['IID_to_Ingredient_Mapping'][iid]



if __name__ == '__main__':
	main()