import utils as ut

def convert_to_edgelist(Graph, Weights, name):
	with open('../data/graphs/{}_edgelist.txt'.format(name), 'w') as out:
		for Edge in Graph.Edges():
			Edge = (Edge.GetSrcNId(), Edge.GetDstNId())
			line = '{} {} {}\n'.format(Edge[0], Edge[1], Weights[Edge])
			out.write(line)

def main():
	# Original Compliment Network
	OCN, OW = ut.load_ocn()
	convert_to_edgelist(OCN, OW, 'ocn')
	# Food Pairing Hypothesis Network
	FPH, FW = ut.load_fph()
	convert_to_edgelist(FPH, FW, 'fph')
	# Updated Compliment Network
	UCN, UW = ut.load_ucn()
	convert_to_edgelist(UCN, UW, 'ucn')
	# Substitution Network
	SN, SW = ut.load_sn()
	convert_to_edgelist(SN, SW, 'sn')

if __name__ == '__main__':
	main()