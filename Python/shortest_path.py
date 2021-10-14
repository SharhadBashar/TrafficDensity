'''
Author: Sharhad Bashar
Class: Shortest_Path
Description: This class takes in the graph, as well as start and end points and returns the shortest path from start to finish
             Input: graph, A, B
             Output: shortest path from A to B
'''

import os
import math
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from data_processing import Data_Processing

class Shortest_Path:
	def __init__(self, A, B, graph_name, draw = False):
		self.graph_name = graph_name
		self.data_folder = '../Data/'
		self.graph_path = self.data_folder + self.graph_name

		if not (os.path.isfile(self.graph_path)):
			print('Graph not found')
			return

		self.G = self.prep_graph(self.graph_path)
		if (draw): 
			self.draw_graph()
		self.path = self.path(self.G, A, B)

	def return_path(self):
		return self.path

	def prep_graph(self, graph_path):
		sources, targets, weights = [], [], []
		graph = pd.read_csv(graph_path, index_col = 0)
		intersections = len(graph.index)
		source_list = graph.index.values
		target_list = graph.columns.values
		
		for source in source_list:
			sources += [source] * 30
		for _ in range(intersections):
			for target in target_list:
				targets.append(int(target))
		if len(sources) != len(targets):
			Print('Length issue')
			return

		for i in range(len(sources)):
			source = sources[i]
			target = targets[i]
			weights.append(graph[str(source)][target])

		so = pd.DataFrame({
			'source': sources,
			'target': targets,
			'weight': weights
		})
		self.sources = sources
		self.targets = targets
		self.weights = weights
		G = nx.from_pandas_edgelist(so, source = 'source', target = 'target', edge_attr = 'weight')
		return G

	def draw_graph(self):
		i = 0
		sources = self.sources
		targets = self.targets
		weights = self.weights
		while i < len(weights):
			if (weights[i] == math.inf):
				sources.pop(i)
				targets.pop(i)
				weights.pop(i)
			else:
				weights[i] = round(weights[i], 1) 
				i += 1
		so = pd.DataFrame({
			'source': sources,
			'target': targets,
			'weight': weights
		})
		G = nx.from_pandas_edgelist(so, source = 'source', target = 'target', edge_attr = 'weight')
		pos = nx.spring_layout(G)

		nx.draw(G, pos = pos, with_labels = True, node_size = 500)
		labels = nx.get_edge_attributes(G,'weight')
		nx.draw_networkx_edge_labels(G, pos = pos, edge_labels = labels)
		plt.show()

	def path(self, G, A, B):
		intersections = Data_Processing().get_intersections()
		if (A not in intersections):
			print('Start not found')
			return
		if (B not in intersections):
			print('End not found')
			return
		return nx.shortest_path(G, source = A, target = B, weight = 'weight', method = 'dijkstra')

if __name__ == 'main':
	Shortest_Path('test.csv', 13060, 41218)