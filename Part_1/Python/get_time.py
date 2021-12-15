import pandas as pd

class Get_Time:
	def __init__(self, path, graph_name = 'test.csv', has_density = True):
		self.jam_density = 30
		self.max_speed = 60
		self.road_len = 1

		self.graph_name = graph_name
		self.data_folder = '../Data/'
		self.graph_path = self.data_folder + self.graph_name

		if not has_density:
			self.zero_density_time(path = path)
			return
		self.travel_time(path, graph_name)

	def __speed_bucket(self, density):
		max_speed = self.max_speed
		jam_density = self.jam_density
		if (density <= 5):
			return max_speed
		elif (density > 5 and density <= 10):
			return max_speed - 10
		elif (density > 10 and density <= 15):
			return max_speed - 20
		elif (density > 15 and density <= 20):
			return max_speed - 30
		elif (density > 20 and density <= 30):
			return max_speed - 40
		elif (density > 30):
			return max_speed - 50
		if (density > jam_density):
			return 10

	def zero_density_time(self, path):
		return ((len(path) - 1) / self.max_speed)

	def travel_time(self, path, graph_name = 'test.csv'):
		data_folder = '../Data/'
		graph_path = data_folder + graph_name
		
		graph = pd.read_csv(graph_path, index_col = 0)
		time = 0
		for i in range(len(path) - 1):
			source = path[i]
			target = path[i + 1]
			weight = graph[str(source)][target]
			speed = self.__speed_bucket(weight)
			time += self.road_len / speed
		return time