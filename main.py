from gp import *

grid_size = (4, 7)
nums = 45 # number of heat values on each map
l = 400 # length of time series

# np.random.seed(1)
# Asin(0.01* t) + B
params = np.random.random(size=(nums, 2))

list_of_graphs = [heatmap_sequence_generator(i, params, grid_size) for i in range(l)]

gp = GraphPredictor(list_of_graphs)

t = time()
gp.convert_graph_list_to_tensor_batch()
print("converting time:\t", time() - t)

t = time()
gp.train()
print("training time:\t", time() - t)
