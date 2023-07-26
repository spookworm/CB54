from lib import custom_functions
import pickle
from IPython import get_ipython


# # Clear workspace
# from numba import cuda
# cuda.select_device(0)
# cuda.close()

get_ipython().run_line_magic('clear', '-sf')


# Load the training history from the pickle file
with open('training_history.pkl', 'rb') as file:
    history = pickle.load(file)
print(history.keys())
# print(history.history.keys())
custom_functions.plot_loss(history)


# print("start stats")
# output_folder = "F:\\instances_output"
# info_dataset = custom_functions.info_data_harvest(output_folder)
# custom_functions.info_data_paired('.\\doc\\_stats\\dataset_instances_output.csv')
