# import numpy as np


# python -m cProfile -o main_plain.profile main_plain.py
# pyprof2calltree -i main_plain.profile -o main_plain.calltree
# kcachegrind main_plain.calltree

# from IPython import get_ipython
# # Clear workspace
# get_ipython().run_line_magic('reset', '-sf')
# get_ipython().run_line_magic('clear', '-sf')
try:
    from lib import scene_gen
except ImportError:
    import scene_gen
import time
# import numpy as np
from matplotlib import pyplot as plt
# from time import perf_counter


start = time.time()

input_disc_per_lambda = scene_gen.input_disc_per_lambda()
# print("input_disc_per_lambda: ", input_disc_per_lambda)
object_name = scene_gen.object_name()
# print("object_name: ", object_name)
path_geo = scene_gen.path_geo()
# print("path_geo: ", path_geo)
path_lut = scene_gen.path_lut()
# print("path_lut: ", path_lut)
image_object = scene_gen.image_object(path_geo, object_name)
# print("image_object: ", image_object)
unique_integers = scene_gen.unique_integers(image_object)
# print("unique_integers: ", unique_integers)
materials_dict = scene_gen.materials_dict(path_lut)
# print("materials_dict: ", materials_dict)
epsilon0 = scene_gen.epsilon0()
# print("epsilon0: ", epsilon0)
input_carrier_frequency = scene_gen.input_carrier_frequency()
# print("input_carrier_frequency: ", input_carrier_frequency)
image_geometry_materials_parse = scene_gen.image_geometry_materials_parse(materials_dict, unique_integers)
# print("image_geometry_materials_parse: ", image_geometry_materials_parse)
mu0 = scene_gen.mu0()
# print("mu0: ", mu0)
angular_frequency = scene_gen.angular_frequency(input_carrier_frequency)
# print("angular_frequency: ", angular_frequency)
image_geometry_materials_full = scene_gen.image_geometry_materials_full(image_geometry_materials_parse, epsilon0, input_carrier_frequency, mu0, angular_frequency)
# print("image_geometry_materials_full: ", image_geometry_materials_full)
lambda_smallest = scene_gen.lambda_smallest(image_geometry_materials_full, epsilon0, mu0, input_carrier_frequency)
# print("lambda_smallest: ", lambda_smallest)
length_x_side = scene_gen.length_x_side(image_object)
# print("length_x_side: ", length_x_side)
length_y_side = scene_gen.length_y_side(image_object)
# print("length_y_side: ", length_y_side)
longest_side = scene_gen.longest_side(length_x_side, length_y_side)
# print("longest_side: ", longest_side)
discretise_side_1 = scene_gen.discretise_side_1(longest_side, lambda_smallest, input_disc_per_lambda)
# print("discretise_side_1: ", discretise_side_1)
delta_1 = scene_gen.delta_1(longest_side, discretise_side_1)
# print("delta_1: ", delta_1)
discretise_side_2 = scene_gen.discretise_side_2(longest_side, delta_1)
# print("discretise_side_2: ", discretise_side_2)
delta_2 = scene_gen.delta_2(longest_side, discretise_side_2)
# print("delta_2: ", delta_2)
resolution_information = scene_gen.resolution_information(longest_side, discretise_side_1, discretise_side_2)
# print("resolution_information: ", resolution_information)
basis_counter = scene_gen.basis_counter(resolution_information)
# print("basis_counter: ", basis_counter)
vacuum_kr = scene_gen.vacuum_kr(image_geometry_materials_full)
# print("vacuum_kr: ", vacuum_kr)
equiv_a = scene_gen.equiv_a(delta_1, delta_2)
# print("equiv_a: ", equiv_a)
input_centre = scene_gen.input_centre()
# print("input_centre: ", input_centre)
start_point = scene_gen.start_point(input_centre, input_disc_per_lambda, length_x_side, length_y_side)
# print("start_point: ", start_point)
position = scene_gen.position(resolution_information, longest_side, start_point)
# print("position: ", position)
G_vector = scene_gen.G_vector(basis_counter, position, equiv_a, image_geometry_materials_full, vacuum_kr)
# print("G_vector: ", G_vector)
image_resize = scene_gen.image_resize(image_object, resolution_information)
# print("image_resize: ", image_resize)
basis_wave_number = scene_gen.basis_wave_number(resolution_information, image_geometry_materials_full, image_resize)
# print("basis_wave_number: ", basis_wave_number)
field_incident_D = scene_gen.field_incident_D(basis_counter, basis_wave_number, vacuum_kr)
# print("field_incident_D: ", field_incident_D)
rfo = scene_gen.rfo(field_incident_D)
# print("rfo: ", rfo)
the_phi = scene_gen.the_phi(resolution_information, image_geometry_materials_full, longest_side, start_point, input_centre, image_resize, position)
# print("the_phi: ", the_phi)
rho = scene_gen.rho(resolution_information, image_geometry_materials_full, longest_side, start_point, input_centre, image_resize, position)
# print("rho: ", rho)
field_incident_V = scene_gen.field_incident_V(basis_counter, rho, the_phi, vacuum_kr)
# print("field_incident_V: ", field_incident_V)
Vred = scene_gen.Vred(rfo, field_incident_V)
# print("Vred: ", Vred)
Vred_2D = scene_gen.Vred_2D(resolution_information, Vred)
# print("Vred_2D: ", Vred_2D)
palette = scene_gen.palette(materials_dict)
# print("palette: ", palette)
image_resize_render = scene_gen.image_resize_render(image_resize, palette)
# print("image_resize_render: ", image_resize_render)
parula_map = scene_gen.parula_map()
# print("parula_map: ", parula_map)
image_render = scene_gen.image_render(image_object, palette)
# print("image_render: ", image_render)
model_guess = scene_gen.model_guess()
# print("model_guess: ", model_guess)
seed = scene_gen.seed()
# print("seed: ", seed)
Ered_load = scene_gen.Ered_load(basis_counter, model_guess)
# print("Ered: ", Ered)
r = scene_gen.r(Ered_load, G_vector, Vred, field_incident_D, resolution_information, rfo)
# print("r: ", r)
p = scene_gen.p(G_vector, field_incident_D, resolution_information, rfo, r)
# print("p: ", p)
input_solver_tol = scene_gen.input_solver_tol()
# print("input_solver_tol: ", input_solver_tol)
solver_error = scene_gen.solver_error(r)
# print("solver_error: ", solver_error)

# start = perf_counter()
krylov_solver = scene_gen.krylov_solver(basis_counter, input_solver_tol, G_vector, field_incident_D, p, r, resolution_information, rfo, Ered_load)
# print("type(krylov_solver): ", type(krylov_solver))
# duration = perf_counter() - start
# print('krylov_solver', duration)


Ered = scene_gen.Ered(krylov_solver)
# print("Ered: ", Ered)
reduced_iteration_error = scene_gen.reduced_iteration_error(krylov_solver)
# print("reduced_iteration_error: ", reduced_iteration_error)

ScatRed = scene_gen.ScatRed(Ered, Vred)

Ered_2D = scene_gen.Ered_2D(resolution_information, Ered)
ScatRed_2D = scene_gen.ScatRed_2D(resolution_information, ScatRed)

[image_Vred_2D_real, image_Vred_2D_imag, image_Vred_2D_abs] = scene_gen.complex_image_render(Vred_2D, parula_map)
[image_Ered_2D_real, image_Ered_2D_imag, image_Ered_2D_abs] = scene_gen.complex_image_render(Ered_2D, parula_map)
[image_ScatRed_2D_real, image_ScatRed_2D_imag, image_ScatRed_2D_abs] = scene_gen.complex_image_render(ScatRed_2D, parula_map)

# PLOT ITERATIONS
# plt.semilogy(reduced_iteration_error[:, 0], reduced_iteration_error[:, 1])
# plt.autoscale(enable=True, axis='x', tight=True)

# CREATE ALL THE PLOTS
# plt.imshow(image_Vred_2D_real)
# # plt.show()
# plt.imshow(image_Vred_2D_imag)
# # plt.show()
# plt.imshow(image_Vred_2D_abs)
# # plt.show()
# plt.imshow(image_Ered_2D_real)
# # plt.show()
# plt.imshow(image_Ered_2D_imag)
# # plt.show()
# plt.imshow(image_Ered_2D_abs)
# # plt.show()
# plt.imshow(image_ScatRed_2D_real)
# # plt.show()
# plt.imshow(image_ScatRed_2D_imag)
# # plt.show()
# plt.imshow(image_ScatRed_2D_abs)
# # plt.show()

end = time.time()
print("code runtime: ", end - start)
############################################################################################
# FRESH WORK BELOW HERE

print('Used frequency of incomming wave = ', input_carrier_frequency,  'Hz')
print('Discretizised space is ', resolution_information["length_x_side"], 'grids by ', resolution_information["length_y_side"], 'grids')
print('with grid size', longest_side[longest_side["name"] == "length_x_side"]["length"] / resolution_information["length_x_side"], 'meter by ', longest_side[longest_side["name"] == "length_y_side"]["length"] / resolution_information["length_y_side"], ' meter')
print('So basis counter goes to ', basis_counter)
print('Number of unknowns in Z is than ', resolution_information["length_x_side"]*resolution_information["length_y_side"]*resolution_information["length_x_side"]*resolution_information["length_y_side"])
print('Number of reduced unknowns is ', sum(rfo)*resolution_information["length_x_side"]*resolution_information["length_y_side"])
print('with ', sum(rfo)/(resolution_information["length_x_side"] * resolution_information["length_y_side"])*100, 'percent of the is filled by contrast')
print('CG iteration error tollerance = ', input_solver_tol)
# print('Duration of reduced CG iteration = %d seconds \n', time_Red)



