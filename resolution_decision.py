from IPython import get_ipython
import numpy as np
# import matplotlib.pyplot as plt
# from skimage import data
# import numpy as np

# Clear workspace
get_ipython().run_line_magic('reset', '-sf')

# image = data.horse()
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# plt.imshow(np.flipud(image), origin='lower')
# plt.show()


def image_object(path_geo, object_name):
    # Generated geometry will be in the form of PNG as it is smaller in memory than CSV or TXT
    from PIL import Image
    return Image.open(path_geo + object_name, mode='r').convert('L')


def path_geo():
    return "./CB54/code_ref/vefie_for_building_streamlined/geometry/"


def object_name():
    # return "object_mp_landscape_empty.txt"
    return "placeholder.png"


def path_lut():
    return "./CB54/code_ref/vefie_for_building_streamlined/lut/materials.json"


def unique_integers(image_object):
    # "Unique Values in Geometry"
    import pandas as pd
    import numpy as np
    unique_elements, counts_elements = np.unique(image_object, return_counts=True)
    return pd.DataFrame({"uint8": unique_elements, "counts_elements": counts_elements})


def materials_dict(path_lut):
    import pandas as pd
    import json
    return pd.DataFrame(json.load(open(path_lut, 'r')))


def image_geometry_materials_parse(materials_dict, unique_integers):
    # "Unique Values in Geometry"
    import pandas as pd
    materials_dict['uint8'] = materials_dict['uint8'].astype(int)
    unique_integers['uint8'] = unique_integers['uint8'].astype(int)
    return pd.merge(unique_integers.set_index(['uint8']), materials_dict.set_index(['uint8']), on='uint8', how='left')


path_geo = path_geo()
path_lut = path_lut()
materials_dict = materials_dict(path_lut)
object_name = object_name()
image_object = image_object(path_geo, object_name)
unique_integers = unique_integers(image_object)
image_geometry_materials_parse = image_geometry_materials_parse(materials_dict, unique_integers)
# print(image_geometry_materials_parse)


discretise_M = 116
discretise_N = 284
basis_wave_number = np.full((discretise_M * discretise_N, 1), 10.0)
print(basis_wave_number.shape)
basis_counter = 0
for ct1 in range(0, discretise_M):
    for ct2 in range(0, discretise_N):
        basis_counter = ct1 * discretise_N + ct2
        basis_wave_number[basis_counter, 0] = basis_counter

print("ct1", ct1)
print("ct2", ct2)
print("basis_counter", basis_counter)
print("basis_wave_number.shape", basis_wave_number.shape)
print("basis_wave_number", basis_wave_number)
