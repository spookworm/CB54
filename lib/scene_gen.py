# import numba


def seed():
    # Seed
    return 42


def path_geo():
    return "./code_ref/vefie_for_building_streamlined/geometry/"


def path_lut():
    return "./code_ref/vefie_for_building_streamlined/lut/materials.json"


def object_name():
    # return "object_mp_landscape_empty.txt"
    return "placeholder.png"


def epsilon0():
    # Vacuum Permittivity (F/m)
    return 8.854187817e-12


def mu0():
    # Vacuum Permeability
    import math
    return 4.0 * math.pi * 1.0e-7

#
# def realmax():
#     # Equivalent of realmax in MATLAB. Used to find the minimum resolution required by the materials present in the scene.
#     import sys
#     return sys.float_info.max


def input_carrier_frequency():
    return 60e6


def input_disc_per_lambda():
    return 10


def input_solver_tol():
    return 1e-1


def angular_frequency(input_carrier_frequency):
    import math
    return 2.0 * math.pi * input_carrier_frequency


def image_object(path_geo, object_name):
    # Generated geometry will be in the form of PNG as it is smaller in memory than CSV or TXT
    from PIL import Image
    return Image.open(path_geo + object_name, mode='r').convert('L')


def image_render(image_object, palette):
    # print(image_object.mode)
    # Gradio may be converting input here before it reaches the operations so bear that in mind.
    image_object.putpalette(palette)
    return image_object

#
# def image_image(image_object):
#     from PIL import Image
#     # return Image.open("./geometry/placeholder.png", mode='r')
#     return Image.fromarray(image_object, 'RGB')


def unique_integers(image_object):
    # "Unique Values in Geometry"
    import pandas as pd
    import numpy as np
    unique_elements, counts_elements = np.unique(image_object, return_counts=True)
    # return pd.DataFrame({"uint8": np.unique(image_object).tolist()})
    return pd.DataFrame({"uint8": unique_elements, "counts_elements": counts_elements})


def materials_dict(path_lut):
    import pandas as pd
    import json
    return pd.DataFrame(json.load(open(path_lut, 'r')))


def palette(materials_dict):
    from PIL import ImageColor
    tuples = []
    for ind in materials_dict.index:
        tuples.append(ImageColor.getcolor(materials_dict['HEX'][ind], "RGB"))
    palette_out = [item for sublist in tuples for item in sublist]
    # print(palette_out)
    return palette_out


def image_geometry_materials_parse(materials_dict, unique_integers):
    # "Unique Values in Geometry"
    import pandas as pd
    materials_dict['uint8'] = materials_dict['uint8'].astype(int)
    unique_integers['uint8'] = unique_integers['uint8'].astype(int)
    # return pd.merge(unique_integers.set_index(['uint8']), materials_dict.set_index(['uint8']), on='uint8', how='left')
    return pd.merge(unique_integers, materials_dict, on='uint8', how='left')


def image_geometry_materials_full(image_geometry_materials_parse, epsilon0, input_carrier_frequency, mu0, angular_frequency):
    # "Calculated properties for specific carrier frequency"
    import math
    # [epsilon,sigma,complexepsilon] = buildingMaterialPermittivity(material,fc)
    fcGHz = input_carrier_frequency/1e9

    # epsilon = a*(fcGHz)^b. See (57) in [1]
    image_geometry_materials_parse["epsilonr"] = image_geometry_materials_parse["a"].astype(float) * pow(fcGHz, image_geometry_materials_parse["b"].astype(float))

    # sigma = c*(fcGHz)^d). See (58) in [1]
    image_geometry_materials_parse["sigma"] = image_geometry_materials_parse["c"].astype(float) * pow(fcGHz, image_geometry_materials_parse["d"].astype(float))

    # complexEpsilon = epsilon - 1i*(sigma/(2*pi*fc*epsilon0). See (9b) in [1]
    image_geometry_materials_parse["epsilonr_complex"] = image_geometry_materials_parse["epsilonr"].astype(float) - (1j * image_geometry_materials_parse["sigma"].astype(float))/(2 * math.pi * input_carrier_frequency * epsilon0)

    # Set relative permiablility to one initially
    image_geometry_materials_parse["mur"] = 1.0
    image_geometry_materials_parse["mur_complex"] = 1.0 - (0.0 * 1j)

    # Set k in material.
    # kr(k) = omega * sqrt(epsilonr(k)*epsilon0*mur(k)*mu0);
    image_geometry_materials_parse["kr"] = angular_frequency * pow(image_geometry_materials_parse["epsilonr"].astype(float)*epsilon0*image_geometry_materials_parse["mur"].astype(float)*mu0, 0.5)
    return image_geometry_materials_parse


def lambda_smallest(image_geometry_materials_full, epsilon0, mu0, input_carrier_frequency):
    # "Choose the smallest lambda (wavelength) of all materials in the configuration."
    # cr(k)=1.0 / sqrt(epsilonr_complex(k)*epsilon0*mur(k)*mu0);
    image_geometry_materials_full["cr"] = 1.0/pow(image_geometry_materials_full["epsilonr"].astype(float)*epsilon0*image_geometry_materials_full["mur"].astype(float)*mu0, 0.5)
    lambda_smallest_val = image_geometry_materials_full["cr"].astype(float).min() / input_carrier_frequency
    return lambda_smallest_val


def length_x_side(image_object):
    return image_object.size[0]


def length_y_side(image_object):
    return image_object.size[1]


def longest_side(length_x_side, length_y_side):
    # Could there be a problem if both are equal in length?
    import pandas as pd
    data = pd.DataFrame(
        {"name": ["length_x_side", "length_y_side"],
         "length": [length_x_side, length_y_side]
         },
        index=[1, 2]
        )
    return data


def discretise_side_1(longest_side, lambda_smallest, input_disc_per_lambda):
    import math
    N = math.floor(longest_side['length'].max()/(abs(lambda_smallest) / input_disc_per_lambda))
    fourth_of_N = math.ceil(N/4)
    while ((N % fourth_of_N) != 0):
        N = N + 1
    return N


def delta_1(longest_side, discretise_side_1):
    return longest_side['length'].max() / discretise_side_1


def discretise_side_2(longest_side, delta_1):
    import math
    N = math.floor(longest_side['length'].min()/delta_1)
    fourth_of_N = math.ceil(N/4)
    while ((N % fourth_of_N) != 0):
        N = N + 1
    return N


def delta_2(longest_side, discretise_side_2):
    return longest_side['length'].min() / discretise_side_2


def equiv_a(delta_1, delta_2):
    import math
    return pow(delta_1*delta_2/math.pi, 0.5)


def resolution_information(longest_side, discretise_side_1, discretise_side_2):
    data = {}
    longest = str(longest_side[longest_side["length"] == longest_side["length"].max()]['name'].iloc[0])
    shortest = str(longest_side[longest_side["length"] == longest_side["length"].min()]['name'].iloc[0])
    data[longest] = discretise_side_1
    data[shortest] = discretise_side_2
    return data


def image_resize(image_object, resolution_information):
    from PIL import Image
    return image_object.resize((resolution_information["length_x_side"], resolution_information["length_y_side"]), Image.Resampling.NEAREST)


def image_resize_render(image_resize, palette):
    # print(image_object.mode)
    # Gradio may be converting input here before it reaches the operations so bear that in mind.

    # print("type(image_resize)", type(image_resize))
    image_resize.putpalette(palette)
    return image_resize


def input_centre():
    return 0.0 + 0.0 * 1j


def start_point(input_centre, input_disc_per_lambda, length_x_side, length_y_side):
    # CHECK, is this correct? like there will have to be multiple resize options so why stick with original lengths etc.
    return input_centre - 0.5 * length_x_side - 0.5 * length_y_side * 1j


def position(resolution_information, longest_side, start_point):
    import numpy as np
    # PRE-POPULATING THE MATRIX WITH THE DOMININANT MATERIAL MIGHT SAVE OPERATIONS
    discretise_M = resolution_information["length_y_side"]
    discretise_N = resolution_information["length_x_side"]
    delta_y = longest_side[longest_side["name"] == "length_y_side"]["length"] / resolution_information["length_y_side"]
    delta_x = longest_side[longest_side["name"] == "length_x_side"]["length"] / resolution_information["length_x_side"]
    delta_y = delta_y.iloc[0]
    delta_x = delta_x.iloc[0]

    # NOT WORTH VECTORISING ... YET!
    # ct1 -> ct1 + 1 ?
    position = np.zeros((discretise_M * discretise_N, 1), dtype=np.complex_)
    basis_counter = 0
    for ct1 in range(0, discretise_M):
        for ct2 in range(0, discretise_N):
            # basis_counter = (ct1 - 1) * discretise_N + ct2
            basis_counter = ct1 * discretise_N + ct2
            # position(basis_counter, 1) = start_pt + (ct2 - 0.5) * delta_x + 1i * (ct1 - 0.5) * delta_y;
            position[basis_counter, 0] = start_point + (ct2 + 0.5) * delta_x + 1j * (ct1 + 0.5) * delta_y
    return position


def rho(resolution_information, image_geometry_materials_full, longest_side, start_point, input_centre, image_resize, position):
    import numpy as np
    # PRE-POPULATING THE MATRIX WITH THE DOMININANT MATERIAL MIGHT SAVE OPERATIONS
    discretise_M = resolution_information["length_y_side"]
    discretise_N = resolution_information["length_x_side"]
    delta_y = longest_side[longest_side["name"] == "length_y_side"]["length"] / resolution_information["length_y_side"]
    delta_x = longest_side[longest_side["name"] == "length_x_side"]["length"] / resolution_information["length_x_side"]
    delta_y = delta_y.iloc[0]
    delta_x = delta_x.iloc[0]

    # NOT WORTH VECTORISING ... YET!
    # ct1 -> ct1 + 1 ?
    # position = np.zeros((discretise_M * discretise_N, 1), dtype=np.complex_)
    rho = np.zeros((discretise_M*discretise_N, 1))
    basis_counter = 0
    for ct1 in range(0, discretise_M):
        for ct2 in range(0, discretise_N):
            # basis_counter = (ct1 - 1) * discretise_N + ct2
            basis_counter = ct1 * discretise_N + ct2
            # position(basis_counter, 1) = start_pt + (ct2 - 0.5) * delta_x + 1i * (ct1 - 0.5) * delta_y;
            # position[basis_counter, 0] = start_point + (ct2 + 0.5) * delta_x + 1j * (ct1 + 0.5) * delta_y
            temp_vec = position[basis_counter, 0] - input_centre
            rho[basis_counter, 0] = abs(temp_vec)
    return rho


def the_phi(resolution_information, image_geometry_materials_full, longest_side, start_point, input_centre, image_resize, position):
    import numpy as np
    import math
    # PRE-POPULATING THE MATRIX WITH THE DOMININANT MATERIAL MIGHT SAVE OPERATIONS
    discretise_M = resolution_information["length_y_side"]
    discretise_N = resolution_information["length_x_side"]
    delta_y = longest_side[longest_side["name"] == "length_y_side"]["length"] / resolution_information["length_y_side"]
    delta_x = longest_side[longest_side["name"] == "length_x_side"]["length"] / resolution_information["length_x_side"]
    delta_y = delta_y.iloc[0]
    delta_x = delta_x.iloc[0]

    # NOT WORTH VECTORISING ... YET!
    # ct1 -> ct1 + 1 ?
    # position = np.zeros((discretise_M * discretise_N, 1), dtype=np.complex_)
    the_phi = np.zeros((discretise_M*discretise_N, 1))
    basis_counter = 0
    for ct1 in range(0, discretise_M):
        for ct2 in range(0, discretise_N):
            # basis_counter = (ct1 - 1) * discretise_N + ct2
            basis_counter = ct1 * discretise_N + ct2
            # position(basis_counter, 1) = start_pt + (ct2 - 0.5) * delta_x + 1i * (ct1 - 0.5) * delta_y;
            # position[basis_counter, 0] = start_point + (ct2 + 0.5) * delta_x + 1j * (ct1 + 0.5) * delta_y
            temp_vec = position[basis_counter, 0] - input_centre
            the_phi[basis_counter, 0] = math.atan2(np.imag(temp_vec), np.real(temp_vec))
    return the_phi


def basis_wave_number(resolution_information, image_geometry_materials_full, image_resize):
    import numpy as np
    discretise_M = resolution_information["length_y_side"]
    discretise_N = resolution_information["length_x_side"]
    image_resize = np.array(image_resize)

    prepop_fill = image_geometry_materials_full[image_geometry_materials_full["counts_elements"] == image_geometry_materials_full["counts_elements"].max()]['kr'].iloc[0]

    # NOT WORTH VECTORISING ... YET!
    # ct1 -> ct1 + 1 ?
    basis_wave_number = np.full((discretise_M * discretise_N, 1), prepop_fill)
    basis_counter = 0
    for ct1 in range(0, discretise_M):
        for ct2 in range(0, discretise_N):
            # basis_counter = (ct1 - 1) * discretise_N + ct2
            basis_counter = ct1 * discretise_N + ct2
            temp_cell = image_geometry_materials_full[image_geometry_materials_full['uint8'] == image_resize[ct1, ct2]]['kr']
            basis_wave_number[basis_counter, 0] = temp_cell.iloc[0]
    return basis_wave_number


def basis_counter(resolution_information):
    discretise_M = resolution_information["length_y_side"]
    discretise_N = resolution_information["length_x_side"]
    basis_counter = discretise_M*discretise_N
    return basis_counter


def vacuum_kr(image_geometry_materials_full):
    vacuum_kr = image_geometry_materials_full[image_geometry_materials_full['name'] == 'vacuum']['kr'].iloc[0]
    return vacuum_kr


def field_incident_V(basis_counter, rho, the_phi, vacuum_kr):
    import cmath
    import numpy as np
    V = np.zeros((basis_counter, 1), dtype=np.complex_)
    for ct1 in range(0, basis_counter):
        # V(ct1) = exp(-1i*kr(1)*rho(ct1)*cos(the_phi(ct1)));
        V[ct1] = cmath.exp(-1j * vacuum_kr * rho[ct1] * cmath.cos(the_phi[ct1]))
    return V


def field_incident_D(basis_counter, basis_wave_number, vacuum_kr):
    # import cmath
    import numpy as np
    D = np.zeros((basis_counter, 1), dtype=np.complex_)

    # D(ct1,1)= (basis_wave_number(ct1,1)*basis_wave_number(ct1,1) - k0*k0); % contrast function
    for ct1 in range(0, basis_counter):
        D[ct1] = basis_wave_number[ct1] * basis_wave_number[ct1] - vacuum_kr * vacuum_kr

    # This is somehow slower!
    # D = [basis_wave_number[ct1] * basis_wave_number[ct1] - vacuum_kr * vacuum_kr for ct1 in range(0, basis_counter)]
    return D


def rfo(field_incident_D):
    # Reduced Foreward Operator
    # Rfo = logical(D);
    import numpy as np
    rfo = abs(np.real(field_incident_D.copy())) + abs(np.imag(field_incident_D.copy()))
    rfo[rfo != 0] = 1

    # unique_elements, counts_elements = np.unique(rfo, return_counts=True)
    # print(unique_elements)
    # print(counts_elements)
    return rfo.astype(int)


def Vred(rfo, field_incident_V):
    # Vred = Rfo .* V; % create V reduced
    Vred = rfo * field_incident_V
    return Vred


def parula_map():
    parula_data = [[0.2422, 0.1504, 0.6603],
                   [0.2444, 0.1534, 0.6728],
                   [0.2464, 0.1569, 0.6847],
                   [0.2484, 0.1607, 0.6961],
                   [0.2503, 0.1648, 0.7071],
                   [0.2522, 0.1689, 0.7179],
                   [0.254, 0.1732, 0.7286],
                   [0.2558, 0.1773, 0.7393],
                   [0.2576, 0.1814, 0.7501],
                   [0.2594, 0.1854, 0.761],
                   [0.2611, 0.1893, 0.7719],
                   [0.2628, 0.1932, 0.7828],
                   [0.2645, 0.1972, 0.7937],
                   [0.2661, 0.2011, 0.8043],
                   [0.2676, 0.2052, 0.8148],
                   [0.2691, 0.2094, 0.8249],
                   [0.2704, 0.2138, 0.8346],
                   [0.2717, 0.2184, 0.8439],
                   [0.2729, 0.2231, 0.8528],
                   [0.274, 0.228, 0.8612],
                   [0.2749, 0.233, 0.8692],
                   [0.2758, 0.2382, 0.8767],
                   [0.2766, 0.2435, 0.884],
                   [0.2774, 0.2489, 0.8908],
                   [0.2781, 0.2543, 0.8973],
                   [0.2788, 0.2598, 0.9035],
                   [0.2794, 0.2653, 0.9094],
                   [0.2798, 0.2708, 0.915],
                   [0.2802, 0.2764, 0.9204],
                   [0.2806, 0.2819, 0.9255],
                   [0.2809, 0.2875, 0.9305],
                   [0.2811, 0.293, 0.9352],
                   [0.2813, 0.2985, 0.9397],
                   [0.2814, 0.304, 0.9441],
                   [0.2814, 0.3095, 0.9483],
                   [0.2813, 0.315, 0.9524],
                   [0.2811, 0.3204, 0.9563],
                   [0.2809, 0.3259, 0.96],
                   [0.2807, 0.3313, 0.9636],
                   [0.2803, 0.3367, 0.967],
                   [0.2798, 0.3421, 0.9702],
                   [0.2791, 0.3475, 0.9733],
                   [0.2784, 0.3529, 0.9763],
                   [0.2776, 0.3583, 0.9791],
                   [0.2766, 0.3638, 0.9817],
                   [0.2754, 0.3693, 0.984],
                   [0.2741, 0.3748, 0.9862],
                   [0.2726, 0.3804, 0.9881],
                   [0.271, 0.386, 0.9898],
                   [0.2691, 0.3916, 0.9912],
                   [0.267, 0.3973, 0.9924],
                   [0.2647, 0.403, 0.9935],
                   [0.2621, 0.4088, 0.9946],
                   [0.2591, 0.4145, 0.9955],
                   [0.2556, 0.4203, 0.9965],
                   [0.2517, 0.4261, 0.9974],
                   [0.2473, 0.4319, 0.9983],
                   [0.2424, 0.4378, 0.9991],
                   [0.2369, 0.4437, 0.9996],
                   [0.2311, 0.4497, 0.9995],
                   [0.225, 0.4559, 0.9985],
                   [0.2189, 0.462, 0.9968],
                   [0.2128, 0.4682, 0.9948],
                   [0.2066, 0.4743, 0.9926],
                   [0.2006, 0.4803, 0.9906],
                   [0.195, 0.4861, 0.9887],
                   [0.1903, 0.4919, 0.9867],
                   [0.1869, 0.4975, 0.9844],
                   [0.1847, 0.503, 0.9819],
                   [0.1831, 0.5084, 0.9793],
                   [0.1818, 0.5138, 0.9766],
                   [0.1806, 0.5191, 0.9738],
                   [0.1795, 0.5244, 0.9709],
                   [0.1785, 0.5296, 0.9677],
                   [0.1778, 0.5349, 0.9641],
                   [0.1773, 0.5401, 0.9602],
                   [0.1768, 0.5452, 0.956],
                   [0.1764, 0.5504, 0.9516],
                   [0.1755, 0.5554, 0.9473],
                   [0.174, 0.5605, 0.9432],
                   [0.1716, 0.5655, 0.9393],
                   [0.1686, 0.5705, 0.9357],
                   [0.1649, 0.5755, 0.9323],
                   [0.161, 0.5805, 0.9289],
                   [0.1573, 0.5854, 0.9254],
                   [0.154, 0.5902, 0.9218],
                   [0.1513, 0.595, 0.9182],
                   [0.1492, 0.5997, 0.9147],
                   [0.1475, 0.6043, 0.9113],
                   [0.1461, 0.6089, 0.908],
                   [0.1446, 0.6135, 0.905],
                   [0.1429, 0.618, 0.9022],
                   [0.1408, 0.6226, 0.8998],
                   [0.1383, 0.6272, 0.8975],
                   [0.1354, 0.6317, 0.8953],
                   [0.1321, 0.6363, 0.8932],
                   [0.1288, 0.6408, 0.891],
                   [0.1253, 0.6453, 0.8887],
                   [0.1219, 0.6497, 0.8862],
                   [0.1185, 0.6541, 0.8834],
                   [0.1152, 0.6584, 0.8804],
                   [0.1119, 0.6627, 0.877],
                   [0.1085, 0.6669, 0.8734],
                   [0.1048, 0.671, 0.8695],
                   [0.1009, 0.675, 0.8653],
                   [0.0964, 0.6789, 0.8609],
                   [0.0914, 0.6828, 0.8562],
                   [0.0855, 0.6865, 0.8513],
                   [0.0789, 0.6902, 0.8462],
                   [0.0713, 0.6938, 0.8409],
                   [0.0628, 0.6972, 0.8355],
                   [0.0535, 0.7006, 0.8299],
                   [0.0433, 0.7039, 0.8242],
                   [0.0328, 0.7071, 0.8183],
                   [0.0234, 0.7103, 0.8124],
                   [0.0155, 0.7133, 0.8064],
                   [0.0091, 0.7163, 0.8003],
                   [0.0046, 0.7192, 0.7941],
                   [0.0019, 0.722, 0.7878],
                   [0.0009, 0.7248, 0.7815],
                   [0.0018, 0.7275, 0.7752],
                   [0.0046, 0.7301, 0.7688],
                   [0.0094, 0.7327, 0.7623],
                   [0.0162, 0.7352, 0.7558],
                   [0.0253, 0.7376, 0.7492],
                   [0.0369, 0.74, 0.7426],
                   [0.0504, 0.7423, 0.7359],
                   [0.0638, 0.7446, 0.7292],
                   [0.077, 0.7468, 0.7224],
                   [0.0899, 0.7489, 0.7156],
                   [0.1023, 0.751, 0.7088],
                   [0.1141, 0.7531, 0.7019],
                   [0.1252, 0.7552, 0.695],
                   [0.1354, 0.7572, 0.6881],
                   [0.1448, 0.7593, 0.6812],
                   [0.1532, 0.7614, 0.6741],
                   [0.1609, 0.7635, 0.6671],
                   [0.1678, 0.7656, 0.6599],
                   [0.1741, 0.7678, 0.6527],
                   [0.1799, 0.7699, 0.6454],
                   [0.1853, 0.7721, 0.6379],
                   [0.1905, 0.7743, 0.6303],
                   [0.1954, 0.7765, 0.6225],
                   [0.2003, 0.7787, 0.6146],
                   [0.2061, 0.7808, 0.6065],
                   [0.2118, 0.7828, 0.5983],
                   [0.2178, 0.7849, 0.5899],
                   [0.2244, 0.7869, 0.5813],
                   [0.2318, 0.7887, 0.5725],
                   [0.2401, 0.7905, 0.5636],
                   [0.2491, 0.7922, 0.5546],
                   [0.2589, 0.7937, 0.5454],
                   [0.2695, 0.7951, 0.536],
                   [0.2809, 0.7964, 0.5266],
                   [0.2929, 0.7975, 0.517],
                   [0.3052, 0.7985, 0.5074],
                   [0.3176, 0.7994, 0.4975],
                   [0.3301, 0.8002, 0.4876],
                   [0.3424, 0.8009, 0.4774],
                   [0.3548, 0.8016, 0.4669],
                   [0.3671, 0.8021, 0.4563],
                   [0.3795, 0.8026, 0.4454],
                   [0.3921, 0.8029, 0.4344],
                   [0.405, 0.8031, 0.4233],
                   [0.4184, 0.803, 0.4122],
                   [0.4322, 0.8028, 0.4013],
                   [0.4463, 0.8024, 0.3904],
                   [0.4608, 0.8018, 0.3797],
                   [0.4753, 0.8011, 0.3691],
                   [0.4899, 0.8002, 0.3586],
                   [0.5044, 0.7993, 0.348],
                   [0.5187, 0.7982, 0.3374],
                   [0.5329, 0.797, 0.3267],
                   [0.547, 0.7957, 0.3159],
                   [0.5609, 0.7943, 0.305],
                   [0.5748, 0.7929, 0.2941],
                   [0.5886, 0.7913, 0.2833],
                   [0.6024, 0.7896, 0.2726],
                   [0.6161, 0.7878, 0.2622],
                   [0.6297, 0.7859, 0.2521],
                   [0.6433, 0.7839, 0.2423],
                   [0.6567, 0.7818, 0.2329],
                   [0.6701, 0.7796, 0.2239],
                   [0.6833, 0.7773, 0.2155],
                   [0.6963, 0.775, 0.2075],
                   [0.7091, 0.7727, 0.1998],
                   [0.7218, 0.7703, 0.1924],
                   [0.7344, 0.7679, 0.1852],
                   [0.7468, 0.7654, 0.1782],
                   [0.759, 0.7629, 0.1717],
                   [0.771, 0.7604, 0.1658],
                   [0.7829, 0.7579, 0.1608],
                   [0.7945, 0.7554, 0.157],
                   [0.806, 0.7529, 0.1546],
                   [0.8172, 0.7505, 0.1535],
                   [0.8281, 0.7481, 0.1536],
                   [0.8389, 0.7457, 0.1546],
                   [0.8495, 0.7435, 0.1564],
                   [0.86, 0.7413, 0.1587],
                   [0.8703, 0.7392, 0.1615],
                   [0.8804, 0.7372, 0.165],
                   [0.8903, 0.7353, 0.1695],
                   [0.9, 0.7336, 0.1749],
                   [0.9093, 0.7321, 0.1815],
                   [0.9184, 0.7308, 0.189],
                   [0.9272, 0.7298, 0.1973],
                   [0.9357, 0.729, 0.2061],
                   [0.944, 0.7285, 0.2151],
                   [0.9523, 0.7284, 0.2237],
                   [0.9606, 0.7285, 0.2312],
                   [0.9689, 0.7292, 0.2373],
                   [0.977, 0.7304, 0.2418],
                   [0.9842, 0.733, 0.2446],
                   [0.99, 0.7365, 0.2429],
                   [0.9946, 0.7407, 0.2394],
                   [0.9966, 0.7458, 0.2351],
                   [0.9971, 0.7513, 0.2309],
                   [0.9972, 0.7569, 0.2267],
                   [0.9971, 0.7626, 0.2224],
                   [0.9969, 0.7683, 0.2181],
                   [0.9966, 0.774, 0.2138],
                   [0.9962, 0.7798, 0.2095],
                   [0.9957, 0.7856, 0.2053],
                   [0.9949, 0.7915, 0.2012],
                   [0.9938, 0.7974, 0.1974],
                   [0.9923, 0.8034, 0.1939],
                   [0.9906, 0.8095, 0.1906],
                   [0.9885, 0.8156, 0.1875],
                   [0.9861, 0.8218, 0.1846],
                   [0.9835, 0.828, 0.1817],
                   [0.9807, 0.8342, 0.1787],
                   [0.9778, 0.8404, 0.1757],
                   [0.9748, 0.8467, 0.1726],
                   [0.972, 0.8529, 0.1695],
                   [0.9694, 0.8591, 0.1665],
                   [0.9671, 0.8654, 0.1636],
                   [0.9651, 0.8716, 0.1608],
                   [0.9634, 0.8778, 0.1582],
                   [0.9619, 0.884, 0.1557],
                   [0.9608, 0.8902, 0.1532],
                   [0.9601, 0.8963, 0.1507],
                   [0.9596, 0.9023, 0.148],
                   [0.9595, 0.9084, 0.145],
                   [0.9597, 0.9143, 0.1418],
                   [0.9601, 0.9203, 0.1382],
                   [0.9608, 0.9262, 0.1344],
                   [0.9618, 0.932, 0.1304],
                   [0.9629, 0.9379, 0.1261],
                   [0.9642, 0.9437, 0.1216],
                   [0.9657, 0.9494, 0.1168],
                   [0.9674, 0.9552, 0.1116],
                   [0.9692, 0.9609, 0.1061],
                   [0.9711, 0.9667, 0.1001],
                   [0.973, 0.9724, 0.0938],
                   [0.9749, 0.9782, 0.0872],
                   [0.9769, 0.9839, 0.0805]]
    map_1 = [item for sublist in parula_data for item in sublist]
    map_2 = [round(i * 255) for i in map_1]
    return map_2


def G_vector(basis_counter, position, equiv_a, image_geometry_materials_full, vacuum_kr):
    # BMT dense matrix, stored as a vector
    import numpy as np
    import math
    from scipy.special import jv, yv
    # import time

    G_vector = np.zeros((basis_counter, 1), dtype=np.complex_)

    # start = time.time()

    for ct1 in range(0, basis_counter):
        # R_mn2 = abs(position(ct1)-position(1));
        R_mn2 = abs(position[ct1]-position[0])
        # besselj -> jv; bessely -> yv
        if ct1 == 0:
            # G_vector(ct1,1)=(1i/4.0)*((2.0*pi*equiv_a/kr(1))*(besselj(1,kr(1)*equiv_a)-1i*bessely(1,kr(1)*equiv_a))-4.0*1i/(kr(1)*kr(1)));
            G_vector[ct1] = (1j / 4.0) * ((2.0 * math.pi * equiv_a / vacuum_kr) * (jv(1, vacuum_kr * equiv_a) - 1j * yv(1, vacuum_kr * equiv_a)) - 4.0 * 1j/(vacuum_kr * vacuum_kr))
        else:
            # G_vector(ct1,1)=(1i/4.0)*(2.0*pi*equiv_a/kr(1))*besselj(1,kr(1)*equiv_a)*(besselj(0,kr(1)*R_mn2)-1i*bessely(0,kr(1)*R_mn2));
            G_vector[ct1] = (1j / 4.0) * (2.0 * math.pi * equiv_a / vacuum_kr) * jv(1, vacuum_kr * equiv_a) * (jv(0, vacuum_kr * R_mn2) - 1j * yv(0, vacuum_kr * R_mn2))

    # end = time.time()
    # print(end - start)
    return G_vector


def model_guess():
    # Placeholder for model guess infusion
    return None


def Ered_load(basis_counter, model_guess):
    # this is the initial guess
    # some sort of if statement will need to go here i think to integrate non-zero guess
    import numpy as np
    # Ered = zeros(basis_counter, 1);
    if model_guess is None:
        Ered = np.zeros((basis_counter, 1), dtype=np.complex_)
    else:
        Ered = model_guess.copy()
    return Ered


def complex_image_render(image_array, colour_map):
    from PIL import Image
    import numpy as np

    image_array_real = np.real(image_array)
    numpy_image_real = Image.fromarray((image_array_real * 255).astype(np.uint8))
    numpy_image_real.putpalette(colour_map)

    image_array_imag = np.imag(image_array)
    numpy_image_imag = Image.fromarray((image_array_imag * 255).astype(np.uint8))
    numpy_image_imag.putpalette(colour_map)

    image_array_abs = abs(image_array)
    numpy_image_abs = Image.fromarray((image_array_abs * 255).astype(np.uint8))
    numpy_image_abs.putpalette(colour_map)

    return numpy_image_real, numpy_image_imag, numpy_image_abs

# # JUST DON'T INCLUDE IT IN THE COMPOSER PART SO IT WON'T MISS IT WHEN RENDERING
#
# def BMT_FFT(G_vector.', D.*Ered, N):
#     return None


# @numba.jit
def BMT_FFT(X, V, N):
    # def BMT_FFT(resolution_information, G_vector):
    import numpy as np
    from scipy import fftpack

    # discretise_M = resolution_information["length_y_side"]
    # discretise_N = resolution_information["length_x_side"]

    # X = G_vector.transpose()
    # N = discretise_N
    # M = discretise_M

    M = int(max(X.shape) / N)
    X.resize(M, N)

    # This does not match back to MATLAB as the final as MATLAB is missing final row and column based on idea.
    X_fliplr = np.flip(X, 1)
    X = np.append(X, np.zeros((M, 1)), axis=1)
    X = np.append(X, X_fliplr, axis=1)
    X_flipud = np.flip(X, 0)
    X = np.append(X, np.zeros((1, 2*N + 1)), axis=0)
    X = np.append(X, X_flipud, axis=0)

    # TO MATCH BACK TO MATLAB THE FOLLOWING TWO OPERATIONS ARE CREATED
    # DELETE OUT THE LAST ROW
    # X = X[:-1, :]

    # DELETE OUT THE LAST COLUMN
    # X = X[:, :-1]

    # Both
    X = X[:-1, :-1]

    # This matches back to MATLAB
    # V = np.zeros((M, N), dtype=np.complex128)
    # V = field_incident_D * Ered
    V.resize(M, N)
    V = np.append(V, np.zeros((M, N)), axis=1)
    V = np.append(V, np.zeros((M, 2*N)), axis=0)
    full_solution = fftpack.ifft2(fftpack.fft2(X)*fftpack.fft2(V))

    output = np.zeros((1, M * N))
    # return np.transpose(np.ndarray.flatten(full_solution[0:M, 0:N]))
    output = np.transpose(np.asmatrix(np.ndarray.flatten(full_solution[0:M, 0:N])))
    # output = np.asmatrix(np.ndarray.flatten(full_solution[0:M, 0:N]))
    return output


def r(Ered_load, G_vector, Vred, field_incident_D, resolution_information, rfo):
    # Z*E - V (= error)
    import numpy as np
    try:
        from lib import scene_gen
    except ImportError:
        import scene_gen
    # r = Rfo .* Ered + Rfo .* BMT_FFT(G_vector.', D.*Ered, N) - Vred; % Z*E - V (= error)
    return np.multiply(rfo, Ered_load) + np.multiply(rfo, scene_gen.BMT_FFT(np.transpose(G_vector), field_incident_D * Ered_load, resolution_information["length_x_side"])) - Vred


def p(G_vector, field_incident_D, resolution_information, rfo, r):
    # -Z'*r
    import numpy as np
    try:
        from lib import scene_gen
    except ImportError:
        import scene_gen
    # p = -(Rfo .* r + conj(D) .* (BMT_FFT(conj(G_vector.'), Rfo.*r, N)));
    return -1*(np.multiply(rfo, r) + np.multiply(np.conjugate(field_incident_D), scene_gen.BMT_FFT(np.conjugate(np.transpose(G_vector)), np.multiply(rfo, r), resolution_information["length_x_side"])))


def solver_error(r):
    # error = abs(r'*r);
    out = abs((r.H) * r)
    return out[0, 0]


def krylov_solver(basis_counter, input_solver_tol, G_vector, field_incident_D, p, r, resolution_information, rfo, Ered_load):
    import numpy as np
    # import time
    try:
        from lib import scene_gen
    except ImportError:
        import scene_gen

    solver_error = scene_gen.solver_error(r)

    # discretise_M = resolution_information["length_y_side"]
    # discretise_N = resolution_information["length_x_side"]

    # iteration counter
    icnt = 0

    # Reduced_iteration_error = zeros(1, 1);
    reduced_iteration_error = np.array([icnt, solver_error], dtype=object)

    Ered = Ered_load.copy()

    print('\nStart reduced CG iteration with 2D FFT\n')
    # start = time.time()

    # while (solver_error > input_solver_tol) && (icnt <= basis_counter)
    while (solver_error > input_solver_tol) and (icnt <= basis_counter):
        icnt = icnt + 1
        if icnt % 50 == 0:
            print(icnt, "th iteration Red")
        # a = (norm(Rfo.*r+conj(D).*BMT_FFT(conj(G_vector.'), Rfo.*r, N)) / norm(Rfo.*p+Rfo.*(BMT_FFT(G_vector.', D.*p, N))))^2; %(norm(Z'*r)^2)/(norm(Z*p)^2);
        a = (np.linalg.norm(np.multiply(rfo, r) + np.multiply(np.conjugate(field_incident_D), scene_gen.BMT_FFT(np.conjugate(np.transpose(G_vector)), np.multiply(rfo, r), resolution_information["length_x_side"]))) / np.linalg.norm(np.multiply(rfo, p) + np.multiply(rfo, (scene_gen.BMT_FFT(np.transpose(G_vector), np.multiply(field_incident_D, p), resolution_information["length_x_side"])))))**2
        # Ered = Ered + a * p;
        Ered = Ered + np.multiply(a, p)
        # r_old = r;
        r_old = r.copy()
        # r = r + a * (Rfo .* p + Rfo .* (BMT_FFT(G_vector.', D.*p, N))); % r = r + a*z*p
        r = r + np.multiply(a, (np.multiply(rfo, p) + np.multiply(rfo, scene_gen.BMT_FFT(np.transpose(G_vector), np.multiply(field_incident_D, p), resolution_information["length_x_side"]))))
        # b = (norm(Rfo.*r+conj(D).*BMT_FFT(conj(G_vector.'), Rfo.*r, N)) / norm(Rfo.*r_old+conj(D).*BMT_FFT(conj(G_vector.'), Rfo.*r_old, N)))^2; %b = (norm(Z'*r)^2) /(norm(Z'*r_old)^2);
        b = (np.linalg.norm(np.multiply(rfo, r) + np.multiply(np.conjugate(field_incident_D), scene_gen.BMT_FFT(np.conjugate(np.transpose(G_vector)), np.multiply(rfo, r), resolution_information["length_x_side"])))/np.linalg.norm(np.multiply(rfo, r_old) + np.multiply(np.conjugate(field_incident_D), scene_gen.BMT_FFT(np.conjugate(np.transpose(G_vector)), np.multiply(rfo, r_old), resolution_information["length_x_side"]))))**2
        # p = -(Rfo .* r + conj(D) .* BMT_FFT(conj(G_vector.'), Rfo.*r, N)) + b * p; % p=-Z'*r+b*p
        p = -1*(np.multiply(rfo, r) + np.multiply(np.conjugate(field_incident_D), scene_gen.BMT_FFT(np.conjugate(np.transpose(G_vector)), np.multiply(rfo, r), resolution_information["length_x_side"]))) + np.multiply(b, p)
        # solver_error = abs(r'*r);
        solver_error = scene_gen.solver_error(r)
        # Reduced_iteration_error(icnt, 1) = abs(r'*r);
        reduced_iteration_error = np.vstack([reduced_iteration_error, [icnt, solver_error]])

    # end = time.time()
    # print("code runtime: ", end - start)
    return Ered, reduced_iteration_error


def Ered(krylov_solver):
    # print(type(krylov_solver))
    Ered = krylov_solver[0]
    return Ered


def reduced_iteration_error(krylov_solver):
    # print(type(krylov_solver))
    reduced_iteration_error = krylov_solver[1]
    return reduced_iteration_error


def ScatRed(Ered, Vred):
    return Ered - Vred


def restore_arrays(resolution_information, array_input):
    # Convert solutions to 2D grid for 3D plots
    import numpy as np
    discretise_M = resolution_information["length_y_side"]
    discretise_N = resolution_information["length_x_side"]
    # return array_input.resize(discretise_M, discretise_N)
    return np.reshape(array_input, (discretise_M, discretise_N))


def Ered_2D(resolution_information, Ered):
    try:
        from lib import scene_gen
    except ImportError:
        import scene_gen
    return scene_gen.restore_arrays(resolution_information, Ered)


def Vred_2D(resolution_information, Vred):
    try:
        from lib import scene_gen
    except ImportError:
        import scene_gen
    return scene_gen.restore_arrays(resolution_information, Vred)


def ScatRed_2D(resolution_information, ScatRed):
    try:
        from lib import scene_gen
    except ImportError:
        import scene_gen
    return scene_gen.restore_arrays(resolution_information, ScatRed)
