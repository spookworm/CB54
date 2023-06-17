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


# def realmax():
#     # Equivalent of realmax in MATLAB. Used to find the minimum resolution required by the materials present in the scene.
#     import sys
#     return sys.float_info.max


def input_carrier_frequency():
    return 60e6


def input_disc_per_lambda():
    return 10


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
    # print("longest", longest)
    # print("shortest", shortest)
    # print("data[longest]", data[longest])
    # print("data[shortest]", data[shortest])
    return data


def image_resize(image_object, resolution_information):
    from PIL import Image
    return image_object.resize((resolution_information["length_x_side"], resolution_information["length_y_side"]), Image.Resampling.NEAREST)


def image_resize_render(image_resize, palette):
    # print(image_object.mode)
    # Gradio may be converting input here before it reaches the operations so bear that in mind.
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
    for ct1 in range(0, basis_counter):
        # D(ct1,1)= (basis_wave_number(ct1,1)*basis_wave_number(ct1,1) - k0*k0); % contrast function
        D[ct1] = basis_wave_number[ct1] * basis_wave_number[ct1] - vacuum_kr * vacuum_kr
    return D


def rfo(field_incident_D):
    # Reduced Foreward Operator
    # Rfo = logical(D);
    field_incident_D[field_incident_D != 0] = 1
    rfo = field_incident_D
    # unique_elements, counts_elements = np.unique(rfo, return_counts=True)
    # print(unique_elements)
    # print(counts_elements)
    return rfo


def Vred(rfo, field_incident_V):
    # Vred = Rfo .* V; % create V reduced
    Vred = rfo * field_incident_V
    return Vred


def Vred_2D(resolution_information, Vred):
    import numpy as np
    discretise_M = resolution_information["length_y_side"]
    discretise_N = resolution_information["length_x_side"]
    Vred_2D = np.reshape(Vred, (discretise_M, discretise_N))
    return Vred_2D


def G_vector(basis_counter, position, equiv_a, image_geometry_materials_full, vacuum_kr):
    # BMT dense matrix, stored as a vector
    import numpy as np
    import math
    from scipy.special import jv, yv
    import time

    G_vector = np.zeros((basis_counter, 1), dtype=np.complex_)

    start = time.time()

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

    end = time.time()
    print(end - start)
    # MATLAB: 0.059997600000000
    return G_vector

