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


def input_palette():
    from PIL import ImageColor
    # CHECK, this needs to come from the JSON LUT and get iterated over.
    tuples = [
        ImageColor.getcolor("#FDFF00", "RGB"),
        ImageColor.getcolor("#A7A9AB", "RGB"),
        ImageColor.getcolor("#D08931", "RGB"),
        ImageColor.getcolor("#B9E8E9", "RGB"),
        ImageColor.getcolor("#ED774C", "RGB"),
        ImageColor.getcolor("#EFEFEE", "RGB"),
        ImageColor.getcolor("#F4F1DB", "RGB"),
        ImageColor.getcolor("#C09B53", "RGB"),
        ImageColor.getcolor("#7A5409", "RGB"),
        ImageColor.getcolor("#909090", "RGB")
        ]
    palette = [item for sublist in tuples for item in sublist]
    return palette


def image_object(path_geo, object_name):
    # Generated geometry will be in the form of PNG as it is smaller in memory than CSV or TXT
    from PIL import Image
    return Image.open(path_geo + object_name, mode='r').convert('L')


def image_render(image_object, input_palette):
    # print(image_object.mode)
    # Gradio may be converting input here before it reaches the operations so bear that in mind.
    image_object.putpalette(input_palette)
    return image_object


# def image_image(image_object):
#     from PIL import Image
#     # return Image.open("./geometry/placeholder.png", mode='r')
#     return Image.fromarray(image_object, 'RGB')


def unique_integers(image_object):
    # "Unique Values in Geometry"
    import pandas as pd
    import numpy as np

    return pd.DataFrame({"uint8": np.unique(image_object).tolist()})


def unique_integers_freq(image_object):

    prepopulation requires freq count of image contents. so replace unique_integers to include a count too
    # "Unique Values in Geometry"
    import pandas as pd
    import numpy as np
    pd.value_counts(image_object)

    return pd.DataFrame({"uint8": np.unique(image_object).tolist()})


def materials_dict(path_lut):
    import pandas as pd
    import json
    return pd.DataFrame(json.load(open(path_lut, 'r')))


def image_geometry_materials_parse(materials_dict, unique_integers):
    # "Unique Values in Geometry"
    materials_dict['uint8'] = materials_dict['uint8'].astype(int)
    return unique_integers.join(materials_dict.set_index(['uint8']), on=['uint8'], how='left')


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
    longest = str(longest_side[longest_side["length"] == longest_side["length"].min()]['name'].iloc[0])
    shortest = str(longest_side[longest_side["length"] == longest_side["length"].max()]['name'].iloc[0])
    data[longest] = discretise_side_1
    data[shortest] = discretise_side_2
    return data


def image_resize(image_object, resolution_information):
    from PIL import Image
    return image_object.resize((resolution_information["length_y_side"], resolution_information["length_x_side"]), Image.Resampling.NEAREST)


def image_resize_render(image_resize, input_palette):
    # print(image_object.mode)
    # Gradio may be converting input here before it reaches the operations so bear that in mind.
    image_resize.putpalette(input_palette)
    return image_resize


def input_centre():
    return 0.0 + 0.0 * 1j


def start_point(input_centre, input_disc_per_lambda, length_x_side, length_y_side):
    # CHECK, is this correct? like there will have to be multiple resize options so why stick with original lengths etc.
    return input_centre - 0.5 * length_x_side - 0.5 * length_y_side * 1j


def basis_specification(resolution_information, image_geometry_materials_full):
    import numpy as np
    basis_counter = 0
    # basis_wave_number(1:N*N, 1) = kr(1)
    print(resolution_information)
    # print(resolution_information["length_y_side"])
    # basis_wave_number = np.full((resolution_information["length_y_side"]*resolution_information["length_y_side"], 1), 10)
    print(image_geometry_materials_full["kr"])
    PRE-POPULATING THE MATRIX WITH THE DOMININANT MATERIAL MIGHT SAVE OPERATIONS?
    image_geometry_materials_full[image_geometry_materials_full["length"] == longest_side["length"].min()]['name'].iloc[0]

    # print(basis_wave_number)
    return basis_counter
