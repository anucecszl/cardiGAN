import numpy as np
import torch
import cardiGAN
import hyper_parameters as parameters
import matminer.utils.data as mm_data
from pymatgen.core.periodic_table import Element

print('start loading parameter calculator...')
# Load the trained phase classifier model.
classifier_path = 'saved_models/classifier_net.pt'
classifier = cardiGAN.Classifier()
classifier.load_state_dict(torch.load(classifier_path))
classifier.eval()
# Load the parameters of the existing CCAs and calculate the mean and std.
data_set = np.loadtxt('data/train_parameter.csv', delimiter=',')
feature = data_set[:, parameters.ANN_param_selection]
feature_mean = torch.from_numpy(np.mean(feature, axis=0)).float()
feature_std = torch.from_numpy(np.std(feature, axis=0)).float()


def calculate_Tm(fake_alloy):
    """
    This function calculates the average melting temperature of the given alloy.
    :param fake_alloy: The element composition of the given alloy.
    :return: The average melting temperature in Kelvin.
    """
    return torch.matmul(fake_alloy, torch.Tensor(parameters.Tm_list))


def calculate_std_Tm(fake_alloy):
    """
    This function calculates the standard deviation of melting temperature of the given alloy.
    :param fake_alloy: The element composition of the given alloy.
    :return: The standard deviation of melting temperature in percentage.
    """
    Tm_list = torch.tensor(parameters.Tm_list, requires_grad=True).view(parameters.num_elements, -1)
    average_Tm = torch.matmul(fake_alloy, Tm_list)
    Tm = torch.ones((fake_alloy.shape[0], parameters.num_elements))
    for k in range(fake_alloy.shape[0]):
        Tm[k] = Tm_list.view(parameters.num_elements)
    parenthesis = 1 - Tm / (average_Tm + 1e-6)
    parenthesis = parenthesis * parenthesis
    parenthesis = torch.matmul(fake_alloy, torch.transpose(parenthesis, 0, 1))
    parenthesis = torch.diag(parenthesis)
    return 100 * torch.sqrt(parenthesis)


def calculate_std_VEC(fake_alloy):
    """
    This function calculates the standard deviation of valence electron concentration of the given alloy.
    :param fake_alloy: The element composition of the given alloy.
    :return: The standard deviation of valence electron concentration.
    """
    vec_list = torch.tensor(parameters.VEC_list, requires_grad=True).view(parameters.num_elements, -1)
    average_vec = torch.matmul(fake_alloy, vec_list)
    vec = torch.ones((fake_alloy.shape[0], parameters.num_elements))
    for k in range(fake_alloy.shape[0]):
        vec[k] = vec_list.view(parameters.num_elements)
    parenthesis = vec - average_vec
    parenthesis = parenthesis * parenthesis
    parenthesis = torch.matmul(fake_alloy, torch.transpose(parenthesis, 0, 1))
    parenthesis = torch.diag(parenthesis)
    return torch.sqrt(parenthesis)


def calculate_VEC(fake_alloy):
    """
    This function calculates the valence electron concentration of the given alloy.
    :param fake_alloy: The element composition of the given alloy.
    :return: The valence electron concentration of the alloy.
    """
    return torch.matmul(fake_alloy, torch.Tensor(parameters.VEC_list))


def calculate_a(fake_alloy):
    """
    This function calculates the average atomic radius of the given alloy.
    :param fake_alloy: The element composition of the given alloy.
    :return: The average atomic radius of the alloy in Angstrom.
    """
    return torch.matmul(fake_alloy, torch.Tensor(parameters.radii_list))


def calculate_delta(fake_alloy):
    """
    This function calculates the atomic size difference (Delta) of the given alloy.
    :param fake_alloy: The element composition of the given alloy.
    :return: The atomic size difference of the alloy in percentage.
    """
    radii_list = torch.tensor(parameters.radii_list, requires_grad=True).view(parameters.num_elements, -1)
    average_radius = torch.matmul(fake_alloy, radii_list)
    radii = torch.ones((fake_alloy.shape[0], parameters.num_elements))
    for k in range(fake_alloy.shape[0]):
        radii[k] = radii_list.view(parameters.num_elements)
    parenthesis = 1 - radii / (average_radius + 1e-6)
    parenthesis = parenthesis * parenthesis
    parenthesis = torch.matmul(fake_alloy, torch.transpose(parenthesis, 0, 1))
    parenthesis = torch.diag(parenthesis)
    return 100 * torch.sqrt(parenthesis)


def calculate_entropy(fake_alloy):
    """
    This function calculates the entropy of mixing of the given alloy.
    :param fake_alloy: The element composition of the given alloy.
    :return: The entropy of mixing of the given alloy in J/(K*mol).
    """
    fake_alloy_modified = fake_alloy + 1e-10 * torch.ones(fake_alloy.shape)
    return -8.31446261815324 * torch.diag(
        torch.matmul(fake_alloy, torch.transpose(torch.log(fake_alloy_modified), 0, 1)))


def calculate_ohm():
    """
    This function calculates the parameter ohm which is used in the calculation of enthalpy of mixing.
    :return: A matrix containing the binary enthalpy of mixing between the 56 selected elements.
    """
    ohm_matrix = torch.zeros(parameters.num_elements, parameters.num_elements)
    for i in range(parameters.num_elements):
        for j in range(parameters.num_elements):
            if i != j:
                ohm_matrix[i][j] = mm_data.MixingEnthalpy().get_mixing_enthalpy(Element(parameters.element_list[i]),
                                                                                Element(parameters.element_list[j]))
    return 4 * ohm_matrix


ohm = calculate_ohm()


def calculate_enthalpy(fake_alloy, ohm_matrix):
    """
    This function calculates the enthalpy of mixing of the given alloy.
    :param fake_alloy: The element composition of the given alloy.
    :param ohm_matrix: A matrix containing the binary enthalpy of mixing between the 56 selected elements.
    :return: The enthalpy of mixing of the given alloy.
    """
    enthalpy = torch.zeros(fake_alloy.shape[0])
    for i in range(fake_alloy.shape[0]):
        alloy = fake_alloy[i].view(parameters.num_elements, -1)
        alloy_matrix = torch.matmul(alloy, torch.transpose(alloy, 0, 1))
        result = torch.sum(alloy_matrix * ohm_matrix) / 2
        enthalpy[i] = result
    return enthalpy


def calculate_std_enthalpy(fake_alloy, H, ohm_matrix):
    """
    This function calculates the standard deviation of enthalpy of the given alloy.
    :param H: The enthalpy of mixing of the given alloy.
    :param fake_alloy: The element composition of the given alloy.
    :param ohm_matrix: A matrix containing the binary enthalpy of mixing between the 56 selected elements.
    :return: The standard deviation of enthalpy of the given alloy.
    """
    h_matrix = ohm_matrix / 4
    std_enthalpy = torch.ones(fake_alloy.shape[0])
    for i in range(fake_alloy.shape[0]):
        alloy = fake_alloy[i].view(parameters.num_elements, -1)
        alloy_matrix = torch.matmul(alloy, torch.transpose(alloy, 0, 1))
        parenthesis = h_matrix - H[i] + H[i] * torch.eye(parameters.num_elements)
        parenthesis = parenthesis * parenthesis
        result = alloy_matrix * parenthesis
        result = torch.sum(result) / 2
        std_enthalpy[i] = torch.sqrt(result)

    return std_enthalpy


def calculate_omega(Tm, S, H):
    """
    This function calculates the Omega value of the given alloy.
    :param Tm: The average melting temperature of the alloy.
    :param S: The entropy of mixing of the alloy.
    :param H: The enthalpy of mixing of the alloy.
    :return: The Omega value of the given alloy.
    """
    return torch.abs(1e-3 * Tm * S / (H + 1e-6))


def calculate_x(fake_alloy):
    """
    This function calculates the electronegativity of the given alloy.
    :param fake_alloy: The element composition of the given alloy.
    :return: The electronegativity of the given alloy.
    """
    return torch.matmul(fake_alloy, torch.Tensor(parameters.X_list))


def calculate_std_x(fake_alloy):
    """
    This function calculates the standard deviation of electronegativity of the given alloy.
    :param fake_alloy: The element composition of the given alloy.
    :return: The standard deviation of electronegativity of the given alloy.
    """
    x_list = torch.tensor(parameters.X_list, requires_grad=True).view(parameters.num_elements, -1)
    average_x = torch.matmul(fake_alloy, x_list)
    x = torch.ones((fake_alloy.shape[0], parameters.num_elements))
    for k in range(fake_alloy.shape[0]):
        x[k] = x_list.view(parameters.num_elements)
    parenthesis = 1 - x / (average_x + 1e-6)
    parenthesis = parenthesis * parenthesis
    parenthesis = torch.matmul(fake_alloy, torch.transpose(parenthesis, 0, 1))
    parenthesis = torch.diag(parenthesis)
    return 100 * torch.sqrt(parenthesis)


def calculate_density(fake_alloy):
    """
    This function calculates the density of the given alloy.
    :param fake_alloy: The element composition of the given alloy.
    :return: The density of the given alloy.
    """
    mass = torch.matmul(fake_alloy, torch.Tensor(parameters.mass_list))
    volume = torch.matmul(fake_alloy, torch.Tensor(parameters.volume_list))
    return mass / (volume + 1e-6)


def calculate_price(fake_alloy):
    """
    This function calculates the estimated element cost of the given alloy.
    :param fake_alloy: The element composition of the given alloy.
    :return: The estimated element cost of the given alloy.
    """
    mass = torch.matmul(fake_alloy, torch.Tensor(parameters.mass_list))
    price = torch.matmul(fake_alloy, torch.Tensor(parameters.mass_list) * torch.Tensor(parameters.price_list))
    return price / (mass + 1e-6)


def predict_phase(features):
    """
    This function applies the pre-trained phase classifier to predict the phases of the given alloy.
    :param features: The calculated empirical parameters of alloy.
    :return: A tensor containing the predicted phases of the alloy.
    """
    norm_feature = (features - feature_mean) / (feature_std + 1e-6)
    return classifier(norm_feature)


def calculate_parameters(fake_alloy):
    """
    This function applies the above functions and calculates the 11 empirical parameters for GAN training.
    :param fake_alloy: The element composition of the given alloy.
    :return: The 11 empirical parameters of the alloy.
    """
    enthalpy = calculate_enthalpy(fake_alloy, ohm).view(fake_alloy.shape[0], -1)
    std_enthalpy = calculate_std_enthalpy(fake_alloy, enthalpy, ohm).view(fake_alloy.shape[0], -1)
    a = calculate_a(fake_alloy).view(fake_alloy.shape[0], -1)
    delta = calculate_delta(fake_alloy).view(fake_alloy.shape[0], -1)
    entropy = calculate_entropy(fake_alloy).view(fake_alloy.shape[0], -1)
    Tm = calculate_Tm(fake_alloy).view(fake_alloy.shape[0], -1)
    std_Tm = calculate_std_Tm(fake_alloy).view(fake_alloy.shape[0], -1)
    x = calculate_x(fake_alloy).view(fake_alloy.shape[0], -1)
    std_x = calculate_std_x(fake_alloy).view(fake_alloy.shape[0], -1)
    vec = calculate_VEC(fake_alloy).view(fake_alloy.shape[0], -1)
    std_vec = calculate_std_VEC(fake_alloy).view(fake_alloy.shape[0], -1)
    params = torch.cat(
        (enthalpy, std_Tm, delta, Tm, std_enthalpy, a, std_vec, entropy, std_x, vec, x,),
        dim=1)
    return params


def calculate_phase_parameters(fake_alloy):
    """
    This function applies the above functions and calculates the 12 empirical parameters of the given alloy.
    :param fake_alloy: The element composition of the given alloy.
    :return: The 12 empirical parameters of the alloy.
    """
    enthalpy = calculate_enthalpy(fake_alloy, ohm).view(fake_alloy.shape[0], -1)
    std_enthalpy = calculate_std_enthalpy(fake_alloy, enthalpy, ohm).view(fake_alloy.shape[0], -1)
    a = calculate_a(fake_alloy).view(fake_alloy.shape[0], -1)
    delta = calculate_delta(fake_alloy).view(fake_alloy.shape[0], -1)
    entropy = calculate_entropy(fake_alloy).view(fake_alloy.shape[0], -1)
    Tm = calculate_Tm(fake_alloy).view(fake_alloy.shape[0], -1)
    std_Tm = calculate_std_Tm(fake_alloy).view(fake_alloy.shape[0], -1)
    omega = calculate_omega(Tm, entropy, enthalpy).view(fake_alloy.shape[0], -1)
    x = calculate_x(fake_alloy).view(fake_alloy.shape[0], -1)
    std_x = calculate_std_x(fake_alloy).view(fake_alloy.shape[0], -1)
    vec = calculate_VEC(fake_alloy).view(fake_alloy.shape[0], -1)
    std_vec = calculate_std_VEC(fake_alloy).view(fake_alloy.shape[0], -1)
    params = torch.cat(
        (enthalpy, std_Tm, delta, Tm, std_enthalpy, a, std_vec, entropy, std_x, omega, vec, x,),
        dim=1)
    return params


def calculate_all_parameters(fake_alloy):
    """
    This function applies all the above functions and calculates the 15 parameters of the given alloy (the 12 empirical
    parameter + phase, density, and estimated element cost).
    :param fake_alloy: The element composition of the given alloy.
    :return: The 15 parameters of the alloy.
    """
    enthalpy = calculate_enthalpy(fake_alloy, ohm).view(fake_alloy.shape[0], -1)
    std_enthalpy = calculate_std_enthalpy(fake_alloy, enthalpy, ohm).view(fake_alloy.shape[0], -1)
    a = calculate_a(fake_alloy).view(fake_alloy.shape[0], -1)
    delta = calculate_delta(fake_alloy).view(fake_alloy.shape[0], -1)
    entropy = calculate_entropy(fake_alloy).view(fake_alloy.shape[0], -1)
    Tm = calculate_Tm(fake_alloy).view(fake_alloy.shape[0], -1)
    std_Tm = calculate_std_Tm(fake_alloy).view(fake_alloy.shape[0], -1)
    omega = calculate_omega(Tm, entropy, enthalpy).view(fake_alloy.shape[0], -1)
    x = calculate_x(fake_alloy).view(fake_alloy.shape[0], -1)
    std_x = calculate_std_x(fake_alloy).view(fake_alloy.shape[0], -1)
    vec = calculate_VEC(fake_alloy).view(fake_alloy.shape[0], -1)
    std_vec = calculate_std_VEC(fake_alloy).view(fake_alloy.shape[0], -1)
    density = calculate_density(fake_alloy).view(fake_alloy.shape[0], -1)
    price = calculate_price(fake_alloy).view(fake_alloy.shape[0], -1)
    features = torch.cat(
        (enthalpy, std_Tm, delta, Tm, std_enthalpy, a, std_vec, entropy, std_x, omega, vec, x,),
        dim=1)
    phase_matrix = torch.softmax(predict_phase(features), dim=1).view(fake_alloy.shape[0], -1)
    phase = phase_matrix[:, [0]].view(fake_alloy.shape[0], -1).float()
    params = torch.cat(
        (enthalpy, std_enthalpy, a, delta, omega, entropy, Tm, std_Tm, x, std_x, vec, std_vec, phase, density, price),
        dim=1)
    return params


print('finish loading parameter calculator.')
