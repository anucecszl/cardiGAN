import csv
import numpy as np
import torch
import parameter_calculator as calculator
import cardiGAN
import hyper_parameters as parameters

GEN_PATH = 'saved_models/sample_generator_net.pt'
RESULT_PATH = 'data/generated_result.csv'

num_samples = 10000  # The number of generated samples.

# Load the trained generator model
generator = cardiGAN.Generator()
generator.load_state_dict(torch.load(GEN_PATH))
generator.eval()

# Set the csv writer.
out = open(RESULT_PATH, 'w', newline='')
csv_write = csv.writer(out, dialect='excel')

print("start generating novel CCA compositions...")
# Create a Gaussian noise and use the trained generator to generate CCA candidates.
g_noise = torch.tensor(np.random.randn(num_samples, parameters.num_latent)).float()
cca_candidates = generator(g_noise) + 1e-6
while torch.isnan(torch.sum(cca_candidates)):
    cca_candidates = generator(g_noise)

# Write the headers for the generated dataset.
csv_write.writerow(['Element', 'Molar_ratio'] + parameters.empirical_params)
# Normalize the compositions and calculate the empirical parameters of the generated candidates.
normalized_compositions = cca_candidates / torch.sum(cca_candidates, dim=1).reshape((cca_candidates.shape[0], -1))
calc_result = calculator.calculate_all_parameters(normalized_compositions).detach().numpy()
normalized_compositions = normalized_compositions.detach().numpy()

# Write the element compositions and empirical parameters of the generated candidates.
for i in range(normalized_compositions.shape[0]):
    element_strings = ''
    mol_ratio_strings = ''
    for m in range(parameters.num_elements):
        if normalized_compositions[i][m] > 0.01:
            element_strings += (parameters.element_list[m] + '-')
            mol_ratio_strings += ("{0:.2f}".format(normalized_compositions[i][m]) + '-')
    csv_write.writerow([element_strings, mol_ratio_strings] + list(calc_result[i]))
print("finish generation of " + str(num_samples) + " novel CCA samples.")
