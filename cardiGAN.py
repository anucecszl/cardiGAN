import numpy as np
import torch
import torch.nn as nn
import phase_classifier
import parameters
import pytorch_calculator as calculator
from torch.utils.data import DataLoader, Dataset

# The path to save the trained generator model.
GEN_PATH = 'saved_models/generator_net.pt'
# The element compositions of the 278 existing CCAs.
cca_compositions = np.genfromtxt('data/train_composition.csv', delimiter=',')
# The empirical parameters of the 278 existing CCAs.
cca_parameters = np.loadtxt('data/train_parameter.csv', delimiter=',')
param_mean = cca_parameters.mean(axis=0)
param_std = cca_parameters.std(axis=0)
# Load the trained phase classifier model.
classifier_path = 'saved_models/classifier_net.pt'
classifier = phase_classifier.Classifier()
classifier.load_state_dict(torch.load(classifier_path))
classifier.eval()


# —————————————————————————————————— Customize the training set ————————————————————————————————————————
class TrainingSet(Dataset):
    """
    A customized Dataset used to train the cardiGAN model. It includes the element compositions and empirical
    parameters of the 278 exisitng CCAs.
    """

    def __init__(self):
        # Load the element compositions of the 278 existing CCAs.
        compositions = np.loadtxt('data/train_composition.csv', delimiter=',')
        compositions = np.concatenate(
            (compositions, parameters.sum_one * np.ones((compositions.shape[0], 1)) - parameters.sum_one), axis=1)

        # Load the empirical parameters and normalize it into a Gaussian distribution.
        cca_params = np.loadtxt('data/train_parameter.csv', delimiter=',')
        cca_params = (cca_params - param_mean) / param_std

        # Build the training set by concatenating the composition and parameter datasets.
        self.data = torch.from_numpy(
            np.concatenate((compositions, cca_params[:, parameters.GAN_param_selection]), axis=1)).float()
        self.len = compositions.shape[0]  # The length of the training set.

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


# ————————————————————————————————— Define the neural networks ————————————————————————————————————

class Generator(nn.Module):
    """
    The generator neural network of the cardiGAN model. This network is trained to learn the mapping between the
    latent space (a Gaussian distribution) to the distribution of existing CCAs. This network has one hidden layer.
    The output layer is activated by ReLU to produce non-negative sparse vectors (the element compositions of novel
    CCA candidates).
    """

    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(parameters.num_latent, parameters.num_elements),
            nn.LeakyReLU(),
            nn.Linear(parameters.num_elements, parameters.num_elements),
            nn.ReLU()
        )

    def forward(self, noise_input):
        cca_candidates = self.model(noise_input)
        return cca_candidates


class Discriminator(nn.Module):
    """
    The discriminator neural network (the critic) of the cardiGAN model. This network is trained to fit a k-Lipschitz
    function that can be used to measure the Wasserstein distance between the generated and training distributions.
    This network has two hidden layers. The output of this network is a scalar value.
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(parameters.num_elements + parameters.num_params + 1,
                      parameters.num_elements + parameters.num_params + 1),
            nn.LeakyReLU(),
            nn.Linear(parameters.num_elements + parameters.num_params + 1,
                      parameters.num_elements + parameters.num_params + 1),
            nn.LeakyReLU(),
            nn.Linear(parameters.num_elements + parameters.num_params + 1, 1),
        )

    def forward(self, x):
        y = self.model(x)
        return y


# ————————————————————————————————— Set up the neural networks ————————————————————————————————————

generator = Generator()
discriminator = Discriminator()

# As recommended in 'Wasserstein GAN' (https://arxiv.org/abs/1701.07875), both networks apply RMSprop optimization.
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=parameters.lr_generator, )
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=parameters.lr_discriminator, )


# ————————————————————————————————— Set up train functions ————————————————————————————————————————


def generate_novel_input(size):
    """
    This function applies the generator and phase classifier to generate novel CCA candidates and calculate their
    empirical parameters.
    :param size: The number of generated candidates.
    :return: The element compositions and empirical parameters of the generated candidates.
    """
    # Use a Gaussian distributed noise to generate novel CCA compositions.
    noise = torch.tensor(np.random.randn(size, parameters.num_latent)).float()
    novel_alloy = generator(noise) + 1e-9
    novel_alloy_norm = novel_alloy / (torch.sum(novel_alloy, axis=1).view(noise.shape[0], -1) + 1e-6)
    # Use the parameter calculator to calculate the empirical parameters of the novel alloys.
    novel_param = (calculator.calculate_parameters(novel_alloy_norm) - torch.tensor(
        param_mean[parameters.GAN_param_selection]).float()) / torch.tensor(
        param_std[parameters.GAN_param_selection]).float()
    phase_param = (calculator.calculate_phase_parameters(novel_alloy_norm) - torch.tensor(
        param_mean[parameters.ANN_param_selection]).float()) / torch.tensor(
        param_std[parameters.ANN_param_selection]).float()
    # Concatenate the generated CCA candidates and their calculated empirical parameters as inputs of the discriminator.
    novel_alloy = torch.cat(
        [novel_alloy_norm, parameters.sum_one * torch.sum(novel_alloy, axis=1).view((-1, 1)) - parameters.sum_one], dim=1)
    novel_input = torch.cat([novel_alloy, novel_param], dim=1)

    return novel_input, phase_param


def train_generator(real_input):
    """
    This function trains the generator network.
    :param real_input: The element compositions and empirical parameters of existing CCAs.
    """
    size = real_input.shape[0]
    optimizer_G.zero_grad()
    # Use the generator to generate CCA candidates and calculate their empirical parameters.
    cca_candidates, candidate_param = generate_novel_input(size)
    # Use the phase classifier to predict the phases of the generated cca candidates and produce a (phase) loss.
    phase_matrix = torch.softmax(classifier(candidate_param), dim=1).view(cca_candidates.shape[0], -1)
    phase_matrix = torch.mul(phase_matrix, phase_matrix)
    phase_loss = torch.mean(phase_matrix)
    # The loss of the generator comprises the Wasserstein distance and the "phase loss".
    g_loss = -torch.mean(discriminator(cca_candidates))
    g_loss = g_loss - abs(g_loss.item()) * phase_loss
    g_loss.backward()

    optimizer_G.step()


def train_discriminator(real_input):
    """
    This function trains the discriminator network.
    :param real_input: The element compositions and empirical parameters of existing CCAs.
    :return: The value of discriminator loss which indicates the Wasserstein distance.
    """
    size = real_input.shape[0]
    optimizer_D.zero_grad()
    # Use the generator to generate CCA candidates and calculate their empirical parameters.
    cca_candidates, candidate_param = generate_novel_input(size)
    # The loss of the discriminator is the Wasserstein distance between the two distributions.
    d_loss = -torch.mean(discriminator(real_input)) + torch.mean(discriminator(cca_candidates.detach()))
    d_loss.backward()

    optimizer_D.step()
    return d_loss.item()


if __name__ == "__main__":
    # ————————————————————————————————— Load the training set ————————————————————————————————————————

    training_set = TrainingSet()
    loader = DataLoader(dataset=training_set, batch_size=parameters.size_batch, shuffle=True, )

    # ————————————————————————————————— Start GAN training ————————————————————————————————————————————

    for epoch in range(parameters.num_epoch):
        sum_d_loss = 0  # The sum of discriminator losses of the current epoch.

        for i, real_cca in enumerate(loader):
            # train the generator network.
            train_generator(real_cca)
            # set up clip value for discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-parameters.clip_range, parameters.clip_range)

            for j in range(5):
                # train the discriminator and accumulate real input and fake input's loss
                sum_d_loss += -train_discriminator(real_cca)

        print('Epoch:', epoch, "discriminator loss:", sum_d_loss)
        # Save the model for every 100 epochs.
        if epoch % 100 == 99:
            torch.save(generator.state_dict(), GEN_PATH)

    torch.save(generator.state_dict(), GEN_PATH)
