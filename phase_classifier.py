import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import parameters

data_set = np.loadtxt('data/train_parameter.csv', delimiter=',')

np.random.shuffle(data_set)

features = data_set[:, parameters.ANN_param_selection]
feature_mean = np.mean(features, axis=0)
feature_std = np.std(features, axis=0)
features = (features - feature_mean)/feature_std

targets = data_set[:, [12]]
for i in range(len(targets)):
    if targets[i][0] == 5 or targets[i][0] == 0 or targets[i][0] == 2:
        targets[i][0] = 0
    else:
        targets[i][0] = 1

train_features = features[:int(data_set.shape[0] * 0.7)]
train_targets = targets[:int(data_set.shape[0] * 0.7)].reshape(train_features.shape[0])
test_features = features[int(data_set.shape[0] * 0.7):]
test_targets = targets[int(data_set.shape[0] * 0.7):].reshape(test_features.shape[0])


class ANNTrainSet(Dataset):
    def __init__(self):
        self.features = torch.from_numpy(train_features).float()
        self.len = self.features.shape[0]
        self.target = torch.from_numpy(train_targets).long()

    def __getitem__(self, index):
        return self.features[index], self.target[index]

    def __len__(self):
        return self.len


class Classifier(nn.Module):
    """
    Build the classifier network
    """

    def __init__(self):
        super(Classifier, self).__init__()

        # Set the model to have two latent layers both with LeakyReLU activation
        self.model = nn.Sequential(
            nn.Linear(12, 12),
            nn.LeakyReLU(),
            nn.Linear(12, 12),
            nn.LeakyReLU(),
            nn.Linear(12, len(parameters.phase_mapping_2)),
        )

    def forward(self, x):
        predicted_phase = self.model(x)
        return predicted_phase


classifier = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=1e-4)

training_set = ANNTrainSet()
loader = DataLoader(dataset=training_set, batch_size=1, shuffle=True, )

train_rate_list = []
test_rate_list = []
train_loss_list = []
test_loss_list = []

lowest_test_loss = 10000

if __name__ == "__main__":
    for epoch in range(parameters.num_epoch_ANN):  # loop over the dataset multiple times

        for i, data in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            optimizer.zero_grad()
            outputs = classifier(inputs)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()

        train_prediction = classifier(torch.from_numpy(train_features).float())
        train_loss = criterion(train_prediction, torch.from_numpy(train_targets).long()).item()
        train_predicted_phases = np.argmax(train_prediction.detach().numpy(), axis=1)

        test_prediction = classifier(torch.from_numpy(test_features).float())
        test_loss = criterion(test_prediction, torch.from_numpy(test_targets).long()).item()
        test_predicted_phases = np.argmax(test_prediction.detach().numpy(), axis=1)

        if test_loss < lowest_test_loss:
            lowest_test_loss = test_loss

        train_pred_rate = 1 - np.count_nonzero(train_predicted_phases - train_targets) / train_features.shape[0]
        test_pred_rate = 1 - np.count_nonzero(test_predicted_phases - test_targets) / test_features.shape[0]

        print('epoch:', epoch)
        print('train accuracy:', train_pred_rate)
        train_rate_list.append(train_pred_rate)
        print('test  accuracy:', test_pred_rate)
        test_rate_list.append(test_pred_rate)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        if test_loss > lowest_test_loss * 1.05:
            torch.save(classifier.state_dict(), 'saved_models/classifier_net.pt')
            break

    all_prediction = classifier(torch.from_numpy(features).float())
    all_loss = criterion(all_prediction, torch.from_numpy(targets.reshape(features.shape[0])).long()).item()
    all_predicted_phases = np.argmax(all_prediction.detach().numpy(), axis=1).reshape(features.shape[0])
    all_pred_rate = 1 - np.count_nonzero(all_predicted_phases - targets.reshape(features.shape[0])) / features.shape[0]
    print('all accuracy', all_pred_rate)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.plot(range(len(train_rate_list)), train_rate_list, c='r', label='train accuracy')
    plt.plot(range(len(test_rate_list)), test_rate_list, c='b', label='test accuracy')
    plt.legend()
    plt.show()

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(len(train_rate_list)), train_loss_list, c='r', label='train loss')
    plt.plot(range(len(test_rate_list)), test_loss_list, c='b', label='test loss')
    plt.legend()
    plt.show()

    print('Finished Training')
