# cardiGAN
Compositionally-complex Alloy Research Directive Inference GAN, a hybrid empirical model and machine learning approach for exploring the Compositionally Complex Alloy (CCA) solution space.

## Description
This project developed a generative adversarial model named cardiGAN to enhance the capacity and efficiency of development of novel CCAs.  

- The cardiGAN model has 4 components:
1. Generator network:  
The generator network is used to generate novel CCA compositions. It is trained to learn the mapping of a multi-dimensional Gaussian distribution to the distribution of the element compositions of existing CCAs.
2. Discriminator network:  
The discriminator network (serving the function of “critic”) can be used to estimate the Wasserstein distance between the distributions of the generated and training data. 
3. Phase classifier network (pre-trained):  
The phase classifier network could classify CCAs into three classes: single solid-solution, mixed solid-solution, and solid-solution with secondary phases. It is used to regularize the training of the generator network. 
4. Empirical parameter calculator:  
The empirical parameter calculator could calculate 12 empirical parameters of CCAs based on their element compositions. The calculator in this project is hard coded in PyTorch. A parameter calculator with user interface can be accessed through https://github.com/ZhipengGaGa/Parameter-Calculator-for-CCA.

- The configuration of the cardiGAN model: [image](http://github.com/ZhipengGaGa/cardiGAN/raw/master/model_configuration.png)

- The model is trained on the element compositions of 278 existing CCAs. The compositions are represented as 56-dimensional vectors with each dimension indicating the molar ratio of a specific element. 

## Installation
1. git clone this repository.
2. Download and install python 3.7 or any version released after python 3.7 (https://www.python.org/downloads/)
3. (Optional) Install anaconda individual edition (https://www.anaconda.com/products/individual)
4. Install virtualenv:
   ```
   $ pip install virtualenv
   ```
5. Create a virtual environment for running, e.g.:
   ```
   $ python -m venv cardigan
   (for Windows)
      $ cardigan\Scripts\activate.bat
   (for MacOS/Linux)
      $ source cardigan/bin/activate
   ```
6. Run environment and install required packages: 
   ```
   $ source activate cardiGAN
   $ conda install pytorch torchvision -c pytorch -y
   $ conda install numpy pandas -y
   $ conda install matsci -y
   $ pip install pymatgen 
   $ pip install matminer
   ```
   
## Usage
1. Run **cardiGAN.py** to train the cardiGAN model.  
The model will be trained on two datasets: **data/train_composition.csv** (the element compositions of existing CCAs) and **data/train_parameter.csv** (the empirical parameters of existing CCAs). The hyper-parameters and constants used for training are located in hyper_parameters.py. Changing these hyper-parameters might affect the performance of the model. The trained generator network is saved as **saved_models/generator_net.pt**
2. Run **novel_alloy_generator.py** to generate novel CCA candidates.  
This file applies the trained generator network to produce sythnesized CCA compositions. The generated results include the element compositions, empirical parameters (calculated by **parameter_calculator.py**), and phases (predicted using **classifier_net.pt**) of the CCA candidates. The generated dataset is saved as **data/generated_result.csv**

## Authors and acknowledgment
Zhipeng Li (u6766505@anu.edu.au), Will Nash, Nick Birbilis  
Zhipeng Li performed the bulk of model design and coding, with guidance from Will Nash. Nick Birbilis supervised the project. 

## License
[MIT](https://choosealicense.com/licenses/mit/)
