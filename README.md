# cardiGAN
Compositionally-complex Alloy Research Directive Inference GAN, a hybrid empirical model and machine learning approach for exploring the Compositionally Complex Alloy (CCA) solution space.

## Description
This project developed a generative adversarial model named cardiGAN to enhance the capacity and efficiency of development of novel CCAs.  

The cardiGAN model has 4 component:
1. Generator network:  
The generator network is used to generate novel CCA compositions. It is trained to learn the mapping of a multi-dimensional Gaussian distribution to the distribution of the element compositions of existing CCAs.
2. Discriminator network:  
The discriminator network (serving the function of “critic”) can be used to estimate the Wasserstein distance between the distributions of the generated and training data. 
3. Phase classifier network (pre-trained):  
The phase classifier network could classify CCAs into three classes: single solid-solution, mixed solid-solution, and solid-solution with secondary phases. It is used to regularize the training of the generator network. 
4. Empirical parameter calculator:  
The empirical parameter calculator could calculate 12 empirical parameters of CCAs based on their element compositions. The calculator in this project is hard-coded in PyTorch. A parameter calculator with user interface can be accessed through https://github.com/ZhipengGaGa/Parameter-Calculator-for-CCA.

The configuration of the cardiGAN model:


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
1. This application supports calculations for 14 empirical parameters of compositionally complex alloys. 

## Authors and acknowledgment
Zhipeng Li (u6766505@anu.edu.au), Will Nash, Nick Birbilis  
Zhipeng Li performed the bulk of model design and coding, with guidance from Will Nash. Nick Birbilis supervised the project. 

## License
[MIT](https://choosealicense.com/licenses/mit/)
