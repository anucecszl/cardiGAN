# cardiGAN
Compositionally-complex Alloy Research Directive Inference GAN, a hybrid empirical model and machine learning approach for exploring the Compositionally Complex Alloy (CCA) solution space.

## Description

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
   $ python -m venv epcalc
   (for Windows)
      $ cardiGAN\Scripts\activate.bat
   (for MacOS/Linux)
      $ source cardiGAN/bin/activate
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
