# cardiGAN
 A hybrid empirical model and machine learning approach for exploring the Compositionally Complex Alloy solution space.


## Installation
1. git clone this repository.
2. Install anaconda individual edition through https://www.anaconda.com/products/individual
3. Create an environment for running, e.g.:
   ```
   $ conda create -n cardiGAN python=3.7
   ```
4. Run environment and install required packages:
   ```
   $ source activate cardiGAN
   $ conda install pytorch torchvision -c pytorch -y
   $ conda install numpy pandas -y
   $ conda install matsci -y
   $ pip install pymatgen 
   $ pip install matminer
   ```
