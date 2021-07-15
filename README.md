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
The empirical parameter calculator could calculate 12 empirical parameters of CCAs based on their element compositions. The calculator in this project is hard coded in PyTorch. A parameter calculator with user interface can be accessed through https://github.com/anucecszl/Parameter-Calculator-for-CCA.
5. HEA dataset:
The file HEA_dataset provides the element compositions and reported phases of the 845 CCAs obtained from 11 literatures. Since there are 845 data and the dataset is manually collected by the author, the references are in the form of 'A-B' where A is number of the referred literature where the data was collected, and B is the original reference in the referred literature. It's noted that the training dataset for the model excludes all the repetitive data and alloys with unknown phases. 

- The configuration of the cardiGAN model: [image](http://github.com/ZhipengGaGa/cardiGAN/raw/master/model_configuration.png)

- The model is trained on the element compositions of 278 existing CCAs. The compositions are represented as 56-dimensional vectors with each dimension indicating the molar ratio of a specific element. 


References for the dataset:     
[1] Toda-Caraballo, I. and Rivera-Díaz-del-Castillo, P.E.J. (2016). A criterion for the formation of high entropy alloys based on lattice distortion. Intermetallics, 71, pp.76–87.   
[2] Qiu, Y., Hu, Y.J., Taylor, A., Styles, M.J., Marceau, R.K.W., Ceguerra, A.V., Gibson, M.A., Liu, Z.K., Fraser, H.L. and Birbilis, N. (2017). A lightweight single-phase AlTiVCr compositionally complex alloy. Acta Materialia, 123, pp.115–124.    
[3] Singh, A.K. and Subramaniam, A. (2014). On the formation of disordered solid solutions in multi-component alloys. Journal of Alloys and Compounds, 587, pp.113–119. 
[4] Zhang, Y., Zhou, Y.  J., Lin, J.  P., Chen, G.  L. and Liaw, P.  K. (2008). Solid-Solution Phase Formation Rules for Multi-component Alloys. Advanced Engineering Materials, 10(6), pp.534–538.  
[5] Zhang, Y., Yang, X. and Liaw, P.K. (2012). Alloy Design and Properties Optimization of High-Entropy Alloys. JOM, 64(7), pp.830–838.   
[6] Miracle, D.B. and Senkov, O.N. (2017). A critical review of high entropy alloys and related concepts. Acta Materialia, [online] 122, pp.448–511. Available at: https://www.sciencedirect.com/science/article/pii/S1359645416306759.  
[7] Feng, R., Gao, M., Lee, C., Mathes, M., Zuo, T., Chen, S., Hawk, J., Zhang, Y. and Liaw, P. (2016). Design of Light-Weight High-Entropy Alloys. Entropy, 18(9), p.333.  
[8] Dong, Y., Lu, Y., Jiang, L., Wang, T. and Li, T. (2014). Effects of electro-negativity on the stability of topologically close-packed phase in high entropy alloys.  Intermetallics, 52, pp.105–109.  
[9] Yang, X. and Zhang, Y. (2012). Prediction of high-entropy stabilized solid-solution in multi-component alloys. Materials Chemistry and Physics, 132(2-3), pp.233–238.  
[10] Anand, G., Goodall, R. and Freeman, C.L. (2016). Role of configurational entropy in body-centred cubic or face-centred cubic phase formation in high entropy alloys. Scripta Materialia, 124, pp.90–94.  
[11] Gorsse, S., Nguyen, M.H., Senkov, O.N. and Miracle, D.B. (2018). Database on the mechanical properties of high entropy alloys and complex concentrated alloys. Data in Brief, 21, pp.2664–2678.

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
