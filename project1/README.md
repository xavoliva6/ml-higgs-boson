# Class Project 1

This repository contains our submission for [project 1](https://github.com/epfml/ML_course/blob/master/projects/project1/project1_description.pdf) of the course Machine Learning CS-433 at EPFL.


### Installation
The project has been developed in Python 3.6. Required packages for running the project are listed in the requirements.txt file and can be installed with:

```
pip3 install --user --requirement requirements.txt
```

Note that we employed the libraries `pandas`, `matplotlib`, `seaborn` and `sklearn` only for visualisation purposes in the preprocessing and hyperparameter-tuning phase and that they are not needed to train or run the models.

To train, test and make predictions with our models on the given data sets, run:

```
  python3 run.py
``` 

The training data can be downloaded as a zip file [here](https://github.com/epfml/ML_course/blob/master/projects/project1/data/train.csv.zip?raw=true=)
and the test data [here](https://github.com/epfml/ML_course/blob/master/projects/project1/data/test.csv.zip?raw=true).


Our final predictions can be found in the file `XXX`.



### Architecture

#### FILES

##### config.py

##### data_loader.py

##### implementations.py
This file contains all our optimization algorithms for the regression tasks, i.e. linear regression using (stochastic) gradient descent, closed form least squares and ridge regression and (regularized logistic regression. All functions return their computed optimal weights and the final loss.

##### preprocessing.py
We gathered all our preprocessing functions in this file, 

##### helpers.py

##### project1.ipynb



##### run.py

##### utils.py

##### visualization.py


### Usage

To run our project, simply execute run.py in your terminal.


### Authors

The authors (team \[ShoesOrLose]) of this repository are:
- Devrim Celik (@Devrim-Celik)
- Xavier Oliva i Jürgens (@xavoliva6)
- Nina Mainusch (@Nina-Mainusch)

