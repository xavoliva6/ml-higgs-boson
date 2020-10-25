# Class Project 1

This repository contains our submission for [project 1](https://github.com/epfml/ML_course/blob/master/projects/project1/project1_description.pdf) of the course Machine Learning CS-433 at EPFL.


### Installation
The project has been developed in Python 3.6. Required packages for running the project are listed in the requirements.txt file and can be installed with:

```
pip3 install --user --requirement requirements.txt
```

Note that we employed the libraries `pandas`, `matplotlib` and `seaborn` only for visualisation purposes in the preprocessing and hyperparameter-tuning phase and that they are not needed to train or run the models.

### Usage

The training data can be downloaded as a zip file [here](https://github.com/epfml/ML_course/blob/master/projects/project1/data/train.csv.zip?raw=true=)
and the test data [here](https://github.com/epfml/ML_course/blob/master/projects/project1/data/test.csv.zip?raw=true).

To train, test and make predictions with our final best models on the given data sets, run in your terminal:
```
  python3 run.py
``` 

Our final predictions can be found in the file `XXX`.

### Project architecture

#### Files:

- **config.py:** contains all our global variables
- **data_loader.py:** downloads the data from the aforementioned github repository, splits it into training- and test data and preprocesses it
- **gridsearch.py:** here we set up a gridsearch to find the optimal parameter values for the preprocessing parameters, the optimal model and the optimal hyperparameters
- **implementations.py:** this file contains all our optimization algorithms for the regression tasks, i.e. linear regression using (stochastic) gradient descent, closed form least squares and ridge regression and (regularized logistic regression. We added two more algorithms: support vector machines and least squares badge gradient descent. All functions return their computed optimal weights and the final loss.
- **preprocessing.py:** all preprocessing functions are in this file
- **proj1_helpers.py:** helper functions for loading the data, creating a submissions and predicting target values
- **project1.ipynb:** contains our fully commented data analysis in one notebook (loading the data, exploratory data analysis, preprocessing, training the models, performance evaluation)
- **run.py:** our final predictions with optimal parameter values for the best models on the given data
- **utils.py:** an assemblance of minor functions used troughout the project
- **visualization.py:** functions used for visualising the data


### Authors

The authors (team \[ShoesOrLose]) of this repository are:
- Devrim Celik ([@Devrim-Celik](https://github.com/Devrim-Celik))
- Xavier Oliva i Jürgens ([@xavoliva6](https://github.com/xavoliva6))
- Nina Mainusch ([@Nina-Mainusch](https://github.com/Nina-Mainusch))

