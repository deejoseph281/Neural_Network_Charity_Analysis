# Neural_Network_Charity_Analysis
Create a deep-learning neural network to analyze and classify the success of charitable donations.

## Project Overview

A neural network is a powerful machine learning technique that is modeled after neurons in the brain. Neural Networks can rival the performance |of the most robust statistical algorithms without having to worry about any statistical theory. Industry leaders such as Google, Facebook, Twitter, and amazon use an advanced form of neural networks called Deep Neural Networks to analyze images and natural language processing datasets. 

In this project, we'll explore and implement neural networks using the TensorFlow platform in Python. We'll discover how neural networks are designed and how effective they can be with complex datasets, With neural networks. We can combine the performance of multiple statistical and machine learning models with minimal effort. In this project, we'll also use some of the techniques shares by the world's top data engenieers. We will prepare input data and create a robust deep learning models for complex and irregular data. By the end of this project, we'll able to design, train, evaluate and export neural network to use in any scenario.

This assignment is related to the Bootcamp Data Analytics from the University of Toronto. It comprises the goals below for this module: 

Follow below the goals for this project:

* Objective 1: Preprocessing Data for a Neural Network Model
* Objective 2: Compile, Train, and Evaluate the Model
* Objective 3: Optimize the Model
* Objective 4: Write a report on the performance of the deep learning model we created for AlphabetSoup

## Resources

* Data Output: [AlphabetSoupCharity.ipynb](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb), [AlphabetSoupCharity.h5](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.h5), [AlphabetSoupCharity.ipynb](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization.ipynb), and [AlphabetSoupCharity_Optimization.h5](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization.h5). The database is available on [charity_data.csv](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/Resources/charity_data.csv) 
* Software & Data Tools: Python 3.8.8, Visual Studio Code 1.64.2, Jupyter Notebook 6.4.8, pandas 1.4.1, numpy 1.20.3 and scikit-learn 1.0.2

## Results & Code

## Objective 1: Preprocessing Data for a Neural Network Model

* Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
* Drop the EIN and NAME columns.

![](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Capture19_1_1.PNG)

* Determine the number of unique values for each column.
* For those columns that have more than 10 unique values, determine the number of data points for each unique value.
* Create a density plot to determine the distribution of the column values.

![](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Capture19_1_2.PNG)

* Use the density plot to create a cutoff point to bin "rare" categorical variables together in a new column, Other, and then check if the binning was successful.

![](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Capture19_1_3.PNG)

* Generate a list of categorical variables.
* Encode categorical variables using one-hot encoding, and place the variables in a new DataFrame.

![](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Capture19_1_4.PNG)
![](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Capture19_1_5.PNG)

* Merge the one-hot encoding DataFrame with the original DataFrame, and drop the originals.
* Split the preprocessed data into features and target arrays.
* Split the preprocessed data into training and testing datasets.
* Standardize numerical variables using Scikit-Learn’s StandardScaler class, then scale the data.

![](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Capture19_1_6.PNG)


## Objective 2: Compile, Train, and Evaluate the Model

* Continue using the AlphabetSoupCharity.ipynb file where you’ve already performed the preprocessing steps from Deliverable 1.
* Create a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras.
* Create the first hidden layer and choose an appropriate activation function.
* If necessary, add a second hidden layer with an appropriate activation function.
* Create an output layer with an appropriate activation function.
* Check the structure of the model.
* Compile and train the model.

![](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Capture19_2_1.PNG)

* Create a callback that saves the model's weights every 5 epochs.

![](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Capture19_2_2.PNG)

* Evaluate the model using the test data to determine the loss and accuracy.
* Save and export your results to an HDF5 file, and name it AlphabetSoupCharity.h5.

![](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Capture19_2_3.PNG)


##  Objective 3: Optimize the Model

* Noisy variables are removed from features 

![](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Capture19_3_1.PNG)

### First Optimization

* Additional neurons are added to hidden layers 
* Additional hidden layers are added 
* The model's weights are saved every 5 epochs 
* The results are saved to an HDF5 file 

![](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Capture19_3_2.PNG)
![](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Capture19_3_3.PNG)


### Second Optimization

* Additional neurons are added to hidden layers 
* Additional hidden layers are added 
* The activation function of hidden layers or output layers is changed for optimization 
* The model's weights are saved every 5 epochs 
* The results are saved to an HDF5 file 

![](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Capture19_3_4.PNG)
![](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Capture19_3_5.PNG)
![](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Capture19_3_6.PNG)


### Third Optimization

* Additional neurons are added to hidden layers 
* Additional hidden layers are added 
* The activation function of hidden layers or output layers is changed for optimization 
* The model's weights are saved every 5 epochs 
* The results are saved to an HDF5 file 

![](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Capture19_3_7.PNG)
![](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Capture19_3_8.PNG)

## Write a report on the performance of the deep learning model we created for AlphabetSoup

## Data Preprocessing

### What variable(s) are considered the target(s) for your model?

The variable considered as focus or target for the model is 'IS_SUCCESSFUL feature

### What variable(s) are considered to be the features for your model?

Following below the variables that we considered on the feature related to the performance of the deep learning model:

  * APPLICATION_TYPE
  * AFFILIATION
  * CLASSIFICATION
  * USE_CASE
  * ORGANIZATION
  * STATUS
  * INCOME_AMT
  * ASK_AMT
  

### What variable(s) are neither targets nor features, and should be removed from the input data?

  * EIN
  * NAME
  * SPECIAL_CONSIDERATIONS

## Compiling, Training, and Evaluating the Model

### How many neurons, layers, and activation functions did you select for your neural network model, and why?

following below is a summary of the three optimizations with results obtained

![](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Capture19_4.PNG)

In the first optimization, we tried to add the fourth layer, changing the neurons and keeping 50 epochs and not changing the activation function of hidden layers or output layers.

In the second optimization, we removed one layer, changing the neurons again around half values compared to the first optimization, keeping 40 epochs. Also, we changed the activation function of hidden layers to 'tanh.'

In the last optimization attempt, we returned similar parameters to the first optimization because we got a better result of accuracy (73.05) than the second optimization (72.97). So, we decided to change the neurons a bit, keeping the values close to the first optimizations; however, in this attempt, we decided to change the activation function of output layers to 'tanh' and the optimizer to 'Nadam". We got the best accuracy result compared with the three optimizations with 73.20.

### Were you able to achieve the target model performance?

No. The best result was 73.20 in the third attempt vs 75% of the target, its means a gap of 1.8%.

### What steps did you take to try and increase model performance?

First Optimization

- Added two additional layers totalizing four.
- hidden nodes layer 1 = 250
- hidden nodes layer 2 = 140
- hidden nodes layer 3 = 50
- hidden nodes layer 4 = 20
- We kept activation function of hidden layers as relu
- We kept the activation function of output layers as sigmoid.
- We kept the optimizer as adam
- We change the epoch to 50

Second Optimzation

- Excluded one layer, totalizing three.
- hidden nodes layer 1 = 100
- hidden nodes layer 2 = 75
- hidden nodes layer 3 = 50
- We changed the activation function of hidden layers to tanh
- We kept the activation function of output layers as sigmoid.
- We kept the optimizer as adam
- We change the epoch to 40

Third Optimzation

- Added one extra layer again, totalizing four.
- hidden nodes layer 1 = 250
- hidden nodes layer 2 = 120
- hidden nodes layer 3 = 35
- hidden nodes layer 4 = 15
- We changed the activation function of hidden layers to relu
- We changed the activation function of output layers as tanh.
- We changed the optimizer as Nadam
- We increased the epoch to 100


## SUMMARY

The neural networks are not a definitive answer for all information data science issues. The figure underneath shows that there are trade-offs to utilizing the new and famous neural network (and deep learning) models over their more established, frequently more lightweight statistics and machine learning counterparts.

![](https://github.com/DougUOT/Neural_Network_Charity_Analysis/blob/main/Resources/Images/Capture19_5.PNG)

## NOTE

Note: All the checkpoints HDF5 Files are available on [HDF5 Files Folder](https://github.com/DougUOT/Neural_Network_Charity_Analysis/tree/main/Resources/Checkpoints%20HDF5%20Files%20saved)

## RECOMMENDATIONS

In the future or as additional analysis, we recommend other different models such as Random Forest, boosting and SVMs in order to increase the result accuracy.
