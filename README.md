# Linear Regression Using Gradient Descent
**Gradient descent** is an optimization algorithm that's commonly used to train machine learning models and neural networks. It works by minimizing the error between predicted and actual results, which improves the model's accuracy over time.

![Alt text](https://miro.medium.com/v2/resize:fit:720/format:webp/1*yUYZ1j2toATCYnKasSncmg.png)

[**For more information, please refer to this link**](https://medium.com/@kavita_gupta/gradient-descent-simplified-for-machine-learning-mastery-407201dae91c).

## Pre-requisites
### Creating Virtual Enviroment
```
conda create --name gradient-descent python=3.10
```
### Activating
```
conda activate gradient-descent
```
### Installing Packages
```
pip install -r requirements.txt
```
### Registrating the environment in a notebook
```
ipython kernel install --name "gradient-descent" --user
```

## Usage
Important: Only uses **numeric features** in nD numpy array for the X and Y matrices.

Feel free to create a pull request with the additional implementation.

## Linear Regression

### Functions
- **LinearRegression(iterations, alpha, stoppage)** :- 
    - **iterations (int)** :- No of times algorithm will run gradient descent, **default value = 1000**
    - **alpha (float)** :- Learning rate, algorithm automatically adjust its value on running gradient descent, **default value = 1**
    - **stoppage (float)** :- Maximum difference of the cost between two iterations, **default value = 0.001**.

- **fit(x, y)** :- This function trains the algorithm, calculates all the parameters required for predictions.
    - **x (numpy.ndarray)** :- Values of the features to train the algorithm
    - **y (numpy.ndarray)** :- Output of the dataset to train the algorithm.

- **predict(x)** :- This function predicts the values for the given values in x.
    - **x (numpy.ndarray)** :- Values of the features for which values need to be predicted
    - returns a **numpy.1d array** containing predicted values

- **score(x, y)** :- This function calculates the **coefficient of determination** for the predicted values and actual values.
    - **x (numpy.ndarray)** :- Values of the features for which values need to be predicte and compare
    - **y (numpy.ndarray)** :- Actual Output, with the help of which coefficient of determination or score will be calculated.
    - **Coefficient of Determination** :- 
    ![Alt Text](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQO5f-66JNgxG7pitR1HcnBZW5Hdrx6o7jrEYM2uzysCupWp4SNGXhEUCCbSDVR1BkKwAo&usqp=CAU)
    - returns a **float** value

- **cost(x, y)** :- This function calculates the cost or error between predict value and actual values.
    - **x (numpy.ndarray)** :- Values for which algorithm, predict the output
    - **y (numpy.ndarray)** :- Actual Outputs, using these cost will be calculated
    - **Cost** :-    
    ![Alt Text](https://editor.analyticsvidhya.com/uploads/272996.PNG)
    - returns a **float** value, average of all the errors with the corresponding values
- **LinearRegression.coef_ (numpy.ndarray)** :- All the coefficients of the features calculated by the algorithm.
- **LinearRegression.intercept_ (float)** :- Y-intercept calculated by the algorithm

### Sample Code
```
import numpy as np
import gradient_descent
from sklearn.model_selection import train_test_split

# Creating a dataset of N features which 1000 random values
N = 10        # No of features
M = 1000       # No of rows
x = np.random.rand(M, N)
# Setting coefficient and intercept 
# (you can add features or change coefficient as per your requirement)
# Make sure that no of coefficient should be same as no of features
actual_m = np.random.rand(N)
actual_c = np.random.rand(1)[0]

# Creating X and Y
y = np.array([(actual_m * x[row]).sum() + actual_c for row in range(M)])

'''
After the training the algorithm, we will test on training data, check how good algorithm
is trained after training with training data and then test will testing_data
'''
# We can run train the algorithm, ntimes and
# check the average training score and testing score

total_Training_Score = 0
total_Test_Score = 0

n = 1
for count in range(n):
    # Since X and Y, as a complete data
    # We need to split into training data and test data
    # So, that we can use training data for training the
    # algorithm, and testing data for testing it

    # Splitting the data into train test, it will split 3:1, mean
    # if dataset has 1000 rows, 750 will be in training and 250 in
    # testing. And the rows will be selected on random basis.
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    algo = gradient_descent.LinearRegression(stoppage = 1e-15, iterations = 2000)
    algo.fit(x_train, y_train)
    total_Training_Score += algo.score(x_train, y_train)
    total_Test_Score += algo.score(x_test, y_test)

# Some Analysis
print('Average Training Score:', total_Training_Score/n)
print('Average Test Score:', total_Test_Score/n)

print('Predicted Coefficients:', algo.coef_)
print('Predicted Intercept:', algo.intercept_)

print('Actual Coefficients:', actual_m)
print('Actual Intercept:', actual_c)
```