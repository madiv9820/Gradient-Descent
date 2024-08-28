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
    ![Alt Text](data:image/png;base64,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)
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