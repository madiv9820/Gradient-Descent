'''
    MULTI-VARIABLE LINEAR REGRESSION USING GRADIENT DESCENT


    Creating a Linear Regression Algorithm using Gradient Descent.
    This algorithm accepts multiple variables or features, from the 
    dataset or dataframe, trainings the algorithms as per the features
    and predicts the value.
'''

# Data will be form of 2D or 1D numpy arrays
import numpy    

# Defining a class Linear Regression
class LinearRegression:

    # Constructor
    def __init__(self, 
                    iterations: int = 1000, 
                    alpha: int = 1, 
                    stoppage: float = 0.001
                ):
        '''
            iterations:- No of times we want to run gradient descent on 
                            the training set.
            alpha:- Learning rate
            stoppage:- Maximum difference between the cost during training. 
                        If difference exceeds, it will stop training.
            coef:- Slopes of all the variables for features
            intercept: Y-intercept
        '''
        self.__iterations = iterations
        self.__alpha = alpha
        self.__stoppage = stoppage
        self.__constants_ = None
        self.coef_ = None
        self.intercept_ = None
        self.logs = []
        
    # String representation of the class
    def __repr__(self) -> str:
        return ('gradient_descent.Linear_Regression' + 
                f'<iterations = {self.__iterations}, ' + 
                f'alpha = {self.__alpha}, ' + 
                f'stoppage = {self.__stoppage}>')
    
    # A private cost function 
    def __cost(self, 
               x: numpy.ndarray, 
               y: numpy.ndarray, 
               m: numpy.ndarray
            ) -> float:
        '''
            Cost = Mean from i = 1 to n 
            (y[ith row] - (m1 *  x1[ith row] + m2 * x2[ith row] + ....... + xn[ith row]) ** 2)
        '''
        return numpy.mean((y - numpy.dot(x, m)) ** 2)
     
    
    # Gradient function to find the current slope of each value in m 
    def __step_gradient_descent(self, 
                                x: numpy.ndarray, 
                                y: numpy.ndarray, 
                                m: numpy.ndarray
                            ) -> numpy.ndarray:
        errors = y - numpy.dot(x, m)
        slopes = (-2/x.shape[0]) * numpy.dot(x.T, errors)

        return slopes   # Returning all the slopes
    
    # Gradient Descent function to train the algorithm
    def __gradient_descent(self, x: numpy.ndarray, y: numpy.ndarray) -> None:
        # Intialising all the coefficients of each feature = 0
        self.__constants_ = numpy.zeros(x.shape[1], dtype = float)
        
        # Training all the algorithm
        for count in range(self.__iterations):
            # Getting slope with the help of step gradient descent
            slopes = self.__step_gradient_descent(x, y, self.__constants_)
            
            '''
                New Coef = Current_Coef - alpha * Slopes
                Slopes = d(Cost)/d(Slope)
            '''
            new_coef = self.__constants_ - self.__alpha * slopes
            
            # Calculating the cost with new coefficient and previous coefficient
            previous_Cost = self.__cost(x, y, self.__constants_)
            current_Cost = self.__cost(x, y, new_coef)

            
            log = (f"Iteration: {count+1}, Previous Cost: {previous_Cost}, " +
                    f"Current Cost: {current_Cost}, Difference: " +
                    f"{abs(previous_Cost - current_Cost)}, Alpha: {self.__alpha}")
            
            self.logs.append(log)

            '''
                If current_Cost > previous_Cost, means value of alpha is very high,
                we need to decrease the value of alpha
            '''
            if count % 100 == 0: self.__alpha *= 2
            if current_Cost > previous_Cost: self.__alpha /= 2
            else: self.__constants_ = new_coef   # Updating the coefficient

            # If difference between the cost is too low, 
            # we can stop training the algorithm
            if abs(current_Cost - previous_Cost) < self.__stoppage: break
            
    # A fit function to train the algorithm
    def fit(self, x: numpy.ndarray, y: numpy.ndarray) -> None:
        '''
            x: all the values of the features in the form 2d array
            y: the result of the features

            y = m1x1 + m2x2 + m3x3 + m4x4 + ... + mnxn + C
            C is the y intercept. 
        '''
        
        '''
            We are not calculating C seperatly, rather than calculating
            it, we know C is constant, independent of m's for that we are
            taking, m = 1. So, we have added 1 to each row.
            y = (m1 X x1) + (m2 X x2) + (m3 X x3) + (m4 X x4) + ..... + (mn X xn) + (1 X C)
            coef = [m1, m2, m3, ......, mn, C]
            x = [x1, x2, x3, ........, xn, 1]
        '''
        x = numpy.append(x, numpy.ones((x.shape[0], 1)), axis = 1)

        # Training the algorithm using gradient descent
        self.__gradient_descent(x, y)

        # Seggregating Coefficient and intercept
        self.coef_ = self.__constants_[:-1]
        self.intercept_ = self.__constants_[-1]

    # A function to predict values to provide values of features
    def predict(self, x: numpy.ndarray) -> numpy.ndarray:
        # If fit function has not run yet
        if self.__constants_ is None: raise RuntimeError('Model has not been fitted yet.')

        # Since we need to m = 1 to finding intercept (C)
        # Appending 1 to each row in y 
        x = numpy.append(x, numpy.ones((x.shape[0], 1)), axis = 1)

        # We have already have coef we can find value
        # i.e. y = mx + C, since value of C is already
        # included in M, so y = mx
        return numpy.dot(x, self.__constants_)

    # A function to determine coefficient of determination or score
    def score(self, x: numpy.ndarray, y: numpy.ndarray) -> float:
        if self.__constants_ is None: raise RuntimeError('Model has not been fitted yet.')
        y_pred = self.predict(x)
        total_Variance = numpy.sum((y - numpy.mean(y)) ** 2)
        explained_Variance = numpy.sum((y - y_pred) ** 2)

        return 1 - explained_Variance/total_Variance

    # A function to get the cost
    def cost(self, x: numpy.ndarray, y: numpy.ndarray) -> float:
        if self.__constants_ is None: raise RuntimeError('Model has not been fitted yet.')
        y_pred = self.predict(x)
        return numpy.mean((y - y_pred) ** 2)