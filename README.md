# Data310_Lab2
### QUESTION 1:

Regardless whether we know or not the shape of the distribution of a random variable, an interval centered around the mean whose total length is 8 standard deviations is guaranteed to include at least a certain percantage of data. This guaranteed minimal value as a percentage is: 

k = 4
(1-1/(k * k))*100 = **93.75**


### QUESTION 2: 

For scaling data by quantiles we need to compute the z-scores first:

**False**


### QUESTION 3

In the 'mtcars' dataset the zscore of an 18.1mpg car is: 

`import numpy as np`

`import pandas as pd`

`data = pd.read_csv('mtcars.csv')`

`mpg = data.mpg.values`

`zs = (18.1 - np.mean(mpg))/np.std(mpg)`

`print(zs)`

**-0.3355723336253064**


### Question 4





### QUESTION 4

In the 'mtcars' dataset determine the percentile of a car that weighs 3520bs is (round up to the nearest percentage point):

`from scipy.stats import percentileofscore`

`wt = data.wt.values`

`for i in range(101):`

 ` if np.percentile(wt, i) >= 3.52:`
  
    `print(i)`
    
    `break`
    
### Question 5
	
A finite sum of squared quantities that depends on some parameters (weights), always has a minimum value: 

**True**

### Question 6

For the 'mtcars' data set use a linear model to predict the mileage of a car whose weight is 2800lbs. The answer with only the first two decimal places and no rounding is:


`from sklearn import linear_model`

`x = data[['wt']]`

`y = data[['mpg']]`

`lm = linear_model.LinearRegression()`

`model = lm.fit(x,y)`

`lm.predict([[2.8]])`

**array([[22.32060576]])**



### Question 7

	
In this problem you will use the gradient descent algorithm as presented in the 'Linear-regression-demo' notebook. For the 'mtcars' data set if the input variable is the weight of the car and the output variable is the mileage, then (slightly) modify the gradient descent algorithm to compute the minimum sum of squared residuals. If, for running the gradient descent algorithm, you consider the learning_rate = 0.01, the number of iterations = 10000 and the initial slope and intercept equal to 0, then the optimal value of the sum of the squared residuals is:


`import numpy

from numpy import reshape

df = pd.read_csv('mtcars.csv')

x = df[['wt']]

y = df[['mpg']]

learning_rate = 0.01

initial_b = 0

initial_m = 0

num_iterations = 10000

data = np.concatenate((x.values,y.values),axis=1)

def compute_cost(n, m, data):

    total_cost = 0
    
    # number of datapoints in training data
    
    N = float(len(data))
    
    # Compute sum of squared errors
    
    for i in range(0, len(data)):
    
        x = data[i, 0]
        
        y = data[i, 1]
        
        total_cost += (y - (m * x + n)) ** 2
        
    # Return average of squared error
    
    return total_cost/(2*N)

def step_gradient(b_current, m_current, data, alpha):

    """takes one step down towards the minima
    
    Args:
    
        b_current (float): current value of b
        
        m_current (float): current value of m
        
        data (np.array): array containing the training data (x,y)
        
        alpha (float): learning rate / step size
    
    
    Returns:
    
        tuple: (b,m) new values of b,m
    """
    
    m_gradient = 0
    
    b_gradient = 0
    
    N = float(len(data))

    # Calculate Gradient
    
    for i in range(0, len(data)):
    
        x = data[i, 0]
        
        y = data[i, 1]
        
        m_gradient += - (2/N) * x * (y - (m_current * x + b_current))
        
        b_gradient += - (2/N) * (y - (m_current * x + b_current))
    
    # Update current m and b
    
    m_updated = m_current - alpha * m_gradient
    
    b_updated = b_current - alpha * b_gradient

    #Return updated parameters
    
    return b_updated, m_updated

def gradient_descent(data, starting_b, starting_m, learning_rate, num_iterations):

    """runs gradient descent
    
    Args:
    
        data (np.array): training data, containing x,y
        
        starting_b (float): initial value of b (random)
        
        starting_m (float): initial value of m (random)
        
        learning_rate (float): hyperparameter to adjust the step size during descent
        
        num_iterations (int): hyperparameter, decides the number of iterations for which gradient descent would run
    
    Returns:
    
        list : the first and second item are b, m respectively at which the best fit curve is obtained, the third and fourth 
      
      items are two lists, which store the value of b,m as gradient descent proceeded.
      
    """

    # initial values
    
    b = starting_b
    
    m = starting_m
    
    # to store the cost after each iteration
    
    cost_graph = []
    
    # to store the value of b -> bias unit, m-> slope of line after each iteration (pred = m*x + b)
    
    b_progress = []
    
    m_progress = []
    
    # For every iteration, optimize b, m and compute its cost
    
    for i in range(num_iterations):
    
        cost_graph.append(compute_cost(b, m, data))
        
        b, m = step_gradient(b, m, data, learning_rate)
        
        b_progress.append(b)
        
        m_progress.append(m)
        
    return [b, m, cost_graph,b_progress,m_progress]
  

n, m, cost_graph,n_progress,m_progress = gradient_descent(data, initial_b, initial_m, learning_rate, num_iterations)

#Print optimized parameters

print ('Optimized n:', n)

print ('Optimized m:', m)

#Print error with optimized parameters

print ('Minimized cost:', compute_cost(n, m, data))`

Optimized n: 37.285117303091454
Optimized m: -5.344469026915932
Minimized cost: 4.348780274117971

**Answer = 4.349**

### Question 8:

	
(True/False) If we have one input variable and one output, the process of determining the line of best fit may not require the calculation of the intercept inside the gradient descent algorithm.

**True**

### Question 9: 

	
For the line of regression in the case of the example we discussed with the 'mtcars' data set the meaning of the intercept is:

**No interpretable meaning **

### Question 10: 

	
The slope of the regression line always remains the same if we scale the data by z-scores:

**False**

