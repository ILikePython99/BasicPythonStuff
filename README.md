# BasicPythonStuff
Basic Python Stuff 


## Basic Imports

```python
# We'll load up some Python modules and set some parameters for making higher quality plots.
%matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from pandas.plotting import lag_plot
import pandas as pd
import numpy as np
```

## Panda

```python

```
## ODEINT

```python
# Import commands
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.integrate import odeint # This one is new to you!

# Derivative function
def derivs(curr_vals, time):
    
    # Declare parameters
    g = 9.81 # m/s^2
    A = 0.1 # m^2
    m = 80.0 # kg
    l_unstretched = 30
    # Unpack the current values of the variables we wish to "update" from the curr_vals list
    l, v, = curr_vals
    
    # Right-hand side of odes, which are used to compute the derivative
    dldt = v
    dvdt = g + ((-0.65 * A * v * abs(v))/m) + (-1 * k * (l - l_unstretched))/m
    return dldt, dvdt

# Declare Variables for initial conditions
h0 = 200 # meters
v0 = 0 # m/s
l0 = 30 # m/s
g = -9.81 # m/s^2
tmax = 60 # seconds
dt = 0.1 # seconds
k = 6

# Define the time array
time = np.arange(0, tmax + dt, dt)

# Store the initial values in a list
init = [l0, v0]

# Solve the odes with odeint
sol = odeint(derivs, init, time)

# Plot the results using the values stored in the solution variable, "sol"

# Plot the height using the "0" element from the solution
plt.figure(1)
plt.plot(time, sol[:,0],color="green")
plt.xlabel('Time [s]')
plt.ylabel('Height [m]')
plt.grid()

# Plot the velocity using the "1" element from the solution
plt.figure(2)
plt.plot(time, sol[:,1],color="purple")
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.grid()
```


## Markov Chain

```python
def calc_total_distance(table_of_distances, city_order):
    '''
    Calculates distances between a sequence of cities.
    
    Inputs: N x N array containing distances between each pair of the N
    cities, as well as an array of length N+1 containing the city order,
    which starts and ends with the same city (ensuring that the path is
    closed)

    Returns: total path length for the closed loop.
    '''
    total_distance = 0.0

    # loop over cities and sum up the path length between successive pairs
    for i in range(city_order.size-1):
        total_distance += table_of_distances[city_order[i]][city_order[i+1]]

    return total_distance


def plot_cities(city_order, city_x, city_y):
    '''
    Plots cities and the path between them.
    
    Inputs:
    
    city_order : the order of cities
    city_x : the x courdinates of each city
    city_y : the y coordinates of each city.  
    
    Returns: a plot showing the cities and the path between them.
    '''
    
    # first make x,y arrays
    x = []
    y = []

    # put together arrays of x and y positions that show the order that the
    # salesman traverses the cities
    for i in range(0, city_order.size):
        x.append(city_x[city_order[i]])
        y.append(city_y[city_order[i]])

    # append the first city onto the end so the loop is closed
    x.append(city_x[city_order[0]])
    y.append(city_y[city_order[0]])

    time.sleep(0.1)  
    clear_output(wait = True)
    display(fig)            # Reset display
    fig.clear()             # clear output for animation

    plt.xlim(-0.4, 20.4)    # give a little space around the edges of the plot
    plt.ylim(-0.4, 20.4)
    
    # plot city positions in blue, and path in red.
    plt.plot(city_x, city_y, 'bo', x, y, 'r-')
    #plt.show()


width = 1
intercept = 0
sigma = 2.0
num_points = 10

xs, ys, ys_noisy = make_noisy_data(width, intercept, sigma, num_points)
draw_system_and_model(xs, ys, ys_noisy, sigma)


# number of cities we'll use.
number_of_cities = 30

# seed for random number generator so we get the same value every time!
# There is a really nice feature of using seeds here: your classmates can also use the same
# seed. As you get results post them on Slack with your seed. Let's see who gets the shortest
# distance! (Or, the same distance when that is appropriate.)
np.random.seed(123456789)

# create random x,y positions for our current number of cities.  (Distance scaling is arbitrary.)
city_x = np.random.random(size = number_of_cities) * 20.0
city_y = np.random.random(size = number_of_cities) * 20.0

# table of city distances - empty for the moment
city_distances = np.zeros((number_of_cities,number_of_cities))

# Calculate distance between each pair of cities and store it in the table.
# Technically we're calculating 2x as many things as we need (as well as the
# diagonal, which should all be zeros).
for a in range(number_of_cities):
    for b in range(number_of_cities):
        city_distances[a][b] = ((city_x[a] - city_x[b])**2 + (city_y[a]-city_y[b])**2 )**0.5

# create the array of cities in the order we're going to go through them
city_order = np.arange(city_distances.shape[0])

# tack on the first city to the end of the array, since that ensures a closed loop
city_order = np.append(city_order, city_order[0])

fig = plt.figure(figsize=(6,6))
plot_cities(city_order, city_x, city_y)

```

```python

def swap_city(city_order):
    '''
    This function randomly swaps two cities in the current path defined by city_order.
    Args:
       city_order: a path for the salesperson.
    Returns:
       new_order: new path with two cities swapped.
    '''
    # This step is important! What is ".copy()" doing? What if we didn't include that?
    new_order = city_order.copy()
    
    # Put your swapping code here
    swap1,swap2 = np.random.choice(new_order[1:-1], 2, replace = False)
    new_order[swap1] = city_order[swap2]
    new_order[swap2] = city_order[swap1]
#     np.insert(new_order,city_order[swap2],swap1)
#     np.insert(city_order,city_order[swap1],swap2)
    return new_order

def find_path(city_order, dist_table, city_x, city_y, N):
    '''
    This function finds the shortest path covering all cities using MC method. 
    Args:
       city_order: a path for the salesperson.
       dist_table: array containing mutual distance between cities.
       city_x: the x coordinate of the cities.
       city_y: the y coordinate of the cities.
       N: the number of iterations for the search.
    Returns:
       best_order: a solution for "best" path.
       dist: list of distances for each iteration
    '''
    # Put your path-finding code here, make sure you plot the cities so that you can see the path change
    dist = []
    total_dist_old = calc_total_distance(dist_table,city_order)
    dist.append(total_dist_old)
    for i in range(N+1):
        #total_dist_old = calc_total_distance(dist_table,city_order)
        new_array = swap_city(city_order)
        total_dist_new = calc_total_distance(dist_table,new_array)
        if total_dist_new < total_dist_old:
            total_dist_old = total_dist_new
            city_order = new_array
        dist.append(total_dist_old)
        plot_cities(city_order, city_x, city_y)
        
    
    
    

    return dist, city_order


```


## Polyfit

```python
parameters_2017 = np.polyfit(temp_file_2017["Unnamed: 0"], temp_file_2017["meantempi"], 2)
print(parameters_2017)

quad = np.poly1d(parameters_2017)
expected_quad = quad(temp_file_2017["Unnamed: 0"])

plt.scatter(temp_file_2017["Unnamed: 0"], temp_file_2017["meantempi"], label = "data")
plt.plot(temp_file_2017["Unnamed: 0"], expected_quad, color = "green", label = "fit")
plt.show()

parameters_2017 = np.polyfit(temp_file_2017["Unnamed: 0"], temp_file_2017["meantempi"], 3)
print(parameters_2017)

quad = np.poly1d(parameters_2017)
expected_quad = quad(temp_file_2017["Unnamed: 0"])

plt.scatter(temp_file_2017["Unnamed: 0"], temp_file_2017["meantempi"], label = "data")
plt.plot(temp_file_2017["Unnamed: 0"], expected_quad, color = "green", label = "fit")
```
