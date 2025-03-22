# Branch and Bound Method for Integer Optimization

This project is done for the "MA4172 Capita Selecta in Applied Math. I" course at Bandung Institute of Technology in the 7th semester of the Mathematics Program. The branch and bound algorithm was created using the Python programming language with several additional libraries, namely

1. Numpy (for matrix definition)
2. Scipy (to perform the simplex method in solving LP relaxation).

## How to run

1. Clone this repository

```
git clone https://github.com/FelixFern/Branch-And-Bound-Method/tree/main
```

2. Install the needed library using

```
pip install requirement.txt
```
or 
```
pip install scipy numpy
```

3. Opened the `Branch and Bound.ipynb` and go to the second cell and update the `c, A, B, integer_var, lb`, and `ub` variable
4. Run the cell

## Algorithm Details

### Function:

This algorithm will be divided into two functions, namely the is_integer and branch_and_bound functions,

1. is_integer Function

This function determines whether the value entered as a parameter is a member of an integer.

Code:

```python
  def is_integer(n):
      if int(n) == n:
          return True
      else:
          return False
```

2. branch_and_bound Function

A recursion function that serves to solve integer optimization problems using the branch and bound algorithm.

Code:
Defining branch_and_bound function

```python
def branch_and_bound(c, A, b, integer_var, lb, ub, isMax = False, depth=1):
```

Solve the LP relaxation of an integer optimization problem and store the optimal result in the variables x_candidate and f_canditate.

```python
    if(isMax):
        optimized = scipy.optimize.linprog(-c, A, b, method="highs", bounds=list(zip(lb, ub)))
    else:
        optimized = scipy.optimize.linprog(c, A, b, method="highs", bounds=list(zip(lb, ub)))

    x_candidate = optimized.x
    f_candidate = optimized.fun
```

Determine whether the result of each variable is a member of an integer.

```python
    var_constraint_fulfilled = True

    if(f_candidate == None):
        if(isMax):
            return [], -np.inf, depth
        else:
            return [], np.inf, depth

    for idx, bool in enumerate(integer_var):
        if(bool and not is_integer(x_candidate[idx])):
            var_constraint_fulfilled = False
            break
        else:
            var_constraint_fulfilled = True
```

Perform a recursion process with the Depth First Search (DFS) method to perform branch and bound so as to obtain the optimal result which is a member of an integer.

```python
    if(var_constraint_fulfilled):
        if(isMax):
            return x_candidate, -f_candidate, depth
        else:
            return x_candidate, f_candidate, depth
    else:
        left_lb = np.copy(lb)
        left_ub = np.copy(ub)

        right_lb = np.copy(lb)
        right_ub = np.copy(ub)

        max_coeff_idx = -1
        for idx, val in enumerate(integer_var):
            if(val and not is_integer(x_candidate[idx])):
                if(max_coeff_idx == -1):
                    max_coeff_idx = idx
                elif(c[max_coeff_idx] < c[idx]):
                    max_coeff_idx = idx

        left_ub[max_coeff_idx] = np.floor(x_candidate[max_coeff_idx])
        right_lb[max_coeff_idx] = np.ceil(x_candidate[max_coeff_idx])

        x_left, f_left, depth_left = branch_and_bound(c, A, b, integer_var, left_lb, left_ub, isMax, depth + 1)
        x_right, f_right, depth_right = branch_and_bound(c, A, b, integer_var, right_lb, right_ub, isMax, depth + 1)

        if(isMax):
            if(f_left > f_right):
                return x_left, f_left, depth_left
            else:
                return x_right, f_right, depth_right
        else:
            if(f_left < f_right):
                return x_left, f_left, depth_left
            else:
                return x_right, f_right, depth_right
```

Using these two functions, the branch and bound algorithm for integer optimization problems can be performed by calling the branch_and_bound function which contains several parameters, as follows:

-   c: Coefficients of the objective function with data type numpy array
-   A: Coefficient of constraint with 2D numpy array data type
-   b: RHS of each constraint with numpy array data type
-   integer_var: A target variable that is a member of an integer with the data type of an array of Booleans
-   lb: Lower bound of each target variable with array data type
-   ub: The upper limit of each target variable with the array data type
-   isMax: A boolean that specifies whether the objective function problem is a maximization problem.
-   depth: The depth of the branch and bound method tree, with a default value of 1.

### Algorithm Test:

The above algorithm will be used to solve the following integer optimization problem,

```math
z = max\{4x_1 + 3x_2 + x_3\}
```

Constraint

```math
3x_1+2x_2+x_3 ≤ 7
```

```math
2x_1+x_2+2x_3≤11
```

```math
x_1≥0,x_2≥0,x_3≥0\: \text{dengan} \; x_2,x_3∈ Z
```

with this that will be inputted into the parameters of the branch_and_bound function are:

```math
c = [4,3,1]^T
```

```math
A = \begin{bmatrix}
3 & 2 & 1 \\
2 & 1 & 2
\end{bmatrix}
```

```math
b = [7,11]^T
```

```math
lb=[0,0,0]
```

or in Python code, it is written as,

```python
c = np.array([4, 3, 1])

A = np.array([[3, 2, 1], [2, 1, 2]])
b = np.array([7, 11])

integer_var = [False, True, True]

lb = [0, 0, 0]
ub = [None, None, None]
```

With this, by performing variable deconstructing on the branch_and_bound function call,

```python
x_optimal, f_optimal, depth_optimal = branch_and_bound(c, A, b, integer_var, lb, ub, True)
```

the results are as following,

```math
x^*=(x_1,x_2,x_3 )=(0.33..,3,0)
```

```math
z=10.33
```
