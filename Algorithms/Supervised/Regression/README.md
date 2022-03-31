# Linear Regression


## Table of Contents

- [Description](#description)
- [Notations](#notations)
- [Definition](#definition)
- [Flowchart](#flowchart)
- [Univariate Linear Regression](#univariate-linear-regression)
    - [Definition](#definition-for-univariate-linear-regression)
    - [Formula](#formula-for-univariate-linear-regression)
    - [Cost Function](#cost-function-for-univariate-linear-regression)
    - [Gradient Descent](#gradient-descent-for-univariate-linear-regression)
- [Multivariate Linear Regression](#multivariate-linear-regression)
    - [Definition](#definition-for-multivariate-linear-regression)
    - [Formula](#formula-for-multivariate-linear-regression)
    - [Cost Function](#cost-function-for-multivariate-linear-regression)
    - [Gradeint Descent](#gradient-descent-for-multivariate-linear-regression)
- [Feature Scaling and Mean Normalization](#feature-scaling-and-mean-normalization)
- [Bias - Variance](#bias---variance)
    - [High Bias](#high-bias)
    - [Just Right](#just-right)
    - [High Variance](#high-variance)
- [Resolving High Variance](#resolving-high-variance)
    - [Cost Function](#cost-function-for-regularization)
    - [Gradeint Descent](#gradient-descent-for-regularization)

## Description
A Mathematical intuition and quick guide and understanding of how Linear Regression Algorithms works. Given links to other study materials in order to understand the concepts more concretly.

## Notations
- `m` 👉 Number of Training Examples.
- `x` 👉 "input" variable / features.
- `y` 👉 "ouput" variable / "target" variable.
- `n` 👉 Number of feature variable `(x)`
- `(x, y)` 👉 One training example.
- `x`<sub>i</sub> , `y`<sub>i</sub>  👉 i<sup>th</sup> training example.
- `x`<sub>i<sub>j</sub></sub> 👉 i<sup>th</sup> training example of the j<sup>th</sup> column / feature.

-----

## Definition
A linear equation that models a function such that if we give any `x` to it, it will predict a value `y` , where both `x and y` are input and output varaibles respectively. These are numerical and continous values.

It is the most simple and well known algorithm used in machine learning.

## Flowchart 

<p align = 'center'><img src = 'Formulas/Linear_Reg_Flowchart.png'></p>

<br>

The above Flowchart represents that we choose our training set, feed it to an algorithm, it will learn the patterns and will output a function called `Hypothesis function 'H(x)'`. We then give any `x` value to that function and it will output an estimated `y` value for it.

For historical reasons, this function `H(x)` is called `hypothesis function.`

-----

## Univariate Linear Regression
### Definition for Univariate Linear Regression
When you have one feature / variable `x` as an input to the function to predict `y`, we call this `Univariate Linear Regression` problem.

### Formula for Univariate Linear Regression

<p align='center'>H(x) = θ<sub>0</sub> + θ<sub>1</sub>x</p>

Other way of representing this formula as what we are familiar with:

<p align='center'>H(x) = b + mx</p>

> Where :
>- b = θ<sub>0</sub> 👉 y intercept
>- m = θ<sub>1</sub> 👉 slope
>- x = x 👉 feature / input variable

<p align = 'center'><img src = 'Formulas/Linear_model_representation.jpg'></p>


<br>




### Cost Function for Univariate Linear Regression
All that said, how do we figure out the best possible straight line to the data that we feed?

**This is where `Cost Function` will help us:**

The best fit line to our data will be where we have least distance between the `predicted 'y' value` and `trained 'y' value`.

#### Formula for Cost Function
<p align = 'center'><img src = 'Formulas/MSE.png'></p>

> Where :
>- h(x<sub>i</sub>) 👉 hypothesis function
>- y<sub>i</sub> 👉 actual values of `y`
>- 1/m 👉 gives Mean of Squared Errors
>- 1/2 👉 Mean is halved as a convenience for the computation of the `Gradient Descent`.


The above formula takes the sum of the distances between <i>`predicted values` and `actual values` of training set, sqaure it, take the average and multiply it by `1/2`.</i>
<br>
<br>
This cost function is also called as `Squared Error Function` or `Mean Squared Error`.
<br>
<br>
🙋‍ Why do we take squares of the error's?<br>
The `MSE` function is commonly used and is a reasonable choice and works well for most Regression problems.
<br>
<br>
Let's subsititute `MSE` function to function `J` :
<p align = 'center'><img src = 'Formulas/MSE1.png'></p>

<br>
<br>




### Gradient Descent for Univariate Linear Regression
So now we have our hypothesis function and we have a way of measuring how well it fits into the data. Now we need to estimate the parameters in the hypothesis function. That's where `Gradient Descent` comes in.<br>
`Gradient Descent` is used to minimize the cost function `J`, minimizing `J` is same as minimizing `MSE` to get best possible fit line to our data.

#### Formula for Gradient Descent
<p align = 'center'><img src = 'Formulas/Gradient_Descent.PNG'></p>

> Where :
>- `:=` 👉 Is the Assignment Operator
>- `α` 👉 is `Alpha`, it's the number which is called learning rate. If its too high it may fail to converge and if too low then descending will be slow.
>- 'θ<sub>j</sub>' 👉 Taking Gradient Descent of a feature or a column of a dataset.
> - ∂/(∂θ<sub>j</sub>) J(θ<sub>0</sub>,θ<sub>1</sub>) 👉 Taking partial derivative of `MSE` cost function.

<br>
<br>



<br>
<br>

**Now Let's apply Gradient Descend to minmize our `MSE` function.**
<br>
In order to apply `Gradient Descent`, we need to figure out the partial derivative term.<br>
So let's solve partial derivative of cost function `J`.

<br>

<p align = 'center'><img src = 'Formulas/Solving_Partial_Derivative.PNG'></p>

<br>

Now let's plug these 2 values to our `Gradient Descent`:

<br>

<p align = 'center'><img src = 'Formulas/Final_Gradient_Descent.PNG'></p>

<br>

> **Note :** 🚩<br>
> - Cost Function for Linear Regression is always going to be Convex or Bowl Shaped Function, so this function doesn't have any local minimum but one global minimum, thus always converging to global minimum.
> - The above hypothesis function has 2 parameters, θ<sub>0</sub> & θ<sub>1</sub>, so Gradient Descent will run on each feature, hence here two times, one for feature and one for base `(y-intercept)`, to get minimum value of `j`. So if we have `n` features, Gradient Descent will run on all `n+1` features.

-----

## Multivariate Linear Regression

### Definition for Multivariate Linear Regression
Its same as `Univariate Linear Regression`, except it has more than one feature variable `(x)` to predict target variable `(y)`.

### Formula for Multivariate Linear Regression
Our hypothesis function for `n` = 4 :
<p align = 'center'><img src = 'Formulas/Multi_Hypo_Func.PNG'></p>

<br>

> Where :
>- θ<sub>0</sub> 👉 y intercept
> - And rest are features `x` to help predict `y` value.
>> **Intuition:**<br>
>>In order to develop an intuition about this function, let's imagine that this function represents price of a house `(y)` based on the features given `(x)`, then we can think of this function as:
>> - θ<sub>0</sub> as the basic price of a house.
>> - θ<sub>1</sub> as price/m<sup>2</sup>.
>> - x<sub>1</sub> as area of a house (m<sup>2</sup>).
>> - θ<sub>2</sub> as price/floor.
>> - x<sub>2</sub> as number of floors.
>> - etc _(You get the idea)_

<br>
<br>

Let's set all the parameters:
<p align='center'><b> θ<sub>0</sub>, θ<sub>1</sub>, θ<sub>2</sub>, θ<sub>3</sub>.......θ<sub>n</sub> = θ </b></p>
<br>
And Let's set all the features:
<p align='center'><b> x<sub>0</sub>, x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>.......x<sub>n</sub> = x </b></p>
<br>

> Where :
>- θ 👉 will be `n+1` dimensional vector because we have θ<sub>0</sub> which is not a feature.
> - x<sub>0</sub> 👉 is added just for convenience so that we can take matrix multiplication of `θ` as θ<sup>T</sup> and `x` and we will set x<sub>0</sub> value to 1, so this doesn't change the values.
> - x 👉 will also be `n+1` dimensional vector.

<br>
<br>

### Cost Function for Multivariate Linear Regression

So, 
<p align='center'>J(θ<sub>0</sub>, θ<sub>1</sub>, θ<sub>2</sub>, θ<sub>3</sub>.......θ<sub>n</sub>) = J(θ)</p>
<br>
Where,
<p align = 'center'><img src = 'Formulas/Multi_Linear_Cost_Func.PNG'></p>

<br>
<br>

### Gradient Descent for Multivariate Linear Regression

<p align = 'center'><img src = 'Formulas/Multi_Linear_Gradient_Descent.PNG'></p>

<br>
<br>

**Appling Gradient Descend to minmize our `MSE` function after solving partial derivative of J(θ), we get :**
<br>
<p align = 'center'><img src = 'Formulas/Final_Gradient_Descent_Multi_Linear.PNG'></p>

<br>
<br>

**For Example : If we had 2 features, this is how gradient descent would run on each parameter:**

<br>

<p align = 'center'><img src = 'Formulas/Param_Final_Gradient_Descent_Multi_Linear.PNG'></p>

<br>

------

## Feature Scaling and Mean Normalization

We can speed up `Gradient Descent` by having each of our input values in roughly the same range. This is because `θ` will descend quickly on small ranges and slowly on large ranges. If we have large ranges, it will oscilate inefficiently down to optimum (minimum) when the variables are very uneven.

<br>

The way to prevent this is to modify the ranges of our input variables so that they are all roughly the same. Ideally between :
<p align = 'center'>-1  ≤  x<sub>i</sub>  ≤  1</p>
<p align = 'center'>OR</p>
<p align = 'center'>-0.5  ≤  x<sub>i</sub>  ≤  0.5</p>

<br>

These aren't exact requirements, we are only trying to speed things up. The goal is to get all input variables into roughly one of these ranges.<br>
This can make `Gradient Descent` run much faster and converge in a lot few iterations.

<br>

<br>Two techniques to help with this are `Feature Scaling` and `Mean Normalization`.</b>

- `Feature Scaling` involves diving the input values by the range (i.e max value - min value) of the input variable, resulting in a new values.
<p align = 'center'><img src = 'Formulas/Feature_Scaling.PNG'></p>

- `Mean Normalization` involves subtracting the average of a feature variable from the values of the feature, dividing by _range_ of values or by _standard deviation_, resulting in a new values.
<p align = 'center'><img src = 'Formulas/Mean_Normalization.PNG'></p>

> Where:
> - μ<sub>j</sub> 👉 Average of a Feature variable `j`.
> - s<sub>i</sub> 👉 Either `Range` or `Standard Deviation` of a Feature `j`.

-----

## Bias - Variance

### High Bias

Consider a problem of predicting `y`. The figure below shows the result of fitting a `hypothesis function` θ<sub>0</sub> + θ<sub>1</sub>x to a dataset.

<p align = 'center'><img src = 'Formulas/fig_1.PNG'></p>

We see that the data doesn't really lie on a straight line and so the fit is not very good.<br>
This figure is an instance of `Underfitting`, in which the data clearly shows structure not captured by the model `h(x)`.<br>
`Underfitting` or `high bias` is when the form of our `h(x)` function maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features.

-----

### Just Right

If we add an extra feature x<sup>2</sup> and fit `hypothesis function` θ<sub>0</sub> + θ<sub>1</sub>x + θ<sub>2</sub>x<sup>2</sup>, then we obtain a slightly better fit to the data.

<br>

<p align = 'center'><img src = 'Formulas/fig_2.PNG'></p>

<br>

------

### High Variance

If we add more features, it would intuitively seem that it could perform much better, however there's also a danger in adding too amny features. The result of fitting 4th order polynomial where our `h(x)` is  θ<sub>0</sub> + θ<sub>1</sub>x + θ<sub>2</sub>x<sup>2</sup> + θ<sub>3</sub>x<sup>3</sup> + θ<sub>4</sub>x<sup>4</sup>

<br>

<p align = 'center'><img src = 'Formulas/fig_3.PNG'></p>

<br>

We see that even though the fitted curve passes through the data perfectly, we would not expect this to be a very good predictor of for example housing prices `y` for different areas `x`.

<br>

`Overfitting` or `High Variance`, is caused by `h(x)` function that fits the available data but does not generalize (unable to accurately predict) well to predict new data. It is usually caused by giving too many features or having a complicated `h(x)` function that creates lots of unnecessory curves and angles unrelated to the data.

<br>

## Resolving High Variance

There are 2 main options to address the issue of Overfitting:

- **Regularization**
    - Keep all features, but reduce the magnitude of parameters θ<sub>j</sub>.
    - It works well when we have a lot of slightly useful features.
- **Reduce number of features**
    - Manually select which features to keep.
    - Use `model selection` algorithm.

<br>

### Cost Function for Regularization

If we have overfitting from our `hypothesis function`, for example like in figure 3, we can reduce the weight that some of the terms in our function carry by increasing their cost.
<br>
For example, let's say we wanted to make the following function more quadratic:<br>
θ<sub>0</sub> + θ<sub>1</sub>x + θ<sub>2</sub>x<sup>2</sup> + θ<sub>3</sub>x<sup>3</sup> + θ<sub>4</sub>x<sup>4</sup><br>
We'll want to eliminate the influence of θ<sub>3</sub>x<sup>3</sup> and θ<sub>4</sub>x<sup>4</sup>, without actually getting rid of these features or changing the form of our hypothesis, we can instead modify our cost function.<br>

<p align = 'center'><img src = 'Formulas/regularized_cost_func_ex.PNG'></p><br>

We've added two extra terms at the end to inflate the cost of θ<sub>3</sub> and θ<sub>4</sub>. Now, in order for the cost function to get close to `0`, we will have to reduce the values θ<sub>3</sub> and θ<sub>4</sub> to near `0`. This will in turn greatly reduce the values of θ<sub>3</sub>x<sup>3</sup> and θ<sub>4</sub>x<sup>4</sup> in our `hypothesis function`.<br>

<p align = 'center'><img src = 'Formulas/fig_4.PNG'></p><br>

As a result we see that the new `H(x)` looks like a quadratic function but fits the data better compared to _figure 3_ due to extra small terms θ<sub>3</sub>x<sup>3</sup> and θ<sub>4</sub>x<sup>4</sup>.

<br>
<br>

We can also regularize all of our parameters in a single summation as:
<br>
<p align = 'center'><img src = 'Formulas/regularized_cost_func.PNG'></p><br>

The `λ`, or `lambda`, is regularization parameter. It determines how much costs of our `θ` parameters are inflated.<br>
Using the above cost function with the extra summation, we can smooth the output of our hypothesis function to reduce overfitting. If `lambda` is choosen to be too large, it may smooth our function too much and causing `High bias` or `Underfitting`.

> **Note:**<br>
> We penalize the parameters from θ<sub>1</sub> .... θ<sub>n</sub> , but we don't penalize θ<sub>0</sub>. We treat this differently.

------

<br>

### Gradient Descent for Regularization

We will modify our gradient descent function to separate out θ<sub>0</sub> from rest of the parameters because we do not want to penalize θ<sub>0</sub>.

<br>

<p align = 'center'><img src = 'Formulas/regularized_gradient_descent.PNG'></p><br>

The term λ/m times θ<sub>j</sub> performs our regularization.<br>

With some manipulation our updated Gradient Descent for θ<sub>j</sub> can also be represented as:<br>

<p align = 'center'><img src = 'Formulas/regularized_gradient_descent_simple.PNG'></p><br>

The first term in the equation <img src = 'Formulas/regularization_first_term.PNG'> will always be less than 1. Intuitively you can see it as reducing the value θ<sub>j</sub> by some amount on every update. And the second term is exactly the same as it was in `Gradient Descent` without applying regularization.


-----
