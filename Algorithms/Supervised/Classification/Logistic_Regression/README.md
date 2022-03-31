# Logistic Regression

## Table of Contents
- [Description](#description)
- [Prerequisite](#prerequisite)
- [Notations](#notations)
- [Defintion](#definition)
- [Hypothesis Function](#hypothesis-function)
- [Decision Boundary](#decision-boundary)
- [Cost Function](#cost-function)
- [Gradient Descent](#gradient-descent)
- [Multiclass Classification](#multiclass-classification)
- [Regularization](#regularization)

## Description
A Mathematical intuition and quick guide and understanding of how Logistic Regression Algorithm works. 

## Notations
- `m` 👉 Number of Training Examples.
- `x` 👉 "input" variable / features.
- `y` 👉 "ouput" variable / "target" variable.
- `n` 👉 Number of feature variable `(x)`
- `(x, y)` 👉 One training example.
- `x`<sub>i</sub> , `y`<sub>i</sub>  👉 i<sup>th</sup> training example.
- `x`<sub>i<sub>j</sub></sub> 👉 i<sup>th</sup> training example of the j<sup>th</sup> column / feature.

## Definition
`Logisitic Regression` is a classification algorithm where a dependent variable `'y'` that we want to predict takes on discrete values, for example `y ϵ {0,1}`. It is the most popular and widely used.

<br>

### Example of Classification Problem

<p align = 'center'><img src = 'Formulas/example.png'></p>

<br>

These are some of the area where `Logistic Regression` is used. Where we want to know whether an email recieved is `'Spam'` or `'Not-Spam'` and then place them to their predicted category. Whether the transaction is fradulent or not and whether the tumor is `'Benign'` or `'Malignant'`.
<br>
The way we approach to these type of classification problems where the prediction variable `'y'` does not take on continous value, is we set `'y'` to take on `'discrete values'`, for example:<br>
<pre align = center>y ϵ {0,1}          0 : 'Nagative Class'
                    1 :  'Positive Class' </pre>

<br>

We can think of predicting value `'y'` taking on two value either `'0'` or `'1'`, either `'Not-Spam'` or `'Spam'`, either `'Benign'` or `'Malignant'` etc.

<br>

Another name for the class that we denote with `'0'` is the `'negative class'` and another name for the class that we denote with `'1'` is `'positive class'`. So `'0'` we denote as `'Not-Spam'` and `'1'` as `'Spam'`. The assignment of these classes is arbitrary and it doesn't really matter but often there is an intuition that a `'negative class' '0'` is conveying the absence of something.

<br>

Classification probelms like these are also called `'Binary Classification'` problem where we have only two outputs, either `'0'` or `'1'`.

## Hypothesis Function
We could appraoch the classification problem ignoring the fact that `'y'` is discrete valued, and use [Linear Regression]( https://github.com/JuzerShakir/Linear_Regression#formula-for-univariate-linear-regression) algorithm to try to predict `'y'` given `'x'`. However, it is easy to construct examples where this method performs very poorly. And also it doesn't make sense for our `'h(x)'` to take values larger than `1` or smaller than `0` when we konw `'y ϵ {0,1}'`. To fix this, we need to change the form of our `'h(x)'` to satisfy 0 ≤ h(x) ≤ 1.
<br>
This is achieved by plugging θ<sup>T</sup>x into the `'Logistic Function'` or also known as `'Sigmoid Function'`.

<br>

**Sigmoid Function:**

<br>

<p align = 'center'><img src = 'Formulas/Sigmoid_Func.PNG'></p>

<br>

**Graph of Sigmoid Function: g(z) with respect to z**
<p align = 'center'><img src = 'Formulas/Sigmoid_Func_Graph.png'></p>

<br>

This function has 2 horizonatal asymptotes. As `'z'` approaches to `-∞` , g(z) approaches to `0` and z approaches to `∞`, g(z) approaches to `1` and `y-intercept` is `0.5` when `'z'` is `0`.<br>
The function `g(z)` shown above, maps to any real number between `0` and `1` interval, making it useful for tranforming an arbitrary valued function into a fucntion better suited for classification.

<br>

Now lets set `'z'` to θ<sup>T</sup>x and pass it to our `'h(x)'`:

<br>

<p align = 'center'><img src = 'Formulas/hypothesis_fun.PNG'></p>

<br>

`h(x)` will give us the probability that our output is 1. For example, `h(x) = 0.7` gives us the probability of `70%` that our output is `1`. Here's how we interpret it:

<br>

<p align = 'center'><img src = 'Formulas/probability_1.PNG'></p>

<br>

Since here the probability of `'y'` is `0.7` then the probability of `'y'` being `0` is `0.3` since both probability should add up to `1`.<br>
Here's how we can interpret it:

<br>

<p align = 'center'><img src = 'Formulas/probability_2.PNG'></p>

<br>

### Setting discrete values
We can translate the output of the `h(x)` function as follows:<br>

<p align = 'center'><img src = 'Formulas/discrete_value_1.PNG'></p><br>

Because the way that our `Logistic Function g(z)` behaves is that when its input is greater than or equal to `0`, its output is greater than equal to `0.5`.

> **Note:**<br>
> if z = 0, then e<sup>0</sup> = 1, ∴ g(z) = 0.5<br>
> if z = ∞, then e<sup>-∞</sup> = 0, ∴ g(z) approaches 1<br>
> if z = -∞, then e<sup>∞</sup> = 1, ∴ g(z) approaches 0<br>

So if our input to the function `g` is θ<sup>T</sup>x, then that means when θ<sup>T</sup>x ≥ 0, then `h(x)` ≥ 0.5.<br>
From all of these statements we can now say:<br>

<p align = 'center'><img src = 'Formulas/discrete_value_2.PNG'></p><br>

## Decision Boundary
The decision boundary is the line that separates the area of `y=1` and `y=0`. It is created by our hypothesis function.

### Linear Decision Boundary
For example : if<br>
<p align = center>h(x) = θ<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>x<sub>2</sub><br>
and θ<sub>0</sub> = 5, θ<sub>1</sub> = -1, θ<sub>2</sub> = 0</p>

So `y = 1` if:
<p align = center>5 + (-1)x<sub>1</sub> + 0x<sub>2</sub> ≥ 0<br>
5-x<sub>1</sub>  ≥ 0 <br>
-x<sub>1</sub> ≥ -5 <br>
x<sub>1</sub> ≤ 5 </p> 

<br>

In this case our decision boundary is a straight line placed on the graph where x<sub>1</sub> = 5 and everything to the left of that denotes `y = 1` while everything to the right of that denotes `y = 0`.

<br>

<p align = 'center'><img src = 'Formulas/linear_decision_graph.PNG'></p>

<br>

### Non-Linear Decision Boundary
The above input to the `Logistic or Sigmoid Function` was linear but 
θ<sup>T</sup>x can also be a function that describes a circle or any other function.<br>

For example:<br>
<p align = center>h(x) = θ<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub><sup>2</sup> + θ<sub>2</sub>x<sub>2</sub><sup>2</sup><br>
and θ<sub>0</sub> = -1, θ<sub>1</sub> = 1, θ<sub>2</sub> = 1</p>

So `y = 1` if:
<p align = center>-1 + x<sub>1</sub><sup>2</sup> + x<sub>2</sub><sup>2</sup> ≥ 0<br>
or <br>
x<sub>1</sub><sup>2</sup> + x<sub>2</sub><sup>2</sup> ≥ 1<br></p> 

<br>

So if we were to plot the decision boundary of this, it would be a circle with radius 1 centered at the origin.

<br>

<p align = 'center'><img src = 'Formulas/non-linear_decision_graph.PNG'></p><br>

Everything outside the circle is `y=1` and inside is `y=0`.

> **Note:**<br>
> We do not need to define the decision boundary. The training set will fit the parameters θ and once you have them then that will define decision boundary.

## Cost Function
If we choose the cost function of [Linear Regression](https://github.com/dsmentors/Machine_Learning_study_Materials/blob/main/Algorithms/Supervised/Regression/README.md) (MSE), it turns out that this would not guarantee that when we run Gradient Desccent, it will converge to global minimum because here our hypothesis function `h(x)` is not linear, it is a sigmoid function and when we plot `J(θ)` with respect to `θ`, this is what it looks like:<br>
<p align = 'center'><img src = 'Formulas/non-convex.PNG'></p><br>
Since this is has many local minimum, the Gradient Descent will not guarantee to converge to global minimum. We can also call this as non-convex function. We need a convex function which has no local minimum but one globla minimum.

<br>

Our cost function for Logistic Regression:<br>
<p align = 'center'><img src = 'Formulas/cost-func_1.PNG'></p><br>

When `y=1`, we get the following plot for `J(θ)` vs `h(x)`:<br>
<p align = 'center'><img src = 'Formulas/cost-func_graph_1.PNG'></p><br>

Few interesting properties about this we see are:
<p align = center>if y = 1 and h(x) = 1, then Cost J(θ) = 0<br>
But as h(x) approaches 0, Cost approaches ∞</p><br>

Similarly When `y = 0`, we get the following plot for `J(θ)` vs `h(x)`:<br>
<p align = 'center'><img src = 'Formulas/cost-func_graph_2.PNG'></p><br>

Few interesting properties about this we see are:
<p align = center>if y = 0 and h(x) = 0, then Cost J(θ) = 0<br>
But as h(x) approaches 1, Cost approaches ∞</p><br>

Writing the cost function this way guarantees that J(θ) is convex for logistic regression.<br>

We can compress our `Cost Function's` two conditional cases into one case:<br>
<p align = 'center'><img src = 'Formulas/cost-func_2.PNG'></p><br>

Notice that when `y=1`, then the second term will be zero and will not affect the result. And if `y=0`, then the first term will be zero and will not affect the result.<br>
Therefore, our `Cost function` is :<br>
<p align = 'center'><img src = 'Formulas/cost-func_3.PNG'></p><br>

## Gradient Descent
General form of Gradient Descent :<br>
<p align = 'center'><img src = 'Formulas/general_gradient_descent.PNG'></p><br>
Working out the derivative part using Calculus we get:<br>
<p align = 'center'><img src = 'Formulas/gradient_descent.PNG'></p><br>

This algorithm looks similar to [Gradient Descent of Linear Regression](https://github.com/dsmentors/Machine_Learning_study_Materials/blob/main/Algorithms/Supervised/Regression/README.md) but its not since `h(x)` here is a `logistic/sigmoid function` and `h(x)` in `linear regression` is θ<sup>T</sup>x.

## Multiclass Classification
Now we will approach the classification of data when we have more than `2 categories`. Instead of y ϵ {0,1} we will expand our definition so that y ϵ {0,1,2....,n}.<br>
We need to predict the probability that `y` is a member of one of our classes from {0,1,2....,n}.<br>
<p align = 'center'><img src = 'Formulas/Multiclass_Probability.PNG'></p><br>

We apply `Logistic Regression` to each class, and then choose hypothesis which returned the highest probability and use to predict new `x` value.

## Regularization
Let's say that we have a function:<br>
h(x) = g(θ<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>x<sub>1</sub><sup>2</sup> + θ<sub>3</sub>x<sub>1</sub><sup>2</sup>x<sub>2</sub> + 
θ<sub>4</sub>x<sub>1</sub><sup>2</sup>x<sub>2</sub><sup>2</sup> + 
θ<sub>5</sub>x<sub>1</sub><sup>2</sup>x<sub>2</sub><sup>3</sup>
............)<br>
and it fits the data as follows:<br>
<p align = 'center'><img src = 'Formulas/overfit.PNG'></p><br>
As we can clearly see this function overfits the data and will not generalize well for new or unseen data.

### Cost Function for Regularization
We'll want to eliminate the influence of the parameters without actually getting rid of these features or changing the form of the hypothesis function. We instead modify our `Cost function.`<br>
Our Cost Function was:<br>

<p align = 'center'><img src = 'Formulas/cost-func_3.PNG'></p><br>

We can regularize this equation by adding a term to the end:<br>
<p align = 'center'><img src = 'Formulas/regularized_cost-func.PNG'></p><br>

The second sum, <img src = 'Formulas/term_1.PNG'>, means to explicitly exclude the bias term, θ<sub>0</sub>, i.e the θ vector is indexed from 0 to n (holding n+1 values, θ<sub>0</sub> through θ<sub>n</sub>), and this sum explicitly skips θ<sub>0</sub>, by running from 1 to n. Thus, when computing the equation, we should continously update the 2 following equation:

### Gradeint Descent
<p align = 'center'><img src = 'Formulas/regularized_gradient_descent.PNG'></p><br>

This may look identical to [Linear Regression's regularization Gradient Descent](https://github.com/dsmentors/Machine_Learning_study_Materials/blob/main/Algorithms/Supervised/Regression/README.md) but the hypothesis function is different, here we have `Sigmoid or Logistic Function` and for Linear we have θ<sup>T</sup>x.
