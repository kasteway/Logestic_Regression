# Logistic_Regression

## from sklearn.linear_model import LogisticRegression -> [Scikit-Learn Logistic Regression]((https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html))  

### Summary:

Logistic regression is a type of regression analysis used for predicting the outcome of a categorical dependent variable based on one or more predictor variables. It's especially used when the dependent variable is binary, meaning it has only two possible outcomes (like "yes" or "no", "win" or "lose"). It's useful for predicting outcomes like whether a patient has a disease, a customer will buy a product, or a student will pass an exam.

The main idea is to find a relationship between features (independent variables) and the probability of a particular outcome. For instance, in a medical context, you might predict the probability of having a disease based on symptoms and test results. Logistic regression is widely used in various fields like medicine, economics, and social sciences. 

##### Example:

A teacher can use logistic regression to predict students' final exam outcomes (pass or fail) based on two factors: their attendance rate and average homework score. By analyzing past data, the logistic regression model establishes a relationship between these variables and the likelihood of passing the exam. For example, the model might indicate that students with over 75% attendance and homework scores above 70 have a high probability of passing. This predictive insight allows the teacher to identify and support students who are at risk of failing. Although the predictions are based on trends and not guaranteed, they provide a valuable tool for informed educational decision-making.



#### How Does it Work?

1. Model Setup: It starts with defining a logistic model, which predicts the probability of an outcome (usually binary, like pass/fail) based on input features (like attendance rate, test scores).

2. Estimation of Coefficients: The model calculates coefficients (weights) for each input feature using a method called maximum likelihood estimation. This method aims to find the set of coefficients that makes the observed outcomes most likely.

3. Applying the Sigmoid Function: The linear combination of input features and their coefficients is passed through a sigmoid (logistic) function. This function maps any real-valued number into a value between 0 and 1, representing a probability.

4. Prediction and Interpretation: Using these probabilities, predictions are made (e.g., whether a student passes or fails). The output is interpreted as the likelihood of the outcome occurring, with values closer to 1 indicating a higher probability of the event (e.g., passing the exam).




#### Why is it Important?

- Probabilistic Interpretation: Unlike other classification methods, logistic regression provides a probability score for observations. This can be more informative than just a classification, as it gives a sense of how confident the model is in its prediction.
- Simplicity and Efficiency: As a relatively simple algorithm, logistic regression can be easier to implement, interpret, and train than more complex models. This makes it a good starting point for binary classification problems.
- Baseline Performance: In machine learning, logistic regression serves as a good baseline model. Due to its simplicity, it's often used to establish an initial performance level that more complex models should exceed.


---
### Advantages & Disadvantages:

#### Advantages:

- Easy to Implement and Interpret: Logistic regression models are straightforward to understand and interpret, making them appealing for many practical applications.

- Efficiency: It is computationally less intensive, which makes it a relatively efficient algorithm, especially for binary classification problems.

- Probabilistic Interpretation: Provides probability scores for observations, offering more nuanced insights than simple binary outcomes.

- Good Performance on Linearly Separable Data: Performs well with datasets where the classes are linearly separable.

- Handles Categorical and Continuous Inputs: Can process both categorical and continuous input variables.

- Robustness: The model is less prone to over-fitting, particularly if the dataset is not too large and the model is not overly complex.

- Extension to Multiclass Problems: Can be extended to multiclass classification problems (though this requires modifications like one-vs-rest or multinomial logistic regression).

#### Disadvantages:

- Assumes Linear Relationships: Assumes a linear relationship between the independent variables and the log-odds of the dependent variable, which may not always be the case.

- Not Suitable for Complex Relationships: Struggles to capture complex relationships in data, unlike more sophisticated models like neural networks.

- Sensitive to Imbalanced Data: Its performance can be affected by imbalanced datasets (where the outcomes are not approximately equally represented).

- Independent Variables Should Be Independent of Each Other: The predictors in logistic regression should be independent of each other. If multicollinearity exists, it can affect the interpretation of the coefficients.

- Outliers Influence: Outliers can have a significant impact on the outcome, leading to skewed results.

- Limited Outcome Structure: Designed only for binary or, with extensions, categorical outcomes, and not suitable for predicting continuous outcomes.

- Not Flexible Enough for Non-linear Data: The model can underperform if the underlying data has a non-linear structure.

---

## Sigmoid Function used for Logistic Regression:

The sigmoid function is a way to transform values into a bounded range of 0 to 1, making it extremely useful in fields like machine learning for tasks such as binary classification. The sigmoid function, often used in logistic regression and neural networks, is a mathematical function that turns any real-valued number into a value between 0 and 1. 

- Shape: It has an "S" shaped curve, known as a sigmoid curve.

- Output Range: The output of the sigmoid function is always between 0 and 1. This makes it particularly useful for situations where we need to interpret the output as a probability.

- Equation: The basic form of the sigmoid function is f(z) = 1 / (1 + e^(-z)), where e is the base of natural logarithms, and x is the input value.

- Behavior: When the input (z) is a large positive number, the sigmoid function approaches 1, and when z is a large negative number, it approaches 0. Around z=0, the sigmoid function shows a rapid transition from 0 to 1.

- Application in Logistic Regression: In logistic regression, the sigmoid function is used to transform the linear equation (like ax + b) into a range between 0 and 1, providing the probability of a particular class or event.

![Screenshot 2024-01-06 at 9 35 10 AM](https://github.com/kasteway/Logestic_Regression/assets/62068733/a81d8f15-e439-41f2-ac68-6adc0756f401)

[Source: Andrew Ng Machine Learning Specialization Coursera Github](https://github.com/greyhatguy007/Machine-Learning-Specialization-Coursera/tree/main/C1%20-%20Supervised%20Machine%20Learning%20-%20Regression%20and%20Classification/week3)

---

## Tips:

- Extension to Multiclass Problems: Although primarily used for binary classification, logistic regression can be extended to multiclass problems (where there are more than two possible outcomes) using techniques like one-vs-rest (OvR) or multinomial logistic regression.
- Non-Convex - will have multiple local minimums if used the squared error cost function to calculate the loss

---

## Resources:

- Wikipedia Logistic Regression Link - [Wikipedia Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
- Scikit-Learn Logistic Regression - [Scikit-Learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Andrew Ng Machine Learning Specialization Coursera Github - [Andrew Ng Machine Learning Specialization Coursera Github](https://github.com/greyhatguy007/Machine-Learning-Specialization-Coursera/tree/main/C1%20-%20Supervised%20Machine%20Learning%20-%20Regression%20and%20Classification/week3)


---
