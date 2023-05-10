#@authors: Elias Stenhede 5900298, ...., .......

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

#read the txt file 
df = pd.read_csv(r'./spam.txt')
df.to_numpy()
target = 'spam'
columns = ['word_freq_make', 'word_freq_address', 'word_freq_all',
           'word_freq_3d', 'word_freq_our', 'word_freq_over',
           'word_freq_remove', 'word_freq_internet', 'word_freq_order',
           'word_freq_mail', 'word_freq_receive', 'word_freq_will',
           'word_freq_people', 'word_freq_report', 'word_freq_addresses',
           'word_freq_free', 'word_freq_business', 'word_freq_email',
           'word_freq_you', 'word_freq_credit', 'word_freq_your',
           'word_freq_font', 'word_freq_000', 'word_freq_money',
           'word_freq_hp', 'word_freq_hpl', 'word_freq_george',
           'word_freq_650', 'word_freq_lab', 'word_freq_labs',
           'word_freq_telnet', 'word_freq_857', 'word_freq_data',
           'word_freq_415', 'word_freq_85', 'word_freq_technology',
           'word_freq_1999', 'word_freq_parts', 'word_freq_pm',
           'word_freq_direct', 'word_freq_cs', 'word_freq_meeting',
           'word_freq_original', 'word_freq_project', 'word_freq_re',
           'word_freq_edu', 'word_freq_table', 'word_freq_conference',
           'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!',
           'char_freq_$', 'char_freq_#', 'capital_run_length_average',
           'capital_run_length_longest', 'capital_run_length_total']

x= df[columns].values
y= df[target].values

test_set = df['test'].to_numpy()
x_train, x_test = x[test_set == 0], x[test_set == 1]
y_train, y_test = y[test_set == 0], y[test_set == 1]

def cross_validation(model, x, y, K, param_name, param_range):
    """Plots the accuracy of a given model with different parameter values."""
    def fit_and_score(model, x_train, y_train, x_test, y_test, param):
        """Calculates the accuracy score of our model given a specific parameter value"""
        model.set_params(**{param_name: param})
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return accuracy_score(y_test, y_pred)

    # Empty arrays will be filled with data for plot
    cross_validation_estimated_accuracy = []
    cross_validation_estimated_accuracy_standard_deviation = []

    # Create folds [[], [], [], ...]
    x_folds = np.array_split(x, K)
    y_folds = np.array_split(y, K)

    for param in param_range:
        accuracies = []
        for k in range(K):
            # For each fold, let the k'th part be test data and the rest training data.
            # Fit the model and calculate the accuracy score.
            x_test, y_test = x_folds[k], y_folds[k]
            x_train = np.concatenate([x_folds[id] for id in range(K) if id != k])
            y_train = np.concatenate([y_folds[id] for id in range(K) if id != k])
            score = fit_and_score(model, x_train, y_train, x_test, y_test, param)
            accuracies.append(score)

        # Save the mean and std for each parameter.
        cross_validation_estimated_accuracy.append(np.mean(accuracies))
        cross_validation_estimated_accuracy_standard_deviation.append(np.std(accuracies))

    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.errorbar(param_range, cross_validation_estimated_accuracy, yerr=cross_validation_estimated_accuracy_standard_deviation, capsize=4, fmt='ro--', ecolor='black', elinewidth=0.5)
    ax.set_xlabel(r'Tuning parameter')
    ax.set_ylabel('')
    ax.set_title('CV estimated accuracy')
    plt.show()
    return

"""
a)
Fit a logistic regression model to the spam dataset using the Python function LogisticRegression
and compute the prediction accuracy over the test set using the function accuracy score.
"""
#Logistic Regression
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy: {:.2f}%".format(acc*100))

K = 5 #TODO how many folds do we want?
#TODO what param ranges are interesting?
"""
b)
Fit a decision tree to the spam dataset using the Python function DecisionTreeClassifier and
compute the prediction accuracy over the test set using the function accuracy score. Write an
implementation of K-fold cross-validation for the decision tree in order to determine the optimal
amount of terminal nodes. All other tuning parameters can be set to their default setting. Plot a
graph that shows the estimated generalisation error (including standard error) against the allowed
amount of terminal nodes.
"""
# Decision tree
dtc = DecisionTreeClassifier()
param_name = "max_leaf_nodes"
param_range = np.array(range(2, 10, 3))
cross_validation(dtc, x, y, K, param_name, param_range)
"""
c)
Fit a random forest to the spam dataset using the Python function RandomForestClassifier and
compute the prediction accuracy over the test set using the function accuracy score. Write an
implementation of K-fold cross-validation for the random forest in order to determine the optimal
amount of terminal nodes per decision tree. All other tuning parameters can be set to their default
setting. Plot a graph that shows the estimated generalisation error against the allowed amount of
terminal nodes per tree. Compute the OOB error of your final model and comment on your results.
"""
# Random forest
rfc = RandomForestClassifier()
param_name = "max_leaf_nodes"
param_range = np.array(range(2, 10, 3))
cross_validation(rfc, x, y, K, param_name, param_range)
# TODO
# Plot a graph that shows the estimated generalisation error against the allowed amount of terminal nodes per tree.
# Currently we only plot accuracy, not error.

optimal_param_value = 14 #TODO determine this from the plot in some nice way.
rfc = RandomForestClassifier(max_leaf_nodes=optimal_param_value, oob_score=True)
rfc.fit(x_train, y_train)
oob_error = 1 - rfc.oob_score_
print("OOB error:", oob_error)

"""
d)
Comment on the difference between the found optimal amount of terminal nodes for the decision
tree of Q2b and the random forest fitted in Q2c. Explain your reasoning.
"""

"""
e)
Fit a series of random forests to the spam data, to explore the sensitivity to the parameter m, the
maximal amount of features considered per split. Plot the OOB error as well as the test error
against a suitably chosen range of values for m. Interpret your results.
"""

# Set up range of values for max_features
max_features_range = range(1, x_train.shape[1]//2 + 1) #TODO determine optimal range
oob_errors = []
test_errors = []

for max_features in max_features_range:
    print(f"{max_features} out of {max_features_range[-1]}")
    rfc = RandomForestClassifier(max_leaf_nodes=optimal_param_value, max_features=max_features, oob_score=True)
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    oob_errors.append(1 - rfc.oob_score_)
    test_errors.append(1 - accuracy_score(y_test, y_pred))

fig, ax = plt.subplots()
ax.plot(max_features_range, oob_errors, label='OOB error')
ax.plot(max_features_range, test_errors, label='Test error')
ax.set_xlabel('max_features')
ax.set_ylabel('Error')
ax.legend()
plt.show()

"""
f)
Fit a neural network with one hidden layer to the spam data using the Python function MLPClassifier.
Use cross-validation to determine the optimal amount of hidden units for the hidden layer. Com-
ment on the obtained network architecture, the classification performance, and the interpretability
of the final model.
"""

# Neural network
mlp = MLPClassifier()
param_name = "hidden_layer_sizes"
param_range = np.array(range(10, 40, 10))
cross_validation(mlp, x, y, K, param_name, param_range)
