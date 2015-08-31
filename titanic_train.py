import pandas

titanic = pandas.read_csv('train.csv')

print('Titanic train data: ')
print(titanic.describe())

# fill the absent data in 'Age' column with the median of 'Age' column values
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
print()

#find all the unique genders in 'Sex' column
print('Unique genders in "Sex" column:', titanic["Sex"].unique())

# replace all the occurencies of male and female with 0 and 1
titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1

# find all the unique labels in 'Embarked' and replace them with numbers
print('Unique labels in "Embarked" column:', titanic["Embarked"].unique())
print()
titanic['Embarked'] = titanic['Embarked'].fillna('S')
titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2

# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold

# The columns used to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize algorithm class
alg = LinearRegression()
# Generate cross validation folds for the titanic dataset.  It returns the row indices corresponding to train and test.
# Set random_state to ensure to get the same splits every time it runs.
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    # The predictors used to train the algorithm. Only take the rows in the train folds.
    train_predictors = titanic[predictors].iloc[train]
    # The target being used to train the algorithm.
    train_target = titanic["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # Possible now to make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test])
    predictions.append(test_predictions)

import numpy as np

# The predictions are in three separate numpy arrays.  Concatenate them into one.
# Concatenate them on axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
lin_score = sum(predictions == titanic['Survived']) / len(predictions)
print('Linear regression accuracy %s' % lin_score)

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

# Initialize algorithm
alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what was done before!)
log_score = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)

# Take the mean of the scores (because it is one for each fold)
print('Logistic regression accuracy %s' % log_score.mean())

titanic_test = pandas.read_csv('test.csv')

titanic_test['Age'] = titanic_test['Age'].fillna(titanic['Age'].median())
titanic_test['Embarked'] = titanic_test['Embarked'].fillna('S')
titanic_test.loc[titanic_test['Embarked'] == 'S', 'Embarked'] = 0
titanic_test.loc[titanic_test['Embarked'] == 'C', 'Embarked'] = 1
titanic_test.loc[titanic_test['Embarked'] == 'Q', 'Embarked'] = 2
titanic_test.loc[titanic_test['Sex'] == 'male', 'Sex'] = 0
titanic_test.loc[titanic_test['Sex'] == 'female', 'Sex'] = 1
titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].median())

# Initialize the algorithm class
alg = LogisticRegression(random_state=1)

# Train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])

# Make predictions using the test set.
predictions = alg.predict(titanic_test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions})

# ---- RANDOM FORESTS ----

from sklearn.ensemble import RandomForestClassifier


# Initialize algorithm with the default paramters
# n_estimators is the number of trees
# min_samples_split is the minimum number of rows needed to make a split
# min_samples_leaf is the minimum number of samples at the place where a tree branch ends (the bottom points of the tree)
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
rf1_scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=3)
print('1st random forests accuracy:', rf1_scores.mean())

alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
rf2_scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=3)
print('2nd random forests accuracy:', rf2_scores.mean())
print()

# Generating a familysize column
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

# The .apply method generates a new series
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))

import re

# A function to get the title from a name.
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Get all the titles and print how often each one occurs.
titles = titanic["Name"].apply(get_title)
print(pandas.value_counts(titles))
print()

# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8,
                 "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v

# Verify that everythig is converted
print('Titles counted:')
print(pandas.value_counts(titles))
print()

# Add in the title column.
titanic["Title"] = titles

import operator

# A dictionary mapping family name to id
family_id_mapping = {}

# A function to get the id given a row
def get_family_id(row):
    # Find the last name by splitting on a comma
    last_name = row['Name'].split(',')[0]
    # Create the family id
    family_id = '{0}{0}'.format(last_name, row['FamilySize'])
    # Look up for the id in the mapping
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 0
        else:
            # Get the maximum id from the mapping and add one if there isn't an id
            current_id = max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

# Get the family ids with the apply method
family_ids = titanic.apply(get_family_id, axis=1)

# There are a lot of family id's so we'll compress all of the families under 3 members into one code
family_ids[titanic['FamilySize'] < 3] = -1

# Print the count for each unique id
print('Unique IDs counted:')
print(pandas.value_counts(family_ids))

titanic['FamilyId'] = family_ids

#---------------------- Univariate Feature Selection -----------------------
from sklearn.feature_selection import SelectKBest, f_classif
from matplotlib import pyplot as plt

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

# Pick only the four best features.
predictors = ["Pclass", "Sex", "Fare", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)
rfKBest_scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=3)
print('KBest forests accuracy:', rfKBest_scores.mean())
print()

from sklearn.ensemble import GradientBoostingClassifier

# The algorithms we want to ensemble.
# Use more linear predictors for the logistic regression, and everything with the gradient boosting classifier.
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

# Initialize the cross validation folds
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        # Select and predict on the test fold.
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_predictions.append(test_predictions)
    # Use a simple ensembling scheme -- just average the predictions to get the final classification.
    test_predictions = (full_predictions[0] + full_predictions[1]) / 2
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

# Put all the predictions together into one array.
predictions = np.concatenate(predictions, axis=0)

# Compute accuracy by comparing to the training data.
accuracy = sum(predictions == titanic["Survived"]) / len(predictions)
print('Ensemble accuracy:', accuracy)
print()

# Add titles to the test set.
titles = titanic_test["Name"].apply(get_title)
# Add the Dona title to the mapping, because it's in the test set, but not the training set
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9,
                 "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
for k,v in title_mapping.items():
    titles[titles == k] = v
titanic_test["Title"] = titles
# Check the counts of each unique title.
print(pandas.value_counts(titanic_test["Title"]))

# Add the family size column.
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test['Parch']

# Add family ids.
family_ids = titanic_test.apply(get_family_id, axis=1)
family_ids[titanic_test["FamilySize"] < 3] = -1
titanic_test["FamilyId"] = family_ids

# The .apply method generates a new series
titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))

# PREDICTION & SUBMISSION

predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

full_predictions = []
for alg, predictors in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(titanic[predictors], titanic["Survived"])
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
    full_predictions.append(predictions)

# The gradient boosting classifier generates better predictions, weight it higher.
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4

predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
predictions = predictions.astype(int)

submission = pandas.DataFrame({
    'PassengerId': titanic_test['PassengerId'],
    'Survived': predictions})

submission.to_csv("kaggle.csv", index=False)