# Horse-Racing-True-Odds-Calculator
This repository contains the codebase and sample data that ingests horse racing data and takes away the overround and gives the true odds.

## Model Selection
Due to the noise and non-linearity of the data, I wanted to use a tree based model. I decided
on an XGBoost model, due to their regularisation methods on tree weights to prevent
overfitting when using relatively high dimensional features, the optimised speed and scope
for scalability, and the integration with SciKit-learn’s calibrated classifier (to offset tree
models natively poor probability calculations).


## Assumptions

I assumed there was no cross race leakage in the feature, and that each feature was
created independently. For example, I assume the normalised position last time out is
untraceable between races.


## Features and Feature Engineering

In regards to features of the data, there is a lot going on. Initially, I tried using a few feature
building packages, such as ‘autofeat’ and ‘PolynomialFeatures’
, followed by a Mutual
Information search to determine feature importance. This did create some interesting
features, but I chose to focus on creating my own metrics. As the brief mentioned focusing
on race-relative features, I made features such as ‘DistanceForm
Rank’ and ‘JockeyRating_ Rank’. These proved to be both significant in the model, and are great to
include to increase the explainability of the model’s features - nothing too obscure.
I created a function that produced around 30 features, but this section of the task alone
could have a week dedicated to it and I would like to continue looking into.


## Model Setup

The model itself I wrapped into a single class, mainly for simplicity at the moment. The class
contains the following methods:

● Load Data - loads, cleans and splits the data by the race ID, to ensure the training
and validation sets are grouped by whole races. Drops any leakage columns and
performs the feature engineering.

● Fit - fits the training data, uses a randomised search cross-validation method, and
calibrates the results.

● Predict - Either predicts on the validation set of the training data, or takes a
dataframe as an input to predict results, ensures raw probabilities are transformed
and returns a dataframe with predicted probabilities.

● Evaluate vs. market - This provides metrics such as log loss, brier score and
accuracy vs. the market, as well as the MAE compared with the market's
probabilities.

● Meta methods - I use a regression model to try and make the probability curve
slightly more accurate to reality - mainly by not overestimating winners and
underestimating longshots. This does use the betfairSP odds, so it may not fit the
criteria to use on the test set, but given general odds are available before a race -
this could be an extremely useful tool. The accuracy of predicting the winner after
meta calibration was typically ~87%.

● Saving results - saves the results of the fit/predict to a JSON.
The model and features were performing relatively well. It was typically outperforming the
market odds at giving the highest odds to the true winner by a few percent - 32-35% vs.
28-31%. The log loss and brier score were roughly similar to that of the market too:

Model Log Loss: 0.2887

Market Log Loss: 0.2871

Model Brier: 0.0828

Market Brier: 0.0832

This is before the meta calibration.

The MAE for a given run generally fell between 5.5-6%.


## Probabilities

To ensure that the probabilities summed to 1 for each race, I took the raw win possibilities,
which represent the model’s estimate of each horse's chance to win, independently, without
context of other runners. I clipped extreme probabilities to ensure no 0 errors, then simply
normalised the results - x: x / x.sum(). I tried to use softmaxing, however this had a tendency
to over smooth lower predictions and give one extreme result. This could possibly be
tweaked to work better going forward.

## Model Limitation

The model uses a classifier with the target of finishing position being 1, and using the
probability function to give the models’ confidence in its answer. This does result in the
outputs being slightly away from the true probabilities, but I am confident with additional
feature engineering and calibration, this can be resolved.


## Challenges

The main challenges I faced were probably to do with scope of feature engineering. I’d have
liked to spend more time mapping out all the potential avenues to explore.
I ran into some issues with computational load when running the feature subset testing -
ideally I would have run this with a GPU and iterated through all the possibilities.


## Further Development

● I had considered creating a neural network to see if a deep learning model could
outperform the gradient boosting model, but that would require considerably more
data. If we compounded the last 10-20 years of racing, this would be possible -
although the results then become a lot less interpretable if being deployed to users.

● Better Feature Engineering - This could include incorporating temporal fields, such as
horse form trajectory or jockey win streaks; rolling statistics over last 3 races etc.

● Splitting the data down and having more specialised models for certain races, for
example have a model that looks specifically at races given a certain grounding or
age category.

● Other horse racing datasets include information about the weight and handicap
weight etc. I think this would further improve model accuracy.
