# Gaussian Naive Bayes
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

# load the iris datasets
dataset = datasets.load_iris()

print(len(dataset.data))

# fit a Naive Bayes model to the data
model = GaussianNB()
model.fit(dataset.data, dataset.target)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


