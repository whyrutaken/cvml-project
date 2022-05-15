# %%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import imblearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def plot_roc_curve(model, x, y):
    plt.figure()
    RocCurveDisplay.from_estimator(model, x, y)
    plt.title('Receiver Operating Characteristic')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()


def print_confusion_matrix(x, y):
    predictions = model1.predict(x)
    cm = confusion_matrix(y, predictions)
    print(cm)
    TN, FP, FN, TP = confusion_matrix(y_val, predictions).ravel()
    print('True Positive(TP)  = ', TP)
    print('False Positive(FP) = ', FP)
    print('True Negative(TN)  = ', TN)
    print('False Negative(FN) = ', FN)
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    print(accuracy)


def predict_and_save_test(model, filepath="predictions.csv"):
    pred_test = pd.DataFrame(model.predict(x_test))
    pred_test.index = np.arange(1, len(pred_test) + 1)

    # print value counts of the classes
    print(pred_test[0].value_counts())

    # save to csv
    pred_test[0] = pred_test[0].astype(int)
    pred_test.to_csv(filepath, header=False, index=True)


## PREPROCESSING
# read in the datasets
train_data = pd.ExcelFile("Train_data.xls").parse(0, index_col=0)
test_data = pd.ExcelFile("Test_data.xls").parse(0, index_col=0)
val_data = pd.ExcelFile("Validation_data.xls").parse(0, index_col=0)


# plot value distributions of the MIG_group column
train_data.iloc[:,-1].value_counts().plot(kind='barh', title="MIG_group")
plt.savefig("imbalanced-train-data.png", format="png")

# sum and print NaN values of the data sets
print(train_data.isna().sum())
print(test_data.isna().sum())
print(val_data.isna().sum())

# split data sets into predictor and target values
x_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:,-1]
x_val, y_val = val_data.iloc[:, :-1], val_data.iloc[:, -1]

# oversampling
oversample = imblearn.over_sampling.RandomOverSampler(sampling_strategy=0.5)
x_train, y_train = oversample.fit_resample(x_train, y_train)

# fill NaN values
x_train = x_train.fillna(x_train.mean())
x_test = test_data.fillna(x_train.mean())
x_val = x_val.fillna(x_train.mean())





# %%

# Model1: PCA+Logreg
scaler = StandardScaler()
pca = PCA()
logreg = LogisticRegression()  # penalty='l2', solver='liblinear'
pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("logreg", logreg)])
# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {
    "pca__n_components": [1, 2, 5, 15],
}
model1 = GridSearchCV(pipe, param_grid, n_jobs=2)
model1.fit(x_train, y_train)

print("Best parameter (CV score=%0.3f):" % model1.best_score_)
print(model1.best_params_)


print_confusion_matrix(x_val, y_val)
metrics.plot_confusion_matrix(model1, x_val, y_val)
plt.show()
plot_roc_curve(model1, x_train, y_train)
predict_and_save_test(model1, "pca-logreg.csv")




#%%

# Model2: SVM
scaler = StandardScaler()
svm = SVC()
pipe = Pipeline(steps=[("scaler", scaler), ("svm", svm)])

# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {
    "svm__kernel": ["linear", "poly", "rbf"]
}
model2 = GridSearchCV(pipe, param_grid, n_jobs=2)
model2.fit(x_train, y_train)

print("Best parameter (CV score=%0.3f):" % model2.best_score_)
print(model2.best_params_)


print_confusion_matrix(x_val, y_val)
metrics.plot_confusion_matrix(model2, x_val, y_val)
plt.show()
plot_roc_curve(model2, x_val, y_val)
predict_and_save_test(model2, "svm.csv")




