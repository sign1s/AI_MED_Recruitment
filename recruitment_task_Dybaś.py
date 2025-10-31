from scipy.spatial.distance import minkowski
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#importing data and scaling

#importing data from csv file, added separator ";", because headings were treated ass one word
#and decimal = ",", because some data were treated as string
data = pd.read_csv("task_data.csv", sep=";", decimal=",")

#selecting columns, which will be used to learn
X = data[[
    "Heart width", "Lung width", "CTR - Cardiothoracic Ratio", "xx", "yy",
    "xy", "normalized_diff", "Inscribed circle radius", "Polygon Area Ratio",
    "Heart perimeter", "Heart area ", "Lung area"
]]

#selecting target column
y = data["Cardiomegaly"]

#spliting data(80% to train and 20% for test)
#set random_state to 42, so it's always the same split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#creating StandardScaler object
scaler=StandardScaler()

#applying scaling to data, so they are on similar scale
X_scaled_train=scaler.fit_transform(X_train)
X_scaled_test=scaler.transform(X_test)

#training(KNN)

#defining pipeline that scale and model
pipe_knn = Pipeline(steps=[
                           ('scaler', StandardScaler()),
                           ("model", KNeighborsClassifier(
                            n_neighbors=3,      #number of neighbours used
                            weights='distance', #influence of neighbour on prediction
                            metric='manhattan', #look up diff distance formulas in documentation
    ))
])

#fit pipeline on training data
#learning
pipe_knn.fit(X_train, y_train)

cv_score = np.round(cross_val_score(pipe_knn, X_train, y_train,), 3)

print("KNN")
print(f"Scores of training:")
list(map(print, cv_score))
print(f"\nCross-validation score: {np.mean(cv_score):.3}")
print(f"Standard deviation of CV score: {np.std(cv_score):.3f}\n")

#GridSearchCV

#making combinations of parameters
#used to find best combination
param_grid = {
            "model__n_neighbors": [3, 5, 7, 9, 11, 15], #number of neighbours
            "model__weights": ["uniform", "distance"],  #their influence
            "model__metric":["minkowski","manhattan","euclidean","chebyshev"] # distance metrics
}

#spliting data into folds while keeping balance between them
rskf = RepeatedStratifiedKFold(n_splits=5, #number of folds
                               n_repeats=100, #n of repeats #for fast checking set repeats on smaller number (basic=100)
                               random_state=None) #random seed
#pipeline to prevent data leakage
pipe_knn = Pipeline(steps=[('scaler', StandardScaler()),("model",KNeighborsClassifier())])

#grid search for KNN model
grid_search = GridSearchCV(
    estimator=pipe_knn, #pipeline
    param_grid=param_grid,
    scoring="accuracy",
    cv=rskf,
    verbose=1,
    n_jobs=-1, #use all available CPU for processing
)

grid_search.fit(X_train, y_train)

#show the best results
print(f"GridSearchCV")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best accuracy(averged CV): {grid_search.best_score_:.4f}\n")

#now we check close surrounding of the best result
param_grid = {
    "model__n_neighbors": [6,7,8],
    "model__weights": ["uniform"],
    "model__metric":["manhattan"],
}

rskf = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=100,
    random_state=None
)
pipe_knn=Pipeline(steps=[
    ('scaler', StandardScaler()),
    ("model", KNeighborsClassifier())
])
grid_search = GridSearchCV(
    estimator=pipe_knn,
    param_grid=param_grid,
    scoring="accuracy",
    cv=rskf,
    verbose=1,
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

print(f"GridSearchCV(narrow)")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best accuracy(averged CV): {grid_search.best_score_:.4f}\n")

#checking optimized para.

#now we put optimized parameters into KNN for more accurate score
pipe_knn = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ("model",KNeighborsClassifier(
                n_neighbors=7,
                weights='uniform',
                metric='manhattan',
    ))
])

pipe_knn.fit(X_train, y_train)

cv_score = np.round(cross_val_score(pipe_knn, X_train, y_train,), 2)

print(f"Scores of training data cross-validation(each fold):")
list(map(print, cv_score))
print(f"\nCross-validation score: {np.mean(cv_score):.3}")
print(f"Standard deviation of CV score: {np.std(cv_score):.3f}\n")

#decision tree

#in Decision Tree we don't use scaling
clf_tree=DecisionTreeClassifier(
    max_depth=7, #limits of how much tree can grow
    criterion="log_loss", #measure quantity of split
    min_samples_split=7, #minimum numer of samples required to split
    min_samples_leaf=5, #minimum numer of samples required to be at a leaf node
    class_weight=None, #no weightening, everything is equal
)

clf_tree.fit(X_train, y_train)

cv_score = np.round(cross_val_score(clf_tree, X_train, y_train,), 2)

print("Decision tree")
print(f"Scores of training data cross-validation(each fold):")
list(map(print, cv_score))
print(f"\nCross-validation score: {np.mean(cv_score):.3}")
print(f"Standard deviation of CV score: {np.std(cv_score):.3f}\n")

#Support Vector Machine

pipe_svc = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ("model", SVC( #SCV - support vector classifier
        kernel="rbf",
        C=3, #regularization strength
        gamma="scale", #scale adapts to data variance
        class_weight=None,
        probability=True, #used for RoC
    ))
])

pipe_svc.fit(X_train, y_train)

cv_score = np.round(cross_val_score(pipe_svc, X_train, y_train,), 2)

print("SVM")
print(f"Scores of training data cross-validation(each fold):")
list(map(print, cv_score))
print(f"\nCross-validation score: {np.mean(cv_score):.3}")
print(f"Standard deviation of CV score: {np.std(cv_score):.3f}\n")

#logistic regression

pipe_log = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ("model", LogisticRegression(
        C=1, #inverted strength{C=3} higher = weaker regularization
        penalty="l1", #drives some coefficients exactly zero
        solver="liblinear", #solver compatible with L1 penalty
        max_iter=1000, #ensure convergence
        class_weight=None # treat classes equally
    ))
])

pipe_log.fit(X_train, y_train)

cv_score = np.round(cross_val_score(pipe_log, X_train, y_train,), 3)

print("logic regression")
print(f"Scores of training data cross-validation(each fold):")
list(map(print, cv_score))
print(f"\nCross-validation score: {np.mean(cv_score):.3}")
print(f"Standard deviation of CV score: {np.std(cv_score):.3f}\n")

#Random forest Classifier

clf_rf = RandomForestClassifier(
    max_depth=6, #limit of depth of each tree
    min_samples_split=6,
    n_estimators=125, #number of trees
    min_samples_leaf=2,
    max_features='sqrt', #number of features to consider when looking for the beset split
    class_weight=None
)

clf_rf.fit(X_train, y_train)

cv_score = np.round(cross_val_score(clf_rf, X_train, y_train,), 3)

print("RFC")
print(f"Scores of training data cross-validation(each fold):")
list(map(print, cv_score))
print(f"\nCross-validation score: {np.mean(cv_score):.3}")
print(f"Standard deviation of CV score: {np.std(cv_score):.3f}\n")

#model on test data

#now I'm checking trained models on test data
y_pred_knn = pipe_knn.predict(X_test)
y_pred_svc = pipe_svc.predict(X_test)
y_pred_log = pipe_log.predict(X_test)
y_pred_tree = clf_tree.predict(X_test)
y_pred_rf = clf_rf.predict(X_test)

acc_knn = accuracy_score(y_test, y_pred_knn)
acc_svc = accuracy_score(y_test, y_pred_svc)
acc_log = accuracy_score(y_test, y_pred_log)
acc_tree = accuracy_score(y_test, y_pred_tree)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"accuracy on test set:")
print(f"- Accuracy of KNN model on test set: {acc_knn:.4f}")
print(f"- Accuracy of SVM model on test set: {acc_svc:.4f}")
print(f"- Accuracy of Logistic Regression model on test set: {acc_log:.4f}")
print(f"- Accuracy of Decision tree model on test set: {acc_tree:.4f}")
print(f"- Accuracy of Random Forest model on test set: {acc_rf:.4f}")

#crating RoC curves

models = {
    "KNN": pipe_knn,
    "SVM": pipe_svc,
    "Logistic Regression": pipe_log,
    "Decision Tree": clf_tree,
    "Random Forest": clf_rf
}

plt.figure(figsize=(10, 7))

for name, model in models.items():
    probs = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

# Baseline
plt.plot([0, 1], [0, 1], linestyle='--')

plt.title("ROC Curves for Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

#matrix, precision, recall, f1 score

models = {
    "KNN": y_pred_knn,
    "SVM": y_pred_svc,
    "Logistic Regression": y_pred_log,
    "Decision Tree": y_pred_tree,
    "Random Forest": y_pred_rf
}

for name, preds in models.items():
    print(f"\n==== {name} ====")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds, digits=3))

