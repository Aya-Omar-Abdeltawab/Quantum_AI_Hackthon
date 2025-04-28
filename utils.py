import pandas as pd
from sklearn.decomposition import PCA
from qiskit_machine_learning.algorithms.classifiers import VQC


# Preprocessing Functions
def load_and_preprocess_data(dataset_path):
    df = pd.read_csv(dataset_path)
    features, labels = df.iloc[:, :-1], df.iloc[:, [-1]]
    labels = labels.to_numpy().ravel()
    features = PCA(n_components=3).fit_transform(features)
    return features, labels

# API Function
def api(model_path, features, labels):
    model = VQC.load(model_path)
    score = model.score(features, labels)
    return score


dataset_path = "C:\\Users\\mozek\Desktop\\final\\test.csv"
model_path = "C:\\Users\\mozek\\Desktop\\final\\v4.model"

features, labels = load_and_preprocess_data(dataset_path)
score = api(model_path, features, labels)
print(score)