from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

iris = load_iris()
iris = dict(iris)

iris_data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                         columns=iris['feature_names'] + ['target'])

file_name = f"Iris.csv"
iris_data.to_csv(file_name)