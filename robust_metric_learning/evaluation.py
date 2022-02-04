from datetime import datetime
import os
from sklearn.metrics import accuracy_score


def clf_predict_evaluate(X_test, y_test,
                         metrics_names, metric_name,
                         clf, errors_dict):
    y_pred = clf.predict(X_test)
    error = 1 - accuracy_score(y_test, y_pred)
    if metric_name not in metrics_names:
        metrics_names.append(metric_name)
        errors_dict[metric_name] = list()
    errors_dict[metric_name].append(error)


def create_directory(name):
    date_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    path = os.path.join('results', name, date_str)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path
