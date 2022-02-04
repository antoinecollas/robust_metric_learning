from datetime import datetime
import os


def create_directory(name):
    date_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    path = os.path.join('results', name, date_str)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path
