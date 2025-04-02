from ucimlrepo import fetch_ucirepo
def load_data():

    credit = fetch_ucirepo(id=27)
    X = credit.data.features
    y = credit.data.targets

    # print(X.info())
    return X, y