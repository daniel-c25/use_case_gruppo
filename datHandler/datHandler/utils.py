import pandas as pd

def features_extractor(df : pd.DataFrame, features : list[str]):

    '''
    Exctracts a set of features out of the specified DataFrame

    Parameters
    ----------
    :param df: The DataFrame from which we want to extract features.
    :param features: List of strings representing the feature names.

    Returns
    -------
    :return: Dictionary containing the feature names as keys and the feature values as values.

    See Also
    --------
    pandas.DataFrame

    Example:
    --------
    >>> data = {
    ... "age": [25, 30, 35, 40],
    ... "height": [170, 180, 175, 165],
    ... "weight": [70, 80, 75, 65]
    ...}
    >>> df = pd.DataFrame(data)
    >>> features = ["age", "height"]
    >>> extracted_features = features_extractor(df, features)

    >>> print(extracted_features)
    {'age': array([25, 30, 35, 40]), 'height': array([170, 180, 175, 165])}
    '''

    features_values = {}
    for feature in features:
        features_values[feature] = df[feature].values

    return features_values