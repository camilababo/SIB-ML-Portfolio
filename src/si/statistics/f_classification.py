from scipy.stats import stats

from si.data.dataset import Dataset


def f_classification(dataset: Dataset):
    """
    Calculates the F value for each feature.
    :param dataset: Dataset object.
    :return: F value for each feature.
    """
    dataset_classes = dataset.get_classes()
    dataset_groups = [dataset.x[dataset.y == c] for c in dataset_classes]  # group the dataset by class
    f_value, p_value = stats.f_oneway(*dataset_groups)

    return f_value, p_value

