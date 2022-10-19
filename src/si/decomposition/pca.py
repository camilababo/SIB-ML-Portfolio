import numpy as np

from si.data.dataset import Dataset


class PCA:
    def __init__(self, n_components: int = 2):
        """
        Initializes the PCA.
        :param n_components: Number of components to keep.
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def _get_components(self, dataset: Dataset) -> np.ndarray:
        """
        Calculates the components of the dataset.
        :param dataset:
        :return: A matrix with the components.
        """
        # Get mean and center data
        self.mean = np.mean(dataset.x, axis=0)  # axis=0 means that we want to calculate the mean for each column
        centered_data = dataset.x - self.mean

        # Get single value decomposition
        self.u_matrix, self.s_matrix, self.v_matrix_t = np.linalg.svd(centered_data, full_matrices=False)

        # Get principal components
        self.components = self.v_matrix_t[:, :self.n_components]  # get the first n_components columns

        return self.components

    def _get_explained_variance(self, dataset: Dataset) -> np.ndarray:
        """
        Calculates the explained variance.
        :param dataset: Dataset object.
        :return: A vector with the explained variance.
        """
        # Get explained variance
        ev_formula = self.s_matrix ** 2 / (len(dataset.x) - 1)
        explained_variance = ev_formula[:self.n_components]

        return explained_variance

    def fit(self, dataset: Dataset):
        """
        Calculates the mean, the components and the explained variance.
        :return: Dataset.
        """
        # Get mean and center data
        self.mean = np.mean(dataset.x, axis=0)  # axis=0 means that we want to calculate the mean for each column
        centered_data = dataset.x - self.mean

        # Get single value decomposition
        u_matrix, s_matrix, v_matrix_t = np.linalg.svd(centered_data, full_matrices=False)

        # Get principal components
        self.components = v_matrix_t[:, :self.n_components]  # get the first n_components columns

        self.components = self._get_components(dataset)
        self.explained_variance = self._get_explained_variance(dataset)

        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transforms the dataset.
        :return: Dataset object.
        """
        if self.components is None:
            raise Exception("You must fit the PCA before transform the dataset.")

        transformed_data = np.dot(dataset.x, self.components)  # multiplication between the dataset and the components

        return Dataset(transformed_data, dataset.y, dataset.features_names, dataset.label_name)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Calculates the mean, the components and the explained variance and transforms the dataset.
        :return: Dataset object.
        """
        self.fit(dataset)
        return self.transform(dataset)
