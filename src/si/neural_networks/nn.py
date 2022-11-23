from si.data.dataset import Dataset


class NN:
    def __init__(self, layers: list):
        # parameters
        self.layers = layers

    def fit(self, dataset: Dataset) -> "NN":
        """
        Trains the neural network.
        :param dataset: dataset to train the neural network
        :return: Returns the trained neural network.
        """
        x = dataset.x  # pointer to the input data, a more correct way would be to copy the data (x = dataset.x.copy())
        for layer in self.layers:
            x = layer.forward(x)  # if we were to use the dataset.x we would be using the original data, but we want
            # to use the data that was already processed by the previous layer

        return self

    # def predict(self, dataset: Dataset) -> np.ndarray:
    #     """
    #     Predicts the classes of the dataset.
    #     :param dataset: dataset to predict the classes
    #     :return: Returns the predicted classes.
    #     """

