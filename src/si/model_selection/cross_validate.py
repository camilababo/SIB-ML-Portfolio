from typing import List, Callable, Dict, Union

import numpy as np
from traitlets import Float

from si.data.dataset import Dataset
from si.model_selection.split import train_test_split

Num = Union[int, float]


def cross_validate(model,
                   dataset: Dataset,
                   scoring: Callable = None,
                   cv: int = 3,
                   test_size: float = 0.3) \
        -> Dict[str, List[Num]]:
    """
    Computes the cross-validated score for the given model and dataset.

    :param model: The model to evaluate.
    :param dataset: The dataset to evaluate.
    :param scoring: The scoring function to use.
    :param cv: The number of folds to use.
    :param test_size: The proportion of the dataset to include in the test split.
    :returns: A dictionary with the cross-validated scores.
    """
    scores = {
        'seed': [],
        'train': [],
        'test': [],
        'parameters': []
    }

    # computes the score for each fold of the score
    for i in range(cv):
        # set the random seed
        random_state = np.random.randint(0, 1000)

        # store the seed
        scores['seed'].append(random_state)

        # splits the train and test
        train, test = train_test_split(dataset=dataset, test_size=test_size, random_state=random_state)

        # trains the model
        model.fit(train)

        # calculates the training score
        if scoring is None:

            # stores the train score
            scores['train'].append(model.score(train))

            # stores the test score
            scores['test'].append(model.score(test))

        else:

            # stores the train score
            scores['train'].append(scoring(train.y, model.score(train)))

            # stores the test score
            scores['test'].append(scoring(test.y, model.score(test)))

    return scores
