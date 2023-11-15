import numpy as np
from sklearn import (datasets, tree, model_selection)

if __name__ == '__main__':
    # Load a dataset
    wdbc = datasets.load_breast_cancer()

    # Train a model
    model = tree.DecisionTreeClassifier(random_state=42) # TODO
    param = {
        'max_depth' : range(1, 101),
        'min_samples_leaf' : range(1, 101),
        'min_samples_split' : range(2,101),
        'criterion' : ['gini', 'entropy'],
        'max_leaf_nodes' : range(2, 101),
        'max_features' : ['sqrt', 'log2', None]
    }
    random_search = model_selection.RandomizedSearchCV(model, param_distributions=param, n_iter=10000, cv=5)
    random_search.fit(wdbc.data, wdbc.target)
    best_params = random_search.best_params_
    print(f"best_params : {best_params}")
    best_model = tree.DecisionTreeClassifier(random_state=42, **best_params)

    cv_results = model_selection.cross_validate(best_model, wdbc.data, wdbc.target, cv=5, return_train_score=True)

    # Evaluate the model
    acc_train = np.mean(cv_results['train_score'])
    acc_test = np.mean(cv_results['test_score'])
    print(f'* Accuracy @ training data: {acc_train:.3f}')
    print(f'* Accuracy @ test data: {acc_test:.3f}')
    print(f'* Your score: {max(10 + 100 * (acc_test - 0.9), 0):.0f}')