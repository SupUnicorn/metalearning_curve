# coding: utf-8

import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split, KFold


def key_value_mapping(key, value):
    key_value_dict = {
        'target_type': {'Categorical': 0.0, 'Binary': 1.0, 'Numerical': 2.0},
        'task': {'regression': 0.0, 'multilabel.classification': 1.0, 'multiclass.classification': 2.0,
                 'binary.classification': 3.0},
        'feat_type': {'Mixed': 0.0, 'Binary': 1.0, 'Numerical': 2.0, 'Categorical': 3.0},
        'metric': {'auc_metric': 0.0, 'f1_metric': 1.0, 'bac_metric': 2.0, 'r2_metric': 3.0, 'a_metric': 4.0,
                   'pac_metric': 5.0}
    }

    if key in key_value_dict:
        value = key_value_dict[key][value]
    return float(value)


class Agent():
    """
    RANDOM SEARCH AGENT
    """

    def __init__(self, number_of_algorithms):
        """
        Initialize the agent

        Parameters
        ----------
        number_of_algorithms : int
            The number of algorithms

        """
        self.nA = number_of_algorithms
        self.algo_name_list = [str(x) for x in range(number_of_algorithms)]
        self.data_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        self.ds_feat_keys = [
            'target_type',
            'task',
            'feat_type',
            'metric',
            'feat_num',
            'target_num', 'label_num',
            'train_num',
            'valid_num', 'test_num',
            'has_categorical', 'has_missing',
            'is_sparse',
            'time_budget'
        ]

        self.model_nums = 5

        self.model = [RandomForestRegressor(n_estimators=50) for _ in range(self.model_nums)]
        self.model1 = RandomForestClassifier(n_estimators=50)
        self.voting = None

        self.hp_keys1 = ['meta_feature_0', 'meta_feature_1', 'meta_feature_2']

    def _process_curve(self, scores, data_sizes, times):
        scores = [max(0.0, float(x)) for x in scores]

        if not scores:
            scores = [0.0]
            data_sizes = [0.0]
            times = [0.0]

        scores = scores + [scores[-1]]
        data_sizes = data_sizes + [1.1]
        times = times + [1.1 * times[-1]]
        new_scores = []
        new_times = []
        for s in self.data_sizes:
            if s < data_sizes[0]:
                new_s = 0.0
                new_t = 0.0
            else:
                j = 0
                while j < len(data_sizes) - 1:
                    if data_sizes[j] <= s < data_sizes[j + 1]:
                        break
                    j += 1
                delta = (s - data_sizes[j]) / (data_sizes[j + 1] - data_sizes[j])
                new_s = scores[j] + (scores[j + 1] - scores[j]) * delta
                new_s = round(new_s, 4)
                new_t = times[j] + (times[j + 1] - times[j]) * delta
                new_t = round(new_t, 4)
            new_scores.append(new_s)
            new_times.append(new_t / 1000)
        return new_scores, new_times

    def get_times_class(self, times):
        ts = []
        for i in times:
            if i < 0.1:
                t = 0
            elif 0.1 <= i < 0.25:
                t = 1
            elif 0.25 <= i < 0.5:
                t = 2
            else:
                t = 3
            ts.append(t)
        return ts

    def _get_dataset_vector(self, dataset_meta_features):
        values = []
        for k in self.ds_feat_keys:
            v = key_value_mapping(k, dataset_meta_features[k]) / self.ds_feat_max_values[k]
            values.append(round(v, 6))
        return values

    def reset(self, dataset_meta_features, algorithms_meta_features):

        self.dataset_meta_features = dataset_meta_features
        self.algorithms_meta_features = algorithms_meta_features

        self.validation_last_scores = [0.0 for _ in self.algo_name_list]
        self.train_last_scores = [0.0 for _ in self.algo_name_list]
        self.total_budget = float(dataset_meta_features['time_budget'])
        self.time_used = 0.0

        self.current_dataset_meta_features = dataset_meta_features

        self.last_delta_t = 0.1
        self.last_index = 0

        ds_vector = self._get_dataset_vector(dataset_meta_features)

        scores_list = []
        times_list = []
        for algo_name in self.algo_name_list:
            algo_vector1 = [
                float(algorithms_meta_features[algo_name][k]) if (k in algorithms_meta_features[algo_name]) else 0.0 for
                k in self.hp_keys1]
            pdata = []
            for sizes in self.data_sizes:
                pdata.append(ds_vector + algo_vector1 + [float(algo_name)] + [sizes])

            scores = self.voting.predict(pdata)
            scores_list.append(scores.tolist())
            times = self.model1.predict(pdata)
            times_list.append(times)

        self.scores_list = scores_list
        self.times_list = times_list

        self.best_algo_list = []
        for i in range(len(self.data_sizes)):
            scores = [x[i] for x in scores_list]
            best_idx = np.argmax(scores)
            self.best_algo_list.append(best_idx)

        self.best_algo_list += [0, 10]
        self.best_algo_list = list(set(self.best_algo_list))

        print('===> reset, best algo', dataset_meta_features['name'], self.best_algo_list)

        self.best_algo_scores = [self.scores_list[i] for i in self.best_algo_list]

        self.best_algo_runtime_sum = [sum(list(self.times_list[i])) for i in self.best_algo_list]

        algo_list = [(self.best_algo_list[i], self.best_algo_runtime_sum[i]) for i in range(len(self.best_algo_list))]
        algo_list = sorted(algo_list, key=lambda x: x[1])
        self.best_algo_list = [a[0] for a in algo_list]

        self.best_algo_runtime = np.array([self.times_list[i] for i in self.best_algo_list])

        self.delta_index_list = [0 for _ in self.algo_name_list]
        self.delta_index_list2 = [0 for _ in self.algo_name_list]
        self.algo_index = 0
        self.step = 0

    def meta_train(self, datasets_meta_features, algorithms_meta_features, train_learning_curves,
                   validation_learning_curves, test_learning_curves):

        self.train_learning_curves = train_learning_curves
        self.validation_learning_curves = validation_learning_curves
        self.test_learning_curves = test_learning_curves
        self.datasets_meta_features = datasets_meta_features
        self.algorithms_meta_features = algorithms_meta_features

        algo_hp_value_dict1 = {}
        for algo_name in self.algo_name_list:
            algo_hp_value_dict1[algo_name] = [float(algorithms_meta_features[algo_name][k]) if (
                    k in algorithms_meta_features[algo_name]) else 0.0 for k in self.hp_keys1] + [float(algo_name)]

        self.ds_feat_max_values = {key: 1.0 for key in self.ds_feat_keys}
        for _, ds in datasets_meta_features.items():
            for key in self.ds_feat_keys:
                v = key_value_mapping(key, ds[key])
                self.ds_feat_max_values[key] = max(self.ds_feat_max_values[key], v)

        ds_vec_dict = {}
        for key, ds in datasets_meta_features.items():
            ds_vec_dict[key] = self._get_dataset_vector(ds)

        inputs_list1 = []
        score_list = []
        ds_name_list = []
        times_list = []
        for key, ds in validation_learning_curves.items():
            ds_vector = ds_vec_dict[key]
            ds_name_list.append(datasets_meta_features[key]['name'])
            for algo_name in self.algo_name_list:
                for sizes in self.data_sizes:
                    inputs_list1.append(ds_vector + algo_hp_value_dict1[algo_name] + [sizes])
                new_scores, new_times = self._process_curve(ds[algo_name].scores,
                                                            list(ds[algo_name].training_data_sizes),
                                                            list(ds[algo_name].times))
                score_list += new_scores

                times_list += self.get_times_class(new_times)

        x1_data = np.array(inputs_list1)
        y1_data = np.array(score_list)
        y2_data = np.array(times_list)

        kf = KFold(n_splits=self.model_nums, shuffle=False)
        for i, d in enumerate(kf.split(x1_data, y1_data)):
            train_index = list(d[0])
            self.model[i].fit(x1_data[train_index], y1_data[train_index])

        self.voting = VotingRegressor(estimators=[("RF_" + str(i), self.model[i]) for i in range(self.model_nums)])
        self.voting.fit(x1_data, y1_data)

        self.model1.fit(x1_data, y2_data)

    def suggest(self, observation):
        """
        Return a new suggestion based on the observation

        Parameters
        ----------
        observation : tuple of (int, float, float, float, float)
            An observation containing: (A, p, t, R_train_A_p, R_validation_A_p)
                1) A: index of the algorithm provided in the previous action,
                2) p: decimal fraction of training data used, with value of p in [0.1, 0.2, 0.3, ..., 1.0]
                3) t: amount of time it took to train A with training data size of p,
                      and make predictions on the training/validation/test sets.
                4) R_train_A_p: performance score on the training set
                5) R_validation_A_p: performance score on the validation set

        Returns
        ----------
        action : tuple of (int, float)
            The suggested action consisting of 2 things:
                (2) A: index of the algorithm to be trained and tested
                (3) p: decimal fraction of training data used, with value of p in [0.1, 0.2, 0.3, ..., 1.0]

        Examples
        ----------
        >>> action = agent.suggest((9, 0.5, 151.73, 0.9, 0.6))
        >>> action
        (9, 0.9)
        """

        # Get observation
        t = 0.0
        p = 0.1
        if observation != None:
            A, p, t, R_train_A_p, R_validation_A_p = observation
            self.train_last_scores[A] = max(self.train_last_scores[A], R_validation_A_p)
            self.validation_last_scores[A] = max(self.validation_last_scores[A], R_validation_A_p)

        best_algo_for_test = np.argmax(self.validation_last_scores)

        new_budget = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        if self.time_used < self.total_budget // 4 or max(self.validation_last_scores) <= 0.1:
            if self.algo_index >= len(self.best_algo_list):
                self.algo_index = 0

            next_algo_to_reveal = self.best_algo_list[self.algo_index]
            delta_index = self.delta_index_list[next_algo_to_reveal]
            if delta_index + 1 < len(new_budget):
                p = new_budget[delta_index]
                self.delta_index_list[next_algo_to_reveal] += 1
            else:
                p = 0.1

            self.algo_index += 1
        else:
            next_algo_to_reveal = best_algo_for_test
            p = new_budget[self.scores_list[next_algo_to_reveal].index(max(self.scores_list[next_algo_to_reveal]))]

        self.time_used += t
        self.step += 1

        action = (next_algo_to_reveal, p)

        return action
