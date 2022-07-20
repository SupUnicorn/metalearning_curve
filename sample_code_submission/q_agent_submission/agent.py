import numpy as np
import random

random.seed(2301)
np.random.seed(795118)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class Agent():
    """
    Class that models a reinforcement learning agent.
    """

    def __init__(self, number_of_algorithms, epsilon=0.2, alpha=0.01, gamma=0.7):

        self.n_actions = number_of_algorithms
        # 20 algoritmi
        self.time_portions = [0.01, 0.0127, 0.0162, 0.0206, 0.0263, 0.0335, 0.0428, 0.0545, 0.0695, 0.0885, 0.1128,
                              0.1438, 0.1832, 0.2335, 0.2976, 0.3792, 0.4, 0.4832, 0.5, 0.6158, 0.7, 0.7847, 0.85, 0.9,
                              0.97, 1.02]
        self.time_portions = [self.time_portions[i] - 0.01 for i in range(len(self.time_portions))]
        self.p_portions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        # self.Q = np.random.rand(len(self.time_portions)-1, self.n_actions)

    def reset(self, dataset_meta_features, algorithms_meta_features):
        """
        Reset the agents' memory for a new dataset
        Parameters
        ----------
        dataset_meta_features : dict of {str : str}
            The meta-features of the dataset at hand, including:
                'usage' : name of the competition
                'name' : name of the dataset
                'task' : type of the task
                'target_type' : target type
                'feat_type' : feature type
                'metric' : evaluatuon metric used
                'time_budget' : time budget for training and testing
                'feat_num' : number of features
                'target_num' : number of targets
                'label_num' : number of labels
                'train_num' : number of training examples
                'valid_num' : number of validation examples
                'test_num' : number of test examples
                'has_categorical' : presence or absence of categorical variables
                'has_missing' : presence or absence of missing values
                'is_sparse' : full matrices or sparse matrices
        algorithms_meta_features : dict of dict of {str : str}
            The meta_features of all algorithms
        Examples
        ----------
        >>> dataset_meta_features
        {'usage': 'Meta-learningchallenge2022', 'name': 'Erik', 'task': 'regression',
        'target_type': 'Binary', 'feat_type': 'Mixed', 'metric': 'f1_metric',
        'time_budget': '600', 'feat_num': '9', 'target_num': '6', 'label_num': '10',
        'train_num': '17', 'valid_num': '87', 'test_num': '72', 'has_categorical': '1',
        'has_missing': '0', 'is_sparse': '1'}
        >>> algorithms_meta_features
        {'0': {'meta_feature_0': '0', 'meta_feature_1': '0.1'},
         '1': {'meta_feature_0': '1', 'meta_feature_1': '0.2'},
         '2': {'meta_feature_0': '0', 'meta_feature_1': '0.3'},
         '3': {'meta_feature_0': '1', 'meta_feature_1': '0.4'},
         ...
         '18': {'meta_feature_0': '1', 'meta_feature_1': '0.9'},
         '19': {'meta_feature_0': '0', 'meta_feature_1': '1.0'},
         }
        """
        self.dataset_metadata = dataset_meta_features
        self.algorithms_metadata = algorithms_meta_features
        self.validation_last_scores = [0.0 for i in range(self.n_actions)]
        self.validation_time_seen = [0.0 for i in range(self.n_actions)]

        self.time_used = 1
        self.used_times = 0
        self.time_budget = float(dataset_meta_features['time_budget'])
        self.used_times = 0
        if self.time_budget not in self.time_budgets_state:
            distances = [abs(self.time_budget - self.time_budgets_state[i]) for i in
                         range(len(self.time_budgets_state))]
            self.time_budget_position = np.argmin(distances)
        else:
            self.time_budget_position = self.time_budgets_state.index(float(self.time_budget))
        ds_features = np.array(self._ds_to_vec(dataset_meta_features, self.ordered_features)).reshape(1, -1)
        ds_features = self.scaler.transform(ds_features)
        self.cluster_label = self.cluster.predict(ds_features)

    def reset_for_train(self, dataset_meta_features, algorithms_meta_features):
        """
        Reset the agents' memory for a new dataset
        Parameters
        ----------
        dataset_meta_features : dict of {str : str}
            The meta-features of the dataset at hand, including:
                'usage' : name of the competition
                'name' : name of the dataset
                'task' : type of the task
                'target_type' : target type
                'feat_type' : feature type
                'metric' : evaluatuon metric used
                'time_budget' : time budget for training and testing
                'feat_num' : number of features
                'target_num' : number of targets
                'label_num' : number of labels
                'train_num' : number of training examples
                'valid_num' : number of validation examples
                'test_num' : number of test examples
                'has_categorical' : presence or absence of categorical variables
                'has_missing' : presence or absence of missing values
                'is_sparse' : full matrices or sparse matrices
        algorithms_meta_features : dict of dict of {str : str}
            The meta_features of all algorithms
        Examples
        ----------
        >>> dataset_meta_features
        {'usage': 'Meta-learningchallenge2022', 'name': 'Erik', 'task': 'regression',
        'target_type': 'Binary', 'feat_type': 'Mixed', 'metric': 'f1_metric',
        'time_budget': '600', 'feat_num': '9', 'target_num': '6', 'label_num': '10',
        'train_num': '17', 'valid_num': '87', 'test_num': '72', 'has_categorical': '1',
        'has_missing': '0', 'is_sparse': '1'}
        >>> algorithms_meta_features
        {'0': {'meta_feature_0': '0', 'meta_feature_1': '0.1'},
         '1': {'meta_feature_0': '1', 'meta_feature_1': '0.2'},
         '2': {'meta_feature_0': '0', 'meta_feature_1': '0.3'},
         '3': {'meta_feature_0': '1', 'meta_feature_1': '0.4'},
         ...
         '18': {'meta_feature_0': '1', 'meta_feature_1': '0.9'},
         '19': {'meta_feature_0': '0', 'meta_feature_1': '1.0'},
         }
        """
        self.dataset_metadata = dataset_meta_features
        self.algorithms_metadata = algorithms_meta_features
        self.validation_last_scores = [0.0 for i in range(self.n_actions)]
        self.validation_time_seen = [0.0 for i in range(self.n_actions)]
        # self.Q = np.random.rand(len(self.time_portions) - 1, self.n_actions)
        self.time_used = 1
        self.used_times = 0
        self.time_budget = float(dataset_meta_features['time_budget'])
        if self.time_budget not in self.time_budgets_state:
            distances = [abs(self.time_budget - self.time_budgets_state[i]) for i in
                         range(len(self.time_budgets_state))]
            self.time_budget_position = np.argmin(distances)
        else:
            self.time_budget_position = self.time_budgets_state.index(float(self.time_budget))

    def meta_train(self, datasets_meta_features, algorithms_meta_features, train_learning_curves,
                   validation_learning_curves, test_learning_curves):

        self.train_datasets_ids = [k for k in test_learning_curves][:25]
        self.ordered_features = self._ds_ordered(datasets_meta_features[random.choice(self.train_datasets_ids)])
        self.cluster = self.kmeans_clustering(datasets_meta_features)
        self.cluster_labels = self.cluster.labels_
        self.time_budgets_state = []
        for ds in datasets_meta_features:
            if float(datasets_meta_features[ds]['time_budget']) not in self.time_budgets_state:
                self.time_budgets_state.append(float(datasets_meta_features[ds]['time_budget']))

        self.Q = np.random.rand(16, len(self.p_portions) - 1, self.n_actions)
        maxit = 3000
        for iteration in range(maxit):
            for idx, episode in enumerate(self.train_datasets_ids):
                self.dataset_num = episode
                self.counters = {i: 0.0 for i in range(
                    self.n_actions)}  # Counters keeping track of the time has been spent for each algorithm
                dataset_meta_features = datasets_meta_features[episode]
                self.cluster_label = self.cluster_labels[idx]
                self.total_time_budget = float(dataset_meta_features['time_budget'])
                self.remaining_time_budget = self.total_time_budget
                self.list_algorithms = [k for k in test_learning_curves[episode].keys()]

                self.reset_for_train(dataset_meta_features, algorithms_meta_features)
                # print(
                #       "\n#===================== Start META-TRAINING on dataset: " + episode + " =====================#")
                # print( "\n#---Dataset meta-features = " + str(datasets_meta_features[episode]))
                # print( "\n#---Algorithms meta-features = " + str(algorithms_meta_features))
                observation = None
                for it in range(len(self.p_portions) - 1):
                    # === Get the agent's suggestion
                    (next_algo, p) = self.suggest_for_train(observation)
                    action = (next_algo, p)
                    times = test_learning_curves[episode][self.list_algorithms[next_algo]].times
                    train_scores = train_learning_curves[episode][self.list_algorithms[next_algo]].scores
                    test_scores = test_learning_curves[episode][self.list_algorithms[next_algo]].scores
                    ps = test_learning_curves[episode][self.list_algorithms[next_algo]].training_data_sizes
                    new_train_scores, new_times = self._process_curve(train_scores, ps, times)
                    new_test_scores, new_times1 = self._process_curve(test_scores, ps, times)
                    self.times = new_times
                    self.test_scores = new_test_scores
                    self.train_scores = new_train_scores
                    self.ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                    R_train_A_p, R_validation_A_p, t = self.get_last_point_within_p(p, self.validation_time_seen[
                        next_algo])
                    self.validation_time_seen[next_algo] = next_algo
                    self.used_times += t
                    observation = (next_algo, p, self.used_times, R_train_A_p, R_validation_A_p)
                    # if self.used_times > self.time_budget:
                    #     break

                    # print( "------------------")
                    # print( "A_star = " + str(action[0]))
                    # print( "A = " + str(action[1]))
                    # print( "delta_t = " + str(action[2]))
                    # print( "remaining_time_budget = " + str((1.01-self.time_portions[self.time_used-1])*self.time_budget))
                    # print( "observation = " + str(observation))
                # print( "[+]Finished META-TRAINING phase")
        #   #self.reset(datasets_meta_features[episode], algorithms_meta_features)

    def suggest_for_train(self, observation):

        next_algo_to_reveal = self.get_action_eps_greedy(self.cluster_label, self.time_used - 1)

        p_suggest = self.p_portions[self.time_used]

        if observation == None:
            best_algo_for_test = None
            self.time_used += 1
            self.old_score = 0
        else:
            A, p, t, R_train_A_p, R_validation_A_p = observation
            self.validation_last_scores[A] = R_validation_A_p
            self.validation_time_seen[A] = A
            weight = ((1.01 - self.time_portions[self.time_used]) * self.time_budget)
            reward = (np.max([R_validation_A_p, self.old_score]) - self.old_score)
            # self.time_used += 1
            self.update_Q(old_state=[self.cluster_label, self.time_used - 2], action=A, reward=reward,
                          new_state=[self.cluster_label, self.time_used - 1])
            if t > self.time_budget * 0.25:
                self.time_used += 1
            best_algo_for_test = np.argmax(self.validation_last_scores)
            self.old_score = np.max([R_validation_A_p, self.old_score])

        self.best_algo_for_test = best_algo_for_test

        action = (next_algo_to_reveal, p_suggest)
        return action

    def suggest(self, observation):

        next_algo_to_reveal = self.get_action_eps_greedy(self.cluster_label, self.time_used - 1)

        p_suggest = self.p_portions[self.time_used]

        if observation == None:
            best_algo_for_test = None
            self.time_used += 1
            self.old_score = 0
        else:
            A, p, t, R_train_A_p, R_validation_A_p = observation
            self.validation_last_scores[A] = R_validation_A_p
            self.validation_time_seen[A] = A
            weight = ((1.01 - self.time_portions[self.time_used]))
            reward = (np.max([R_validation_A_p, self.old_score]) - self.old_score)
            # self.time_used += 1
            if t > self.time_budget * 0.25:
                self.time_used += 1
            best_algo_for_test = np.argmax(self.validation_last_scores)
            self.old_score = np.max([R_validation_A_p, self.old_score])

        self.best_algo_for_test = best_algo_for_test
        action = (next_algo_to_reveal, p_suggest)
        return action

    def get_action_eps_greedy(self, r, c):
        """
        Epsilon-greedy sampling of next action given the current state.
        Parameters
        ----------
        r: int
          Current `y` position in the labyrinth
        c: int
          Current `x` position in the labyrinth
        Returns
        -------
        action: int
          Action sampled according to epsilon-greedy policy.
        """
        eps = random.random()
        if eps < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return self.Q[r, c].argmax()

    def get_action_greedy(self, r, c):
        """
        Greedy sampling of next action given the current state.
        Parameters
        ----------
        r: int
          Current `y` position in the labyrinth
        c: int
          Current `x` position in the labyrinth
        Returns
        -------
        action: int
          Action sampled according to greedy policy.
        """
        return self.Q[r, c].argmax()

    def update_Q(self, old_state, action, reward, new_state):
        """
        Update action-value function Q
        Parameters
        ----------
        old_state: tuple
          Previous state of the Environment
        action: int
          Action performed to go from `old_state` to `new_state`
        reward: int
          Reward got after action `action`
        new_state: tuple
          Next state of the Environment
        Returns
        -------
        None
        """

        self.Q[old_state[0], old_state[1], action] = \
            self.Q[old_state[0], old_state[1], action] + \
            self.alpha * (reward + self.gamma * self.Q[new_state[0], new_state[1]].max() - \
                          self.Q[old_state[0], old_state[1], action])

    def _process_curve(self, scores, data_sizes, times):
        scores = list(scores)
        data_sizes = list(data_sizes)
        times = list(times)
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
        for s in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
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

    def get_last_point_within_p(self, p, C):
        """
        Return the last achievable point on the learning curve given the allocated time budget delta_t
        Parameters
        ----------
        delta_t : float
            Allocated time budget given by the agent.
        C : float
            The timestamp of the last point on the learning curve (x-coordinate of current position on the learning curve)
        Returns
        ----------
        score : float
            The last achievable score within delta_t
        timestamp : float
            The timestamp associated with the last achievable score
        Examples
        ----------
        """

        temp_time = C + (self.times[list(self.ps).index(p)]) * self.time_budget

        for i in range(len(self.times)):
            if temp_time < self.times[i]:
                if i == 0:  # if delta_t is not enough to get the first point, the agent wasted it for nothing!
                    train_score, test_score, timestamp = 0.0, 0.0, 0.0
                else:  # return the last achievable point
                    train_score, test_score, timestamp = self.train_scores[i - 1], self.test_scores[i - 1], \
                                                         self.times[i - 1]
                return train_score, test_score, timestamp * self.time_budget

        # If the last point on the learning curve is already reached, return it
        train_score, test_score, timestamp, p = self.train_scores[-1], self.test_scores[-1], self.times[-1], self.ps[-1]
        return train_score, test_score, timestamp * self.time_budget

    def _ds_ordered(self, ds):
        self.ordered_features = []
        for k in ds.keys():
            self.ordered_features.append(k)
        return self.ordered_features

    def kmeans_dist(self, ds_vec1, ds_vec2):
        dist = 0
        for i in range(len(ds_vec1)):
            if ds_vec1[i] == ds_vec2[i]:
                dist += 1
        return dist

    def _ds_to_vec(self, ds, ordered_features):

        conver = {
            "usage": None,
            "name": None,
            "task": {
                "binary.classification": '0',
                "multiclass.classification": '1',
                "multilabel.classification": '2',
                "regression": '3'
            },
            "target_type": {
                "Binary": '0',
                "Categorical": '1',
                "Numerical": '2',
            },
            "feat_type": {
                "Binary": '0',
                "Categorical": '1',
                "Numerical": '2',
                "Mixed": '3',
            },
            "metric": {
                "bac_metric": '0',
                "auc_metric": '1',
                "f1_metric": '2',
                "pac_metric": '3',
                "abs_metric": '4',
                "r2_metric": '5',
                'a_metric': '6',
            },
        }

        dv = []
        for keys in ordered_features:
            for k, v in ds.items():
                cd = conver.get(k, {})
                if cd is None:
                    continue

                item = cd.get(v, None)
                if item is None:
                    item = ds[k]
                if k == keys:
                    dv.append(item)

        return dv

    def _algo_to_vec(self, algo):
        algov = []
        for k, v in algo.items():
            algov.append(algo[k])
        return algov

    def kmeans_clustering(self, datasets_meta_features):

        ds_features = [self._ds_to_vec(datasets_meta_features[i], self.ordered_features) for i in
                       self.train_datasets_ids]

        ds_features = np.array(ds_features)

        self.scaler = StandardScaler()
        # transform data
        ds_features = self.scaler.fit_transform(ds_features)

        kmeans = KMeans(n_clusters=16, random_state=0).fit(ds_features)

        print('hello')

        return kmeans
