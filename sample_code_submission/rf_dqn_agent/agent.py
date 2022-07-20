# coding: utf-8

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F


class GDQN(nn.Module):
    def __init__(self, states_num, meta_num, actions_nums):
        super(GDQN, self).__init__()
        self.fc1 = nn.Linear(meta_num, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.gru1 = nn.GRU(states_num, 128)
        self.gru2 = nn.GRU(128, 128)
        self.layer_num = nn.LayerNorm(states_num)
        self.fc4 = nn.Linear(256 + meta_num, 128)
        nn.init.xavier_normal_(self.fc4.weight)
        self.A = nn.Linear(128, actions_nums)
        self.V = nn.Linear(128, 1)
        nn.init.xavier_normal_(self.A.weight)
        nn.init.xavier_normal_(self.V.weight)

    def forward(self, x, meta_x):
        c = F.relu(self.fc1(meta_x))
        c = F.relu(self.fc2(c))
        c = F.relu(self.fc3(c))
        c = torch.cat((c, meta_x), dim=1)
        x = self.layer_num(x)
        x, h = self.gru1(x)
        x, h = self.gru2(F.relu(x))
        x = F.relu(x)[:, -1, :]
        x = torch.cat((x, c), dim=1)

        x = F.relu(self.fc4(x))
        return self.A(x), self.V(x)


class DQN():
    def __init__(self, states_num, meta_num, epsilon=0.5):
        super(DQN, self).__init__()
        self.action_eval_net, self.action_target_net = GDQN(states_num, meta_num, 40), GDQN(states_num, meta_num, 40)
        self.epsilon = epsilon
        self.A_optimizer = torch.optim.Adam(self.action_eval_net.parameters(), lr=0.1)  # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.gamma = 0.1
        self.loss_func = nn.MSELoss()
        self.total_loss = 0.0
        self.p_select = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # self.device = torch.cuda.is_available()
        self.device = False
        # if self.device:
        #     self.action_eval_net.to("cuda")
        #     self.action_target_net.to("cuda")

    def choose_action_greedy(self, x, meta_x, rf_score, epsilon=None):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        rf_score = torch.unsqueeze(torch.FloatTensor(rf_score), 0)
        meta_x = torch.unsqueeze(torch.FloatTensor(meta_x), 0)
        if self.device:
            x = x.cuda()
            meta_x = meta_x.cuda()
        if not epsilon:
            epsilon = self.epsilon
        if np.random.uniform() > epsilon:
            actions_value, _ = self.action_eval_net.forward(x, meta_x)
            actions_value = rf_score * 0.3 + actions_value.cpu() * 0.7
            actions = torch.topk(actions_value, 4)[1].data.numpy()
            action = actions[0][np.random.choice(a=4, p=[0.4, 0.3, 0.2, 0.1])]
        else:  # 随机选择动作
            actions = torch.topk(rf_score, 10)[1].data.numpy()
            action = actions[0][np.random.choice(a=10)]

        # A = action // 40
        # p = (action % 10 + 1) / 10
        # if np.random.uniform() < epsilon:
        #     p = min(p, p_max)
        # return A, p
        return action

    def learn(self, x, meta_x, action, reward, stop=False):
        action = int(action)
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        meta_x = torch.unsqueeze(torch.FloatTensor(meta_x), 0)
        if self.device:
            x = x.cuda()
            meta_x = meta_x.cuda()
        a_eval, v_eval = self.action_eval_net(x, meta_x)

        a_next, v_next = self.action_target_net(x, meta_x)

        v_next = v_next.detach()

        q_eval = v_eval + a_eval[:, action] - torch.mean(a_eval, dim=-1, keepdim=True)
        q_next = v_next + a_next[:, action] - torch.mean(a_next, dim=-1, keepdim=True)
        q_target = reward + self.gamma * q_next
        loss = self.loss_func(q_eval, q_target)
        self.total_loss += loss.item()

        self.A_optimizer.zero_grad()
        loss.backward()
        self.A_optimizer.step()
        if stop:
            self.action_target_net.load_state_dict(self.action_eval_net.state_dict())


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

        self.model = RandomForestRegressor(n_estimators=50)

        self.p_portions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        self.hp_keys1 = ['meta_feature_0', 'meta_feature_1', 'meta_feature_2']
        self.dqn = DQN(53, 14)

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
            new_times.append(new_t)
        return new_scores, new_times

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
        self.validation_time_seen = [0.0 for _ in self.algo_name_list]
        self.train_last_scores = [0.0 for _ in self.algo_name_list]
        self.total_budget = float(dataset_meta_features['time_budget'])
        self.time_used = 0.0
        self.used_p = 0

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
            scores = self.model.predict(pdata)
            scores_list.append(scores.tolist())

        self.scores_list = scores_list
        self.times_list = times_list

        self.best_algo_list = []
        for i in range(len(self.data_sizes)):
            scores = [x[i] for x in scores_list]
            best_idx = np.argmax(scores)
            self.best_algo_list.append(best_idx)

        self.best_algo_list += [0]
        self.best_algo_list = list(set(self.best_algo_list))
        print('===> reset, best algo', dataset_meta_features['name'], self.best_algo_list)

        self.best_algo_scores = [self.scores_list[i] for i in self.best_algo_list]

        self.delta_index_list = [0 for _ in self.algo_name_list]
        self.algo_index = 0

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

                times_list += new_times

        x1_data = np.array(inputs_list1)
        y1_data = np.array(score_list)

        self.model.fit(x1_data, y1_data)

        self.train_datasets_ids = [k for k in test_learning_curves]
        for iteration in range(32):
            print("--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--")
            print("Iteration:", iteration)
            for idx, episode in enumerate(self.train_datasets_ids):
                dataset_meta_feature = datasets_meta_features[episode]

                self.total_time_budget = float(dataset_meta_feature['time_budget'])
                self.remaining_time_budget = self.total_time_budget
                self.list_algorithms = [k for k in test_learning_curves[episode].keys()]

                self.reset_for_train(dataset_meta_feature, algorithms_meta_features)

                observation = None
                for choice in range(256):
                    self.suggest_list = []
                    for it in range(16):
                        # === Get the agent's suggestion
                        (next_algo, p) = self.suggest_for_train(observation,
                                                                must_suggest_by_model=True if choice > 128 else False)

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
                        if self.time_used > self.total_time_budget * 5:
                            break
                        observation = (next_algo, p, t, R_train_A_p, R_validation_A_p)
            print("Average Total Loss:", self.dqn.total_loss)
            self.dqn.total_loss = 0.0

    def cal_alc(self):
        suggest_list = self.suggest_list
        time_budget = self.total_budget
        dots = [(s[50], s[52]) for s in suggest_list]
        normalize_dots = [(a[0] / time_budget, a[1]) if a[0] < time_budget else (1.0, "None") for a in dots]
        alc = 0
        for i in range(len(normalize_dots)):
            if normalize_dots[i][1] == "None":
                continue
            if i == 0:
                alc += normalize_dots[i][1] * normalize_dots[i][0]
            else:
                alc += (normalize_dots[i][1] - normalize_dots[i - 1][1]) * (1 - normalize_dots[i][0])
        if alc <= 0:
            alc = -np.e / 16
        return alc / 5

    def reset_for_train(self, dataset_meta_feature, algorithms_meta_feature):
        self.dataset_meta_feature = dataset_meta_feature
        self.algorithms_meta_feature = algorithms_meta_feature

        self.validation_last_scores = [0.0 for _ in self.algo_name_list]
        self.validation_time_seen = [0.0 for _ in self.algo_name_list]
        self.train_last_scores = [0.0 for _ in self.algo_name_list]
        self.total_budget = float(dataset_meta_feature['time_budget'])
        self.time_used = 0.0
        self.used_p = 0

        self.current_dataset_meta_features = dataset_meta_feature

        self.last_delta_t = 0.1
        self.last_index = 0

        ds_vector = self._get_dataset_vector(dataset_meta_feature)

        scores_list = []

        for algo_name in self.algo_name_list:
            algo_vector1 = [
                float(algorithms_meta_feature[algo_name][k]) if (k in algorithms_meta_feature[algo_name]) else 0.0 for
                k in self.hp_keys1]
            pdata = []
            for sizes in self.data_sizes:
                pdata.append(ds_vector + algo_vector1 + [float(algo_name)] + [sizes])
            scores = self.model.predict(pdata)
            scores_list.append(scores.tolist())

        self.scores_list = scores_list

        self.delta_index_list = [0 for _ in self.algo_name_list]

        self.algo_index = 0
        self.suggest_list = []

    def to_one_hot(self, x, num):
        one_hot = np.eye(num)
        return one_hot[int(x)]

    def suggest_for_train(self, observation, must_suggest_by_model=False):
        if observation == None:
            A, p, t, R_train_A_p, R_validation_A_p = 0, 0.1, 0, 0, 0
        else:
            A, p, t, R_train_A_p, R_validation_A_p = observation
            self.validation_last_scores[A] = max(self.validation_last_scores[A], R_validation_A_p)

        self.time_used += t
        if np.random.uniform() < 0.5:
            p_suggest = self.p_portions[self.used_p]
        else:
            p_suggest = self.p_portions[np.random.randint(0, 9)]

        p_score = np.array(self.scores_list)[:, self.data_sizes.index(p_suggest)]
        dataset_vector = self._get_dataset_vector(self.dataset_meta_feature)
        # a = np.hstack((self.to_one_hot(A, 40), self.to_one_hot(p * 10 - 1, 10),
        #                np.array([self.time_used, R_train_A_p, R_validation_A_p]), p_score, np.array(dataset_vector)))
        a = np.hstack((self.to_one_hot(A, 40), self.to_one_hot(p * 10 - 1, 10),
                       np.array([self.time_used, R_train_A_p, R_validation_A_p])))
        self.suggest_list.append(a)

        # next_algo_to_reveal, p_suggest = self.dqn.choose_action_greedy(
        #     torch.FloatTensor(np.array(self.suggest_list)), p_max=np.random.randint(1, 10) / 10)
        epsilon = 0.7 if not must_suggest_by_model else 0.07
        next_algo_to_reveal = self.dqn.choose_action_greedy(
            torch.FloatTensor(np.array(self.suggest_list)), torch.FloatTensor(dataset_vector),
            torch.FloatTensor(p_score), epsilon=epsilon)

        if observation == None:
            best_algo_for_test = None

            self.old_score = 0
        else:
            best_algo_for_test = np.argmax(self.validation_last_scores)
            self.old_score = np.max([R_validation_A_p, self.old_score])

            self.validation_time_seen[A] = A

            time_used = t
            weight = (self.total_time_budget * .5 - time_used) / self.total_time_budget

            stop = False
            # reward = (np.max([R_validation_A_p, self.old_score]) - self.old_score) * weight * 0.1
            reward = (R_validation_A_p - self.old_score)
            reward = (reward * abs(weight) if reward > 0 else reward * weight) * .2
            if self.time_used >= self.total_time_budget * 5:
                alc = self.cal_alc()
                reward = alc * 0.8 + reward
                stop = True

            self.dqn.learn(torch.FloatTensor(self.suggest_list), torch.FloatTensor(dataset_vector), A, reward,
                           stop=stop)

            self.used_p += 1

            self.used_p = self.used_p if self.used_p <= 9 else 1

            self.validation_time_seen[A] = A

        self.best_algo_for_test = best_algo_for_test

        action = (next_algo_to_reveal, p_suggest)
        return action

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
        if observation == None:
            A, p, t, R_train_A_p, R_validation_A_p = 0, 0.1, 0, 0, 0
        else:
            A, p, t, R_train_A_p, R_validation_A_p = observation
            self.validation_last_scores[A] = max(self.validation_last_scores[A], R_validation_A_p)
        self.time_used += t
        p_suggest = self.p_portions[self.used_p]

        p_score = np.array(self.scores_list)[:, self.data_sizes.index(p_suggest)]

        dataset_vector = self._get_dataset_vector(self.dataset_meta_feature)
        # a = np.hstack((self.to_one_hot(A, 40), self.to_one_hot(p * 10 - 1, 10),
        #                np.array([self.time_used, R_train_A_p, R_validation_A_p]), p_score, np.array(dataset_vector)))
        a = np.hstack((self.to_one_hot(A, 40), self.to_one_hot(p * 10 - 1, 10),
                       np.array([self.time_used, R_train_A_p, R_validation_A_p])))
        self.suggest_list.append(a)

        next_algo_to_reveal = self.dqn.choose_action_greedy(torch.FloatTensor(self.suggest_list),
                                                            torch.FloatTensor(dataset_vector),
                                                            torch.FloatTensor(p_score),
                                                            epsilon=0.1)

        self.delta_index_list[next_algo_to_reveal] = max(self.data_sizes.index(p_suggest),
                                                         self.delta_index_list[next_algo_to_reveal])
        if observation == None:
            best_algo_for_test = None
            self.used_p += 1
            self.old_score = 0
        else:
            best_algo_for_test = np.argmax(self.validation_last_scores)
            self.old_score = np.max([R_validation_A_p, self.old_score])
            new_budget = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            if self.time_used < self.total_budget // 2:
                # self.validation_last_scores[A] = R_validation_A_p
                self.validation_time_seen[A] = A

                if self.time_used < self.total_time_budget // 3:
                    self.used_p += 1

                self.used_p = self.used_p if self.used_p <= 9 else 1
            else:
                next_algo_to_reveal = best_algo_for_test
                p_suggest = self.data_sizes[self.delta_index_list[next_algo_to_reveal] + 1 if self.delta_index_list[
                                                                                                  next_algo_to_reveal] < 9 else 9]

        self.best_algo_for_test = best_algo_for_test

        action = (next_algo_to_reveal, p_suggest)
        return action

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

        temp_time = C + (self.times[list(self.ps).index(p)])

        for i in range(len(self.times)):
            if temp_time < self.times[i]:
                if i == 0:  # if delta_t is not enough to get the first point, the agent wasted it for nothing!
                    train_score, test_score, timestamp = 0.0, 0.0, 0.0
                else:  # return the last achievable point
                    train_score, test_score, timestamp = self.train_scores[i - 1], self.test_scores[i - 1], \
                                                         self.times[i - 1]
                return train_score, test_score, timestamp

        # If the last point on the learning curve is already reached, return it
        train_score, test_score, timestamp, p = self.train_scores[-1], self.test_scores[-1], self.times[-1], self.ps[-1]
        return train_score, test_score, timestamp
