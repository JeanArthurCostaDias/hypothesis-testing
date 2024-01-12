import numpy as np
from scipy.stats import beta
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt


class AdBannerReward:
    def __init__(self, ctr):
        self.ctr = ctr

    def __call__(self, **kwargs):
        rnd = np.random.uniform(0, 1)
        return int(rnd <= self.ctr)


class Action:
    def __init__(self, name, reward_function):
        self.name = name
        self.reward_function = reward_function

    def get_reward(self, **action_kwargs):
        return self.reward_function(**action_kwargs)


class TestEnvironment:
    def __init__(self, actions):
        self.actions_dict = {action.name: action for action in actions}

    def get_reward(self, action_name, **action_kwargs):
        assert action_name in self.actions_dict, 'No such action in the environment!'
        return self.actions_dict[action_name].get_reward(**action_kwargs)


class Sampler:
    def __init__(self, action_names):
        self.action_names = action_names
        self.priors = self._init_priors()

    def _init_priors(self):
        raise NotImplementedError('Needs to be implemented')

    def sample(self):
        raise NotImplementedError('Needs to be implemented')

    def update(self, action_name, reward):
        raise NotImplementedError('Needs to be implemented')


class RandomSampler(Sampler):
    def __init__(self, action_names):
        super().__init__(action_names=action_names)

    def _init_priors(self):
        return np.ones(len(self.action_names)) / len(self.action_names)

    def sample(self):
        return np.random.choice(self.action_names, p=self.priors)

    def update(self, action_name, reward):
        pass


class ThompsonSampler(Sampler):
    def __init__(self, action_names):
        super().__init__(action_names=action_names)
        self.current_stats = self.priors

    def _init_priors(self):
        return {action_name: {'a': 1, 'b': 1} for action_name in self.action_names}

    def sample(self):
        probs = [
            beta.rvs(
                a=self.current_stats[action_name]['a'],
                b=self.current_stats[action_name]['b'],
                size=1
            )
            for action_name in self.action_names
        ]

        return self.action_names[np.argmax(probs)]

    def update(self, action_name, reward):
        self.current_stats[action_name]['a'] += reward
        self.current_stats[action_name]['b'] += 1 - reward


class MABandit:
    def __init__(
        self,
        action_names: list[str],
        strategy: str,
        test_environment: TestEnvironment,
        epsilon: float = 0.1,
        termination_epsilon: float = 0.01
    ):
        self.test_environment = test_environment
        self.action_names = action_names
        self.strategy = strategy
        self.epsilon = epsilon
        self.termination_epsilon = termination_epsilon
        self.sampler = self._get_sampler()
        self.action_stats = self._init_stats()
        self.history = {'actions': [], 'rewards': []}
        self.mean_rewards = self._init_mean_rewards()

    def _init_mean_rewards(self):
        mean_rewards = {action_name: [] for action_name in self.action_names + ['total']}
        return mean_rewards

    def _init_stats(self):
        stats = {action_name: {'rewards': 0, 'trials': 0} for action_name in self.action_names}
        return stats

    def reset(self):
        self.sampler = self._get_sampler()
        self.action_stats = self._init_stats()
        self.history = {'actions': [], 'rewards': []}
        self.mean_rewards = self._init_mean_rewards()

    def _get_sampler(self):
        if self.strategy in ('epsilon-greedy', 'epsilon-greedy-decay'):
            sampler = RandomSampler(self.action_names)
        elif self.strategy == 'thompson':
            sampler = ThompsonSampler(self.action_names)
        else:
            raise NotImplementedError('Sampler not implemented!')
        return sampler

    def _epsilon_step_type(self):
        rnd = np.random.uniform(0, 1)
        return 1 if rnd <= self._get_epsilon() else 0

    def _get_epsilon(self):
        if self.strategy == 'epsilon-greedy':
            epsilon = self.epsilon
        elif self.strategy == 'epsilon-greedy-decay':
            epsilon = min(1.0, self.epsilon / (len(self.history['actions']) + 1))
        else:
            raise NotImplementedError('Epsilon for this strategy is not supported')
        return epsilon

    def _update(self, action_name, reward):
        self.action_stats[action_name]['rewards'] += reward
        self.action_stats[action_name]['trials'] += 1
        self.history['actions'].append(action_name)
        self.history['rewards'].append(reward)
        self.sampler.update(action_name=action_name, reward=reward)

        total_reward = 0
        total_trials = 0
        for name in self.action_names:
            rewards = self.action_stats[name]['rewards']
            n_trials = self.action_stats[name]['trials']
            total_reward += rewards
            total_trials += n_trials
            reward_mean = rewards/n_trials if n_trials else 0
            self.mean_rewards[name].append(reward_mean)

        self.mean_rewards['total'].append(total_reward / total_trials)

    def _get_action_from_sampler(self):
        return self.sampler.sample()

    def _run_epsilon_greedy(self):
        explore = self._epsilon_step_type()
        action_choice = self._get_action_from_sampler()
        if not explore:
            mean_rewards = [self.mean_rewards[action_name][-1] if self.mean_rewards[action_name] else 0 for action_name in self.action_names]
            if min(mean_rewards) != max(mean_rewards):
                action_choice = self.action_names[np.argmax(mean_rewards)]

        reward = self.test_environment.get_reward(action_choice)
        self._update(action_choice, reward)

    def _epsilon_above_limit(self):
        return self._get_epsilon() > self.termination_epsilon

    def _run_thompson(self):
        action_choice = self._get_action_from_sampler()
        reward = self.test_environment.get_reward(action_choice)
        self._update(action_choice, reward)

    def run(self, n_iter: int = 2000):
        self.reset()
        if self.strategy == 'epsilon-greedy':
            for _ in range(n_iter):
                self._run_epsilon_greedy()
        elif self.strategy == 'epsilon-greedy-decay':
            while self._epsilon_above_limit():
                self._run_epsilon_greedy()
        elif self.strategy == 'thompson':
            for _ in range(n_iter):
                self._run_thompson()
        else:
            raise NotImplementedError('Sampler not implemented!')


def calculate_sd(environment: TestEnvironment, action_names: list[str], n_samples: int) -> int:
    sample_a = np.array([environment.get_reward(action_names[0]) for _ in range(n_samples)])
    sample_b = np.array([environment.get_reward(action_names[1]) for _ in range(n_samples)])
    sd = np.sqrt(sample_a.std()**2 + sample_b.std()**2)
    return sd


def calculate_sample_size(prac_sig: float, alpha: float, beta: float, sd) -> int:
    z_alpha = abs(stats.norm.ppf(1 - alpha))
    z_beta = abs(stats.norm.ppf(beta))
    n = np.ceil(sd**2*(z_alpha + z_beta)**2/prac_sig**2)
    return int(n)


class AB_test:
    def __init__(self, environment: TestEnvironment, action_names: list[str], sample_size: int):
        self.environment = environment
        self.action_names = action_names
        self.sample_size = sample_size
        self.history = {'actions': [], 'rewards': []}
        self.history_per_actions = {name: [] for name in self.action_names}
        self.mean_rewards = {name: [] for name in self.action_names + ['total']}

    def reset(self):
        self.history = {'actions': [], 'rewards': []}
        self.history_per_actions = {name: [] for name in self.action_names}
        self.mean_rewards = {name: [] for name in self.action_names + ['total']}

    def calculate_stats(self):
        mean_a = np.mean(self.history_per_actions[self.action_names[0]])
        mean_b = np.mean(self.history_per_actions[self.action_names[1]])

        std_a = np.std(self.history_per_actions[self.action_names[0]])
        std_b = np.std(self.history_per_actions[self.action_names[1]])
        se = np.sqrt((std_a**2 + std_b**2)/len(self.action_names[0]))
        z = (mean_b - mean_a)/se
        p_value = stats.norm.cdf(z)
        return z, p_value

    def run(self, **kwargs):
        self.reset()
        total_rewards = [0]*len(self.action_names)
        for step in range(self.sample_size):
            name = np.random.choice(self.action_names)
            self.history['actions'].append(name)
            i = self.action_names.index(name)
            reward = self.environment.get_reward(name)
            self.history_per_actions[name].append(reward)
            total_rewards[i] += reward
            for i, name in enumerate(self.action_names):
                self.mean_rewards[name].append(total_rewards[i]/(len(self.history_per_actions[name]) + 1))
            self.mean_rewards['total'].append(np.sum(total_rewards)/(step + 1))
        z, p_val = self.calculate_stats()
        return z, p_val


def get_sample_size(prac_sig: float, environment: TestEnvironment, alpha: float, beta: float, action_names: list[str] = ['A', 'B']) -> int:
    sd = calculate_sd(environment, action_names, n_samples=1000)
    sample_size = calculate_sample_size(prac_sig, alpha, beta, sd)
    return sample_size


def ab_baseline(environment: TestEnvironment, sample_size: int, action_names: list[str] = ['A', 'B']) -> dict:
    abtest = AB_test(environment, action_names, sample_size)
    z, p_val = abtest.run()
    history = abtest.mean_rewards
    return z, p_val, history


def count_probs(x: np.ndarray) -> np.ndarray:
    n_rows = x.shape[0]
    action_names = np.unique(x)
    probs = {name: np.zeros(x.shape[1]) for name in action_names}
    for col in range(x.shape[1]):
        unique_values, counts = np.unique(x[:, col], return_counts=True)
        for name, count in zip(unique_values, counts):
            probs[name][col] = count/n_rows
    return probs


def run_n_times(n_runs: int, test: object, **kwargs):
    histories = {name: [] for name in test.action_names + ['total']}
    selections = []
    for _ in range(n_runs):
        test.reset()
        test.run(**kwargs)
        for key in histories.keys():
            histories[key].append(test.mean_rewards[key])
        selections.append(test.history['actions'])
    probs = count_probs(np.stack(selections))
    results = defaultdict(dict)
    for key in histories.keys():
        histories[key] = np.stack(histories[key])
        results[key]['mean'] = np.mean(histories[key], axis=0)
        results[key]['sd'] = np.std(histories[key], axis=0)
    return results, probs


def calculate_epsilon(k: int, c: float, BM_max: float, PS: float) -> float:
    '''
    Calculate the exploration-exploitation trade-off parameter (epsilon) for an epsilon-greedy multi-armed bandit algorithm.

    Parameters:
    - k (int): Number of bandits (arms).
    - c (float): Constant multiplier controlling exploration rate.
    - BM_max (float): Largest plausible value of the bandit mean.
    - PS (float): Practical significance, representing the minimum difference in bandit means considered practically significant.

    Returns:
    float: The calculated epsilon, determining the probability of exploration in the epsilon-greedy strategy.

    Formula:
    epsilon = k * c * (BM_max / PS) ** 2
    '''
    epsilon = k*c*(BM_max/PS)**2
    return epsilon


def get_cost_regret(history: dict, best_mean_reward: float = 0.007, cpc: float = 0.1):
    n = len(history['total']['mean'])
    regret = np.cumsum(np.array([cpc*best_mean_reward]*n) - cpc*np.array(history['total']['mean']))

    return regret


def plot_history(mean_reward_history, probabilities):
    """
    Plot the history of mean rewards with two vertical subplots.

    Parameters:
    - mean_reward_history (dict): Dictionary containing mean reward history for each action and the total.

    Example:
    mean_reward_history = {'action1': [1, 2, 3, ...],
                           'action2': [2, 3, 4, ...],
                           'total': [3, 5, 7, ...]}
    """
    
    regret = get_cost_regret(mean_reward_history)

    # Create a figure with two vertical subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot mean rewards for each action
    for action_name in mean_reward_history:
        mean_values = mean_reward_history[action_name]['mean']
        sd_values = mean_reward_history[action_name]['sd']
        
        axes[0].plot(mean_values, label=f'{action_name} (mean)')
        axes[0].fill_between(range(len(mean_values)),
                         mean_values - sd_values,
                         mean_values + sd_values,
                         alpha=0.3, label=f'{action_name} (±1 sd)')
        if action_name != 'total':
            axes[1].plot(probabilities[action_name],'o', markersize=3, alpha=0.2, label=f'{action_name} prob')

    # Plot total mean reward
    axes[2].plot(regret, color='red', label='Mean Regret')

    # Set labels and title
    axes[0].set_ylabel('Mean Reward')
    axes[2].set_xlabel('Episode')
    axes[1].set_ylabel('Probability')
    axes[2].set_ylabel('Mean Regret')
    axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axes[2].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axes[0].set_ylim(bottom=0, top=0.009)
    axes[1].set_ylim(bottom=0, top=1)
    plt.suptitle('AB Experiment Results')

    # Show the plot
    plt.show()


if __name__ == '__main__':
    rwrd1 = AdBannerReward(ctr=0.2)
    rwrd2 = AdBannerReward(ctr=0.4)
    sample = [rwrd1() for _ in range(100)]
    print(f'Mean of sample: {np.mean(sample)}')

    # Example of TestEnvironment
    action1 = Action(name='A', reward_function=rwrd1)
    action2 = Action(name='B', reward_function=rwrd2)

    environment = TestEnvironment(actions=[action1, action2])

    action_request = {'action_name': 'A', 'size': 2, 'color': 'blue'}
    reward = [environment.get_reward(**action_request) for _ in range(200)]
    print(f'Reward for action1: {np.mean(reward)}')

    action_request = {'action_name': 'B', 'duration': 10}
    reward = [environment.get_reward(**action_request) for _ in range(200)]
    print(f'Reward for action2: {np.mean(reward)}')

    print('START')
    b = MABandit(['A', 'B'], 'epsilon-greedy', environment)
    b.run(80000)
    print(b.mean_rewards['A'][-5:])
    print(b.mean_rewards['B'][-5:])
    print(b.mean_rewards['total'][:5])
    print(b.mean_rewards['total'][-5:])
    '''
    for _ in range(10):
        b.run()
    print(len(b.history['actions']))

    b = MABandit(['A', 'B'], 'epsilon-greedy-decay', epsilon=50, test_environment=environment)
    for _ in range(10):
        b.run()
    print(len(b.history['actions']))

    b = MABandit(['A', 'B'], 'thompson', environment)
    for _ in range(10):
        b.run()
    print(len(b.history['actions']))
    '''