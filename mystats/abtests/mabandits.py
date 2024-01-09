import numpy as np
from scipy.stats import beta


class AdBannerReward:
    def __init__(self, ctr):
        self.ctr = ctr

    def __call__(self, **kwargs):
        rnd = np.random.random()
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
        return np.ones(len(self.action_names))/len(self.action_names)
    
    def sample(self):
        return np.random.choice(self.action_names, p=self.priors)
    
    def update(self, action_name, reward):
        pass


class ThompsonSampler(Sampler):
    def __init__(self, action_names):
        super().__init__(action_names=action_names)
        self.current_stats = self.priors

    def _init_priors(self):
        '''
        return alpha_i, beta_i parameters of Beta Distribution
        assuming uniform priors
        alpha - successes, beta - failures
        '''
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
    def __init__(self, action_names: list[str], strategy: str, test_environment: TestEnvironment, epsilon: float =0.01):
        self.test_environment = test_environment
        self.action_names = action_names
        self.strategy = strategy
        self.epsilon = epsilon
        self.sampler = self._get_sampler()
        self.action_stats = self._init_stats()
        self.history = {'actions': [], 'rewards': []}

    def _init_stats(self):
        stats = {action_name: {'rewards': []} for action_name in self.action_names}
        return stats

    def _get_sampler(self):
        if self.strategy == 'epsilon-greedy':
            sampler = RandomSampler(self.action_names)
        elif self.strategy == 'thompson':
            sampler = ThompsonSampler(self.action_names)
        else:
            raise NotImplementedError('Sampler not implemented!')
        return sampler

    def _epsilon_step_type(self):
        rnd = np.random.random(1)
        if rnd <= self.epsilon:
            return 1
        else:
            return 0

    def _update(self, action_name, reward):
        self.action_stats[action_name]['rewards'].append(reward)
        self.history['actions'].append(action_name)
        self.history['rewards'].append(reward)
        self.sampler.update(action_name=action_name, reward=reward)

    def _get_action_from_sampler(self):
        action_name = self.sampler.sample()
        return action_name

    def _run_epsilon_greedy(self):
        explore = self._epsilon_step_type()
        if explore:
            action_choice = self._get_action_from_sampler(self)
        else:
            mean_rewards = [np.mean(self.action_stats[action_name]['rewards']) for action_name in self.action_names]
            action_choice = self.action_names[np.argmax(mean_rewards)]
        reward = self.test_environment.get_reward(action_choice)
        self._update(action_choice, reward)

    def _run_thompson(self):
        action_choice = self._get_action_from_sampler()
        reward = self.test_environment.get_reward(action_choice)
        self._update(action_choice, reward)

    def run(self):
        if self.strategy == 'epsilon-greedy':
            self._run_epsilon_greedy()
        elif self.strategy == 'thompson':
            self._run_thompson()
        else:
            raise NotImplementedError('Sampler not implemented!') 


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

    b = MABandit(['A', 'B'], 'epsilon-greedy', environment)
    for _ in range(10):
        b.run()
    print(b.history)

    b = MABandit(['A', 'B'], 'thompson', environment)
    for _ in range(10):
        b.run()
    print(b.history)
