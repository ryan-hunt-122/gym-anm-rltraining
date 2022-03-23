import numpy as np


def evaluate(model, num_episodes=10, deterministic=True):
    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        T = 0
        while not done and T < 1000:
            action, _states = model.predict(obs, deterministic=deterministic)

            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            T += 1
        # print(f"Episode {i} rewards: {sum(episode_rewards)}")

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    # print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward