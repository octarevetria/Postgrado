import numpy as np
import matplotlib.pyplot as plt

def plot_rewards(rewards, average_range = 1000):
    episode_ticks = int(len(rewards) / average_range)

    avg_rewards = np.array(rewards).reshape((episode_ticks, average_range))
    avg_rewards = np.mean(avg_rewards, axis=1)

    plt.plot([i * average_range for i in range(episode_ticks)], avg_rewards)
    plt.title("Episode Accumulated Reward")
    plt.xlabel("Episode Number")
    plt.ylabel("Reward")
    plt.show()
    
def plot_epsilon(epsilons):
    plt.plot(epsilons)
    plt.title("Epsilon over Episodes")
    plt.xlabel("Episode Number")
    plt.ylabel("Epsilon")
    plt.show()


def plot_multiple_rewards(rewards_dict, average_range=1000):
    """
    Plottea recompensa promedio para varios modelos en un mismo par de ejes

    Args:
        rewards_dict (dict): Diccionario con los nombres de modelo como keys y la lista de recompensas de cada modelo
        average_range (int): Rango de episodios para promediar la recompensa
    """
    plt.figure(figsize=(10, 6))
    for model_name, rewards in rewards_dict.items():
        episode_ticks = int(len(rewards) / average_range)
        if len(rewards) % average_range != 0:
            rewards = rewards[:-(len(rewards) % average_range)]
        
        if len(rewards) == 0:
            print(f"No hay rewards para '{model_name}'")
            continue

        avg_rewards = np.array(rewards).reshape((episode_ticks, average_range))
        avg_rewards = np.mean(avg_rewards, axis=1)
        plt.plot([i * average_range for i in range(episode_ticks)], avg_rewards, label=f'{model_name} Rewards')

    plt.title("Episode Accumulated Reward (Varios Modelos)")
    plt.xlabel("Episode Number")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_multiple_epsilons(epsilons_dict):
    """
    Plottea epsilons para varios modelos en un mismo par de ejes

    Args:
        rewards_dict (dict): Diccionario con los nombres de modelo como keys y la lista de epsilons para cada modelo
    """
    plt.figure(figsize=(10, 6))
    for model_name, epsilons in epsilons_dict.items():
        plt.plot(epsilons, label=f'{model_name} Epsilon')

    plt.title("Epsilon over Episodes (Varios Modelos)")
    plt.xlabel("Episode Number")
    plt.ylabel("Epsilon")
    plt.legend()
    plt.grid(True)
    plt.show()
