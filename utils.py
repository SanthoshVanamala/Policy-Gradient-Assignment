import random
import numpy as np
import torch.cuda


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# Util function to apply reward-to-go scheme on a list of instant-reward (from eq 7)
def apply_reward_to_go(raw_reward):
    # TODO: Compute rtg_reward (as a list) from raw_reward
    # HINT: Reverse the input list, keep a running-average. Reverse again to get the correct order.
    rtg_reward = [] #rtg values of each time step
    rwrd_sum =0#discounted sum of future rewards
    for i in reversed(raw_reward):
        rwrd_sum =  i + rwrd_sum * 0.99#updates discounted sum of future rewards
        rtg_reward.append(rwrd_sum)
        rtg_reward.reverse()#reversed
    # Normalization
    rtg_reward = np.array(rtg_reward)
    rtg_reward = rtg_reward - np.mean(rtg_reward) / (np.std(rtg_reward) + np.finfo(np.float32).eps)
    return torch.tensor(rtg_reward, dtype=torch.float32, device=get_device())


# Util function to apply reward-discounting scheme on a list of instant-reward (from eq 8)
def apply_discount(raw_reward, gamma=0.99):
    # TODO: Compute discounted_rtg_reward (as a list) from raw_reward
    # HINT: Reverse the input list, keep a running-average. Reverse again to get the correct order.
    discounted_rtg_reward = []#rtg values of each time step
    rwrd_sum =0#discounted sum of future rewards
    for i in reversed(raw_reward):
        rwrd_sum =  i + rwrd_sum * gamma#updates discounted sum of future rewards
        discounted_rtg_reward.append(rwrd_sum)
        discounted_rtg_reward.reverse()#reversed
    # Normalization
    discounted_rtg_reward = np.array(discounted_rtg_reward)
    discounted_rtg_reward = discounted_rtg_reward - np.mean(discounted_rtg_reward) / (np.std(discounted_rtg_reward) + np.finfo(np.float32).eps)
    return torch.tensor(discounted_rtg_reward, dtype=torch.float32, device=get_device())


# Util function to apply reward-return (cumulative reward) on a list of instant-reward (from eq 6)
def apply_return(raw_reward):
    # Compute r_reward (as a list) from raw_reward
    r_reward = [np.sum(raw_reward) for _ in raw_reward]
    return torch.tensor(r_reward, dtype=torch.float32, device=get_device())

