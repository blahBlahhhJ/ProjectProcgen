import numpy as np
# from matplotlib import pyplot as plt

# from trees import SumTree, MinTree

class ReplayBuffer:
    def __init__(self, mem_size=1000000):
        self.curr_size = 0
        self.next_idx = 0
        self.mem_size = mem_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def store(self, s, a, r, sp, d):
        if self.curr_size < self.mem_size:
            self.curr_size += 1
            self.states.append(s)
            self.actions.append(a)
            self.rewards.append(r)
            self.next_states.append(sp)
            self.dones.append(d)
        else:
            self.states[self.next_idx] = s
            self.actions[self.next_idx] = a
            self.rewards[self.next_idx] = r
            self.next_states[self.next_idx] = sp
            self.dones[self.next_idx] = d
        self.next_idx = (self.next_idx + 1) % self.mem_size
        

    def sample(self, batch_size, random_idxs=None):
        if random_idxs is None:
            random_idxs = np.random.randint(self.curr_size, size=batch_size)

        sampled_states = []
        sampled_actions = []
        sampled_rewards = []
        sampled_next_states = []
        sampled_dones = []
        for idx in random_idxs:
            sampled_states.append(self.states[idx])
            sampled_actions.append(self.actions[idx])
            sampled_rewards.append(self.rewards[idx])
            sampled_next_states.append(self.next_states[idx])
            sampled_dones.append(self.dones[idx])

        return sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones


# class PrioritizedReplayBuffer(ReplayBuffer):
#     def __init__(self, mem_size=1000000, alpha=1):
#         super(PrioritizedReplayBuffer, self).__init__(mem_size)
#         self.alpha = alpha

#         it_capacity = 1
#         while it_capacity < mem_size:
#             it_capacity *= 2

#         self.sum_tree = SumTree(it_capacity)
#         self.min_tree = MinTree(it_capacity)
#         self.max_priority = 1.0

#     def store(self, s, a, r, sp, d):
#         idx = self.next_idx
#         super().store(s, a, r, sp, d)
#         self.sum_tree[idx] = self.max_priority ** self.alpha
#         self.min_tree[idx] = self.max_priority ** self.alpha

#     def sample(self, batch_size, beta=0):
#         idxs = []
#         p_total = self.sum_tree.sum(0, self.curr_size - 1)
#         range_len = p_total / batch_size

#         for i in range(batch_size):
#             prefix_sum = (np.random.rand() + i) * range_len
#             idx = self.sum_tree.find_idx(prefix_sum)
#             idxs.append(idx)
        
#         weights = []
#         p_min = self.min_tree.min() / p_total
#         max_weight = (p_min * self.curr_size) ** (-beta)

#         for idx in idxs:
#             p_sample = self.sum_tree[idx] / p_total
#             weight = (p_sample * self.curr_size) ** (-beta)
#             weights.append(weight / max_weight)
        
#         samples = super().sample(batch_size, idxs)
#         return (*samples, weights, idxs)

#     def update_priorities(self, idxs, priorities):
#         for idx, priority in zip(idxs, priorities):
#             self.sum_tree[idx] = priority ** self.alpha
#             self.min_tree[idx] = priority ** self.alpha

#             self.max_priority = max(self.max_priority, priority)
        

# if __name__ == "__main__":
    # mem = PrioritizedReplayBuffer(mem_size=100, alpha=1)
    # dic = {}
    # for i in range(100):
    #     mem.store(i, i, i, i, i)
    #     mem.update_priorities([i], [i])
    #     dic[i] = 0


    # for i in range(1000):
    #     sample_state, *_ = mem.sample(2)
    #     for s in sample_state:
    #         dic[s] += 1
    
    # plt.bar(range(100), [dic[i] for i in range(100)])
    # plt.show()