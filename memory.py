import numpy as np
import torch

class Memory:
    def __init__(self, max_num_transitions, mini_batch_size):
        self.lst = []
        self.max_num_transitions = max_num_transitions
        self.mini_batch_size = mini_batch_size

    # this function returns the number of stored transitions
    def __len__(self):
        return len(self.lst)
    
    def append(self, curr_state, action, reward, next_state, done):
        if len(self.lst) == self.max_num_transitions:
            self.lst.pop(0)

        trans = (curr_state, action, reward, next_state, done)
        self.lst.append(trans)
    
    def sample_mini_batch(self):
        if len(self.lst) < self.mini_batch_size:
            raise Exception('Don\'t try to sample mini-batches while number of stored transitions < mini_batch_size')
        
        idxs = np.random.randint(0, len(self.lst), self.mini_batch_size)
        
        samples = (self.lst[idx] for idx in idxs)
        
        # changed the name to mini_batch in order to make the namings for similar to the workflow we use
        mini_batch = tuple(map(torch.tensor, map(np.array, zip(*samples))))
        return mini_batch