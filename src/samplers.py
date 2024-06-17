from torch.utils.data import BatchSampler, SubsetRandomSampler, Sampler
from typing import List

class BatchSubsetsSampler(Sampler[List[int]]):
    def __init__(self, subset_to_sampler: List[SubsetRandomSampler], batch_size: int, resample_sample_till_all_done: bool):
        
        if not isinstance(resample_sample_till_all_done, bool):
            raise ValueError(f"resample_sample_till_all_done should be a boolean value, but got resample_sample_till_all_done={resample_sample_till_all_done}")
        
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")

        self.subset_to_sampler = subset_to_sampler
        self.batch_size = batch_size
        self.resample_sample_till_all_done = resample_sample_till_all_done
    
    def __len__(self):
        return len(self.subset_to_sampler) * len(max(self.subset_to_sampler.values(), key=len)) // self.batch_size
        
    
    def __iter__(self):
        samplers = {subset_name: iter(sampler) for subset_name,sampler in self.subset_to_sampler.items()}
        sampler_done_flag = {name: False for name in samplers.keys()}
        while all(sampler_done_flag.values()) == False:
            for sampler_name,sampler in samplers.items():
                for idx_in_batch in range(self.batch_size):
                    try:
                        idx = next(sampler)
                        yield idx
                    except StopIteration:
                        if self.resample_sample_till_all_done:
                            sampler_done_flag[sampler_name] = True
                            sampler = iter(self.subset_to_sampler[sampler_name])
                            samplers[sampler_name] = sampler
                            idx = next(sampler)
                            yield idx
                        else:
                            raise NotImplementedError("Not implemented yet")
                            continue
       
    
            
                
            