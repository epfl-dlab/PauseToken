from typing import List,Union
import torch


class AbstractReward:
    """ Abstract class for reward functions. This class should be subclassed and the reward_fn method should be overriden. 
    Additonally, the get_max_reward and get_min_reward methods should be overriden to return the maximum and minimum reward values respectively.
    Optionally, the batch_call method can be overriden to compute the reward more efficiently in batch.
    """
    def __call__(
        self,
        model_output: Union[List[int], List[List[int]], torch.LongTensor],
        ground_truth: Union[List[int], List[List[int]], torch.LongTensor],
    ) -> Union[float, List[float], torch.Tensor]:
        """ Call method for the reward function, this method should not be overriden. It checks the type of the input and calls the appropriate method. If the input is a batch of sequences, it calls the batch_call method, otherwise it calls the reward_fn method
        
        :param model_output: Model output
        :type model_output: Union[List[int], List[List[int]], torch.LongTensor]
        :param ground_truth: Ground truth
        :type ground_truth: Union[List[int], List[List[int]], torch.LongTensor
        :return: Reward
        :rtype: Union[float, List[float], torch.Tensor]    
        
        """
        assert isinstance(model_output, type(ground_truth)), "model_output and ground_truth must be of the same type"
        
        # Type checking + determine if it is a batch call or not
        is_batch_call = None
        
        #Case 1: LongTensor
        if isinstance(model_output, torch.LongTensor):
            #if there's only one dimension, it is not a batch call
            if len(model_output.shape) == 1:
                is_batch_call = False
            #if there are two dimensions, it is a batch call
            elif len(model_output.shape) == 2:
                is_batch_call = True                
        
        #Case 2: List[int] or List[List[int]]
        elif isinstance(model_output, list):
            # if the first element is an int, it is not a batch call
            if isinstance(model_output[0], int):
                is_batch_call = False
            # if the first element is a list of ints, it is a batch call
            elif isinstance(model_output[0], list) and isinstance(model_output[0][0], int):
                is_batch_call = True
                
            #convert to torch tensor
            model_output = torch.tensor(model_output)
            ground_truth = torch.tensor(ground_truth)
        
        #Case 3: Invalid type
        if is_batch_call is None:
            raise ValueError(
                "model_output and ground_truth must be either a LongTensor of shape (batch_size, seq_len) or (seq_len), \
                    or a List[List[int]] or List[int]")
        
        #call the appropriate method
        if is_batch_call:
            return self.batch_call(model_output, ground_truth)
        else:
            return self.reward_fn(model_output, ground_truth)
    
    def batch_call(self, model_output: torch.LongTensor, ground_truth: torch.LongTensor) -> List[float]:
        """ Batch call method for the reward function, this method can be overriden by the subclass if the reward function can be computed more efficiently in batch (e.g. using tensor operations). Note: returning a tensor is permitted
        
        :param model_output: Model output
        :type model_output: torch.LongTensor
        :param ground_truth: Ground truth
        :type ground_truth: torch.LongTensor
        :return: corresponding rewards
        :rtype: Union[List[float], torch.Tensor]
        """
        
        return [
                self.reward_fn(model_output, ground_truth) 
                for model_output, ground_truth in zip(model_output, ground_truth)
            ]
        
    def reward_fn(self, model_output: torch.LongTensor, ground_truth: torch.LongTensor):
        """ Reward function, this method should be overriden by the subclass. It should return a float, the reward value.
        
        :param model_output: Model output
        :type model_output: torch.LongTensor
        :param ground_truth: Ground truth
        :type ground_truth: torch.LongTensor
        :return: Reward
        :rtype float
        """
        raise NotImplementedError
        
    def get_max_reward(self):
        """ This method should be overriden by the subclass. Get the maximum reward value
        
        :return: Maximum reward value
        :rtype: float
        """
        raise NotImplementedError
    
    def get_min_reward(self):
        """ This method should be overriden by the subclass. Get the minimum reward value
        
        :return: Minimum reward value
        :rtype: float
        """
        raise NotImplementedError