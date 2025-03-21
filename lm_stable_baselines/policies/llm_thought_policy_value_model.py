from lm_stable_baselines.policies.llm_base_policy_value_model import LLMBasePolicyValueModel
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Optional, Dict
import torch
from lm_stable_baselines.utils import add_filler_tokens,unhash_ids_and_hidden_states, pad_hidden_states, hash_ids_and_hidden_states, remove_filler_tokens, remove_filler_tokens_from_hashed_array

class LLMThoughtPolicyValueModel(LLMBasePolicyValueModel):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.eos_token_id = ( len(self.tokenizer) + self.tokenizer.eos_token_id) if self.lm.thought_mode=='always' else self.tokenizer.eos_token_id

    def extract_features(self, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None) -> PyTorchObs:
        if isinstance(obs, dict):
            if "last_hidden_states" in obs:
                return obs
            else:
                features = obs
        else:
            features = unhash_ids_and_hidden_states(obs)
        
        obs_to_pass = features["input_ids"]
        device = obs_to_pass.device
        obs_to_pass = [ obs[obs != self.filler_token] for obs in obs_to_pass]
        feature = self.tokenizer.pad({"input_ids": obs_to_pass}, return_tensors="pt", padding=True).to(device)
        if feature["input_ids"].dtype == torch.float32:
            feature["input_ids"] = feature["input_ids"].long()

        padding_side = self.tokenizer.padding_side
        if (features["last_hidden_states"] == self.filler_token).all():
            feature["last_hidden_states"] = None
        else:
            feature["last_hidden_states"] = pad_hidden_states(
                last_hidden_states=features["last_hidden_states"],
                attention_mask=feature["attention_mask"],
                filler_token=0,
                padding_side=padding_side
            )
        return feature

    def forward(self, obs: PyTorchObs, labels = None, return_hidden_state=False) -> torch.Tensor:
        """
        Forward pass in the policy. This is used to compute the loss in the training loop.
        and called by the rl algorithm to sample and fill the rollout buffer.
        """
        # generate the actions, get the next_observation=obs+actions and cutaway the excessive pads
        _, actions, _ = self._predict(obs, return_dict= True).values()
        # obs["last_hidden_states"] = obs_dict["last_hidden_states"][]
        # get as input EXACTLY what the rollout buffer will later give to the policy to be trained.
        values, log_probs, _ = self.evaluate_actions(obs, actions) 

        return actions, values, log_probs
    
    def post_predict(self, inputs: torch.Tensor, outputs: Dict[str, torch.Tensor], return_dict = False) -> torch.Tensor:
        # for on-policy replay buffer, we need to pad the actions to the max length of the action space, to append to
        # the actions matrix in the buffer.
        #remove the input tokens from the output 
        # bsize, seq_len, emb_dim = outputs['last_hidden_states'].size()
        # seq_len = seq_len+1
        # outputs['last_hidden_states'] = torch.cat([outputs['last_hidden_states'], torch.zeros((bsize, 1, emb_dim), device=outputs['last_hidden_states'].device)], dim=1)
        next_obs =  hash_ids_and_hidden_states(**outputs)
        hashed_action = next_obs[:, inputs.shape[-1]:].clone()
    
        filler_token_maxlen_action = hashed_action.clone()
        #replace all pad tokens with filler tokens
        mask = (filler_token_maxlen_action[...,-1] == self.tokenizer.pad_token_id)
        filler_token_maxlen_action[mask] = self.filler_token
        
        
        action_space_dim = self.action_space.shape[0]

        filler_token_maxlen_action = add_filler_tokens(filler_token_maxlen_action, action_space_dim, self.filler_token, dim=1)

        if return_dict:
            return {'next_observation': hash_ids_and_hidden_states(**outputs), 'filler_token_maxlen_actions': filler_token_maxlen_action, 'padded_actions': hashed_action}
        else:
            return filler_token_maxlen_action
    
    def _predict(self, observation: PyTorchObs, deterministic: bool = False, return_dict: bool = False):
        """
        Get the action according to the policy for a given observation.

        By default provides a dummy implementation -- not all BasePolicy classes
        implement this, e.g. if they are a Critic in an Actor-Critic method.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        was_in_training = self.lm.training
        assert self.generation_params_to_use is not None, \
            "You've never set the generation config to use. Please set it using the set_generation_cfg method. Options are 'train' or 'test'"
        generation_params = self.generation_params[self.generation_params_to_use]

        # self.lm.eval()
        og_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        feature = self.extract_features(observation)
        inputs = feature["input_ids"]
        self.pre_predict(feature)

        if not self.use_peft_at_inference:
            self.lm.disable_adapter_layers()

        already_terminated_sequences = (inputs == self.tokenizer.eos_token_id).any(dim = 1) 

        assert generation_params["generation_config"].return_dict_in_generate == True, \
            "return_dict_in_generate must be the same as the return_dict argument"

        if not already_terminated_sequences.all(): 
            # generate those that are not terminated:
            with torch.no_grad():
                inputs_to_generate = inputs[already_terminated_sequences == False]
                attention_mask = feature["attention_mask"][already_terminated_sequences == False]
                outputs = self.lm.generate(
                    inputs = inputs_to_generate,
                    attention_mask = attention_mask,
                    tokenizer=self.tokenizer,
                    **generation_params,
                )
                output_ids = outputs.sequences
                hidden_states = torch.cat([hid[-1] for hid in outputs.hidden_states], dim = 1)
                hidden_states = torch.cat([torch.zeros((hidden_states.shape[0], 1, hidden_states.shape[2]), dtype=hidden_states.dtype, device=hidden_states.device), hidden_states], dim=1)

            outputs_ids = torch.full((inputs.shape[0], output_ids.shape[1]), self.tokenizer.pad_token_id, dtype = output_ids.dtype, device = output_ids.device)
            outputs_ids[already_terminated_sequences, :inputs.shape[1]] = inputs[already_terminated_sequences]
            outputs_ids[~already_terminated_sequences] = output_ids
            outputs_hidden_states = torch.full((inputs.shape[0], hidden_states.shape[1], hidden_states.shape[2]), 0, dtype = hidden_states.dtype, device = hidden_states.device)
            outputs_hidden_states[~already_terminated_sequences] = hidden_states

            # for those who where already completed sequences, do one painful autoregressive forward to get all the thoughts...
            if already_terminated_sequences.any():
                already_term_seq_hidden_states = self.lm.forward(inputs[already_terminated_sequences], attention_mask = feature["attention_mask"][already_terminated_sequences]).hidden_states[-1]
                outputs_hidden_states[already_terminated_sequences, :already_term_seq_hidden_states.shape[1]] = already_term_seq_hidden_states

        else:
            outputs_ids = inputs
            outputs_hidden_states = self.lm.forward(inputs, attention_mask = feature["attention_mask"]).hidden_states[-1]
            
        if not self.use_peft_at_inference:
            self.lm.enable_adapter_layers()
        
        outputs = {"input_ids": outputs_ids, "last_hidden_states": outputs_hidden_states}
        outputs =  self.post_predict(inputs, outputs, return_dict = return_dict)

        if was_in_training:
            self.lm.train()
        self.tokenizer.padding_side = og_padding_side  
        return outputs
    

    def get_next_observation(self, observations, actions, compute_hidden_states = True, device = None):
        
        if device is not None:
            for key in observations:
                observations[key] = observations[key].to(device)
                actions[key] = actions[key].to(device)
        
        # get hidden states for observations only
        if compute_hidden_states:
            input_ids = observations['input_ids']
            input_attention_mask = observations['attention_mask']
            with torch.no_grad():
                hidden_states = self.lm.forward(input_ids, attention_mask=input_attention_mask).hidden_states[-1].detach()
            # assert actions['last_hidden_states'][0][0] == hidden_states[0][max(torch.where(input_attention_mask[0])[0])]
            hidden_states = torch.cat([torch.zeros((hidden_states.shape[0], 1, hidden_states.shape[2]), device=hidden_states.device, dtype=hidden_states.dtype), hidden_states[:, :-1, :]], dim=1)
            observations['last_hidden_states'] = hidden_states
        
        next_obs_input_ids = []
        next_obs_hidden_states = []
        #remove the filler tokens from the actions and observations
        obs_list = remove_filler_tokens_from_hashed_array(
            hash_ids_and_hidden_states(
                input_ids=observations["input_ids"],
                last_hidden_states=observations["last_hidden_states"]
                ),
            self.tokenizer.pad_token_id
        )
        actions_list = remove_filler_tokens_from_hashed_array(
            hash_ids_and_hidden_states(
                input_ids=actions["input_ids"],
                last_hidden_states=actions["last_hidden_states"]
                ),
            self.tokenizer.pad_token_id
        )

        #concatenate the observations and actions
        for obs, action in zip(obs_list, actions_list):
            tmp_dict_obs = unhash_ids_and_hidden_states(obs)
            tmp_dict_action = unhash_ids_and_hidden_states(action)
            next_obs_input_ids.append(torch.cat([tmp_dict_obs["input_ids"], tmp_dict_action["input_ids"]]))
            next_obs_hidden_states.append(torch.cat([tmp_dict_obs["last_hidden_states"], tmp_dict_action["last_hidden_states"]]))

        #pad the observations
        new_observations = self.tokenizer.pad({"input_ids": next_obs_input_ids}, return_tensors="pt", 
                                              padding=True, padding_side="right").to(self.device)
        
        new_observations["last_hidden_states"] = torch.zeros(
            (new_observations["input_ids"].shape[0], new_observations["input_ids"].shape[1], observations["last_hidden_states"].shape[2]),
            dtype=observations["last_hidden_states"].dtype,
            device=observations["last_hidden_states"].device
        )   
        for i in range(new_observations["input_ids"].shape[0]):
            new_observations["last_hidden_states"][i, :next_obs_hidden_states[i].shape[0]] = next_obs_hidden_states[i]
        return new_observations

    def evaluate_actions(self, obs, acts, lm=None):
        """
        Evaluate actions. Used in the training loop to train the policy.
        Returns:
            - values: Predicted state values (for value loss).
            - log_prob: Log probability of the actions (for policy gradient loss).
            - entropy: Entropy of the policy (for exploration bonus).
        """
        # if there are filler tokens in the actions or observations, exchange them with pad tokens (filler tokens 
        # are used to mask the actions in the observations
        if lm is None:
            lm = self.lm
        observations = self.extract_features(obs)
        actions = self.extract_features(acts)
        # Compute next observations and prepare for LM processing
        next_obs = self.get_next_observation(observations, actions) # Assuming this is defined elsewhere
        # forward pass through the model
        
        outputs = lm(**next_obs, output_hidden_states=True)
        logits = outputs.logits  # Forward pass through LM
        all_logprobs = torch.log_softmax(logits, dim=-1)  # Convert logits to log-probabilities
        
        # if the model is being finetuned on the demonstrations too, then remove those from the obs and
        # append to the actions.
        reduced_observations, _ = self.augment_actions_reduce_observations(next_obs['input_ids'])

        # position that produced thought have an id of (true_id + vocab_size)
        next_obs["input_ids"] = torch.where(
            next_obs["input_ids"] >= len(self.tokenizer),
            next_obs["input_ids"] - len(self.tokenizer),
            next_obs["input_ids"]
        ).to(self.device)

        if not self.ft_on_action_only:
            action_start_indices = (reduced_observations['input_ids'] != self.tokenizer.pad_token_id).sum(dim=1) - 1
        elif self.ft_on_question_too:
            action_start_indices = torch.zeros(observations['input_ids'].size(0), device=observations['input_ids'].device, dtype=torch.long)
        else:
            # Compute action log probabilities
            action_start_indices = (observations['input_ids'] != self.tokenizer.pad_token_id).sum(dim=1) - 1

        log_probs = self._compute_logprobs(
            all_logprobs[:, :-1, ...], next_obs['input_ids'][:, 1:], 
            action_start_indices, per_token_log_prob=self.per_token_log_prob
        )

        # Compute values
        if self.use_same_model_for_value:
            raw_latent = outputs.hidden_states
        else:
            raw_latent = self.value_lm(**observations, output_hidden_states=True).hidden_states
        
        # get observation mask in next_obs, only attend the obs!
        obs_mask = torch.ones(raw_latent[-1].size(0), raw_latent[-1].size(1), device=raw_latent[-1].device, dtype=torch.bool)
        if self.value_function_only_on_question:
            for i in range(obs_mask.size(0)):
                obs_length_i = reduced_observations['attention_mask'][i].sum().item()
                obs_mask[i, obs_length_i:] = 0
        else:    
            for i in range(obs_mask.size(0)):
                obs_mask[i, action_start_indices[i]:] = 0
       
        values = self.value_forward_pass(raw_latent, obs_mask)
        entropy = - (log_probs * log_probs.exp()).sum(dim=-1).mean()
        return values, log_probs, entropy
  


########################################################################################################################
# some code that might be useful for debugging later:

###################################################
# No need for any of this! can be used in _predict to check if the sequential forward is the same as normal forward
# att_mask = (output_ids != self.tokenizer.pad_token_id).long()
# no need for thought attention mask. it's simply simply simply calculated for token_id>vocab_size.
# action_start_index = attention_mask.shape[1]
# thought_attn_mask = att_mask[:, :-1].clone()
# thought_attn_mask = torch.cat([torch.zeros((thought_attn_mask.shape[0], 1,), dtype=thought_attn_mask.dtype, device=tmp_hidden_states.device), thought_attn_mask], dim=1)
# shifting the hidden states one to the right, because the thought of M(x_<t) is generates x_t and is summed with x_t to get x_t+1
# tmp_hidden_states = torch.cat([torch.zeros((tmp_hidden_states.shape[0], 1, tmp_hidden_states.shape[2]), dtype=tmp_hidden_states.dtype, device=tmp_hidden_states.device), tmp_hidden_states], dim = 1)
# with torch.no_grad():
    # hidden_states = self.lm.forward(output_ids, attention_mask=att_mask, last_hidden_states=tmp_hidden_states).hidden_states[-1]
    # stupid_hidden_75 = self.lm.forward(output_ids[:1, :75], attention_mask=att_mask[:1, :75], last_hidden_states=tmp_hidden_states[:1, :75]).hidden_states[-1]
    # stupid_hidden_76 = self.lm.forward(output_ids[:1, :76], attention_mask=att_mask[:1, :76], last_hidden_states=tmp_hidden_states[:1, :76]).hidden_states[-1]

# att_mask = (output_ids != self.tokenizer.pad_token_id).long()
# u = torch.where(att_mask[0])[0]
# min_u = u[0] - 1
# max_u = u[-1] + 2
# leftzero_h = torch.cat([torch.zeros((hidden_states.shape[0], 1, hidden_states.shape[2]), dtype=hidden_states.dtype, device=hidden_states.device), hidden_states], dim = 1)
# rightzero_h = torch.cat([hidden_states, torch.zeros((hidden_states.shape[0], 1, hidden_states.shape[2]), dtype=hidden_states.dtype, device=hidden_states.device)], dim = 1)
# left_thought_attn_mask = att_mask[:, :-1].clone()
# left_thought_attn_mask = torch.cat([torch.zeros((left_thought_attn_mask.shape[0], 1,), dtype=left_thought_attn_mask.dtype, device=left_thought_attn_mask.device), left_thought_attn_mask], dim=1)

# recomputed_hidden_states = self.lm.forward(output_ids, attention_mask=att_mask, last_hidden_states=hidden_states).hidden_states[-1]
# leftzero_hidden_states = self.lm.forward(output_ids, attention_mask=att_mask, last_hidden_states=leftzero_h, thought_attention_mask=left_thought_attn_mask).hidden_states[-1]
# rightzer_hidden_states = self.lm.forward(output_ids, attention_mask=att_mask, last_hidden_states=rightzero_h).hidden_states[-1]

# recomputed_hidden_states[0, min_u:max_u] - hidden_states[0, min_u:max_u]
# leftzero_hidden_states[0, min_u:max_u] - hidden_states[0, min_u:max_u]
# rightzer_hidden_states[0, min_u:max_u] - hidden_states[0, min_u:max_u]


# self.lm.eval()
# left_thought_pert = self.lm.thought_embedding_head(leftzero_h, attention_mask=left_thought_attn_mask) 
# thought_pert = self.lm.thought_embedding_head(hidden_states, attention_mask=att_mask[:, :-1]) 

###################################################

