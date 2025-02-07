from torch import nn
from transformers.utils import ModelOutput
from transformers import PreTrainedModel,AutoModelForCausalLM
from dataclasses import dataclass
from typing import Optional, Callable, List, Union
import torch
from peft import PeftConfig, PeftModel
import warnings
import os
from src.model.components.control_token_wrappers.base_control_token_wrapper import \
    BaseCtrlTokConfig, BaseControlTokenWrapper
from src.utils.constants import CTRL_TOKEN_LABEL, LM_HEAD_LABEL, IGNORE_LABEL
import hydra
from omegaconf.omegaconf import DictConfig
from src.model.components.control_token_wrappers.base_control_token_wrapper import SequenceClassifierOutputWithPastForCtrlTokens
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateOutput,GenerateNonBeamOutput, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
from transformers.generation.configuration_utils import GenerationMode


class ThoughtEncodingMethod:
    MUST_ENCODE = "must-encode"
    ALREADY_ENCODED = "already-encoded"

class ThoughtPerturbatorConfig(BaseCtrlTokConfig):    
    def __init__(
        self,
        thought_token_id: int = None,
        thought_token_id_token_name: str = "<|thought|>",
        **kwargs
    ):
        if kwargs.get("control_token_to_id") is not None:
            kwargs["control_token_to_id"][thought_token_id] = thought_token_id_token_name
        else:
            kwargs["control_token_to_id"] = {thought_token_id: thought_token_id_token_name}
        kwargs["add_ctrl_tok_to_lm_head"] = False
        kwargs["add_ctrl_tok_to_embeddings"] = False
        super().__init__(**kwargs)
        self.thought_token_id = thought_token_id
        self.thought_token_id_token_name = thought_token_id_token_name
        
class ThoughtPerturbator(BaseControlTokenWrapper):
    config_class = ThoughtPerturbatorConfig

    def __init__(self, thought_embedding_head: DictConfig, **kwargs):
        super().__init__(**kwargs)
        self.thought_embedding_head = hydra.utils.instantiate(thought_embedding_head, _recursive_=False).to(next(self.language_model.parameters()).dtype)
    
    def ctrl_tok_execute(self, labels: torch.LongTensor, token_name: str, **kwargs):
        """ Execute function of pause token. Returns CTRL_TOKEN_LABEL anywhere the pause token is present in the labels tensor and LM_HEAD_LABEL elsewhere. 
        This function is used to determine whether each token in the input sequence is a control token (or part of a control token) or not. It's also used to determine the loss of the model.
        
        :param labels: torch.LongTensor of shape (batch_size, seq_len) containing the labels of the input sequence
        :param token_name: str, name of the token to execute
        :returns: torch.LongTensor of shape (batch_size, seq_len) containing the labels of the input sequence with the pause token replaced by CTRL_TOKEN_LABEL and the other tokens replaced by LM_HEAD_LABEL
        """
        if token_name == self.config.thought_token_id:
            return torch.where(labels == self.config.thought_token_id, CTRL_TOKEN_LABEL, LM_HEAD_LABEL)
        raise ValueError(f"Token name {token_name} not recognized")
    
    def thought_perturbator_forward(self, latent, attention_mask):
        perturbated_thoughts = self.value_head(latent, attention_mask=attention_mask)
        return perturbated_thoughts

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if generation_mode not in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            raise ValueError(f"Generation mode {generation_mode} not supported for generation. Only {GenerationMode.SAMPLE} and {GenerationMode.GREEDY_SEARCH} are supported.")
    
        return super().generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,
        )

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        pass
        # # init values
        # pad_token_id = generation_config._pad_token_tensor
        # output_attentions = generation_config.output_attentions
        # output_hidden_states = True
        # output_scores = generation_config.output_scores
        # output_logits = generation_config.output_logits
        # return_dict_in_generate = generation_config.return_dict_in_generate
        # max_length = generation_config.max_length
        # has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        # do_sample = generation_config.do_sample

        # # init attention / hidden states / scores tuples
        # scores = () if (return_dict_in_generate and output_scores) else None
        # raw_logits = () if (return_dict_in_generate and output_logits) else None
        # decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        # cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        # decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        # if return_dict_in_generate and self.config.is_encoder_decoder:
        #     encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        #     encoder_hidden_states = (
        #         model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        #     )

        # # keep track of which sequences are already finished
        # batch_size, cur_len = input_ids.shape
        # this_peer_finished = False
        # unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        # model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        # while self._has_unfinished_sequences(
        #     this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
        # ):
        #     # prepare model inputs
        #     model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        #     # prepare variable output controls (note: some models won't accept all output controls)
        #     model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        #     model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        #     # forward pass to get next token
        #     outputs = self(**model_inputs, return_dict=True)

        #     # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        #     model_kwargs = self._update_model_kwargs_for_generation(
        #         outputs,
        #         model_kwargs,
        #         is_encoder_decoder=self.config.is_encoder_decoder,
        #     )
        #     if synced_gpus and this_peer_finished:
        #         continue

        #     # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        #     # (the clone itself is always small)
        #     next_token_logits = outputs.logits.clone()[:, -1, :].float()
        #     next_token_logits = next_token_logits.to(input_ids.device)

        #     # pre-process distribution
        #     next_token_scores = logits_processor(input_ids, next_token_logits)

        #     # Store scores, attentions and hidden_states when required
        #     if return_dict_in_generate:
        #         if output_scores:
        #             scores += (next_token_scores,)
        #         if output_logits:
        #             raw_logits += (next_token_logits,)
        #         if output_attentions:
        #             decoder_attentions += (
        #                 (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
        #             )
        #             if self.config.is_encoder_decoder:
        #                 cross_attentions += (outputs.cross_attentions,)

        #         if output_hidden_states:
        #             decoder_hidden_states += (
        #                 (outputs.decoder_hidden_states,)
        #                 if self.config.is_encoder_decoder
        #                 else (outputs.hidden_states,)
        #             )

        #     # token selection
        #     if do_sample:
        #         probs = nn.functional.softmax(next_token_scores, dim=-1)
        #         # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
        #         next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        #     else:
        #         next_tokens = torch.argmax(next_token_scores, dim=-1)

        #     # finished sentences should have their next token be a padding token
        #     if has_eos_stopping_criteria:
        #         next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        #     # update generated ids, model inputs, and length for next step
        #     input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        #     if streamer is not None:
        #         streamer.put(next_tokens.cpu())

        #     unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        #     this_peer_finished = unfinished_sequences.max() == 0
        #     cur_len += 1

        #     # This is needed to properly delete outputs.logits which may be very large for first iteration
        #     # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        #     del outputs

        # if streamer is not None:
        #     streamer.end()

        # if return_dict_in_generate:
        #     if self.config.is_encoder_decoder:
        #         return GenerateEncoderDecoderOutput(
        #             sequences=input_ids,
        #             scores=scores,
        #             logits=raw_logits,
        #             encoder_attentions=encoder_attentions,
        #             encoder_hidden_states=encoder_hidden_states,
        #             decoder_attentions=decoder_attentions,
        #             cross_attentions=cross_attentions,
        #             decoder_hidden_states=decoder_hidden_states,
        #             past_key_values=model_kwargs.get("past_key_values"),
        #         )
        #     else:
        #         return GenerateDecoderOnlyOutput(
        #             sequences=input_ids,
        #             scores=scores,
        #             logits=raw_logits,
        #             attentions=decoder_attentions,
        #             hidden_states=decoder_hidden_states,
        #             past_key_values=model_kwargs.get("past_key_values"),
        #         )
        # else:
        #     return input_ids
    

    def make_input_embeddings_from_input_ids(self, input_ids: torch.LongTensor, thought_hidden_states: torch.Tensor, thought_mask: torch.LongTensor):
        # find locations where thought token is present (when token_id > vocab_size)
        input_embeddings = self.language_model.get_input_embeddings()

        ids = torch.where( thought_mask.bool(), input_ids - self.language_model.config.vocab_size, input_ids)

        inputs_embeds = input_embeddings(ids)
        
        return inputs_embeds

    def make_thought_hidden_state_from_input_embeddings(
            self,
            inputs_embeds: torch.Tensor = None,
            attention_mask: torch.LongTensor = None,
            thought_hidden_states: torch.Tensor = None,
            thought_mask: torch.LongTensor = None,
        ):
        
        
        thought_encoding_method = self.determine_thought_encoding_method(thought_mask, thought_hidden_states)
        
        if thought_encoding_method == ThoughtEncodingMethod.MUST_ENCODE:

            embeds = inputs_embeds.clone()

            thought_pos = thought_mask.nonzero(as_tuple=True)
            thought_seq_positions = thought_pos[1] if len(thought_pos) > 1 else []
   
            unique_thought_seq_positions_sorted = [] if len(thought_seq_positions) == 0 else thought_seq_positions.unique(sorted=True)
            
            for position in unique_thought_seq_positions_sorted:
                _, _, _, thought_hidden_states, _  = self.forward_(
                    input_ids=None,
                    inputs_embeds=embeds[:, :position+1, :],
                    attention_mask=attention_mask[:, :position+1],
                )
                thought_hidden_states = thought_hidden_states[-1]
                if thought_mask.any():
                    thought_perturbance = self.thought_embedding_head(thought_hidden_states)
                    embeds[:, position, :] += thought_mask.unsqueeze(-1)[:,position,:] * thought_perturbance[:,position,:]

            if len(unique_thought_seq_positions_sorted) == 0 or unique_thought_seq_positions_sorted[-1] != embeds.size(1) - 1:
                _, _, _, thought_hidden_states, _  = self.forward_(
                        input_ids=None,
                        inputs_embeds=embeds,
                        attention_mask=attention_mask,
                    )
                thought_hidden_states = thought_hidden_states[-1]

            return thought_hidden_states
        
        else:
            return thought_hidden_states

    def _validate_input_ids_and_thoughts(self, input_ids: torch.LongTensor, thought_mask: torch.Tensor):
        thought_mask_ = input_ids >= self.language_model.config.vocab_size
        assert (thought_mask == thought_mask_).all(), "thought_mask and input_ids mismatch. They must be consistent"

    def _validate_input_arguments(
            self,
            input_ids: torch.LongTensor,
            inputs_embeds: torch.Tensor
        ):

        assert (input_ids is not None and inputs_embeds is None) or (input_ids is None and inputs_embeds is not None), \
            "Either input_ids or input_embeds should be provided (but not both)"

    def make_thought_mask(self, input_ids: torch.LongTensor):
        return (input_ids >= self.language_model.config.vocab_size).long()

    def determine_thought_encoding_method(self, thought_mask: torch.LongTensor, thought_hidden_states: torch.Tensor):
        
        ##### LOGIC IF FLAWED HERE #######
        # I need to make it fit with the for loop.
        # Somehow When you're trying to decode the thought in the last sequence position, it's different from when you're trying to decode it in the middle of the sequence.
        # Also, the if thought_hidden_states is None is not sufficient, there are also cases where thought_hidden_states is not None but we have to Encode
        # Think about the shape of the hidden state ? Needs to match the shape of the thought mask maybe ?
        return ThoughtEncodingMethod.MUST_ENCODE if (thought_mask.any()) else ThoughtEncodingMethod.ALREADY_ENCODED
        
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            inputs_embeds: torch.Tensor = None,
            attention_mask: torch.LongTensor = None,
            thought_hidden_states: torch.Tensor = None,
            thought_mask: torch.LongTensor = None,
            labels: Optional[torch.Tensor] = None,
            *args,
            **kwargs
        ):
        
        self._validate_input_arguments(input_ids, inputs_embeds)

        if input_ids is not None:
            #If no thought mask is provided, create one from input_ids
            thought_mask = self.make_thought_mask(input_ids) if thought_mask is None else thought_mask
            # Make sure thought mask is consistent with input_ids
            self._validate_input_ids_and_thoughts(input_ids, thought_mask)

            # Convert input_ids to inputs_embeds. Masani Idea; for a thought, input_id + vocab_size = thought_id
            inputs_embeds = self.make_input_embeddings_from_input_ids(input_ids, thought_hidden_states, thought_mask)


        thought_hidden_states = self.make_thought_hidden_state_from_input_embeddings(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            thought_hidden_states=thought_hidden_states,
            thought_mask=thought_mask,
        )

        #Add thoughts to input embeddings in the case there are any thoughts to be added
        if thought_mask.any():
            thought_perturbance = self.thought_embedding_head(thought_hidden_states)
            inputs_embeds += thought_mask.unsqueeze(-1) * thought_perturbance
        
        reduce_mean = kwargs.pop("reduce_mean",True)
        lm_logits, ctrl_tok_logits, past_key_values, hidden_states, attentions  = \
            self.forward_(input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, *args, **kwargs)
        
        if labels is not None:
            
            loss, lm_loss, ctrl_tok_loss = self.compute_loss(
                labels=labels,
                lm_logits=lm_logits,
                ctrl_tok_logits=ctrl_tok_logits,
                reduce_mean=reduce_mean,
            )
        else:
            loss = None
            lm_loss = None
            ctrl_tok_loss = None
                
        return SequenceClassifierOutputWithPastForCtrlTokens(
            loss = loss,
            logits = lm_logits,
            past_key_values = past_key_values,
            hidden_states = hidden_states,
            attentions = attentions,
            control_token_logits = ctrl_tok_logits,
            lm_logits = lm_logits,
            lm_loss = lm_loss,
            ctrl_tok_loss = ctrl_tok_loss,
        )
        


def load_model(thought_embed_config):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    lm = AutoModelForCausalLM.from_pretrained("gpt2")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    #get the pause token id
    thought_token_id = tokenizer.vocab_size
    config = ThoughtPerturbatorConfig(thought_token_id=thought_token_id, thought_token_name="<|thought|>")
    model = ThoughtPerturbator(thought_embedding_head = thought_embed_config, config = config, language_model = lm)
    return model, tokenizer

def test_forward_pass():
    model, tokenizer = load_model(TRANSFORMER_THOUGHT_CONFIG)
    inputs = tokenizer(["Hello world", "Bye world"], return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    thought_token = 100 + tokenizer.vocab_size
    input_ids = torch.cat([input_ids, torch.tensor([[thought_token], [thought_token]])], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones(2, 1)], dim=1)
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    inputs = tokenizer(["Hello world", "Bye world"], return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    thought_token = 100 + tokenizer.vocab_size
    input_ids = torch.cat([input_ids, torch.tensor([[thought_token], [thought_token]]), input_ids, torch.tensor([[thought_token], [thought_token]]),input_ids ], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones(2, 1), attention_mask, torch.ones(2, 1),attention_mask], dim=1)
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    
# def test_inference():
#     model, tokenizer = load_model()

#     tokenized_seq = tokenizer("<|endoftext|> Hello world", return_tensors="pt")
    
#     output = model(**tokenized_seq)
    
#     print("next_predicted_token", tokenizer.decode(output.logits[...,-1].argmax(dim=-1).item()))
    
# def test_save_load_peft():
#     def same_state_dicts(dict1, dict2):
#         for key in dict1:
#             if key not in dict2:
#                 return False
#             elif isinstance(dict1[key], dict):
#                 if not same_state_dicts(dict1[key], dict2[key]):
#                     return False
#             elif not torch.allclose(dict1[key], dict2[key]):
#                 return False
#         return True
#     from peft import LoraConfig, TaskType, get_peft_model
#     from transformers import AutoTokenizer
#     model, tokenizer = load_model()
#     model.save_pretrained("pre_lora")
#     peft_config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM, 
#         inference_mode=False,
#         target_modules=["c_attn"]
#     )
    
#     model = get_peft_model(model, peft_config)
#     model.save_pretrained("test_pause_model")
#     tokenizer.save_pretrained("test_pause_model")
    
#     model2 = PauseClassifierWrapper.from_pretrained("test_pause_model")
#     tokenizer2 = AutoTokenizer.from_pretrained("test_pause_model")
    
#     assert same_state_dicts(model.state_dict(), model2.state_dict())
#     #delete directory
#     os.system("rm -r test_pause_model")
#     os.system("rm -r pre_lora")

TRANSFORMER_THOUGHT_CONFIG = {
    "_target_": "src.model.components.thought_embeddings.torch_transformer.ThoughtTransformer",
    "hidden_dim": 768,
    "transformer_config": {
        "_target_": "transformers.GPT2Model",
        "config": {
            "_target_": "transformers.GPT2Config",
            "vocab_size": 0,
            "n_embd": 768,
            "n_layer": 8,
            "n_head": 8,
            "n_positions": 1025,
        }
    }
}



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print("testing model loading...")
    test_forward_pass()
    print("testing inference...")
    # test_inference()
    print("testing inference done")
    print("testing model loading and saving...")
    # test_save_load_peft()
    # print("testing model loading and saving done")
    
