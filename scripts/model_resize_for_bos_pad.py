from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
model_name = "gpt2-xl"
save_dir = "./logs/checkpoints/resized_gpt2-xl"

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    assert model.config.vocab_size == len(tokenizer), "Vocab size mismatch"

    # Explicitly add separate <PAD> and <BOS> tokens
    added_tokens = {}
    
    if tokenizer.pad_token_id == tokenizer.eos_token_id or tokenizer.pad_token_id is None:
        added_tokens["pad_token"] = "<PAD>"
    
    if tokenizer.bos_token_id == tokenizer.eos_token_id or tokenizer.bos_token_id is None:
        added_tokens["bos_token"] = "<BOS>"

    if added_tokens:
        tokenizer.add_special_tokens(added_tokens)

    # Ensure tokenizer recognizes the new tokens
    tokenizer.pad_token = "<PAD>"
    tokenizer.bos_token = "<BOS>"

    new_pad_token_id = tokenizer.pad_token_id
    new_bos_token_id = tokenizer.bos_token_id
    
    new_pad_token_id = tokenizer.pad_token_id
    new_bos_token_id = tokenizer.bos_token_id
    
    # Resize the model to 50256 tokens
    model.resize_token_embeddings(len(tokenizer) + 2)
    
    model.config.pad_token_id = new_pad_token_id
    model.config.bos_token_id = new_bos_token_id
     
    new_embedding = model.get_input_embeddings().weight.clone()
        
    eos_token_id = tokenizer.eos_token_id

    eos_token_embed = new_embedding[eos_token_id]
    
   # make the last 2 embeddings of the model to be the eos token embedding
    new_embedding[-2] = eos_token_embed.clone()
    new_embedding[-1] = eos_token_embed.clone()
    
    # make embedding contiguous
    model.set_input_embeddings(torch.nn.Embedding.from_pretrained(new_embedding.contiguous(), freeze=False))

    # assert that the last 2 embeddings are the same
    model_embeds = model.get_input_embeddings()
    assert torch.equal(model_embeds.weight[-1], model_embeds.weight[-2]), "Embeddings are not the same"
    assert torch.equal(model_embeds.weight[-1], eos_token_embed), "Embeddings are not the same"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Tokenize something to make sure the model is working
    text = ["Hello, my name is", "Jerry has 2 apples and one orange"]
    input_ids = tokenizer(text, padding= True, return_tensors="pt")
    text = tokenizer.batch_decode(input_ids["input_ids"], skip_special_tokens=False)
    
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    

    
    
    
    
    