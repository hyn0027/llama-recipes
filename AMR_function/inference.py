# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
from tqdm import tqdm

import torch
from transformers import LlamaTokenizer
from transformers.utils.logging import set_verbosity_error


from llama_recipes.inference.safety_utils import get_safety_checker, AgentType
from llama_recipes.inference.model_utils import load_model, load_peft_model
from custom_dataset import AMRDataset

from accelerate.utils import is_xpu_available

def main(
    model_name,
    data_path: str=None,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens=100, #The maximum numbers of tokens to generate
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    output_path: str=None,
    **kwargs
):
    set_verbosity_error()
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = load_model(model_name, quantization)
    
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()
    
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    if data_path is None:
        raise ValueError("data_path must be provided")
    dataset = AMRDataset(tokenizer, data_path, "test")
    data_num = len(dataset)
    result = []
    for i in tqdm(range(data_num)):
        user_prompt = dataset.get_item_without_answer(i)
        batch = tokenizer(user_prompt, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt", verbose=False)
        if is_xpu_available():
            batch = {k: v.to("xpu") for k, v in batch.items()}
        else:
            batch = {k: v.to("cuda") for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs 
            )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True, verbose=False)
        
        # keep the portion after Transflation: 
        output_text = output_text[output_text.find("Translation: ")+len("Translation: "):]
        result.append(output_text)
    
    if output_path is not None:
        with open(output_path, "w") as f:
            for line in result:
                f.write(line+"\n")
                

if __name__ == "__main__":
    fire.Fire(main)
