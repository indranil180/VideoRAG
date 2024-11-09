import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor, pipeline

base_model = "meta-llama/Llama-3.2-11B-Vision-Instruct"

hf_auth= "add_the_token"
processor = AutoProcessor.from_pretrained(base_model,token=hf_auth)

model = MllamaForConditionalGeneration.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=hf_auth
)

def lvlm_inference_module(result):
    
    messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Refer to the below context to answer the query accurately. Avoid making unsupported assumptions or generating information that isn't present in the context.\n"+result["prompt"]}
    ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(result["image"], input_text, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=122,pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id, temperature=0.95 )
    return (processor.decode(output[0]).replace(input_text,""))