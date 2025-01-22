from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
model_path = "./lora_finetuned_model"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Ensure the pad_token_id is set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Function to generate text based on a prompt
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
if __name__ == "__main__":
    while True:
        prompt = input("Enter a prompt (type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
        generated_text = generate_text(prompt)
        print("Generated Text:")
        print(generated_text)