import base64
import torch

from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, Qwen2VLProcessor
from datasets import load_dataset
from io import BytesIO
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from qwen_vl_utils import process_vision_info

PROMPT = """
Given 3 functions to move the robot 'down', 'left', 'right', use a composition of these functions tomove the end effector neat the Brown Cup.
Answer in json format containg a list of functions like: {"actions": ["down", "down"]} or {"actions": ["left", "down", "down"]}
"""


dataset_id = "calebgcc/llama-moss"

dataset = load_dataset(dataset_id)

print(dataset['train'][140]['label'])
dataset['train'][140]['image'].show()

label_mapping = {
    0: '{"actions":["down", "down"]}',
    1: '{"actions":["left", "down", "down"]}',
    2: '{"actions":["left", "left", "down", "down"]}',
    3: '{"actions":["right", "down", "down"]}',
}

def format_data(sample):
    buffered = BytesIO()
    sample['image'].save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an export robot controller"}]
            }, 
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label_mapping[sample['label']]}]
            }
        ]
    }


dataset = [format_data(sample) for sample in dataset["train"]]

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=bnb_config)
processor = AutoProcessor.from_pretrained(model_id)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

args = SFTConfig(
    output_dir="llama-moss-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    lr_scheduler_type="constant",
    logging_steps=5,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    push_to_hub=True,
    report_to="tensorboard",
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True}
)

args.remove_unused_columns = False

def collate_fn(examples):
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    if isinstance(processor, Qwen2VLProcessor):
        image_tokens = [151652,151653,151655]
    else: 
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch


trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collate_fn,
    dataset_text_field="",
    peft_config=peft_config,
    tokenizer=processor.tokenizer,
)

trainer.train()
trainer.save_model(args.output_dir)