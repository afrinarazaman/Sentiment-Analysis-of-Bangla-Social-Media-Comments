from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "sagorsarker/bangla-bert-base",
    use_fast=True
)

tokenizer.save_pretrained("model_parameter/tokenizer")
