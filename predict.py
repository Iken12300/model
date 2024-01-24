
from transformers import AutoTokenizer, MT5ForConditionalGeneration
import torch

def predict(text, tokenizer, model):
    tokenized_text = tokenizer.encode(text, return_tensors="pt").to('cuda')

    summary_ids = model.generate(
        tokenized_text,
        max_length=150,
        num_beams=5,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained('google/mt5-small')
    model = MT5ForConditionalGeneration.from_pretrained("\model")
    model.to('cuda')
    print(predict("el es fabian", tokenizer, model))
