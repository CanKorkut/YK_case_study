from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_t5_model():
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
    return tokenizer, t5_model