import numpy as np
from create_dataset import create_dataset
from init_model import load_t5_model

def generate_response(query):
    index, contexts, model = create_dataset()

    query_embedding = model.encode([query])


    k = 3  
    distances, indices = index.search(np.array(query_embedding), k)

    closest_docs = [contexts[i] for i in indices[0]]

    tokenizer, t5_model = load_t5_model()

    combined_docs = " ".join(closest_docs)

    input_text = f"summarize: {combined_docs}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    outputs = t5_model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response