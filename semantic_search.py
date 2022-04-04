import nltk
import pandas as pd
import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer, util

model_name = 'sentence-transformers/msmarco-distilbert-base-v4'
max_sequence_length = 512
embeddings_filename = 'df10k_embeddings_msmarco-distilbert-base-v4.npy'
nltk.download('punkt')
filename = 'gs_10k_2021.txt'

# open 10k text file
textfile = open(filename, 'r')
text_corpus = textfile.read()

corpus = []
sentence_count = []
sentences = nltk.tokenize.sent_tokenize(text_corpus, language='english')
sentence_count.append(len(sentences))
for _, s in enumerate(sentences):
    corpus.append(s)
print(f'Number of sentences: {len(corpus)}')

# Load pre-embedded corpus
corpus_embeddings = np.load("df10k_embeddings_msmarco-distilbert-base-v4.npy")
print(f'Number of embeddings: {corpus_embeddings.shape[0]}')

# Load embedding model
model = SentenceTransformer(model_name)
model.max_seq_length = max_sequence_length


def find_sentences(query, hits):
    query_embedding = model.encode(query)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=hits)
    hits = hits[0]
    print(hits)
    print(hits)

    output = pd.DataFrame(columns=['Text', 'Score'])
    for hit in hits:
        corpus_id = hit['corpus_id']
        # Find source document based on sentence index
        count = 0
        new_row = {
            'Text': corpus[corpus_id],
            'Score': '{:.2f}'.format(hit['score'])
        }
        output = output.append(new_row, ignore_index=True)
    print(output)
    return output


def process(query):
    text = query
    return text, find_sentences(text, 5)


# Gradio inputs
text_query = gr.inputs.Textbox(lines=1, label='Text input', default='Great Opportunity')

# Gradio outputs
speech_query = gr.outputs.Textbox(type='auto', label='Query string')
results = gr.outputs.Dataframe(
    headers=['Text', 'Score'],
    label='Query results')

# Gradio interface
iface = gr.Interface(
    theme='huggingface',
    description='Great Opportunity in business',
    fn=process,
    inputs=[text_query],
    outputs=[speech_query, results],
    examples=[
        ['Great Opportunity in business'],
        ['LIBOR replacement'],
        ['Marquee'],
    ],
    allow_flagging=False
)
iface.launch()
