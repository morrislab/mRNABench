import torch

assert torch.cuda.is_available()
device = torch.device("cuda")

# Orthrus
from mamba_ssm.modules.mamba_simple import Mamba

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2).to("cuda")
y = model(x)
print("Mamba functional.")

# RNA-FM
import fm
model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model = model.to(device).eval()

data = [("RNA3", "CGAUUCNCGUUCCC--CCGCCUCCA")]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

with torch.no_grad():
    results = model(batch_tokens.to(device), repr_layers=[12])
token_embeddings = results["representations"][12]

print("RNA-FM functional.")

# DNABERT2
from transformers import AutoTokenizer, AutoModel
from transformers import logging

logging.set_verbosity_error()

tokenizer = AutoTokenizer.from_pretrained(
    "zhihan1996/DNABERT-2-117M",
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    "zhihan1996/DNABERT-2-117M",
    trust_remote_code=True
).to(device)

dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna, return_tensors='pt')["input_ids"].to(device)
hidden_states = model(inputs)[0]

print("DNA-BERT2 functional.")

# NT
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
)
model = AutoModelForMaskedLM.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
)
max_length = tokenizer.model_max_length

sequence = "ATTCCGATTCCGATTCCG"
tokens_ids = tokenizer.encode_plus(
    sequence,
    return_tensors="pt",
)["input_ids"]

attention_mask = tokens_ids != tokenizer.pad_token_id
torch_outs = model(
    tokens_ids,
    attention_mask=attention_mask,
    encoder_attention_mask=attention_mask,
    output_hidden_states=True
)

embeddings = torch_outs['hidden_states'][-1]


print("NT functional.")

# HyenaDNA
from transformers import AutoModelForSequenceClassification

checkpoint = 'LongSafari/hyenadna-tiny-1k-seqlen-hf'
max_length = 10

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

sequence = 'ACTG' * int(max_length / 4)
tok_seq = tokenizer(sequence)
tok_seq = tok_seq["input_ids"]  # grab ids

tok_seq = torch.LongTensor(tok_seq).unsqueeze(0)  # unsqueeze for batch dim
tok_seq = tok_seq.to(device)

# prep model and forward
model.to(device)
model.eval()
with torch.inference_mode():
    embeddings = model(tok_seq)

print("HyenaDNA functional.")
