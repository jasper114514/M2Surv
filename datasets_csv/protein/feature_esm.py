import pandas as pd
import torch
import esm

checkpoint_path = 'esm2_t33_650M_UR50D.pt'

model_data = torch.load(checkpoint_path, map_location='cpu')

model, alphabet = esm.pretrained.load_model_and_alphabet_core(
    "esm2_t33_650M_UR50D",
    model_data
)
print(model)
batch_converter = alphabet.get_batch_converter()
model.eval() 

#
csv_file = 'csv/combined_protein_targets.csv'
df = pd.read_csv(csv_file)

#
for index, row in df.iterrows():
    peptide_target = row['peptide_target']
    sequence = row['sequence']

    if pd.isna(sequence):
        print(f"Skipping {peptide_target} due to missing sequence.")
        continue

    
    data = [(peptide_target, sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    #
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    
    token_representations = results["representations"][33]

    #
    sequence_representation = token_representations[0, 1 : len(sequence) + 1].mean(0)
    print(sequence_representation.shape)
    #
    output_file = f'embeddings/{peptide_target}.pt'
    torch.save(sequence_representation, output_file)

print("Embeddings have been saved as .pt files.")