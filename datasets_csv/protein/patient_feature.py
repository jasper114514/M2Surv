import pandas as pd
import torch
for st in ['blca', 'brca', 'coadread', 'hnsc', 'stad']:
    csv_file = f'csv/{st}.csv'
    df = pd.read_csv(csv_file)

    for index, row in df.iterrows():
        patient_id = row['patient_id']
        protein_tensors = []

        for i in range(1, 101): 
            expression_value = row.get(f'expression_{i}')
        
            if expression_value: 
                peptide_target, value_str = expression_value.split(':')
                value = float(value_str)

                embedding_path = f'tcga_{st}/{peptide_target}.pt'
                try:
                    embedding_tensor = torch.load(embedding_path)
                    combined_tensor = embedding_tensor + value  #
                    protein_tensors.append(combined_tensor)
                except FileNotFoundError:
                    print(f"Embedding for {peptide_target} not found. Skipping this protein.")

        if protein_tensors:
            patient_tensor = torch.stack(protein_tensors)
            print(patient_tensor.shape)
            output_file = f'patient_protein/{st}/{patient_id}.pt'
            torch.save(patient_tensor, output_file)
            print(f"Saved embeddings for patient {patient_id} to {output_file}")
        else:
            print(f"No valid proteins found for patient {patient_id}. No file saved.")