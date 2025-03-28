

from oaklib import get_adapter
import pandas as pd

df = pd.read_csv('output_data/efo_test_to_hp_mappings_results.csv')

hp = get_adapter('sqlite:obo:hp')

# for each distinct value of the "EFO Term" column

out_rows = []

for efo_term in df['EFO Term'].unique():
    print(f"Processing EFO term: {efo_term}")

    rows = df[df['EFO Term'] == efo_term]

    all_candidates = set()

    for index, row in rows.iterrows():
        mapping = row.to_dict()
        all_candidates.add(mapping['candidate_original_id'])

    for index, row in rows.iterrows():

        mapping = row.to_dict()

        subsumed = False

        for other_candidate in all_candidates:
            if row['candidate_original_id'] == other_candidate:
                continue
            if other_candidate in hp.ancestors(row['candidate_original_id']):
                candidate_label = hp.label(row['candidate_original_id'])
                other_candidate_label = hp.label(other_candidate)
                print(f"Skipping '{candidate_label}' because it is subsumed by '{other_candidate_label}'")
                subsumed = True
                break

        if not subsumed:
            print(mapping)
            out_rows += [mapping]
            
out_df = pd.DataFrame(out_rows)
out_df.to_csv('output_data/efo_test_to_hp_mappings_results_pruned.csv', index=False)
