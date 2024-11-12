from sklearn.preprocessing import LabelEncoder
import pandas as pd
from datetime import datetime


def load_data(gene):
    genotypes_df = pd.read_csv('genotypes.csv')
    genotypes_df =  genotypes_df[['ParticipantID','CYP2D6', 'CYP2C9', 'CYP2C19', 'TPMT', 'UGT1A1', 'SLCO1B1', 'CYP3A5', 'DPYD']]
    demographic_df = pd.read_csv('demographic.csv')
    demographic_df.rename(columns={'Participant_ID': 'ParticipantID'}, inplace=True)
    current_year = datetime.now().year
    demographic_df['Date_of_Death'] = pd.to_datetime(demographic_df['Date_of_Death'], errors='coerce')
    demographic_df['age'] = demographic_df.apply(
        lambda row: (row['Date_of_Death'].year - row['Birth_Year']) if pd.notna(row['Date_of_Death']) else (
                current_year - row['Birth_Year']),
        axis=1
    )

    demographic_df = demographic_df[["ParticipantID", "Sex", "BMI", "age", "Ethnicity"]]
    diseases_df = pd.read_csv('diseases_mapped_all_participants.csv')
    diseases_df.rename(columns={'eid': 'ParticipantID'}, inplace=True)
    prescriptions_df = pd.read_csv('prescriptions_alldrugs_filtered_participants.csv')
    phenotype_df = pd.read_csv(f'{gene}_Diplotype_Phenotype_Table.csv')
    phenotype_df = phenotype_df[[f"{gene} Diplotype", 'Coded Diplotype/Phenotype Summary']]
    available_diplotypes = genotypes_df[gene].unique()
    phenotype_df = phenotype_df[phenotype_df[f"{gene} Diplotype"].isin(available_diplotypes)]

    # Encode phenotypes
    le = LabelEncoder()
    phenotype_df['Encoded Phenotype'] = le.fit_transform(phenotype_df['Coded Diplotype/Phenotype Summary'])
    phenotype_df.rename(columns={'Coded Diplotype/Phenotype Summary': 'Phenotype'}, inplace=True)

    return genotypes_df, demographic_df, diseases_df, phenotype_df, prescriptions_df


def merge_genotype_with_phenotype(merged_df, gene, phenotype_df):
    # Merge the phenotype data with the main merged DataFrame
    merged_df = merged_df.merge(phenotype_df[[f"{gene} Diplotype", 'Phenotype']], left_on=gene,
                                right_on=f"{gene} Diplotype", how='left')
    merged_df.rename(columns={'Phenotype': f'{gene}_Phenotype'}, inplace=True)
    merged_df.drop(columns=[f'{gene} Diplotype'], inplace=True)

    return merged_df


def process_diseases(diseases_df):
    # Process disease counts
    disease_counts = diseases_df.groupby(['ParticipantID', 'Disease_Index']).size().unstack(fill_value=0)
    disease_counts.columns = [f'Disease_Index_{int(col)}' for col in disease_counts.columns]

    return disease_counts


def process_prescriptions(prescription_df):
    # Extract the first letter of the ATC code and count occurrences for each participant
    prescription_df['atc_code_start'] = prescription_df['atc_code'].str[0]
    prescription_counts = prescription_df.groupby(['Participant ID', 'atc_code_start']).size().unstack(fill_value=0)
    prescription_counts.columns = [f'Prescription_{col}' for col in prescription_counts.columns]

    return prescription_counts


def merge_all_genes_with_phenotype(genotypes, demographic, diseases, prescriptions, genes):
    merged_df = genotypes.merge(demographic, on='ParticipantID', how='left')

    # Loop through each gene to merge phenotypes
    for gene in genes:
        _, _, _, phenotype_df, _ = load_data(gene)  # Only unpack the required phenotype_df
        merged_df = merge_genotype_with_phenotype(merged_df, gene, phenotype_df)
    disease_counts = process_diseases(diseases)
    merged_df = merged_df.merge(disease_counts, on='ParticipantID', how='left')
    prescription_counts = process_prescriptions(prescriptions)
    merged_df = merged_df.merge(prescription_counts, left_on='ParticipantID', right_on='Participant ID', how='left')

    return merged_df


genes = ['CYP2D6', 'CYP2C9', 'CYP2C19', 'TPMT', 'UGT1A1', 'CYP3A5',  'SLCO1B1', 'DPYD',]

# Load data for the first gene (for initialization)
genotypes, demographic, diseases, _, prescriptions = load_data('CYP2D6')
merged_df_with_phenotypes = merge_all_genes_with_phenotype(genotypes, demographic, diseases, prescriptions, genes)

# Save the final merged DataFrame to a CSV file
merged_df_with_phenotypes.to_csv("merged_training.csv", index=False)

# Print the merged DataFrame
print(merged_df_with_phenotypes)
