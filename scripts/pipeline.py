from datetime import datetime
import pandas as pd
import requests
import seaborn as sns
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
from lifelines import KaplanMeierFitter

# Define gene list
gene_list = ["CFTR", "CYP2C19", "CYP2C8", "CYP2C9",
             "CYP2D6", "CYP3A4", "CYP3A5", "DPYD",
             "SLCO1B1", "TPMT", "UGT1A1", "VKORC1", "NAT2"]

all_prescriptions = pd.read_csv("prescriptions_alldrugs_filtered_participants.csv")
prescription_cancer = pd.read_csv("prescription_cancerdrugs_clean.csv").rename(
    columns={'Date prescription was issued': 'date'})
genotypes = pd.read_csv("genotypes.csv")
diagnosis_all_participants = pd.read_csv("diseases_mapped_all_participants.csv")
diagnosis_all_participants = diagnosis_all_participants.rename(columns={"eid": "ParticipantID"})
cancer_drugs_info = pd.read_csv("filtered_cancerdrugsdb.csv").drop(columns={"Product"})
genes_related_to_cancer_drugs = {y for x in cancer_drugs_info["Targets"].apply(lambda x: x.split(",")) for y in x}
prescriptions = pd.read_csv("final_prescription_with_ATC_codes.csv")
death_records = pd.read_csv("death_records.csv")

# Filter participants and prescriptions
all_cancer_atc = cancer_drugs_info["ATC"]
filtered_participants = prescription_cancer[prescription_cancer['atc_code'].isin(all_cancer_atc)][
    'Participant ID'].unique()
death_records = death_records[death_records["ParticipantID"].isin(filtered_participants)]
death_records.drop(columns='Cause_of_Death_ICD10', inplace=True)

import re

# Detailed ICD-10 cancer codes within each category
cancer_categories_detailed = {
    "Malignant neoplasms of lip, oral cavity and pharynx": {
        "C00": "Malignant neoplasm of lip",
        "C01": "Malignant neoplasm of base of tongue",
        "C02": "Malignant neoplasm of other and unspecified parts of tongue",
        "C03": "Malignant neoplasm of gum",
        "C04": "Malignant neoplasm of floor of mouth",
        "C05": "Malignant neoplasm of palate",
        "C06": "Malignant neoplasm of other and unspecified parts of mouth",
        "C07": "Malignant neoplasm of parotid gland",
        "C08": "Malignant neoplasm of other and unspecified major salivary glands",
        "C09": "Malignant neoplasm of tonsil",
        "C10": "Malignant neoplasm of oropharynx",
        "C11": "Malignant neoplasm of nasopharynx",
        "C12": "Malignant neoplasm of pyriform sinus",
        "C13": "Malignant neoplasm of hypopharynx",
        "C14": "Malignant neoplasm of other and ill-defined sites in the lip, oral cavity and pharynx"
    },
    "Malignant neoplasms of digestive organs": {
        "C15": "Malignant neoplasm of esophagus",
        "C16": "Malignant neoplasm of stomach",
        "C17": "Malignant neoplasm of small intestine",
        "C18": "Malignant neoplasm of colon",
        "C19": "Malignant neoplasm of rectosigmoid junction",
        "C20": "Malignant neoplasm of rectum",
        "C21": "Malignant neoplasm of anus and anal canal",
        "C22": "Malignant neoplasm of liver and intrahepatic bile ducts",
        "C23": "Malignant neoplasm of gallbladder",
        "C24": "Malignant neoplasm of other and unspecified parts of biliary tract",
        "C25": "Malignant neoplasm of pancreas",
        "C26": "Malignant neoplasm of other and ill-defined digestive organs"
    },
    "Malignant neoplasms of respiratory and intrathoracic organs": {
        "C30": "Malignant neoplasm of nasal cavity and middle ear",
        "C31": "Malignant neoplasm of accessory sinuses",
        "C32": "Malignant neoplasm of larynx",
        "C33": "Malignant neoplasm of trachea",
        "C34": "Malignant neoplasm of bronchus and lung",
        "C37": "Malignant neoplasm of thymus",
        "C38": "Malignant neoplasm of heart, mediastinum and pleura",
        "C39": "Malignant neoplasm of other and ill-defined sites in the respiratory system and intrathoracic organs"
    },
    "Malignant neoplasms of bone and articular cartilage": {
        "C40": "Malignant neoplasm of bone and articular cartilage of limbs",
        "C41": "Malignant neoplasm of bone and articular cartilage of other and unspecified sites"
    },
    "Malignant neoplasms of mesothelial and soft tissue": {
        "C45": "Mesothelioma",
        "C46": "Kaposi's sarcoma",
        "C47": "Malignant neoplasm of peripheral nerves and autonomic nervous system",
        "C48": "Malignant neoplasm of retroperitoneum and peritoneum",
        "C49": "Malignant neoplasm of other connective and soft tissue"
    },
    "Melanoma and other malignant neoplasms of skin": {
        "C43": "Malignant melanoma of skin",
        "C44": "Other and unspecified malignant neoplasm of skin",
        "C4A": "Merkel cell carcinoma"
    },
    "Malignant neoplasms of breast": {
        "C50": "Malignant neoplasm of breast"
    },
    "Malignant neoplasms of female genital organs": {
        "C51": "Malignant neoplasm of vulva",
        "C52": "Malignant neoplasm of vagina",
        "C53": "Malignant neoplasm of cervix uteri",
        "C54": "Malignant neoplasm of corpus uteri",
        "C55": "Malignant neoplasm of uterus, part unspecified",
        "C56": "Malignant neoplasm of ovary",
        "C57": "Malignant neoplasm of other and unspecified female genital organs",
        "C58": "Malignant neoplasm of placenta"
    },
    "Malignant neoplasms of male genital organs": {
        "C60": "Malignant neoplasm of penis",
        "C61": "Malignant neoplasm of prostate",
        "C62": "Malignant neoplasm of testis",
        "C63": "Malignant neoplasm of other and unspecified male genital organs"
    },
    "Malignant neoplasms of urinary tract": {
        "C64": "Malignant neoplasm of kidney, except renal pelvis",
        "C65": "Malignant neoplasm of renal pelvis",
        "C66": "Malignant neoplasm of ureter",
        "C67": "Malignant neoplasm of bladder",
        "C68": "Malignant neoplasm of other and unspecified urinary organs"
    },
    "Malignant neoplasms of eye, brain and central nervous system": {
        "C69": "Malignant neoplasm of eye and adnexa",
        "C70": "Malignant neoplasm of meninges",
        "C71": "Malignant neoplasm of brain",
        "C72": "Malignant neoplasm of spinal cord, cranial nerves and other parts of central nervous system"
    },
    "Malignant neoplasms of thyroid and other endocrine glands": {
        "C73": "Malignant neoplasm of thyroid gland",
        "C74": "Malignant neoplasm of adrenal gland",
        "C75": "Malignant neoplasm of other endocrine glands and related structures"
    },
    "Malignant neoplasms of ill-defined, other secondary, and unspecified sites": {
        "C76": "Malignant neoplasm of other and ill-defined sites",
        "C77": "Secondary and unspecified malignant neoplasm of lymph nodes",
        "C78": "Secondary malignant neoplasm of respiratory and digestive organs",
        "C79": "Secondary malignant neoplasm of other and unspecified sites",
        "C80": "Malignant neoplasm without specification of site"
    },
    "Malignant neuroendocrine tumors": {
        "C7A": "Malignant neuroendocrine tumors"
    },
    "Secondary neuroendocrine tumors": {
        "C7B": "Secondary neuroendocrine tumors"
    },
    "Malignant neoplasms of lymphoid, hematopoietic, and related tissue": {
        "C81": "Hodgkin lymphoma",
        "C82": "Follicular lymphoma",
        "C83": "Non-follicular lymphoma",
        "C84": "Mature T/NK-cell lymphomas",
        "C85": "Other specified and unspecified types of non-Hodgkin lymphoma",
        "C86": "Other specified types of T/NK-cell lymphoma",
        "C88": "Malignant immunoproliferative diseases and certain other B-cell lymphomas",
        "C90": "Multiple myeloma and malignant plasma cell neoplasms",
        "C91": "Lymphoid leukemia",
        "C92": "Myeloid leukemia",
        "C93": "Monocytic leukemia",
        "C94": "Other leukemias of specified cell type",
        "C95": "Leukemia of unspecified cell type",
        "C96": "Other and unspecified malignant neoplasms of lymphoid, hematopoietic and related tissue"
    }
}

def get_icd10_codes(keyword):
    input_value = keyword.lower()

    matching_codes = []
    for family, codes in cancer_categories_detailed.items():
        # If the input matches a family name, return all codes within that family
        if input_value in family.lower():
            return list(codes.keys())

        # If the input matches a specific ICD-10 code or cancer description, return the matching code(s)
        for code, description in codes.items():
            if input_value == code.lower() or input_value in description.lower():
                matching_codes.append(code)

    # If no matches, return None
    if not matching_codes:
        return None

    # Return the list of matching codes
    return matching_codes




def get_drugs_for_indication(keyword, df):
    return df[df["Indications"].str.contains(keyword.lower(), case=False, na=False)]


def get_participants_with_disease(keyword, icd10_codes, diagnosis_df, exclude_other_cancers=True):
    print(f"ICD-10 codes used for filtering: {icd10_codes}")

    # Truncate ICD-10 codes in the diagnosis dataframe to the first three characters
    diagnosis_df['icd10'] = diagnosis_df['icd10'].str[:3]

    participants_with_disease = diagnosis_df[diagnosis_df["icd10"].isin(icd10_codes)].drop(columns=["Disease_Index"])

    print(
        f"Participants with disease before cancer exclusion: {len(participants_with_disease['ParticipantID'].unique())}")

    if exclude_other_cancers:
        patients_to_exclude = diagnosis_df[
            (diagnosis_df["icd10"].str.startswith("C")) & (~diagnosis_df["icd10"].isin(icd10_codes))
            ]["ParticipantID"].unique()

        # filter participants to only include those not diagnosed with other cancers
        participants_with_disease = participants_with_disease[
            ~participants_with_disease["ParticipantID"].isin(patients_to_exclude)]

    print(
        f"Participants with disease after excluding other cancers: {len(participants_with_disease['ParticipantID'].unique())}")

    participants_with_disease.to_csv(f"{keyword}_cancer_diagnosis.csv", index=False)
    return participants_with_disease

def analyze_keyword(keyword, filtered_cancer_drugs, diagnosis_all_participants, prescriptions, cancer_drugs,
                    specific_drugs=False, exclude_other_cancers=True):
    icd10_codes = get_icd10_codes(keyword)
    print("ICD10 codes: ", icd10_codes)

    drugs_for_keyword = cancer_drugs
    genes_related_to_drugs = genes_related_to_cancer_drugs
    participants_with_disease = get_participants_with_disease(
        keyword, icd10_codes, diagnosis_all_participants, exclude_other_cancers)
    print("Participants with disease", participants_with_disease)
    diag_participants_df = pd.read_csv(f"{keyword}_cancer_diagnosis.csv")
    patients = diag_participants_df["ParticipantID"].unique()
    print(f"Patients with {keyword} cancer : {len(patients)}" )
    all_patient_diagnoses = diagnosis_all_participants[diagnosis_all_participants["ParticipantID"].isin(patients)]

    patient_keyword_prescriptions = prescriptions[
        prescriptions["Participant ID"].isin(diag_participants_df['ParticipantID'])]
    patient_keyword_prescriptions = patient_keyword_prescriptions[
        patient_keyword_prescriptions["atc_code"].isin(drugs_for_keyword['ATC'])]

    patient_cancer_prescriptions = prescriptions[
        prescriptions["Participant ID"].isin(diag_participants_df['ParticipantID'])]
    patient_cancer_prescriptions = patient_cancer_prescriptions[
        patient_cancer_prescriptions["atc_code"].isin(cancer_drugs['ATC'])]

    # Return the results based on the specific_drugs parameter
    return {
        'icd10_codes': list(icd10_codes),  # ICD-10 codes related to the keyword
        'drugs_related': list(drugs_for_keyword['ATC']),  # ATC codes for the drugs (specific or general)
        'genes_related': genes_related_to_drugs,  # Related genes for the drugs (specific or general)
        'total_patients': {
            'count': diag_participants_df['ParticipantID'].nunique(),
            'ids': diag_participants_df['ParticipantID'].unique().tolist()
        },
        'patients_took_cancer_drugs': {
            'count': patient_cancer_prescriptions["Participant ID"].nunique(),
            'ids': patient_cancer_prescriptions["Participant ID"].unique().tolist()
        },
        'patients_took_keyword_drugs': {
            'count': patient_keyword_prescriptions["Participant ID"].nunique(),
            'ids': patient_keyword_prescriptions["Participant ID"].unique().tolist()
        }
    }


def calculate_diagnosis_survival(row, current_date, icd10_codes):
    for icd10_code in icd10_codes:
        if icd10_code in row.index:
            diagnosis_date = row[icd10_code]
            death_date = row['Date_of_Death']

            if pd.isnull(death_date):
                death_date = current_date

            if pd.notnull(diagnosis_date):
                row[icd10_code] = (pd.to_datetime(death_date) - pd.to_datetime(diagnosis_date)).days
            else:
                row[icd10_code] = None
    return row


def process_survival_df(df, patient_death_records, codes):
    df.loc[:, 'date'] = pd.to_datetime(df['date'], errors='coerce')
    patient_death_records.loc[:, 'Date_of_Death'] = pd.to_datetime(patient_death_records['Date_of_Death'])
    pivot_df = df.pivot_table(index='ParticipantID', columns='icd10', values='date', aggfunc='first')
    print("pivot", pivot_df)
    df_result = pivot_df.reset_index()
    print("reset index df ", df_result)
    df_merged = pd.merge(df_result, patient_death_records, on='ParticipantID', how='left')
    print("df merged  : \n", df_merged)
    current_date = pd.to_datetime(datetime.now().date())
    current_date = pd.to_datetime(datetime.now().date()).strftime('%Y-%m-%d')

    survival_df = df_merged.apply(lambda row: calculate_diagnosis_survival(row, current_date, codes), axis=1)
    survival_df = survival_df.drop(columns='Cause_of_Death_ICD10', errors='ignore')

    return survival_df


def calculate_prescription_survival(prescriptions_df, death_records_df):
    pivot_prescriptions = prescriptions_df.pivot_table(
        index='ParticipantID',
        columns='atc_code',
        values='date',
        aggfunc='first'
    ).reset_index()
    df_merged = pd.merge(pivot_prescriptions, death_records_df, on='ParticipantID', how='left')
    for atc_code in pivot_prescriptions.columns[1:]:
        df_merged[atc_code] = (df_merged['Date_of_Death'] - df_merged[atc_code]).dt.days
    return df_merged


def add_sample_size(ax, data, x_col, y_col):
    counts = data.groupby(x_col)[y_col].count()
    for tick, label in enumerate(ax.get_xticklabels()):
        genotype = label.get_text()
        if genotype in counts:
            count = counts[genotype]
            ax.text(tick, ax.get_ylim()[1], f'N={count}',
                    horizontalalignment='center', size='small', color='black')


def plot_batch(df, genes, batch_size=3):
    num_batches = int(np.ceil(len(genes) / batch_size))  # Number of figures required

    for batch_num in range(num_batches):
        plt.figure(figsize=(12, batch_size * 4))
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(genes))
        batch_genes = genes[start_idx:end_idx]

        for i, gene in enumerate(batch_genes):
            ax = plt.subplot(batch_size, 1, i + 1)
            gene_data = df[df['Gene'] == gene]
            sns.boxplot(x='Genotype', y='max_survival_days', data=gene_data, ax=ax)
            add_sample_size(ax, gene_data, 'Genotype', 'max_survival_days')
            ax.set_title(f'Survival Days by Genotype for {gene}')
            ax.set_xlabel('Genotype')
            ax.set_ylabel('Max Survival Days')
            tick_labels = ax.get_xticklabels()
            ax.set_xticks(np.arange(len(tick_labels)))  # Set the tick positions
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')  # Set the tick labels with rotation

        plt.tight_layout()
        plt.show()


def process_and_compare_genotype_survival(diagnosis_survival_no_drugs_df,
                                          diagnosis_survival_took_drug_df,
                                          patient_genotype_df,
                                          batch_size=2,
                                          min_sample_size=5):
    def prepare_data(diagnosis_survival_df, group_label):
        # Print lengths before processing
        print(f"Initial length of diagnosis_survival_df (group {group_label}): {len(diagnosis_survival_df)}")

        # Extract numeric columns for survival days calculation
        numeric_cols = diagnosis_survival_df.select_dtypes(include=['number']).columns
        print(f"Numeric columns for survival days (group {group_label}): {numeric_cols}")

        numeric_cols = numeric_cols.drop('ParticipantID', errors='ignore')
        diagnosis_survival_df['max_survival_days'] = diagnosis_survival_df[numeric_cols].max(axis=1)

        # Merge survival and genotype data
        merged_df = pd.merge(diagnosis_survival_df[['ParticipantID', 'max_survival_days']],
                             patient_genotype_df, on='ParticipantID', how='left')

        print(
            f"Number of participants after merging with genotype data (group {group_label}): {len(merged_df['ParticipantID'].unique())}")
        print(f"Merged dataframe (group {group_label}):\n{merged_df.head()}")

        # Melt the genotype data for easier pivoting into wide format
        melted_genotype_df = pd.melt(merged_df,
                                     id_vars=['ParticipantID', 'max_survival_days'],
                                     value_vars=list(patient_genotype_df.columns[1:]),  # All genotype columns
                                     var_name='Gene',
                                     value_name='Genotype')

        print("Melted genotype df before dropping NA:", melted_genotype_df.head())

        # Drop rows with missing values in either Genotype or max_survival_days
        melted_genotype_df = melted_genotype_df.dropna(subset=['Genotype', 'max_survival_days'])
        print(
            f"Number of participants after dropping missing genotype/survival data (group {group_label}): {len(melted_genotype_df)}")

        # Add a column to distinguish the groups (drug/no-drug)
        melted_genotype_df['Group'] = group_label

        # Pivot the data to get one column per gene
        pivoted_df = melted_genotype_df.pivot_table(index=['ParticipantID', 'max_survival_days', 'Group'],
                                                    columns='Gene',
                                                    values='Genotype',
                                                    aggfunc='first').reset_index()

        # Show the pivoted dataframe for debugging
        print(f"Pivoted dataframe (group {group_label}):\n{pivoted_df.head()}")

        return pivoted_df

    # Process both the no-drug group and took-drug group
    no_drugs_df = prepare_data(diagnosis_survival_no_drugs_df, 0)
    took_drug_df = prepare_data(diagnosis_survival_took_drug_df, 1)

    # Combine the two dataframes
    combined_df = pd.concat([no_drugs_df, took_drug_df], axis=0)

    print(f"Combined dataframe length: {len(combined_df)}")

    return combined_df


def main(keyword, specific_drugs=True, exclude_other_cancers=True):
    result_specific = analyze_keyword(
        keyword=keyword,
        filtered_cancer_drugs=cancer_drugs_info,
        diagnosis_all_participants=diagnosis_all_participants,
        prescriptions=prescriptions,
        cancer_drugs=cancer_drugs_info,
        specific_drugs=specific_drugs,
        exclude_other_cancers=exclude_other_cancers
    )

    icd10_codes = result_specific['icd10_codes']
    filtered_genotype_df = genotypes[genotypes['ParticipantID'].isin(result_specific['total_patients']['ids'])]
    patient_genotype_df = filtered_genotype_df[['ParticipantID'] + list(result_specific['genes_related'])]

    patients_cancer_prescriptions = prescription_cancer[
        prescription_cancer['Participant ID'].isin(result_specific['total_patients']['ids'])
    ].rename(columns={'Participant ID': 'ParticipantID'})

    patients_cancer_prescriptions['date'] = pd.to_datetime(patients_cancer_prescriptions['date'])
    patients_keyword_prescriptions = patients_cancer_prescriptions[
        patients_cancer_prescriptions['atc_code'].isin(result_specific['drugs_related'])
    ]

    diagnosis_all_participants['icd10'] = diagnosis_all_participants['icd10'].apply(
        lambda x: x[:3])  # Truncate to 3 characters
    patients_diagnosis = diagnosis_all_participants[diagnosis_all_participants['icd10'].isin(icd10_codes)]
    patient_death_records = death_records[death_records['ParticipantID'].isin(result_specific['total_patients']['ids'])]
    patient_death_records['Date_of_Death'] = pd.to_datetime(patient_death_records['Date_of_Death'])

    # Process survival data for patients who took drugs and those who did not
    drug_participants = patients_cancer_prescriptions['ParticipantID'].unique()
    patient_diagnosis_took_cdrug = patients_diagnosis[patients_diagnosis['ParticipantID'].isin(drug_participants)]
    patients_diagnosis_no_drugs = patients_diagnosis[~patients_diagnosis['ParticipantID'].isin(drug_participants)]

    diagnosis_survival_took_drug = process_survival_df(patient_diagnosis_took_cdrug, patient_death_records, icd10_codes)
    diagnosis_survival_no_drugs_df = process_survival_df(patients_diagnosis_no_drugs, patient_death_records,
                                                         icd10_codes)

    # Combine genotype survival analysis
    combined_df = process_and_compare_genotype_survival(diagnosis_survival_no_drugs_df, diagnosis_survival_took_drug,
                                                        patient_genotype_df)
    print(combined_df)
    # Calculate prescription survival data for keyword-specific and general cancer drugs
    prescription_survival_keyword_specific_drugs = calculate_prescription_survival(patients_keyword_prescriptions,
                                                                                   patient_death_records)
    filtered_prescription_df = patients_cancer_prescriptions[
        patients_cancer_prescriptions['ParticipantID'].isin(drug_participants)]
    prescription_survival_general_cancer_drugs = calculate_prescription_survival(filtered_prescription_df,
                                                                                 patient_death_records)

    exclusion_suffix = "_no_other_cancers" if exclude_other_cancers else "_all_participants"

    # Save the results to CSV files
    prescription_survival_general_cancer_drugs.to_csv(f"{keyword}_general_cancer_drugs_survival{exclusion_suffix}.csv",
                                                      index=False)
    prescription_survival_keyword_specific_drugs.to_csv(
        f"{keyword}_specific_prescription_survival{exclusion_suffix}.csv", index=False)
    combined_df.to_csv(f'{keyword}_genotype_survival{exclusion_suffix}.csv', index=False)

    print("Results have been saved:")
    print(f"{keyword}_general_cancer_drugs_survival{exclusion_suffix}.csv")
    print(f"{keyword}_specific_prescription_survival{exclusion_suffix}.csv")
    print(f"{keyword}_genotype_survival{exclusion_suffix}.csv")


if __name__ == "__main__":
    main(keyword="lymph", specific_drugs=True, exclude_other_cancers=False)
