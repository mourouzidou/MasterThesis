import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Create the disease index mapping as a dictionary (shortened version of each description for plotting)
disease_index_mapping = {
    0: "Infectious Diseases",
    1: "Neoplasms",
    2: "Blood & Immune",
    3: "Endocrine & Metabolic",
    4: "Mental & Neuro",
    5: "Nervous System",
    6: "Eye Diseases",
    7: "Ear & Mastoid",
    8: "Circulatory System",
    9: "Respiratory System",
    10: "Digestive System",
    11: "Skin & Subcutaneous Tissue",
    12: "Musculoskeletal",
    13: "Genitourinary",
    14: "Pregnancy & Childbirth",
    15: "Perinatal Period",
    16: "Congenital Abnormalities",
    17: "Symptoms & Signs",
    18: "Injury & Poisoning",
    19: "Special Codes",
    20: "External Causes",
    21: "Factors Influencing Health"
}


# Function to calculate months from first diagnosis
def calculate_months_since_first_diagnosis(diagnoses_df, first_diagnosis_dates):
    diagnoses_df = pd.merge(diagnoses_df, first_diagnosis_dates, on='eid', how='left',
                            suffixes=('', '_first_diagnosis'))

    # Filter out rows with missing 'date_first_diagnosis'
    diagnoses_df = diagnoses_df[diagnoses_df['date_first_diagnosis'].notna()]

    # Calculate months since the first diagnosis
    diagnoses_df['months_since_first_diagnosis'] = (
            (diagnoses_df['date'] - diagnoses_df['date_first_diagnosis']).dt.days / 30.44).fillna(0).round().astype(int)

    return diagnoses_df
def compute_difference_heatmap(data1, data2, yaxis_range, total1, total2, month_interval):
    # Prepare the data for both groups
    data1['month_intervals'] = (data1['months_since_first_diagnosis'] // month_interval) * month_interval
    data2['month_intervals'] = (data2['months_since_first_diagnosis'] // month_interval) * month_interval

    # Pivot tables for both genotypes
    pivot1 = data1.pivot_table(index='month_intervals', columns='Disease_Index', values='eid',
                               aggfunc=pd.Series.nunique, fill_value=0).reindex(columns=yaxis_range).fillna(0).reset_index()
    pivot2 = data2.pivot_table(index='month_intervals', columns='Disease_Index', values='eid',
                               aggfunc=pd.Series.nunique, fill_value=0).reindex(columns=yaxis_range).fillna(0).reset_index()

    # Normalize the counts by total participants for each group
    normalized1 = pivot1.set_index('month_intervals').values.T / total1
    normalized2 = pivot2.set_index('month_intervals').values.T / total2

    # Align the rows (month intervals) of both matrices
    aligned_intervals = sorted(set(pivot1['month_intervals']).intersection(set(pivot2['month_intervals'])))
    normalized1_aligned = pivot1[pivot1['month_intervals'].isin(aligned_intervals)].set_index('month_intervals').values.T / total1
    normalized2_aligned = pivot2[pivot2['month_intervals'].isin(aligned_intervals)].set_index('month_intervals').values.T / total2

    # Compute the difference
    difference = normalized1_aligned - normalized2_aligned  # Signed difference

    return difference, aligned_intervals

# Function to plot genotype heatmaps and return figure and xaxis range
# Function to plot genotype heatmaps with differences
def plot_genotype_heatmaps_with_differences(diagnoses_genotype1, diagnoses_genotype2, diagnoses_genotype1_group0,
                                            diagnoses_genotype2_group0, gene, genotype1, genotype2, drug,
                                            first_diag_months_genotype1, first_diag_months_genotype2, max_months,
                                            month_interval, min_samples):
    # Create figure with subplots for heatmaps and difference heatmaps
    fig = make_subplots(rows=2, cols=4,
                        subplot_titles=[f'{genotype1} - {drug}', f'{genotype2} - {drug}', f'{genotype1} - Group 0',
                                        f'{genotype2} - Group 0',
                                        f'Difference: {genotype1} vs {genotype2}',
                                        f'Difference: {genotype1} vs Group 0',
                                        f'Difference: {genotype2} vs Group 0'])

    # Set x-axis range
    xaxis_range = [-max_months, max_months]

    # Ensure both heatmaps show all disease indexes
    yaxis_range = list(disease_index_mapping.keys())

    # Check the number of participants for both genotypes
    total_genotype1 = diagnoses_genotype1['eid'].nunique()
    total_genotype2 = diagnoses_genotype2['eid'].nunique()
    total_group0_genotype1 = diagnoses_genotype1_group0['eid'].nunique()
    total_group0_genotype2 = diagnoses_genotype2_group0['eid'].nunique()

    # Plot heatmaps
    if total_genotype1 >= min_samples:
        plot_heatmap_for_genotype(diagnoses_genotype1, fig, first_diag_months_genotype1, max_months, row=1, col=1,
                                  yaxis_range=yaxis_range, month_interval=month_interval,
                                  total_participants=total_genotype1)

    if total_genotype2 >= min_samples:
        plot_heatmap_for_genotype(diagnoses_genotype2, fig, first_diag_months_genotype2, max_months, row=1, col=2,
                                  yaxis_range=yaxis_range, month_interval=month_interval,
                                  total_participants=total_genotype2)

    if total_group0_genotype1 >= min_samples:
        plot_heatmap_for_genotype(diagnoses_genotype1_group0, fig, first_diag_months_genotype1, max_months, row=1,
                                  col=3, yaxis_range=yaxis_range, month_interval=month_interval,
                                  total_participants=total_group0_genotype1)

    if total_group0_genotype2 >= min_samples:
        plot_heatmap_for_genotype(diagnoses_genotype2_group0, fig, first_diag_months_genotype2, max_months, row=1,
                                  col=4, yaxis_range=yaxis_range, month_interval=month_interval,
                                  total_participants=total_group0_genotype2)

    # Compute difference heatmaps
    diff_genotype1_genotype2, intervals = compute_difference_heatmap(diagnoses_genotype1, diagnoses_genotype2,
                                                                     yaxis_range, total_genotype1, total_genotype2,
                                                                     month_interval)

    diff_genotype1_group0, intervals = compute_difference_heatmap(diagnoses_genotype1, diagnoses_genotype1_group0,
                                                                  yaxis_range, total_genotype1, total_group0_genotype1,
                                                                  month_interval)

    diff_genotype2_group0, intervals = compute_difference_heatmap(diagnoses_genotype2, diagnoses_genotype2_group0,
                                                                  yaxis_range, total_genotype2, total_group0_genotype2,
                                                                  month_interval)

    # Plot difference heatmaps
    plot_difference_heatmap(fig, diff_genotype1_genotype2, intervals, row=2, col=1, title=f'{genotype1} vs {genotype2}')
    plot_difference_heatmap(fig, diff_genotype1_group0, intervals, row=2, col=2, title=f'{genotype1} vs Group 0')
    plot_difference_heatmap(fig, diff_genotype2_group0, intervals, row=2, col=3, title=f'{genotype2} vs Group 0')

    # Show the final figure
    fig.update_layout(height=800, width=1800, title=f'Genotype Comparison for {gene} - Drug: {drug}')
    fig.show()

    return fig, xaxis_range
def plot_heatmap_for_genotype(diagnoses_genotype, fig, first_diag_months, max_months, row, col, yaxis_range, month_interval, total_participants):
    heatmap_data = diagnoses_genotype.copy()

    # Calculate months relative to the first diagnosis
    heatmap_data['months_since_first_diagnosis'] = ((heatmap_data['date'] - heatmap_data['date_first_diagnosis']).dt.days / 30.44).round().astype(int)

    # Restrict data to a defined range of months
    heatmap_data = heatmap_data[(heatmap_data['months_since_first_diagnosis'] >= -120) & (heatmap_data['months_since_first_diagnosis'] <= 120)]

    # Group the data based on the specified month_interval
    heatmap_data['month_intervals'] = (heatmap_data['months_since_first_diagnosis'] // month_interval) * month_interval

    # Pivot table to get the count of participants diagnosed with each disease index at each specified month interval
    pivot_genotype = heatmap_data.pivot_table(index='month_intervals', columns='Disease_Index', values='eid',
                                              aggfunc=pd.Series.nunique, fill_value=0).reindex(columns=yaxis_range).fillna(0).reset_index()

    # Normalize the counts by the total number of participants
    normalized_z = pivot_genotype.set_index('month_intervals').values.T / total_participants

    # Apply logarithmic transformation
    log_z = np.log10(normalized_z + 1e-3)  # Small offset to avoid log(0)

    # Plot the heatmap
    fig.add_trace(
        go.Heatmap(z=log_z, x=pivot_genotype['month_intervals'],
                   y=pivot_genotype.columns[1:], colorscale='Plasma', zmin=-3, zmax=0,
                   colorbar=dict(title='Log10(Participants %)')),
        row=row, col=col
    )

    # Add a vertical line for the first diagnosis
    fig.add_vline(x=first_diag_months, line_dash="dash", line_color="red", row=row, col=col)

def plot_difference_heatmap(fig, difference, intervals, row, col, title):
    fig.add_trace(
        go.Heatmap(z=difference, x=intervals,
                   y=list(disease_index_mapping.keys()), colorscale='RdBu', zmid=0,
                   colorbar=dict(title='Difference %')),
        row=row, col=col
    )
    fig.update_yaxes(title_text=title, row=row, col=col)


def dynamic_plot_genotypes_with_differences(significant_pairs, genotypes, prescriptions, diagnoses, max_months=240,
                                            month_interval=1, min_samples=10):
    # Filter Group 0 participants
    group0_participants = genotypes[genotypes['Group'] == 0]

    for idx, row in significant_pairs.iterrows():
        gene = row['Gene']
        drug = row['Drug']
        genotype1 = row['Genotype1']
        genotype2 = row['Genotype2']

        # Determine the length of the drug name to decide how to filter prescriptions
        drug_length = len(drug)

        # Adjust prescription filtering based on the drug length
        if drug_length < len(prescriptions.columns[1]):  # If drug is a prefix
            # Filter columns that start with the drug prefix
            drug_columns = [col for col in prescriptions.columns if col.startswith(drug)]
        else:
            # Use the exact drug name as the column
            drug_columns = [drug] if drug in prescriptions.columns else []

        # Check if there are any matching columns
        if not drug_columns:
            print(f"No matching columns found for drug: {drug}")
            continue

        # Identify participants who took the relevant drug(s)
        participants_with_drug = prescriptions[prescriptions[drug_columns].notna().any(axis=1)]['ParticipantID'].unique()

        # Filter participants by drug prescriptions and genotype
        participants_genotype1 = genotypes[(genotypes['ParticipantID'].isin(participants_with_drug)) &
                                           (genotypes[gene] == genotype1)]['ParticipantID']
        participants_genotype2 = genotypes[(genotypes['ParticipantID'].isin(participants_with_drug)) &
                                           (genotypes[gene] == genotype2)]['ParticipantID']

        # Filter control group (Group 0) participants based on genotype
        group0_genotype1 = group0_participants[group0_participants[gene] == genotype1]['ParticipantID']
        group0_genotype2 = group0_participants[group0_participants[gene] == genotype2]['ParticipantID']

        # Filter diagnoses for genotypes
        diagnoses_genotype1 = diagnoses[diagnoses['eid'].isin(participants_genotype1)]
        diagnoses_genotype2 = diagnoses[diagnoses['eid'].isin(participants_genotype2)]
        diagnoses_genotype1_group0 = diagnoses[diagnoses['eid'].isin(group0_genotype1)]
        diagnoses_genotype2_group0 = diagnoses[diagnoses['eid'].isin(group0_genotype2)]

        # Skip plotting if there are fewer participants than min_samples
        if diagnoses_genotype1['eid'].nunique() < min_samples or diagnoses_genotype2['eid'].nunique() < min_samples:
            print(f"Skipping {gene} - {genotype1} vs {genotype2} for {drug} due to insufficient sample size.")
            continue

        # Calculate the number of months corresponding to the first diagnosis for each genotype group
        first_diag_months_genotype1 = ((diagnoses_genotype1['date_first_diagnosis'] -
                                        diagnoses_genotype1['date_first_diagnosis']).dt.days / 30.44).fillna(0).round().astype(int).min()
        first_diag_months_genotype2 = ((diagnoses_genotype2['date_first_diagnosis'] -
                                        diagnoses_genotype2['date_first_diagnosis']).dt.days / 30.44).fillna(0).round().astype(int).min()

        # Plot the heatmaps for both genotypes and their respective Group 0 controls with differences
        fig, xaxis_range = plot_genotype_heatmaps_with_differences(diagnoses_genotype1, diagnoses_genotype2,
                                                                   diagnoses_genotype1_group0,
                                                                   diagnoses_genotype2_group0, gene, genotype1,
                                                                   genotype2, drug, first_diag_months_genotype1,
                                                                   first_diag_months_genotype2, max_months,
                                                                   month_interval, min_samples)



# Main execution function
def main(keyword, exclusion_suffix='_all_participants', max_months=240, month_interval=1, min_samples=10):
    all_diagnoses = pd.read_csv("diseases_mapped_all_participants.csv")
    keyword_diagnoses = pd.read_csv(f"{keyword}/{keyword}_cancer_diagnosis.csv")
    prescriptions = pd.read_csv(f"{keyword}/{keyword}_general_cancer_drugs_survival{exclusion_suffix}.csv")
    genotypes = pd.read_csv(f"{keyword}/{keyword}_genotype_survival{exclusion_suffix}.csv")
    significant_pairs = pd.read_csv(f"{keyword}/{keyword}_significant_wilcoxon_genotype_drug_pairs.csv")

    keyword_diagnoses['date'] = pd.to_datetime(keyword_diagnoses['date'], errors='coerce')
    all_diagnoses['date'] = pd.to_datetime(all_diagnoses['date'], errors='coerce')

    first_diagnosis_dates = keyword_diagnoses.groupby('ParticipantID').apply(lambda x: x.nsmallest(1, 'date')).reset_index(
        drop=True)
    first_diagnosis_dates = first_diagnosis_dates[['ParticipantID', 'date']]
    print(all_diagnoses, first_diagnosis_dates)
    merged_diagnoses = pd.merge(all_diagnoses, first_diagnosis_dates,left_on="eid", right_on='ParticipantID', how='left',
                                suffixes=('', '_first_diagnosis'))

    # Filter out rows with missing 'date_first_diagnosis'
    merged_diagnoses = merged_diagnoses[merged_diagnoses['date_first_diagnosis'].notna()]

    # Calculate months since first diagnosis
    merged_diagnoses['months_since_first_diagnosis'] = (
            (merged_diagnoses['date'] - merged_diagnoses['date_first_diagnosis']).dt.days / 30.44).fillna(
        0).round().astype(int)

    dynamic_plot_genotypes_with_differences(significant_pairs, genotypes, prescriptions, merged_diagnoses, max_months,
                                            month_interval, min_samples)


# Run the script
if __name__ == "__main__":
    keyword = "skin"
    month_interval = 24
    min_samples = 10
    main(keyword, max_months=400, month_interval=24, min_samples=min_samples)
