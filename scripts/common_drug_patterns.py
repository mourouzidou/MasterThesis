import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go

def plot_genotype_heatmaps(prescriptions_genotype1, prescriptions_genotype2, gene, genotype1, genotype2, drug,
                           top_atc_codes, max_periods, period_length_days):
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        f'{gene} - {genotype1}', f'{gene} - {genotype2}', 'Difference'))

    # Calculate the periods since the first prescription based on the input period length (e.g., 180 days for 6 months)
    prescriptions_genotype1['Periods Since First Prescription'] = (
        pd.to_datetime(prescriptions_genotype1['Date prescription was issued']) -
        pd.to_datetime(prescriptions_genotype1['First_Prescription_Date'])
    ).dt.days // period_length_days

    prescriptions_genotype2['Periods Since First Prescription'] = (
        pd.to_datetime(prescriptions_genotype2['Date prescription was issued']) -
        pd.to_datetime(prescriptions_genotype2['First_Prescription_Date'])
    ).dt.days // period_length_days

    # Filter out any missing values in 'Periods Since First Prescription'
    prescriptions_genotype1 = prescriptions_genotype1.dropna(subset=['Periods Since First Prescription']).copy()
    prescriptions_genotype2 = prescriptions_genotype2.dropna(subset=['Periods Since First Prescription']).copy()

    # Calculate percentage values for heatmap normalization
    total_genotype1 = len(prescriptions_genotype1['Participant ID'].unique())
    total_genotype2 = len(prescriptions_genotype2['Participant ID'].unique())

    # Genotype 1 heatmap
    pivot_genotype1 = plot_heatmap_for_genotype(prescriptions_genotype1, fig, top_atc_codes, max_periods, row=1, col=1,
                                                zmax=100, total_participants=total_genotype1)

    # Genotype 2 heatmap
    pivot_genotype2 = plot_heatmap_for_genotype(prescriptions_genotype2, fig, top_atc_codes, max_periods, row=1, col=2,
                                                zmax=100, total_participants=total_genotype2)

    # Calculate normalized difference heatmap (-1 to +1)
    pivot_difference = pivot_genotype2.set_index('Periods Since First Prescription').subtract(
        pivot_genotype1.set_index('Periods Since First Prescription'), fill_value=0)

    total_samples = total_genotype1 + total_genotype2

    # Normalize the difference by the total number of participants for both genotypes
    pivot_normalized_difference = pivot_difference / total_samples

    # Add the difference heatmap to the figure
    fig.add_trace(
        go.Heatmap(
            z=pivot_normalized_difference.T.values,
            x=pivot_normalized_difference.index,
            y=pivot_normalized_difference.columns,
            colorscale='PuOr',
            zmin=-1,
            zmax=1,
            colorbar=dict(title='Normalized Difference'),
        ),
        row=1, col=3
    )

    # Update layout and annotate the plot
    fig.update_layout(
        title=f'Drug Prescription Patterns Over Time for {gene} ({genotype1} vs {genotype2}) - Drug: {drug}',
        annotations=[
            dict(
                x=0.18,
                y=1.1,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12)
            ),
            dict(
                x=0.55,
                y=1.1,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12)
            ),
            dict(
                x=0.9,
                y=1.1,
                xref="paper",
                yref="paper",
                text=f"Difference: Normalized (-1 to +1) | Interval: {period_length_days // 30} months",
                showarrow=False,
                font=dict(size=12)
            )
        ],
        height=600,
        width=1600
    )

    fig.show()


def plot_heatmap_for_genotype(prescriptions_genotype, fig, top_atc_codes, max_periods, row, col, zmax, total_participants):
    heatmap_data = prescriptions_genotype.copy()
    heatmap_data = heatmap_data[heatmap_data['Periods Since First Prescription'] <= max_periods]
    heatmap_data = heatmap_data[heatmap_data['atc_code'].isin(top_atc_codes)]

    pivot_genotype = heatmap_data.pivot_table(
        index='Periods Since First Prescription',
        columns='atc_code',
        aggfunc='size',
        fill_value=0
    ).reset_index()

    # Normalize the values to percentages
    pivot_genotype.iloc[:, 1:] = (pivot_genotype.iloc[:, 1:] / total_participants) * 100

    fig.add_trace(
        go.Heatmap(
            z=pivot_genotype.set_index('Periods Since First Prescription').values.T,
            x=pivot_genotype['Periods Since First Prescription'],
            y=pivot_genotype.columns[1:],  # Skip 'Periods Since First Prescription' column
            colorscale='darkmint',
            zmin=0,  # Set minimum value for color scale
            zmax=zmax,  # Use the maximum percentage (100%)
            colorbar=dict(title='% of Participants'),
        ),
        row=row, col=col
    )

    return pivot_genotype


def dynamic_plot_genotypes(significant_pairs, genotype_survival, prescriptions, patients_cancer_prescriptions,
                           max_periods=40, period_length_days=180):
    # Dictionary to store the results for each gene-drug-genotype combination
    results = {}

    for idx, row in significant_pairs.iterrows():
        gene = row['Gene']
        drug = row['Drug']
        genotype1 = row['Genotype1']
        genotype2 = row['Genotype2']

        # Get participants who took the relevant drug
        participants_with_drug = patients_cancer_prescriptions[patients_cancer_prescriptions[drug].notna()][
            'ParticipantID'].unique()
        drug_df = genotype_survival[genotype_survival['ParticipantID'].isin(participants_with_drug.tolist())]

        # Filter by the relevant genotypes for the current gene
        filtered_drug_df = drug_df[drug_df[gene].isin([genotype1, genotype2])]
        filtered_drug_df = filtered_drug_df[['ParticipantID', 'max_survival_days', gene]].copy()

        prescriptions['Date prescription was issued'] = pd.to_datetime(prescriptions['Date prescription was issued'],
                                                                       errors='coerce')

        # Prepare prescription data for filtered participants
        prescriptions_filtered = prescriptions[
            (prescriptions['Participant ID'].isin(filtered_drug_df['ParticipantID'])) &
            (prescriptions['atc_code'] == drug)
        ]

        # Find the first prescription date for each participant
        first_prescription_dates = prescriptions_filtered.groupby('Participant ID')[
            'Date prescription was issued'].min().reset_index()
        first_prescription_dates.rename(columns={'Date prescription was issued': 'First_Prescription_Date'},
                                        inplace=True)

        # Merge first prescription dates with filtered participants
        filtered_participants = filtered_drug_df.merge(first_prescription_dates, left_on='ParticipantID',
                                                       right_on='Participant ID', how='left')
        filtered_participants.drop(columns='Participant ID', inplace=True)

        # Store data per gene and genotype for further use without overwriting
        results[f"{gene}_{genotype1}_{drug}"] = filtered_participants[filtered_participants[gene] == genotype1].copy()
        results[f"{gene}_{genotype2}_{drug}"] = filtered_participants[filtered_participants[gene] == genotype2].copy()

        # Continue with prescription data
        prescriptions_with_dates = prescriptions.merge(
            filtered_participants[['ParticipantID', 'First_Prescription_Date']],
            left_on='Participant ID',
            right_on='ParticipantID',
            how='inner'
        )
        prescriptions_with_dates['Date prescription was issued'] = pd.to_datetime(
            prescriptions_with_dates['Date prescription was issued'], errors='coerce')
        prescriptions_with_dates.dropna(subset=['Date prescription was issued'], inplace=True)

        # Filter prescriptions after the first prescription date
        prescriptions_after_first = prescriptions_with_dates[
            prescriptions_with_dates['Date prescription was issued'] > prescriptions_with_dates['First_Prescription_Date']]

        # Find the top N most frequent ATC codes
        n = 300
        top_atc_codes = prescriptions_after_first['atc_code'].value_counts().nlargest(n).index

        # Separate participants into two groups based on genotype1 and genotype2
        participants_genotype1 = filtered_participants[filtered_participants[gene] == genotype1]['ParticipantID']
        participants_genotype2 = filtered_participants[filtered_participants[gene] == genotype2]['ParticipantID']

        prescriptions_genotype1 = prescriptions_after_first[
            prescriptions_after_first['Participant ID'].isin(participants_genotype1)].copy()
        prescriptions_genotype2 = prescriptions_after_first[
            prescriptions_after_first['Participant ID'].isin(participants_genotype2)].copy()

        plot_genotype_heatmaps(prescriptions_genotype1, prescriptions_genotype2, gene, genotype1, genotype2, drug,
                               top_atc_codes, max_periods, period_length_days)


def main(keyword, exclusion_suffix='_all_participants', max_periods=40, months_per_period=6):
    period_length_days = months_per_period * 30  # Convert months to days (e.g., 6 months = 180 days)
    diagnoses = pd.read_csv("diseases_mapped_all_participants.csv")
    significant_pairs = pd.read_csv(f"{keyword}_significant_wilcoxon_genotype_drug_pairs.csv")
    genotype_survival = pd.read_csv(f'{keyword}_genotype_survival{exclusion_suffix}.csv')
    prescriptions = pd.read_csv('final_prescription_with_ATC_codes.csv')
    patients_cancer_prescriptions = pd.read_csv(f"{keyword}_general_cancer_drugs_survival{exclusion_suffix}.csv")

    dynamic_plot_genotypes(significant_pairs, genotype_survival, prescriptions, patients_cancer_prescriptions,
                           max_periods, period_length_days)


# Call the main function
if __name__ == "__main__":
    keyword = "respiratory"
    main(keyword, months_per_period=12)
