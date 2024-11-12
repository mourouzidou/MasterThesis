# from datetime import datetime
# import pandas as pd
# import scikit_posthocs as sp
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import kruskal
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# import numpy as np
#
# # Create the disease index mapping as a dictionary (shortened version of each description for plotting)
# disease_index_mapping = {
#     0: "Infectious Diseases",
#     1: "Neoplasms",
#     2: "Blood & Immune",
#     3: "Endocrine & Metabolic",
#     4: "Mental & Neuro",
#     5: "Nervous System",
#     6: "Eye Diseases",
#     7: "Ear & Mastoid",
#     8: "Circulatory System",
#     9: "Respiratory System",
#     10: "Digestive System",
#     11: "Skin & Subcutaneous Tissue",
#     12: "Musculoskeletal",
#     13: "Genitourinary",
#     14: "Pregnancy & Childbirth",
#     15: "Perinatal Period",
#     16: "Congenital Abnormalities",
#     17: "Symptoms & Signs",
#     18: "Injury & Poisoning",
#     19: "Special Codes",
#     20: "External Causes",
#     21: "Factors Influencing Health"
# }
#
#
# # Function to cluster participants based on survival days
# def cluster_participants(df, num_clusters=3):
#     clustering_data = df[['max_survival_days']].copy()
#     scaler = StandardScaler()
#     clustering_data_scaled = scaler.fit_transform(clustering_data)
#     kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#     df['cluster'] = kmeans.fit_predict(clustering_data_scaled)
#
#     return df, kmeans
#
#
# # Function to calculate months from first diagnosis
# def calculate_months_since_first_diagnosis(diagnoses_df, first_diagnosis_dates):
#     diagnoses_df = pd.merge(diagnoses_df, first_diagnosis_dates, on='eid', how='left',
#                             suffixes=('', '_first_diagnosis'))
#
#     # Filter out rows with missing 'date_first_diagnosis'
#     diagnoses_df = diagnoses_df[diagnoses_df['date_first_diagnosis'].notna()]
#
#     # Calculate months since the first diagnosis
#     diagnoses_df['months_since_first_diagnosis'] = (
#             (diagnoses_df['date'] - diagnoses_df['date_first_diagnosis']).dt.days / 30.44).fillna(0).round().astype(int)
#
#     return diagnoses_df
#
#
# def compute_difference_heatmap(data1, data2, yaxis_range, total1, total2, month_interval):
#     # Prepare the data for both groups
#     data1['month_intervals'] = (data1['months_since_first_diagnosis'] // month_interval) * month_interval
#     data2['month_intervals'] = (data2['months_since_first_diagnosis'] // month_interval) * month_interval
#
#     # Pivot tables for both clusters
#     pivot1 = data1.pivot_table(index='month_intervals', columns='Disease_Index', values='eid',
#                                aggfunc=pd.Series.nunique, fill_value=0).reindex(columns=yaxis_range).fillna(
#         0).reset_index()
#     pivot2 = data2.pivot_table(index='month_intervals', columns='Disease_Index', values='eid',
#                                aggfunc=pd.Series.nunique, fill_value=0).reindex(columns=yaxis_range).fillna(
#         0).reset_index()
#
#     # Normalize the counts by total participants for each group
#     normalized1 = pivot1.set_index('month_intervals').values.T / total1
#     normalized2 = pivot2.set_index('month_intervals').values.T / total2
#
#     # Align the rows (month intervals) of both matrices
#     aligned_intervals = sorted(set(pivot1['month_intervals']).intersection(set(pivot2['month_intervals'])))
#     normalized1_aligned = pivot1[pivot1['month_intervals'].isin(aligned_intervals)].set_index(
#         'month_intervals').values.T / total1
#     normalized2_aligned = pivot2[pivot2['month_intervals'].isin(aligned_intervals)].set_index(
#         'month_intervals').values.T / total2
#
#     # Compute the difference
#     difference = normalized1_aligned - normalized2_aligned  # Signed difference
#
#     return difference, aligned_intervals
#
#
# # Function to plot difference heatmap
# def plot_difference_heatmap(fig, difference, intervals, row, col, title, cluster1_label, cluster2_label):
#     fig.add_trace(
#         go.Heatmap(z=difference, x=intervals,
#                    y=list(disease_index_mapping.keys()), colorscale='RdBu', zmid=0,
#                    colorbar=dict(title='Difference %')),
#         row=row, col=col
#     )
#     fig.update_yaxes(title_text=title, row=row, col=col)
#
#     # Annotate most red and most blue regions
#     max_red = np.unravel_index(np.argmax(difference), difference.shape)
#     max_blue = np.unravel_index(np.argmin(difference), difference.shape)
#
#     # Adding annotations
#     fig.add_annotation(x=intervals[max_red[1]], y=max_red[0],
#                        text=f'More {cluster1_label}', showarrow=True,
#                        arrowhead=2, arrowsize=1, arrowcolor="red",
#                        ax=-50, ay=-30, bordercolor="red",
#                        row=row, col=col)
#
#     fig.add_annotation(x=intervals[max_blue[1]], y=max_blue[0],
#                        text=f'More {cluster2_label}', showarrow=True,
#                        arrowhead=2, arrowsize=1, arrowcolor="blue",
#                        ax=-50, ay=-30, bordercolor="blue",
#                        row=row, col=col)
#
#
# # Function to plot cluster heatmaps and return figure and x-axis range
# def plot_cluster_heatmaps_with_differences(diagnoses_cluster1, diagnoses_cluster2, diagnoses_cluster1_group0,
#                                            diagnoses_cluster2_group0, drug, first_diag_months_cluster1,
#                                            first_diag_months_cluster2, max_months, month_interval, min_samples):
#     # Create figure with subplots for heatmaps and difference heatmaps
#     fig = make_subplots(rows=2, cols=4,
#                         subplot_titles=[f'Cluster 1 - {drug}', f'Cluster 2 - {drug}', f'Cluster 1 - Group 0',
#                                         f'Cluster 2 - Group 0',
#                                         f'Difference: Cluster 1 vs Cluster 2',
#                                         f'Difference: Cluster 1 vs Group 0',
#                                         f'Difference: Cluster 2 vs Group 0'])
#
#     # Set x-axis range
#     xaxis_range = [-max_months, max_months]
#
#     # Ensure both heatmaps show all disease indexes
#     yaxis_range = list(disease_index_mapping.keys())
#
#     # Check the number of participants for both clusters
#     total_cluster1 = diagnoses_cluster1['eid'].nunique()
#     total_cluster2 = diagnoses_cluster2['eid'].nunique()
#     total_group0_cluster1 = diagnoses_cluster1_group0['eid'].nunique()
#     total_group0_cluster2 = diagnoses_cluster2_group0['eid'].nunique()
#
#     # Plot heatmaps
#     if total_cluster1 >= min_samples:
#         plot_heatmap_for_cluster(diagnoses_cluster1, fig, first_diag_months_cluster1, max_months, row=1, col=1,
#                                  yaxis_range=yaxis_range, month_interval=month_interval,
#                                  total_participants=total_cluster1)
#
#     if total_cluster2 >= min_samples:
#         plot_heatmap_for_cluster(diagnoses_cluster2, fig, first_diag_months_cluster2, max_months, row=1, col=2,
#                                  yaxis_range=yaxis_range, month_interval=month_interval,
#                                  total_participants=total_cluster2)
#
#     if total_group0_cluster1 >= min_samples:
#         plot_heatmap_for_cluster(diagnoses_cluster1_group0, fig, first_diag_months_cluster1, max_months, row=1,
#                                  col=3, yaxis_range=yaxis_range, month_interval=month_interval,
#                                  total_participants=total_group0_cluster1)
#
#     if total_group0_cluster2 >= min_samples:
#         plot_heatmap_for_cluster(diagnoses_cluster2_group0, fig, first_diag_months_cluster2, max_months, row=1,
#                                  col=4, yaxis_range=yaxis_range, month_interval=month_interval,
#                                  total_participants=total_group0_cluster2)
#
#     # Compute difference heatmaps
#     diff_cluster1_cluster2, intervals = compute_difference_heatmap(diagnoses_cluster1, diagnoses_cluster2,
#                                                                    yaxis_range, total_cluster1, total_cluster2,
#                                                                    month_interval)
#
#     diff_cluster1_group0, intervals = compute_difference_heatmap(diagnoses_cluster1, diagnoses_cluster1_group0,
#                                                                  yaxis_range, total_cluster1, total_group0_cluster1,
#                                                                  month_interval)
#
#     diff_cluster2_group0, intervals = compute_difference_heatmap(diagnoses_cluster2, diagnoses_cluster2_group0,
#                                                                  yaxis_range, total_cluster2, total_group0_cluster2,
#                                                                  month_interval)
#
#     # Plot difference heatmaps
#     plot_difference_heatmap(fig, diff_cluster1_cluster2, intervals, row=2, col=1, title=f'Cluster 1 vs Cluster 2',
#                             cluster1_label='Cluster 1', cluster2_label='Cluster 2')
#     plot_difference_heatmap(fig, diff_cluster1_group0, intervals, row=2, col=2, title=f'Cluster 1 vs Group 0',
#                             cluster1_label='Cluster 1', cluster2_label='Group 0')
#     plot_difference_heatmap(fig, diff_cluster2_group0, intervals, row=2, col=3, title=f'Cluster 2 vs Group 0',
#                             cluster1_label='Cluster 2', cluster2_label='Group 0')
#
#     # Show the final figure
#     fig.update_layout(height=800, width=1800, title=f'Cluster Comparison - Drug: {drug}')
#     fig.show()
#
#     return fig, xaxis_range
#
#
# def plot_heatmap_for_cluster(diagnoses_cluster, fig, first_diag_months, max_months, row, col, yaxis_range,
#                              month_interval, total_participants):
#     heatmap_data = diagnoses_cluster.copy()
#
#     # Calculate months relative to the first diagnosis
#     heatmap_data['months_since_first_diagnosis'] = (
#                 (heatmap_data['date'] - heatmap_data['date_first_diagnosis']).dt.days / 30.44).round().astype(int)
#
#     # Restrict data to a defined range of months
#     heatmap_data = heatmap_data[
#         (heatmap_data['months_since_first_diagnosis'] >= -120) & (heatmap_data['months_since_first_diagnosis'] <= 120)]
#
#     # Group the data based on the specified month_interval
#     heatmap_data['month_intervals'] = (heatmap_data['months_since_first_diagnosis'] // month_interval) * month_interval
#
#     # Pivot table to get the count of participants diagnosed with each disease index at each specified month interval
#     pivot_cluster = heatmap_data.pivot_table(index='month_intervals', columns='Disease_Index', values='eid',
#                                              aggfunc=pd.Series.nunique, fill_value=0).reindex(
#         columns=yaxis_range).fillna(0).reset_index()
#
#     # Normalize the counts by the total number of participants
#     normalized_z = pivot_cluster.set_index('month_intervals').values.T / total_participants
#
#     # Apply logarithmic transformation
#     log_z = np.log10(normalized_z + 1e-3)  # Small offset to avoid log(0)
#
#     # Plot the heatmap
#     fig.add_trace(
#         go.Heatmap(z=log_z, x=pivot_cluster['month_intervals'],
#                    y=pivot_cluster.columns[1:], colorscale='Plasma', zmin=-3, zmax=0,
#                    colorbar=dict(title='Log10(Participants %)')),
#         row=row, col=col
#     )
#
#     # Add a vertical line for the first diagnosis
#     fig.add_vline(x=first_diag_months, line_dash="dash", line_color="red", row=row, col=col)
#
#
# # Main function to dynamically plot clusters and their differences
# def dynamic_plot_clusters_with_differences(clusters, prescriptions, diagnoses, max_months=240,
#                                            month_interval=1, min_samples=10):
#     # Filter Group 0 participants
#     group0_participants = clusters[clusters['Group'] == 0]
#
#     for cluster_label in clusters['cluster'].unique():
#         drug_columns = prescriptions.columns.difference(['ParticipantID'])
#
#         for drug in drug_columns:
#             participants_with_drug = prescriptions[prescriptions[drug].notna()]['ParticipantID'].unique()
#             cluster1_participants = clusters[
#                 (clusters['ParticipantID'].isin(participants_with_drug)) & (clusters['cluster'] == cluster_label)]
#
#             # Filter participants for Group 0
#             group0_cluster_participants = group0_participants[group0_participants['cluster'] == cluster_label]
#
#             # Filter diagnoses for clusters
#             diagnoses_cluster1 = diagnoses[diagnoses['eid'].isin(cluster1_participants['ParticipantID'])]
#             diagnoses_group0_cluster = diagnoses[diagnoses['eid'].isin(group0_cluster_participants['ParticipantID'])]
#
#             # Skip plotting if there are fewer participants than min_samples
#             if diagnoses_cluster1['eid'].nunique() < min_samples:
#                 print(f"Skipping cluster {cluster_label} for {drug} due to insufficient sample size.")
#                 continue
#
#             # Calculate the number of months corresponding to the first diagnosis for each cluster
#             first_diag_months_cluster1 = ((diagnoses_cluster1['date_first_diagnosis'] - diagnoses_cluster1[
#                 'date_first_diagnosis']).dt.days / 30.44).fillna(0).round().astype(int).min()
#             first_diag_months_cluster2 = ((diagnoses_group0_cluster['date_first_diagnosis'] - diagnoses_group0_cluster[
#                 'date_first_diagnosis']).dt.days / 30.44).fillna(0).round().astype(int).min()
#
#             # Plot the heatmaps for both clusters and their Group 0 controls with differences
#             fig, xaxis_range = plot_cluster_heatmaps_with_differences(diagnoses_cluster1, diagnoses_group0_cluster,
#                                                                       diagnoses_cluster1, diagnoses_group0_cluster,
#                                                                       drug, first_diag_months_cluster1,
#                                                                       first_diag_months_cluster2, max_months,
#                                                                       month_interval, min_samples)
#
#
# # Main execution function
# def main(keyword, exclusion_suffix='_all_participants', max_months=240, month_interval=1, min_samples=10,
#          num_clusters=3):
#     # Load data
#     all_diagnoses = pd.read_csv("diseases_mapped_all_participants.csv")
#     keyword_diagnoses = pd.read_csv(f"{keyword}_cancer_diagnosis.csv")
#     prescriptions = pd.read_csv(f"{keyword}_general_cancer_drugs_survival{exclusion_suffix}.csv")
#     genotypes = pd.read_csv(f"{keyword}_genotype_survival{exclusion_suffix}.csv")
#
#     keyword_diagnoses['date'] = pd.to_datetime(keyword_diagnoses['date'], errors='coerce')
#     all_diagnoses['date'] = pd.to_datetime(all_diagnoses['date'], errors='coerce')
#
#     first_diagnosis_dates = keyword_diagnoses.groupby('eid').apply(lambda x: x.nsmallest(1, 'date')).reset_index(
#         drop=True)
#     first_diagnosis_dates = first_diagnosis_dates[['eid', 'date']]
#     merged_diagnoses = pd.merge(all_diagnoses, first_diagnosis_dates, on='eid', how='left',
#                                 suffixes=('', '_first_diagnosis'))
#
#     # Filter out rows with missing 'date_first_diagnosis'
#     merged_diagnoses = merged_diagnoses[merged_diagnoses['date_first_diagnosis'].notna()]
#
#     # Calculate months since first diagnosis
#     merged_diagnoses['months_since_first_diagnosis'] = (
#             (merged_diagnoses['date'] - merged_diagnoses['date_first_diagnosis']).dt.days / 30.44).fillna(
#         0).round().astype(int)
#
#     # Cluster the participants based on their survival days
#     clusters, kmeans = cluster_participants(genotypes, num_clusters=num_clusters)
#
#     # Run the plotting with clusters and differences
#     dynamic_plot_clusters_with_differences(clusters, prescriptions, merged_diagnoses, max_months, month_interval,
#                                            min_samples)
#
#
# if __name__ == "__main__":
#     keyword = "bone"
#     month_interval = 24
#     min_samples = 10
#     num_clusters = 3
#     main(keyword, max_months=400, month_interval=24, min_samples=min_samples, num_clusters=num_clusters)

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go

def retrieve_patients_with_disease_and_cluster(diagnoses, cluster_df, disease_class, cluster_label):
    diagnoses_disease_class = diagnoses[diagnoses['Disease_Index'] == disease_class]
    patients_cluster = cluster_df[cluster_df['genotype_cluster'] == cluster_label]
    merged_patients = pd.merge(diagnoses_disease_class, patients_cluster, left_on='eid', right_on='ParticipantID')
    return merged_patients['eid'].unique()


# Function to plot genotype (cluster) heatmaps for drug intakes
def plot_genotype_heatmaps(prescriptions_cluster1, prescriptions_cluster2, cluster1_label, cluster2_label, drug,
                           top_atc_codes, max_months):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Cluster {cluster1_label}', f'Cluster {cluster2_label}'))

    prescriptions_cluster1['Months Since First Prescription'] = (
                                                                        pd.to_datetime(prescriptions_cluster1[
                                                                                           'Date prescription was issued']) -
                                                                        pd.to_datetime(prescriptions_cluster1[
                                                                                           'First_Prescription_Date'])
                                                                ).dt.days // 30

    prescriptions_cluster2['Months Since First Prescription'] = (
                                                                        pd.to_datetime(prescriptions_cluster2[
                                                                                           'Date prescription was issued']) -
                                                                        pd.to_datetime(prescriptions_cluster2[
                                                                                           'First_Prescription_Date'])
                                                                ).dt.days // 30

    # Filter out any missing values
    prescriptions_cluster1 = prescriptions_cluster1.dropna(subset=['Months Since First Prescription']).copy()
    prescriptions_cluster2 = prescriptions_cluster2.dropna(subset=['Months Since First Prescription']).copy()

    # Plot heatmaps for drug prescription patterns
    plot_heatmap_for_genotype(prescriptions_cluster1, fig, top_atc_codes, max_months, row=1, col=1)
    fig.update_layout(
        title=f'Drug Prescription Patterns Over Time for Cluster {cluster1_label} vs Cluster {cluster2_label} - Drug: {drug}',
        xaxis_title='Months Since First Prescription',
        yaxis_title='Drugs (ATC Code)',
        height=600,
        width=1200
    )
    fig.show()


# Function to plot heatmap for drug intake over time
def plot_heatmap_for_genotype(prescriptions_cluster, fig, top_atc_codes, max_months, row, col):
    heatmap_data = prescriptions_cluster.copy()
    heatmap_data = heatmap_data[heatmap_data['Months Since First Prescription'] <= max_months]
    heatmap_data = heatmap_data[heatmap_data['atc_code'].isin(top_atc_codes)]

    pivot_genotype = heatmap_data.pivot_table(
        index='Months Since First Prescription',
        columns='atc_code',
        aggfunc='size',
        fill_value=0
    ).reset_index()

    fig.add_trace(
        go.Heatmap(
            z=pivot_genotype.set_index('Months Since First Prescription').values.T,
            x=pivot_genotype['Months Since First Prescription'],
            y=pivot_genotype.columns[1:],  # Skip 'Months Since First Prescription' column
            colorscale='YlGnBu',
            zmin=0,
            zmax=pivot_genotype.iloc[:, 1:].values.max(),  # Use max for color scale
            colorbar=dict(title='Frequency')
        ),
        row=row, col=col
    )


# Main function to dynamically retrieve and plot cluster data
def dynamic_plot_clusters(cluster_df, diagnoses, prescriptions, max_months=240):
    
    disease_class = 17  # ICD Class 17: Symptoms & Signs
    cluster_label = 2  # Cluster 2
    patients_in_cluster_with_disease = retrieve_patients_with_disease_and_cluster(diagnoses, cluster_df, disease_class,
                                                                                  cluster_label)

    prescriptions_cluster = prescriptions[prescriptions['Participant ID'].isin(patients_in_cluster_with_disease)]
    first_prescription_dates = prescriptions_cluster.groupby('Participant ID')[
        'Date prescription was issued'].min().reset_index()
    first_prescription_dates.rename(columns={'Date prescription was issued': 'First_Prescription_Date'}, inplace=True)
    
    prescriptions_cluster = pd.merge(prescriptions_cluster, first_prescription_dates, on='Participant ID', how='left')
    top_atc_codes = prescriptions_cluster['atc_code'].value_counts().nlargest(10).index
    plot_genotype_heatmaps(prescriptions_cluster, prescriptions_cluster, cluster_label, cluster_label, 'Drug',
                           top_atc_codes, max_months)


def main():
    from sklearn.cluster import KMeans
    import pandas as pd
    import pandas as pd
    import numpy as np

    num_participants = 100
    np.random.seed(42)

    # Participant IDs
    participant_ids = np.arange(1, num_participants + 1)
    survival_days = np.random.randint(100, 2000, size=num_participants)
    genes = np.random.choice(['DPYD', 'CYP2C8'], size=num_participants)
    genotypes_dpy = ['*2A/*2A', '*2A/*2B', '*2B/*2B']
    genotypes_cyp = ['*4/*4', '*1/*4', '*1/*1']

    genotypes = [
        np.random.choice(genotypes_dpy if gene == 'DPYD' else genotypes_cyp)
        for gene in genes
    ]
    genotype_survival = pd.DataFrame({
        'ParticipantID': participant_ids,
        'max_survival_days': survival_days,
        'Gene': genes,
        'Genotype': genotypes
    })

    genotype_survival.to_csv('genotype_survival_data.csv', index=False)

    print("Generated 'genotype_survival_data.csv' with simulated data:")
    print(genotype_survival.head())
    genotype_survival = pd.read_csv('genotype_survival_data.csv')

    clustering_data = genotype_survival[['max_survival_days']]

    kmeans = KMeans(n_clusters=3, random_state=42)
    genotype_survival['genotype_cluster'] = kmeans.fit_predict(clustering_data)
    genotype_survival.to_csv('cluster_genotype_survival.csv', index=False)

    diagnoses = pd.read_csv("diseases_mapped_all_participants.csv")
    cluster_df = pd.read_csv("cluster_genotype_survival.csv")  # This contains clusters from the clustering step
    prescriptions = pd.read_csv("final_prescription_with_ATC_codes.csv")

    dynamic_plot_clusters(cluster_df, diagnoses, prescriptions, max_months=240)


# Run the script
if __name__ == "__main__":
    main()
