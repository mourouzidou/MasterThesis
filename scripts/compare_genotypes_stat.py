import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib import pyplot as plt
from scipy.stats import mannwhitneyu
import seaborn as sns
import numpy as np



def plot_drug_usage(ax, df, title):
    drug_usage = df.drop(columns=['ParticipantID', 'Date_of_Death']).count()
    drug_usage.plot(kind='bar', ax=ax, color='lightcoral', edgecolor='black')
    ax.set_title(f'Drug Usage Distribution - {title}')
    ax.set_ylabel('Number of Participants')
    ax.set_xlabel('Drug (ATC Code)')
    ax.tick_params(axis='x', rotation=45)


# Function to calculate the median survival for treated and untreated groups
def calculate_median_survival(df):
    treated_median = df[df['Group'] == 1]['max_survival_days'].median()
    untreated_median = df[df['Group'] == 0]['max_survival_days'].median()
    return treated_median, untreated_median


# Function to perform the log-rank test
def perform_logrank_test(df):
    treated = df[df['Group'] == 1]
    untreated = df[df['Group'] == 0]
    results = logrank_test(treated['max_survival_days'], untreated['max_survival_days'],
                           event_observed_A=(treated['max_survival_days'] > 0),
                           event_observed_B=(untreated['max_survival_days'] > 0))
    return results.p_value


# Function to plot Kaplan-Meier survival curves and statistics
def plot_kaplan_meier_with_statistics(ax, df):
    kmf = KaplanMeierFitter()

    treated = df[df['Group'] == 1]
    untreated = df[df['Group'] == 0]

    # Fit and plot survival for treated group
    kmf.fit(treated['max_survival_days'], event_observed=(treated['max_survival_days'] > 0), label='Treated')
    kmf.plot_survival_function(ax=ax)

    # Fit and plot survival for untreated group
    kmf.fit(untreated['max_survival_days'], event_observed=(untreated['max_survival_days'] > 0), label='Untreated')
    kmf.plot_survival_function(ax=ax)

    # Calculate statistics
    treated_median, untreated_median = calculate_median_survival(df)
    p_value = perform_logrank_test(df)

    # Annotate the plot with statistics
    ax.set_title('Kaplan-Meier Survival Curves')
    ax.set_ylabel('Survival Probability')
    ax.set_xlabel('Days')
    ax.text(0.6, 0.2, f"Treated Median: {treated_median} days", transform=ax.transAxes, fontsize=10)
    ax.text(0.6, 0.15, f"Untreated Median: {untreated_median} days", transform=ax.transAxes, fontsize=10)
    ax.text(0.6, 0.1, f"Log-rank p-value: {p_value:.4f}", transform=ax.transAxes, fontsize=10)


def plot_survival_distribution(ax, df):
    ax.hist(df['max_survival_days'], bins=60, color='skyblue', edgecolor='black')
    ax.set_title('Distribution of Max Survival Days')
    ax.set_xlabel('Max Survival Days')
    ax.set_ylabel('Number of Participants')
    ax.grid(True)
def plot_summary(genotype_survival, general_prescription_survival, specific_prescription_survival):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 6))
    plot_drug_usage(axs[0,1], general_prescription_survival, title="General Cancer Drugs")
    plot_drug_usage(axs[1, 1],specific_prescription_survival , title="Specific Cancer Drugs")
    plot_survival_distribution(axs[1, 0], genotype_survival)
    plot_kaplan_meier_with_statistics(axs[0,0], genotype_survival)
    plt.tight_layout()
    plt.show()



def calculate_median_survival(df):
    treated_median = df[df['Group'] == 1]['max_survival_days'].median()
    untreated_median = df[df['Group'] == 0]['max_survival_days'].median()
    return treated_median, untreated_median

def plot_drug_usage(ax, df, title, drug_prefix_length=2):
    drug_usage = df.drop(columns=['ParticipantID', 'Date_of_Death']).count()
    drug_usage.plot(kind='bar', ax=ax, color='lightcoral', edgecolor='black')
    ax.set_title(f"Drug Usage Distribution - {title}")
    ax.set_ylabel('Number of Participants')
    ax.set_xlabel('Drug (ATC Code)' if drug_prefix_length != 2 else 'Any Antineoplastic and Immunomodulating Agent')
    ax.tick_params(axis='x', rotation=45)

def perform_logrank_test(df):
    treated = df[df['Group'] == 1]
    untreated = df[df['Group'] == 0]
    results = logrank_test(treated['max_survival_days'], untreated['max_survival_days'],
                           event_observed_A=(treated['max_survival_days'] > 0),
                           event_observed_B=(untreated['max_survival_days'] > 0))
    return results.p_value

def plot_kaplan_meier_with_statistics(ax, df):
    kmf = KaplanMeierFitter()
    treated, untreated = df[df['Group'] == 1], df[df['Group'] == 0]

    kmf.fit(treated['max_survival_days'], event_observed=(treated['max_survival_days'] > 0), label='Treated')
    kmf.plot_survival_function(ax=ax)

    kmf.fit(untreated['max_survival_days'], event_observed=(untreated['max_survival_days'] > 0), label='Untreated')
    kmf.plot_survival_function(ax=ax)

    treated_median, untreated_median = calculate_median_survival(df)
    p_value = perform_logrank_test(df)

    ax.set_title('Kaplan-Meier Survival Curves')
    ax.set_ylabel('Survival Probability')
    ax.set_xlabel('Days')
    ax.text(0.6, 0.2, f"Treated Median: {treated_median} days", transform=ax.transAxes, fontsize=10)
    ax.text(0.6, 0.15, f"Untreated Median: {untreated_median} days", transform=ax.transAxes, fontsize=10)
    ax.text(0.6, 0.1, f"Log-rank p-value: {p_value:.4f}", transform=ax.transAxes, fontsize=10)

def plot_survival_distribution(ax, df):
    ax.hist(df['max_survival_days'], bins=60, color='skyblue', edgecolor='black')
    ax.set_title('Distribution of Max Survival Days')
    ax.set_xlabel('Max Survival Days')
    ax.set_ylabel('Number of Participants')
    ax.grid(True)

def plot_summary(genotype_survival, general_prescription_survival, specific_prescription_survival, drug_prefix_length=2):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 6))
    plot_drug_usage(axs[0, 1], general_prescription_survival, title="General Cancer Drugs", drug_prefix_length=drug_prefix_length)
    plot_drug_usage(axs[1, 1], specific_prescription_survival, title="Specific Cancer Drugs", drug_prefix_length=drug_prefix_length)
    plot_survival_distribution(axs[1, 0], genotype_survival)
    plot_kaplan_meier_with_statistics(axs[0, 0], genotype_survival)
    plt.tight_layout()
    plt.show()


def analyze_genotype_drug_pairs_wilcoxon(diagnosis_path, genotype_path, prescriptions_path, keyword,
                                         p_value_threshold=0.05, min_samples_per_genotype=5, drug_prefix_length=2):
    # Read input files
    diagnosis_df = pd.read_csv(diagnosis_path)
    genotype_survival_df = pd.read_csv(genotype_path)
    prescriptions_df = pd.read_csv(prescriptions_path)

    drug_columns = prescriptions_df.columns[1:-1]

    # Group drugs by prefix
    general_cancer_drugs_grouped = pd.DataFrame()
    general_cancer_drugs_grouped['ParticipantID'] = prescriptions_df['ParticipantID']

    for prefix, group in prescriptions_df[drug_columns].groupby(lambda x: x[:drug_prefix_length], axis=1):
        general_cancer_drugs_grouped[prefix] = group.apply(lambda x: x.min(skipna=True), axis=1)

    general_cancer_drugs_grouped['Date_of_Death'] = prescriptions_df['Date_of_Death']

    earliest_diagnosis = diagnosis_df.groupby('ParticipantID')['date'].min().reset_index()
    earliest_diagnosis.columns = ['ParticipantID', 'Earliest_Diagnosis_Date']
    merged_df = genotype_survival_df.merge(earliest_diagnosis, on='ParticipantID', how='inner')
    merged_df = merged_df.merge(general_cancer_drugs_grouped, on='ParticipantID', how='inner')

    filtered_df = merged_df.dropna(subset=['max_survival_days'])

    significant_results = []
    for drug in general_cancer_drugs_grouped.columns[1:-1]:
        participants_with_drug = filtered_df[~filtered_df[drug].isna()]
        if len(participants_with_drug) > 1:
            gene_columns = ['CYP2C19', 'CYP2C8', 'CYP2C9', 'CYP2D6', 'CYP3A4', 'CYP3A5', 'DPYD', 'NAT2', 'SLCO1B1',
                            'TPMT', 'UGT1A1']

            for gene in gene_columns:
                genotype_groups = participants_with_drug.groupby(gene)
                genotypes = list(genotype_groups.groups.keys())

                for i in range(len(genotypes)):
                    for j in range(i + 1, len(genotypes)):
                        genotype1, genotype2 = genotypes[i], genotypes[j]
                        group1 = genotype_groups.get_group(genotype1)['max_survival_days']
                        group2 = genotype_groups.get_group(genotype2)['max_survival_days']

                        if len(group1) >= min_samples_per_genotype and len(group2) >= min_samples_per_genotype:
                            stat, p_value = mannwhitneyu(group1, group2)
                            if p_value < p_value_threshold:
                                significant_results.append({
                                    'Gene': gene,
                                    'Drug': drug,
                                    'Genotype1': genotype1,
                                    'Genotype2': genotype2,
                                    'Wilcoxon p-value': p_value
                                })

    significant_results_df = pd.DataFrame(significant_results)
    output_file = f'{keyword}_significant_wilcoxon_genotype_drug_pairs.csv'
    significant_results_df.to_csv(output_file, index=False)
    print(f"Analysis complete. Significant genotype-drug pairs saved to {output_file}.")

    plot_boxplots_for_significant_genotypes(filtered_df, significant_results_df)

    return significant_results_df

def plot_boxplots_for_significant_genotypes(filtered_df, significant_results_df):
    for idx, row in significant_results_df.iterrows():
        gene, drug = row['Gene'], row['Drug']
        genotype1, genotype2 = row['Genotype1'], row['Genotype2']

        data_to_plot = filtered_df[filtered_df[drug].notna() & filtered_df[gene].isin([genotype1, genotype2])]

        plt.figure(figsize=(8, 6))
        sns.boxplot(x=gene, y='max_survival_days', data=data_to_plot)

        genotype_counts = data_to_plot[gene].value_counts()
        for genotype, count in genotype_counts.items():
            median_val = data_to_plot[data_to_plot[gene] == genotype]['max_survival_days'].median()
            plt.text(
                x=genotype, y=median_val, s=f'n={count}',
                color='black', ha='center', va='center', fontsize=10, weight='bold'
            )

        plt.title(f"Survival Days by Genotype for {gene} (Drug: {drug})")
        plt.ylabel("Survival Days after Diagnosis")
        plt.xlabel(f"{gene} Genotype")
        plt.tight_layout()
        plt.show()

def get_took_significant_and_took_other(genotypes_df, significant_df, prescriptions_df, drug_prefix_length):
    took_significant = set()
    took_other = set()

    for _, row in significant_df.iterrows():
        drug_prefix = row['Drug'][:drug_prefix_length]

        drug_columns = [col for col in prescriptions_df.columns if col[:drug_prefix_length] == drug_prefix]
        took_significant_ids = prescriptions_df[prescriptions_df[drug_columns].notna().any(axis=1)]['ParticipantID'].tolist()
        took_significant.update(took_significant_ids)
        other_drug_columns = [col for col in prescriptions_df.columns[1:-1] if col not in drug_columns]  # Exclude ParticipantID and Date_of_Death
        took_other_ids = prescriptions_df[prescriptions_df[other_drug_columns].notna().any(axis=1)]['ParticipantID'].tolist()

        took_other_ids = [id_ for id_ in took_other_ids if id_ not in took_significant_ids]
        took_other.update(took_other_ids)
    print(genotypes_df.columns)
    untreated_ids = genotypes_df[genotypes_df['Group'] == 0]['ParticipantID'].tolist()

    return list(took_significant), list(took_other), untreated_ids

def create_combined_df_for_plotting(significant_df, genotype_df, prescriptions_df, drug_prefix_length):
    combined_df_list = []

    # Loop through each significant row in significant_df
    for _, row in significant_df.iterrows():
        gene = row['Gene']
        drug = row['Drug']
        genotype1 = row['Genotype1']
        genotype2 = row['Genotype2']

        # Extract the relevant participant IDs based on the drug prefix
        drug_prefix = drug[:drug_prefix_length]
        drug_columns = [col for col in prescriptions_df.columns if col.startswith(drug_prefix)]
        took_significant_ids = prescriptions_df[prescriptions_df[drug_columns].notna().any(axis=1)]['ParticipantID']

        # Filter the genotype_df for the participants who took the significant drug
        significant_participants = genotype_df[(genotype_df['ParticipantID'].isin(took_significant_ids)) &
                                               (genotype_df[gene].isin([genotype1, genotype2]))]
        significant_participants['Group'] = drug

        # Get untreated participants from genotype_df where Group == 0
        untreated_participants = genotype_df[(genotype_df['Group'] == 0) & (genotype_df[gene].isin([genotype1, genotype2]))]
        untreated_participants['Group'] = 'Untreated'

        # Get participants who took any other drug
        other_drug_columns = [col for col in prescriptions_df.columns if col not in drug_columns + ['ParticipantID', 'Date_of_Death']]
        took_other_ids = prescriptions_df[prescriptions_df[other_drug_columns].notna().any(axis=1)]['ParticipantID']
        took_other_participants = genotype_df[(genotype_df['ParticipantID'].isin(took_other_ids)) &
                                              (genotype_df[gene].isin([genotype1, genotype2]))]
        took_other_participants['Group'] = 'Other Cancer Drugs'

        # Combine significant, untreated, and other drug participants into a single dataframe
        combined_df = pd.concat([significant_participants, untreated_participants, took_other_participants])

        # Add this dataframe to the list for further plotting
        combined_df_list.append(combined_df)

    # Combine all gene dataframes together
    return pd.concat(combined_df_list)

def plot_gene_survival_by_genotype_and_drugs(significant_df, genotype_df, prescriptions_df, drug_prefix_length):

    unique_genes = significant_df['Gene'].unique()
    num_genes = len(unique_genes)

    # Define number of rows and columns for the subplots
    cols = 2  # We want 2 columns of subplots
    rows = (num_genes + 1) // 2  # Calculate the number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
    axes = axes.flatten()  # Flatten for easy indexing

    for g_idx, gene in enumerate(unique_genes):
        gene_significant = significant_df[significant_df['Gene'] == gene]
        combined_df = create_combined_df_for_plotting(gene_significant, genotype_df, prescriptions_df, drug_prefix_length)

        # Create the boxplot
        sns.boxplot(x=gene, y='max_survival_days', hue='Group', data=combined_df, ax=axes[g_idx], palette='Set3')

        # Annotate sample sizes on the plot
        for genotype in combined_df[gene].unique():
            for group in combined_df['Group'].unique():
                sample_data = combined_df[(combined_df[gene] == genotype) & (combined_df['Group'] == group)]
                sample_size = sample_data.shape[0]
                if sample_size > 0:
                    median_y = sample_data['max_survival_days'].median()
                    x_position = list(combined_df[gene].unique()).index(genotype) + (list(combined_df['Group'].unique()).index(group) - 1.3) * 0.17
                    axes[g_idx].text(x=x_position, y=median_y + 10, s=f'n={sample_size}', horizontalalignment='center', color='black')

        axes[g_idx].set_title(f'Survival by {gene} Genotype Grouped by Drugs (Including Untreated & Other Cancer Drugs)')
        axes[g_idx].set_ylabel('Max Survival Days')
        axes[g_idx].set_xlabel(f'{gene} Genotype')

    # Remove any unused axes (if the number of genes is odd)
    if num_genes % 2 != 0:
        fig.delaxes(axes[-1])  # Remove the last empty axis if it's not used

    plt.tight_layout()
    plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()



def run_analysis(keyword, p_value_threshold=0.05, min_samples_per_genotype=5, drug_prefix_length=2, exclusion_suffix="all_participants"):

    diagnosis_path = f'{keyword}/{keyword}_cancer_diagnosis.csv'
    genotype_path = f'{keyword}/{keyword}_genotype_survival_{exclusion_suffix}.csv'
    prescriptions_path = f'{keyword}/{keyword}_general_cancer_drugs_survival_{exclusion_suffix}.csv'
    specific_prescriptions = pd.read_csv(f"{keyword}/{keyword}_specific_prescription_survival_{exclusion_suffix}.csv")

    genotype_df = pd.read_csv(genotype_path)
    prescriptions_df = pd.read_csv(prescriptions_path)
    plot_summary(genotype_df, prescriptions_df, specific_prescriptions)
    significant = analyze_genotype_drug_pairs_wilcoxon(diagnosis_path, genotype_path, prescriptions_path, keyword,
                                                       p_value_threshold, min_samples_per_genotype, drug_prefix_length)

    took_significant, took_other, untreated = get_took_significant_and_took_other(genotype_df, significant, prescriptions_df, drug_prefix_length)

    print("Significant:", significant)
    print("Took Significant:", took_significant)
    print("Took Other:", took_other)
    print("Untreated:", untreated)
    plot_gene_survival_by_genotype_and_drugs(significant, genotype_df, prescriptions_df, drug_prefix_length)


run_analysis('skin', p_value_threshold=0.05, min_samples_per_genotype=10, drug_prefix_length=7)
