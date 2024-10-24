from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, chi2_contingency
import seaborn as sns
import scikit_posthocs as sp
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def load_data(keyword, exclusion_suffix):
    general_cancer_drugs_survival = pd.read_csv(f'{keyword}_general_cancer_drugs_survival{exclusion_suffix}.csv')
    specific_cancer_drugs_survival = pd.read_csv(f'{keyword}_specific_prescription_survival{exclusion_suffix}.csv')
    genotype_survival = pd.read_csv(f'{keyword}_genotype_survival{exclusion_suffix}.csv')
    demographic = pd.read_csv("demographic.csv")
    return general_cancer_drugs_survival, specific_cancer_drugs_survival, genotype_survival, demographic



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



def add_age_of_diagnosis(demographic, genotype_survival, current_date):
    demographic = demographic[demographic['Participant_ID'].isin(genotype_survival['ParticipantID'])]
    demographic.rename(columns={"Participant_ID": "ParticipantID"}, inplace=True)
    demographic.drop(columns=['Index'], inplace=True)
    demographic['Date_of_Death'] = pd.to_datetime(demographic['Date_of_Death'], errors='coerce')

    def calculate_age_of_diagnosis(row):
        if pd.notnull(row['Date_of_Death']):
            diagnosis_date = row['Date_of_Death'] - pd.to_timedelta(row['max_survival_days'], unit='D')
        else:
            diagnosis_date = current_date - pd.to_timedelta(row['max_survival_days'], unit='D')

        birth_year = pd.to_datetime(f"{int(row['Birth_Year'])}-01-01")
        age_of_diagnosis = (diagnosis_date - birth_year).days / 365.25
        return age_of_diagnosis

    demographic = pd.merge(demographic, genotype_survival[['ParticipantID', 'max_survival_days']],
                           on='ParticipantID', how='left')
    demographic['Age_of_Diagnosis'] = demographic.apply(calculate_age_of_diagnosis, axis=1)
    return demographic

def plot_survival_by_genotypes_and_drugs_with_significant(keyword, selected_genes, genotype_survival,
                                                          specific_prescription_survival,
                                                          drug_threshold=10, genotype_threshold=5,
                                                          p_value_threshold=0.05,
                                                          apply_bonferroni=True):
    drug_columns = [col for col in specific_prescription_survival.columns if
                    col != 'ParticipantID' and col != 'Date_of_Death']
    genotype_and_drugs = pd.merge(genotype_survival, specific_prescription_survival, on='ParticipantID')

    plot_data = []
    significant_results = []

    for selected_gene in selected_genes:
        for drug in drug_columns:
            patients_on_drug = genotype_and_drugs[genotype_and_drugs[drug].notna()]
            if patients_on_drug.shape[0] >= drug_threshold:
                drug_genotype_counts = patients_on_drug[selected_gene].value_counts()
                drug_valid_genotypes = drug_genotype_counts[drug_genotype_counts >= genotype_threshold].index.tolist()
                if len(drug_valid_genotypes) < 2:
                    continue
                patients_on_drug = patients_on_drug[patients_on_drug[selected_gene].isin(drug_valid_genotypes)]
                patients_on_drug['Drug'] = drug

                if not patients_on_drug.empty:
                    p_adjust_method = 'bonferroni' if apply_bonferroni else None
                    dunn_results = sp.posthoc_dunn(patients_on_drug, val_col='max_survival_days',
                                                   group_col=selected_gene,
                                                   p_adjust=p_adjust_method)

                    significant_pairs = dunn_results[dunn_results < p_value_threshold].stack().index.tolist()
                    if len(significant_pairs) > 0:
                        plot_data.append((patients_on_drug, drug, selected_gene, significant_pairs))
                        for pair in significant_pairs:
                            g1, g2 = pair
                            group1 = patients_on_drug[patients_on_drug[selected_gene] == g1]['max_survival_days']
                            group2 = patients_on_drug[patients_on_drug[selected_gene] == g2]['max_survival_days']
                            kw_pvalue = kruskal(group1, group2).pvalue
                            significant_results.append([selected_gene, drug, g1, g2, kw_pvalue])

    significant_results_df = pd.DataFrame(significant_results,
                                          columns=['Gene', 'Drug', 'Genotype1', 'Genotype2', 'Kruskal-Wallis p-value'])
    significant_results_df['Genotype_Sorted'] = significant_results_df.apply(
        lambda x: tuple(sorted([x['Genotype1'], x['Genotype2']])), axis=1)
    significant_pairs_unique = significant_results_df.drop_duplicates(subset=['Gene', 'Drug', 'Genotype_Sorted'])
    significant_pairs_unique = significant_pairs_unique.drop(columns=['Genotype_Sorted'])
    significant_pairs_unique.to_csv(f"{keyword}_significant_genotype_pairs.csv", index=False)

    num_plots = len(plot_data)
    if num_plots > 0:
        fig, axs = plt.subplots(nrows=(num_plots // 2) + (num_plots % 2), ncols=2, figsize=(15, 8 * (num_plots // 2)))
        axs = axs.flatten()

        for idx, (filtered_df, drug, selected_gene, significant_pairs) in enumerate(plot_data):
            significant_genotypes = set([g for pair in significant_pairs for g in pair])
            filtered_df = filtered_df[filtered_df[selected_gene].isin(significant_genotypes)]
            sns.boxplot(x=selected_gene, y='max_survival_days', hue=selected_gene, data=filtered_df, palette='Set3',
                        ax=axs[idx])
            for genotype in filtered_df[selected_gene].unique():
                sample_data = filtered_df[(filtered_df[selected_gene] == genotype) & (filtered_df['Drug'] == drug)]
                sample_size = sample_data.shape[0]
                if sample_size > 0:
                    median_y = sample_data['max_survival_days'].median()
                    x_position = list(filtered_df[selected_gene].unique()).index(genotype)
                    axs[idx].text(x=x_position, y=median_y + 10, s=f'n={sample_size}',
                                  horizontalalignment='center', color='black')

            for pair in significant_pairs:
                g1, g2 = pair
                axs[idx].plot([], [], ' ', label=f"Significant: {g1} vs {g2} (Drug: {drug})")

            axs[idx].set_title(f'Survival by {selected_gene} Genotype (Drug: {drug})')
            axs[idx].set_ylabel('Max Survival Days')
            axs[idx].set_xlabel(f'{selected_gene} Genotype')

        plt.tight_layout()
        plt.legend(title='Genotype', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
    else:
        print("No valid plots to display.")

    return significant_results


def plot_significant_survival_results(significant_results, genotype_survival, prescription_survival):
    significant_results_df = pd.DataFrame(significant_results,
                                          columns=['Gene', 'Drug', 'Genotype1', 'Genotype2', 'Kruskal-Wallis p-value'])
    genotype_and_drugs = pd.merge(genotype_survival, prescription_survival, on='ParticipantID')
    selected_genes = significant_results_df['Gene'].unique()
    num_genes = len(selected_genes)
    drug_columns = [col for col in prescription_survival.columns if col not in ['ParticipantID', 'Date_of_Death']]

    untreated_group = genotype_survival[genotype_survival['ParticipantID'].isin(
        prescription_survival[prescription_survival[drug_columns].isna().all(axis=1)]['ParticipantID']
    )]

    fig, axs = plt.subplots(nrows=(num_genes // 2) + (num_genes % 2), ncols=2, figsize=(15, 8 * ((num_genes + 1) // 2)))
    axs = axs.flatten()

    for g_idx, selected_gene in enumerate(selected_genes):
        gene_significant = significant_results_df[significant_results_df['Gene'] == selected_gene]

        filtered_df = pd.DataFrame()

        for _, row in gene_significant.iterrows():
            drug = row['Drug']
            genotype1, genotype2 = row['Genotype1'], row['Genotype2']
            if drug not in drug_columns:
                continue

            patients_on_drug = genotype_and_drugs[genotype_and_drugs[drug].notna()]
            valid_genotypes = [genotype1, genotype2]
            patients_on_drug = patients_on_drug[patients_on_drug[selected_gene].isin(valid_genotypes)]
            patients_on_drug['Drug'] = drug
            filtered_df = pd.concat([filtered_df, patients_on_drug], axis=0)

        untreated_patients = untreated_group[untreated_group[selected_gene].isin(valid_genotypes)]
        untreated_patients['Drug'] = 'Untreated'
        filtered_df = pd.concat([filtered_df, untreated_patients], axis=0)

        other_drug_patients = genotype_and_drugs[
            genotype_and_drugs[selected_gene].isin(valid_genotypes) &
            genotype_and_drugs[drug_columns].notna().any(axis=1)
            ]
        other_drug_patients['Drug'] = 'Other Cancer Drugs'
        filtered_df = pd.concat([filtered_df, other_drug_patients], axis=0)

        if not filtered_df.empty:
            sns.boxplot(x=selected_gene, y='max_survival_days', hue='Drug', data=filtered_df, palette='Set3',
                        ax=axs[g_idx])

            for genotype in filtered_df[selected_gene].unique():
                for drug_idx, drug in enumerate(filtered_df['Drug'].unique()):
                    sample_data = filtered_df[(filtered_df[selected_gene] == genotype) & (filtered_df['Drug'] == drug)]
                    sample_size = sample_data.shape[0]

                    if sample_size > 0:
                        median_y = sample_data['max_survival_days'].median()
                        x_position = list(filtered_df[selected_gene].unique()).index(genotype) + (drug_idx - 1.3) * 0.17
                        axs[g_idx].text(x=x_position, y=median_y + 10, s=f'n={sample_size}',
                                        horizontalalignment='center', color='black')

            axs[g_idx].set_title(
                f'Survival by {selected_gene} Genotype Grouped by Drugs (Including Untreated & Other Cancer Drugs)')
            axs[g_idx].set_ylabel('Max Survival Days')
            axs[g_idx].set_xlabel(f'{selected_gene} Genotype')

    if num_genes < len(axs):
        for idx in range(num_genes, len(axs)):
            fig.delaxes(axs[idx])

    plt.tight_layout()
    plt.legend(title='Drug', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


def compare_demographics_for_significant_pairs(significant_pairs, demographic_data, genotype_survival,
                                               prescription_survival, demographic_variables):
    for result in significant_pairs:
        selected_gene = result[0]
        selected_drug = result[1]
        genotype1 = result[2]
        genotype2 = result[3]

        if selected_drug not in prescription_survival.columns:
            continue

        patients_on_drug = prescription_survival[prescription_survival[selected_drug].notna()]
        patients_with_genotype = pd.merge(genotype_survival, patients_on_drug, on='ParticipantID')
        valid_genotypes = [genotype1, genotype2]
        patients_with_genotype = patients_with_genotype[patients_with_genotype[selected_gene].isin(valid_genotypes)]
        patients_with_demographics = pd.merge(patients_with_genotype, demographic_data, on='ParticipantID')

        for var in demographic_variables:
            if pd.api.types.is_numeric_dtype(patients_with_demographics[var]):
                groups = [group[var].dropna().values for _, group in patients_with_demographics.groupby(selected_gene)]
                if len(groups) > 1:
                    stat, p_value = kruskal(*groups)
                    if p_value < 0.05:
                        plt.figure(figsize=(8, 6))
                        sns.boxplot(x=selected_gene, y=var, data=patients_with_demographics)
                        for genotype in patients_with_demographics[selected_gene].unique():
                            sample_data = patients_with_demographics[
                                patients_with_demographics[selected_gene] == genotype]
                            sample_size = sample_data.shape[0]
                            median_y = sample_data[var].median()
                            x_position = list(patients_with_demographics[selected_gene].unique()).index(genotype)
                            plt.text(x=x_position, y=median_y + 10, s=f'n={sample_size}', horizontalalignment='center',
                                     color='black')
                        plt.title(f"{var} Comparison between {selected_gene} Genotypes (p = {p_value:.4f})")
                        plt.ylabel(var)
                        plt.xlabel(f"{selected_gene} Genotype")
                        plt.tight_layout()
                        plt.show()
            else:
                contingency_table = pd.crosstab(patients_with_demographics[selected_gene],
                                                patients_with_demographics[var])
                chi2, p_value, _, _ = chi2_contingency(contingency_table)
                if p_value < 0.05:
                    plt.figure(figsize=(8, 6))
                    sns.countplot(x=selected_gene, hue=var, data=patients_with_demographics)
                    for genotype in patients_with_demographics[selected_gene].unique():
                        sample_data = patients_with_demographics[patients_with_demographics[selected_gene] == genotype]
                        sample_size = sample_data.shape[0]
                        plt.text(list(patients_with_demographics[selected_gene].unique()).index(genotype),
                                 sample_size + 10,
                                 s=f'n={sample_size}', horizontalalignment='center', color='black')
                    plt.title(f"{var} Comparison between {selected_gene} Genotypes (p = {p_value:.4f})")
                    plt.ylabel("Count")
                    plt.xlabel(f"{selected_gene} Genotype")
                    plt.tight_layout()
                    plt.show()


def main():
    keyword = "respiratory"
    exclusion_suffix = "_all_participants"
    # exclusion_suffix = "_no_other_cancers"
    general_cancer_drugs_survival, specific_cancer_drugs_survival, genotype_survival, demographic = load_data(keyword,
                                                                                                              exclusion_suffix)

    print("")
    plot_summary(genotype_survival, general_cancer_drugs_survival, specific_cancer_drugs_survival)

    current_date = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))
    demographic = add_age_of_diagnosis(demographic, genotype_survival, current_date)
    genes_to_plot = genotype_survival.columns[3:]
    significant_pairs = plot_survival_by_genotypes_and_drugs_with_significant(keyword, genes_to_plot, genotype_survival,
                                                                              general_cancer_drugs_survival,
                                                                              drug_threshold=1, genotype_threshold=1,
                                                                              p_value_threshold=0.05,
                                                                              apply_bonferroni=False)

    plot_significant_survival_results(significant_pairs, genotype_survival, general_cancer_drugs_survival)
    demographic_variables = ['Age_of_Diagnosis', 'BMI', 'Sex', 'Smoking_Status', 'Alcohol_Status']
    compare_demographics_for_significant_pairs(significant_pairs, demographic, genotype_survival,
                                               general_cancer_drugs_survival, demographic_variables)


if __name__ == "__main__":
    main()
