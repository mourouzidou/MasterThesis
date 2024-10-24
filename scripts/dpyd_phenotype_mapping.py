import yaml
import pandas as pd


def load_haplotype_mapping(yml_file):
    with open(yml_file, 'r') as file:
        data = yaml.safe_load(file)

    haplotype_mapping = {}

    for allele, details in data['alleles'].items():
        # If 'label' exists, use it, otherwise use the allele name
        if 'label' in details and details['label']:
            label = details['label']
            if '*' in label:
                key = '*' + label.split('*')[1]  # Use the second part after '*'
            else:
                key = label  # Use the full label if no '*' is present
        else:
            if '*' in allele:
                key = '*' + allele.split('*')[1]
            else:
                key = allele  # Use the allele as-is if no '*' is present

        # Get the activity for the allele
        value = details['activity']

        # Convert 'function not assigned' and 'uncertain function' to 'unknown'
        if value in {'function not assigned', 'uncertain function'}:
            value = 'DPYD Indeterminate'

        # Store the key-value pair in the mapping dictionary
        haplotype_mapping[key] = value

    return haplotype_mapping


def assign_functionality(allele1_func, allele2_func):
    """Combine the functionalities of two alleles into a final functionality."""
    if allele1_func == 'decreased function' and allele2_func == 'decreased function':
        return 'decreased'
    elif (allele1_func == 'decreased function' and allele2_func == 'normal function') or \
            (allele1_func == 'normal function' and allele2_func == 'decreased function'):
        return 'moderate_to_normal'
    elif (allele1_func == 'decreased function' and allele2_func == 'no function') or \
            (allele1_func == 'no function' and allele2_func == 'decreased function'):
        return 'decreased_to_no'
    elif allele1_func == 'normal function' and allele2_func == 'normal function':
        return 'normal'
    elif (allele1_func == 'normal function' and allele2_func == 'no function') or \
            (allele1_func == 'no function' and allele2_func == 'normal function'):
        return 'moderate_to_low'
    elif allele1_func == 'no function' and allele2_func == 'no function':
        return 'no'
    else:
        return 'DPYD Indeterminate'


def create_diplotype_functionality_df(genotypes_df, haplotype_mapping, gene_column):
    # Get the unique diplotypes from the specified gene column
    diplotypes = genotypes_df[gene_column].unique()

    # Initialize lists to store diplotypes and their corresponding functionalities
    diplotype_list = []
    functionality_list = []

    for diplotype in diplotypes:
        # Split the diplotype by '/'
        alleles = diplotype.split('/')

        if len(alleles) != 2:
            # Skip or handle unexpected formats
            functionality_list.append('DPYD Indeterminate')
            continue

        allele1, allele2 = alleles[0], alleles[1]

        # Get the functionality from the haplotype mapping for both alleles
        allele1_func = haplotype_mapping.get(allele1, 'DPYD Indeterminate')
        allele2_func = haplotype_mapping.get(allele2, 'DPYD Indeterminate')

        # Assign a combined functionality based on the two alleles
        combined_functionality = assign_functionality(allele1_func, allele2_func)

        # Store the diplotype and its assigned functionality
        diplotype_list.append(diplotype)
        functionality_list.append(combined_functionality)

    # Create a DataFrame from the lists
    diplotype_functionality_df = pd.DataFrame({
        'DPYD Diplotype': diplotype_list,
        'Coded Diplotype/Phenotype Summary': functionality_list
    })

    return diplotype_functionality_df


# Example usage
genotypes_df = pd.read_csv('genotypes.csv')
haplotype_mapping = load_haplotype_mapping('dpyd.yml')
diplotype_functionality_df = create_diplotype_functionality_df(genotypes_df, haplotype_mapping, gene_column='DPYD')

diplotype_functionality_df.to_csv('DPYD_Diplotype_Phenotype_Table.csv', index=False)
print(diplotype_functionality_df)
