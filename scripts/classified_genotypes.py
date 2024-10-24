import pandas as pd


def load_predicted_phenotypes(gene):
    # Load the predicted phenotypes for a gene
    predicted_df = pd.read_csv(f'{gene}_predicted_phenotypes.csv')
    predicted_df.columns = [gene, f'{gene}_Predicted_Phenotype']
    return predicted_df


def replace_genotypes_with_phenotypes(genotypes_df, genes_to_replace):
    # Iterate over each gene and replace the genotype with the predicted phenotype
    for gene in genes_to_replace:
        print(f"Processing {gene}...")
        predicted_df = load_predicted_phenotypes(gene)

        # Ensure that the predicted_df and genotypes_df have the same data types for the merge key
        genotypes_df[gene] = genotypes_df[gene].astype(str)
        predicted_df[gene] = predicted_df[gene].astype(str)

        # Merge smaller chunks to avoid memory issues
        chunk_size = 50000  # Adjust chunk size based on available memory
        chunks = [genotypes_df[i:i + chunk_size] for i in range(0, genotypes_df.shape[0], chunk_size)]

        merged_chunks = []
        for chunk in chunks:
            merged_chunk = chunk.merge(predicted_df, on=gene, how='left')
            # Replace the genotype column with the predicted phenotype
            merged_chunk[gene] = merged_chunk[f'{gene}_Predicted_Phenotype']
            # Drop the temporary column
            merged_chunk.drop(columns=[f'{gene}_Predicted_Phenotype'], inplace=True)
            merged_chunks.append(merged_chunk)

        # Concatenate all chunks
        genotypes_df = pd.concat(merged_chunks, ignore_index=True)

    return genotypes_df


def main():
    # Load the original genotypes dataframe
    genotypes_df = pd.read_csv('genotypes.csv')

    # Define the genes whose genotypes need to be replaced with predicted phenotypes
    genes_to_replace = ['CYP2D6', 'CYP2C9', 'CYP2C19', 'DPYD', 'TPMT', 'UGT1A1']

    # Replace genotypes with predicted phenotypes for the specified genes
    genotypes_df = replace_genotypes_with_phenotypes(genotypes_df, genes_to_replace)

    # Save the final dataframe to a new CSV file
    output_file = 'genotypes_with_predicted_phenotypes.csv'
    genotypes_df.to_csv(output_file, index=False)

    print(f"Saved final genotypes with predicted phenotypes to {output_file}")


if __name__ == "__main__":
    main()
