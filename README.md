The log-rank test compares the survival curves of treated vs. untreated groups.
Since the p-value is extremely small (essentially zero - 1.4226700137847922e-113), this indicates that there is a highly significant difference in the survival curves between treated and untreated patients.
There is strong evidence that the treatment has a **negative** impact on survival.


![image](https://github.com/user-attachments/assets/3481e5d9-9df7-4f27-aca7-465a689ab4ea)


*
keyword = skin
pvalue threshold 0.15 :
UGT1A1 *1/*1 unstreated are the froup with the lowest survival for this genotype 
![image](https://github.com/user-attachments/assets/45851507-a638-40e7-880e-95315c142373)
https://www.icd10data.com/ICD10CM/Codes/C00-D49

skin - unknown advantage 
![image](https://github.com/user-attachments/assets/1213bf1a-f66a-476d-a38f-9fc630d2f900)


Breast - CYP2C19, Unknown diplotypes for significant gene CYP2C19: ['*38/*38+rs17878459' '*38/*38+rs181297724' '*38/*38+rs17879685'
 '*38+rs17878459/*38+rs17878459'] 
 *38/*38 IS CONNSIDERED NORMAL/ low risk diplotype however combined with some snps it shows a significantly lower survival
Unknown diplotypes for significant gene CYP2C19: ['*38/*38+rs17878459' '*38/*38+rs181297724' '*38/*38+rs17879685'
 '*38+rs17878459/*38+rs17878459']


 from CYP2D6 diplotype-phenotype classification occurs that *1+rs72549358/*39 CYP2D6 Intermediate Metabolizer while *1/*39 is a normal metabolizer:
which drugs does CYP2D6 metabolize? are there any differences in the diseases timeline heatmap that are different among the two groups?
where is this SNP located?



# phenotype classification:
DPYD has not assigned phenotype thus we created a custom mapping according to pharmvar haplotype activity definitions
SLCO1B1 has not specified information about the haplotypes or diplotypes activity/functionality thus we handled the categories as unique diplotypes. The restricted unique number of diplotypes (homozygous and heterozygous combinations of (*1, *19, *41) facilitates this assumption.
CFTR has also two diplotypes, the wild type *1/*1 and the heterozygous *D1270N/*1, similarly to SLCL1B1 will be handled as two seperate categories
Smiilarly we hve no Phenotype ifnormation about NAT2, VKORC1, CYP3A4 and we will use the diplotypes as classes.

