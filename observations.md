
* Lymph keyword
Significant Survival Difference for UGT1A1 Genotypes and L02BG04 Drug:

    The Kruskal-Wallis test for the UGT1A1 gene and the L02BG04 drug shows a significant p-value of 0.0022.
    This suggests that there is a statistically significant difference in survival days between different UGT1A1 genotypes when treated with the L02BG04 drug

    ![image](https://github.com/user-attachments/assets/d32df9c0-97dc-47e5-8c3f-c54fa44dff6c)
  ![image](https://github.com/user-attachments/assets/13a5139d-ee34-4976-b460-0d53aa0e4375)

The Kruskal-Wallis test for Age_of_Diagnosis also shows a significant p-value of 0.0021. This means that the age at which participants were diagnosed differs significantly between different UGT1A1 genotypes for patients treated with L02BG04.
This could imply that patients with certain UGT1A1 genotypes may be diagnosed earlier or later compared to others.

For patients that have taken L02BG04(letrozole) UGT1A1*80/*93 seems to have greater survival rates (approximately 3 years more) than *1/*1.
We further checked other demographic factors that could potenitally contribute to this deviation such as age of diagnosis, bmi, smoking and alcohol status, and performed Kruskal-Wallis (for continuous) and Chi-Square test (for categorical data). It appeared the only parameter that significantly contributed to this deviation is age of diagnosis. More specifically genotypes with *80/*93 variations seemed to have a lower median age of diagnosis at 58 years old, compared to the wild type *1/*1 that had a median age of diagnosis at 66 years old. This leads us to investigate further two scenarios: 1) that patients with *80/93 for UGT1A1 gene that have taken letrozole better metabolize the drug - higher efficacy of letrozole is observed for *80/*93 patients compared to the wild type (is UGT1A1 *80/93 a potential biomarker for the prediction of letrozole response?), 2) that patients with *80/*93 were diagnosed earlier (genetic predisposition or more symptoms) than *1/*1 patients, and this led to a more immediate treatment which is more likely to be succesfull. 

 ![image](https://github.com/user-attachments/assets/6ede6758-2927-4311-8ae5-9a08ae1f22d1)





.





.

![image](https://github.com/user-attachments/assets/8712e406-799f-42ab-89ec-b08f86673fe1)


***Higher survival rate** *9+rs17376848/*rs17376848 
https://www.pharmgkb.org/clinicalAnnotation/1451287440
https://www.pharmgkb.org/variant/PA166153874/variantAnnotation

![image](https://github.com/user-attachments/assets/c0b36b3b-8c72-49a4-bb00-40f0178b70cd)

![image](https://github.com/user-attachments/assets/4c1273b9-b340-41c7-ac04-fe388413bf95)



*1/*4 vs *1+rs10509681/*4

![image](https://github.com/user-attachments/assets/e293193e-510e-4c5a-be85-43bc8f202116)




* 191 patients/2381 (total lung cancer patients) received cancer related drug `filtered_cancer_drugs.csv`
* 24 cancer related drugs in total


## Generate violin plot for genotypes of interest
* Filter samples that have taken at least one cancer drug and color each genotype by drug
* Normalize frequencies
![image](https://github.com/user-attachments/assets/d1bc6bac-13b4-4fdf-9f51-7a5fa7e6e221)


![image](https://github.com/user-attachments/assets/6d21ba9f-5872-4647-8ecd-bc09d29480c9)

![image](https://github.com/user-attachments/assets/24a55093-75f2-47c0-9f53-dce9d456a4c1)


## CYP2C8 *rs10509681 , unknown genotypes : higher survival comparing to *1/*1 ?
  * test significance, impact of rs10509681
![image](https://github.com/user-attachments/assets/88d93376-ceb8-46de-a609-03e73f0022f8)

## Statistical_summary_DPYD table
_generate a statistic summary table to store the p-values of different survival periods among different genotypes given a certain drug each time_
| Drug (ATC Code) | Genotype 1  | Genotype 2  |   Mean Difference |        p-value | Significant   |
|:----------------|:------------|:------------|------------------:|---------------:|:--------------|
| L02BA01         | *5/*9       | *9/*rs2297595 |       -245.765676 |  7.12727e-06   | True          |
| L02BA01         | *5/*9       | Other       |     -1676.866142   |  1.94308e-13   | True          |
| L02BA01         | *5/*9       | *1/*5       |        -70.343752 |  0.155492      | False         |


##  HORMONE ANTAGONISTS AND RELATED AGENTS
https://atcddd.fhi.no/atc_ddd_index/?code=L02BA&showdescription=yes
L02BA Anti-estrogens
L02BB Anti-androgens
L02BG Aromatase inhibitors
![image](https://github.com/user-attachments/assets/e07b7d43-5024-4a09-8c76-7676d80de91c)



 ## *9/*rs17376848 is significantly distant from *9/*rs17376848+rs1801265 in terms of survival rates after taking the drug L02BA01
 Is rs1801265 a key factor?
 
![image](https://github.com/user-attachments/assets/9ede6505-0b35-40e5-a914-b763cfff4480)

![image](https://github.com/user-attachments/assets/2f7ffefe-3aeb-414c-8f8d-cdea7787a91b)
https://www.ensembl.org/Homo_sapiens/Tools/VEP/Results?tl=fbEIoq70y6CfJDRI-10419986


![image](https://github.com/user-attachments/assets/bd756f79-a6d0-4e42-a5a6-03fa6ce73461)


