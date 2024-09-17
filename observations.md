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







