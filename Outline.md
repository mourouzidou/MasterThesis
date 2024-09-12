# Genomic data from UKBioBank extraction - preparation

* Retrieve WGS data from UKBB in bfiles format
* Filter data
* Star allele calling for 13 pharmacogenes [CFTR,CYP2C19,CYP2C8,CYP2C9,CYP2D6,CYP3A4,CYP3A5,DPYD,SLCO1B1,TPMT,UGT1A1,VKORC1,NAT2]
* Generate a genotype table 

| ParticipantID | CFTR  | CYP2C19 | CYP2C8 | CYP2C9 | CYP2D6 | CYP3A4 | CYP3A5 | DPYD | SLCO1B1 | TPMT | UGT1A1 | VKORC1 | NAT2 | HLA-A | HLA-B
|---------------|-------|-|--------|--|---|---|---|---|---|---|---|---|--|-|--|
| 4192264       |*1/*1 | *38/*38 | *1/*1|*1/*1 | *39/*39| *1/*1| *1/*1| *1/*rs17376848| *1/*1|*1/*1|*1/*80|*H6+rs9934438/*T|*4/*5 | |
| 3736652       |*1/*1|*38/*38|*1/*1|*1/*1|*39/*39|*1/*1|*1/*1|*1/*6|*1/*1|*1/*1|*1/*1|*H6+rs9934438/*T|*5/*5||


![image](https://github.com/user-attachments/assets/c53ff2cd-0b7f-4584-b069-14a000695009)

# Potential Thesis Subject: 

### Publication based idea : https://pubmed.ncbi.nlm.nih.gov/37372990/

Explore DPYD (and CYP2C8) variants that affect chemotherapy response for lung cancer patients (disease icd10 codes: C340,C341,C342,C343,C348,C349)

##### Currently working on this:
* Distribution of survival time after the diagnosis with one of the 6 icd10 codes.
* e.g. *1/*1  (*1/*1+rs10509681) seems to have an overall higher survival expectancy after diagnosis for all of the diseases (and are not diagnosed with more than one lung cancer types)
* e.g. We observe patterns in the lower boundaries of y axis (less days of survival after diagnosis) where genotypes like *4/*4 , *2/*2 that exhibit coexistance of multiple lung cancer types and lower survival rate (survival days < 2000)
![image](https://github.com/user-attachments/assets/df341b3a-0fc6-470d-acf1-e7a027188565)


**TODO**: 
* Cluster genotypes both for CYP2C8 and DPYD and observe patterns in the survival rate
* Significant differences among genotypes:
  * Check if the mortality rate for certain genotypes is due to their inability to metabolize certain cancer treatments 
  * Check whether there are some treatments(/or dosages) that work better for certain genotypes - Risk ndex / Treatment recommendation based on genotype clusters
  
ParticipantID 	|Drug name |	Date prescription was issued 	|ATC_code |
| -----| ----| ----|---|
4192264 	| Amivantamab | 	2004-08-17 	| L01FX18 | 
4949601 	| Cisplatin 10mg/10ml solution for infusion vials | 	2020-11-12 | L01XA01 |

ParticipantID | C340 | C341 | C342 | C343 |C348 | C349 | CYP2C8 | DPYD|
|---|--|--|--|--|--|--|--|--|
|1003076 |,|,|,|,|,|,|2019-05-25 |*1/*4|*5+rs17376848/*9|
|1006134|,|,|2015-01-16|,|,|,|,|,|*1/*1|*5/*9|


**Useful Literature Info** 
* *2 + *3 + *4 is associated with increased overall survival when treated with carboplatin or cisplatin in people with Carcinoma, Non-Small-Cell Lung as compared to CYP2C8 *1/*1.
* match cancer related drugs with atc codes and filter drug dataset https://www.anticancerfund.org/database-cancer-drugs
* Check other Malignant neoplasms of respiratory and intrathoracic organs C30-C39 https://www.icd10data.com/ICD10CM/Codes/C00-D49/C30-C39

  
