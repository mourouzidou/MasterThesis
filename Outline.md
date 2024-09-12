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


![image](https://github.com/user-attachments/assets/eae26eb5-9778-49db-a7f8-270df2267d18)





    ADR
    Check survival timeline
    Dosages or alernative therapies for lung cancer
    Suugest alternative?
    we have no data for carboplatin - drug of interest - platin cbased chemotherapy cisplatin

https://www.pharmgkb.org/literature/15145754/variantAnnotation variant annotation
*CYP2C8

*2 + *3 + *4 is associated with increased overall survival when treated with carboplatin or cisplatin in people with Carcinoma, Non-Small-Cell Lung as compared to CYP2C8 *1/*1.
*DPYD

Reference is associated with increased progression-free survival and overall survival when treated with carboplatin or cisplatin in people with Carcinoma, Non-Small-Cell Lung.
Step 1 : Check which drugs are used for lung cancer (Non-Small (cisplatin) and others)

List of drugs: https://www.cancer.gov/about-cancer/treatment/drugs/lung

Drugs Approved for Non-Small Cell Lung Cancer
