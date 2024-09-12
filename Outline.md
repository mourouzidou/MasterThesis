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
