# Genomic Data : 
  * filter hwe, phase, star allele calling:
 final:
    
| ParticipantID | CFTR  | CYP2C19 | CYP2C8 | CYP2C9 | CYP2D6 | CYP3A4 | CYP3A5 | DPYD | SLCO1B1 | TPMT | UGT1A1 | VKORC1 | NAT2 | HLA-A | HLA-B
|---------------|-------|-|--------|--|---|---|---|---|---|---|---|---|--|-|--|
| 4192264       |*1/*1 | *38/*38 | *1/*1|*1/*1 | *39/*39| *1/*1| *1/*1| *1/*rs17376848| *1/*1|*1/*1|*1/*80|*H6+rs9934438/*T|*4/*5 | |
| 3736652       |*1/*1|*38/*38|*1/*1|*1/*1|*39/*39|*1/*1|*1/*1|*1/*6|*1/*1|*1/*1|*1/*1|*H6+rs9934438/*T|*5/*5||




# Demographic
Smoking_status, BMI, Sex, Birth_Year, **Death_Date, Death_Cause**



# Prescriptions

ParticipantID 	|Drug name |	Date prescription was issued 	|ATC_code |
| -----| ----| ----|---|
4192264 	| Amivantamab | 	2004-08-17 	| L01FX18 | 
4949601 	| Cisplatin 10mg/10ml solution for infusion vials | 	2020-11-12 | L01XA01 |

Filter cancer drugs file https://sciencedata.anticancerfund.org/pages/cancerdrugsdb.txt to include:
  Only cancer related drugs that target to at least one of our available genes  

  Drugs (78 in total, 26 in our data)
  
    ['Acalabrutinib', 'Alectinib', 'Amsacrine', 'Belinostat', 'Binimetinib', 'Bosutinib', 'Busulfan', 'Cabazitaxel', 'Capecitabine', 'Carboplatin', 'Carmustine', 'Ceritinib', 'Cisplatin', 'Cytarabine', 'Dacarbazine', 'Dactinomycin', 'Daunorubicin', 'Dexamethasone', 'Docetaxel', 'Doxorubicin', 'Doxorubicin Liposome', 'Epirubicin', 'Erdafitinib', 'Eribulin', 'Erlotinib', 'Etoposide', 'Fludarabine', 'Fluorouracil', 'Flutamide', 'Gefitinib', 'Gemcitabine', 'Gemtuzumab Ozogamicin', 'Goserelin', 'Hydroxyurea', 'Ibrutinib', 'Idarubicin', 'Imatinib', 'Irinotecan', 'Irinotecan Liposome', 'Ixabepilone', 'Lapatinib', 'Lenvatinib', 'Letrozole', 'Leuprolide', 'Medroxyprogesterone', 'Melphalan', 'Melphalan Flufenamide', 'Methotrexate', 'Mitomycin', 'Mitoxantrone', 'Nab-Paclitaxel', 'Nilotinib', 'Nilutamide', 'Oxaliplatin', 'Paclitaxel', 'Pazopanib', 'Pegaspargase', 'Prednisolone', 'Prednisone', 'Quizartinib', 'Ripretinib', 'Rucaparib', 'Ruxolitinib', 'Sacituzumab Govitecan', 'Sorafenib', 'Streptozocin', 'Sunitinib', 'Tamoxifen', 'Thalidomide', 'Thioguanine', 'Toremifene', 'Tretinoin', 'Vandetanib', 'Vemurafenib', 'Vinblastine', 'Vincristine', 'Vindesine', 'Vorinostat']
    

 

# Clinical - Diagnosis

 Input: "cancer type keyword" to search in the indications column

Cancer related icd10 codes start with either 
  * C : codes cover malignant neoplasms (cancers).
  * D : codes include benign neoplasms, in situ neoplasms, and some diseases that can evolve into cancer (e.g., precancerous conditions).


Build API endpoint where we can set a keyword as an input and will return icd10 codes that describe a specific type (keyword) of cancer:
  * Example : `input` : "lung", `output` : C46.50, C46.51, C78.01, C34.2, D86.0 ...

