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



ParticipantID 	| ICD10|	Diagnosis_Date |
| -----| ----| ----|
4192264 	| C34.2 | 	2004-08-17 	
4949601 	| D38.1 10mg/10ml solution for infusion vials | 	2020-11-12 


# TODO

* Dataframe:
  example `selected_keyword` : "lung"
  filters  all participants with icd10 related to lung cancer
  filters all cancer drugs related to lung cancer
  

| ParticipantID | Gene1  | Gene2 | GeneX | ICD10_A | ICD10_B | ICD10_C | ATC_A | ATC_B |
|---------------|------- |-------|-------|---------|---------|---------|-------|-------|
| 4192264       |*1/*1 | *38/*38 | *1/*1| Δdate(Death-Diagnosis)=1000| Δdate(Death-Diagnosis)=100 | NA| Δdate(Death-LastPrescription)= 800|Δdate(Death-LastPrescription)= 100 | 


 OR
 selectone specific icd10 code vs related drugs 

 * Search for correlation among ATC codes / Genotypes - Significant survival pattern given a certain cancer related icd10
 * Compare the significant pairs ATC/Genotypes between two groups:
     * Group A: Participants having the disease but did not take any cancer related drug
     * Group B: Participants having the disease and receivd at least one cancer related drug
  

 * Compare the significant survival period (either lower or higher) to the corresponding life expenctancy of this disease according to the literature
 * Retrieve info for the significant genotype(s) and their association to this drug (PharmGKB)

 SCENARIO 1
 * Make meaningful assumptions : e.g. how variation V affects the response to drug D according to its location/type ?
   
 SCENARIO 2
 * Build model on Drug/Genotype/Life Expectancy data and generate a risk index for each genotype associated to a drug
   

![image](https://github.com/user-attachments/assets/21d219a4-f801-4c79-a549-7a6b6054eb89)










