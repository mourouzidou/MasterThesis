Do certain genotypes respond better to specific treatments in terms of survival time?
Are there any genotypes that consistently show worse survival, even with drug treatments?

















####  Filter cancer drugs file https://sciencedata.anticancerfund.org/pages/cancerdrugsdb.txt to include:
  Only cancer related drugs that target to at least one of our available genes  

  Drugs (78 in total, 26 in our data)
  

    
  Genes (13)
    
      gene_list = ["CFTR","CYP2C19","CYP2C8","CYP2C9","CYP2D6","CYP3A4","CYP3A5","DPYD", "SLCO1B1","TPMT","UGT1A1","VKORC1","NAT2"]

  Participants (6979 of total participants have taken at least one of the cancer drugs related to any of the 13 genes)
  
  #### Lung cancer case
  keyword = "lung"
  
Aiming to further explore the potential association among the platin based chemotherapy treatments on Non Small Cell Lung Cancer (NSCLC) patients and certain    variants of DPYD gene [DOI:10.3390/ijms24129843] we decided to follow this inspiration and conduct some exploratory analysis on UKBioBank data. Thus, we took advantage of the exploratory tool we built for the scope of this project. The keyword "lung" was set as an input, expecting to retrieve some brief information related to our data (_Table 1.1_):

   #Participants diagnosed with any type of lung cancer or benign precancerous states = 4333
   #Participants diagnosed with any type of lung cancer or benign precancerous states AND took at least one lung cancer related drug = 
   
  | ICD10 | ATC | GENE |  






## Automation

  Input: "cancer type keyword" to search in the indications column

Cancer related icd10 codes start with either 
  * C : codes cover malignant neoplasms (cancers).
  * D : codes include benign neoplasms, in situ neoplasms, and some diseases that can evolve into cancer (e.g., precancerous conditions).


Build API endpoint where we can set a keyword as an input and will return icd10 codes that describe a specific type (keyword) of cancer:
  * Example : `input` : "lung", `output` : C46.50, C46.51, C78.01, C34.2, D86.0 ...




    
