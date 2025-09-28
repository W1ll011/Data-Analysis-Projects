# Data Analysis Projects
A repository containing a few of my data analysis projects

[Yield Analysis using Synthetic Data](#project-1-yield-analysis-using-synthetic-data)

[Ship Fuel and CO2 Emission](#project-2-ship-fuel-and-co2-emission) 

[Analysis of US Regional Sales](#project-3-analysis-of-us-regional-sales)

## Project 1: Yield Analysis using Synthetic Data

The dataset for this project contains synthetic data for a 300 mm wafer (containing 576 useable die) over 31 days of manufacturing in a fab. Present in the dataset are key electrical measures & test results, environmental variables, die location co-ordinates, and more.  The aim of this project was to perform yield visualization and analysis to determine potential causes for die failures.

### Methodology

1. Prompted Chat GPT to write a Python script that would simulate a typical dataset from a semiconductor fab 
2. Made modifications script 
3. Imported dataset into Power BI
4. Created useful Power BI measures from data
5. Built a dashboard to display key overall yield metrics
6. Analyzed data for potential source(s) of die failure

### Some Visuals

<img width="730" height="395" alt="image" src="https://github.com/user-attachments/assets/09bfe6cd-7f3a-465f-a3ff-fc15df28a3bd" />

<img width="723" height="378" alt="image" src="https://github.com/user-attachments/assets/88e0ac70-a0c6-4ea9-8d37-62193c7fd9e8" />

<img width="584" height="370" alt="image" src="https://github.com/user-attachments/assets/3e9d2466-d9bb-4b05-bab3-d21304909872" />

<img width="631" height="394" alt="image" src="https://github.com/user-attachments/assets/01dfc2fc-aa01-4e07-aedf-87c7e35ce281" />

<img width="761" height="516" alt="image" src="https://github.com/user-attachments/assets/3efd702c-e491-4d1a-91b9-1e9d04796d73" />

<img width="758" height="521" alt="image" src="https://github.com/user-attachments/assets/928017df-30eb-40f7-8c4e-aa0e4743405a" />

<img width="749" height="507" alt="image" src="https://github.com/user-attachments/assets/57a332e4-ffae-47fe-a90d-df1d318a1522" />

<img width="748" height="512" alt="image" src="https://github.com/user-attachments/assets/d40be9a2-a07c-4498-9664-97fba987fcd8" />

### Findings and Analysis

The wafer had an overall die yield of **90.64%**, with **93,587.52 DPM**. By far, the most common cause of die failure was parametric: drain-source current (Idss) being larger than the chosen boundary of 1 nA. 

The wafer was divided into 12 30 degree sectors to aid in analyzing defect location. Sector 8 contained the fewest failures (17), whilst Sector 4 had the most with 33 - all parametric. The 13 physical defects were spread across 7 sectors, with Sector 6 having the most (5). There was a crack in this sector, which can be what led to the oxide defect, contamination and short circuits. All sectors contained a minimum of 15 parametric failures.  



---
## Project 2: Ship Fuel and CO2 Emission
The dataset for this project contains ship fuel consumption and CO2 emission information - among other parameters - for 4 ship types traveling across Nigerian waterways over a 1 year period. The primary aims for this project were to to check for correlation between fuel consumption and CO2 emission, check which ships emitt the most CO2, and whether any of the other given parameters affect the amount of CO2 emission. 

### Methodology
1. Imported a Kaggle dataset into Power BI
2. Used Power Query to change data types into their correct forms
3. Used a combination of Power BI charts and Python scripts to visualize data
4. Performed statistical tests like correlation, t-tests, and ANOVA

### Some Visuals
<img width="930" height="517" alt="image" src="https://github.com/user-attachments/assets/1f85107d-8000-4827-ab70-7e1ac52bc2da" />

<img width="894" height="504" alt="image" src="https://github.com/user-attachments/assets/f29f961d-c9c0-4924-b006-227d0e807141" />

<img width="880" height="478" alt="image" src="https://github.com/user-attachments/assets/33441c98-fc38-4927-87f1-2be799d10fa0" />

<img width="705" height="467" alt="image" src="https://github.com/user-attachments/assets/2f632691-f8e3-477b-a60b-09ead4e89e85" />

<img width="901" height="99" alt="image" src="https://github.com/user-attachments/assets/8ed167e1-e8ea-4615-83e2-372e16815bf1" /> 


### Findings and Analysis 

To determine wheter it was necessary to split and analyze the data based on the two fuel types (HFO and diesel), a t-test was performed using **alpha = 0.05**. A **p-value of 0.357** was calculated, and thus it was concluded that there not a significant difference for CO2 emission based on fuel type. For this t-test the data was filtered for a specific ship type, and for distances greater than 100 miles.

Correlation analysis indeed revealed a strong positve correlation **(R=1)** between fuel and CO2 emission. Tanker ships accounted for 61.16% of total CO2 emission amongst the four ship types. This made sense given they are the largest of the ships. A line graph revealed that voyages during the first half of the year, produced more CO2 than trips during the second half. One likely contributing factor for that, though, is that 8,640 more miles were traveled during the first half of the year. ANOVA also revealed **(p= 0.654 > alpha= 0.05; eta^2= 0.002)** that there was no significant difference for CO2 emission amongst the various weather conditions for tanker ships. 

---
## Project 3: Analysis of US Regional Sales 

The aims of this project were to build a dashboard that displays important sales metrics, and determine most profitable stores/sales channels etc.

### Methodology

1. Imported a Kaggle dataset into Power BI
2. Transformed data in Power Query
3. Created necessary Power BI measures (total revenue/loss/profit etc.)
4. Created a combination of Python and Power BI charts for visualization and analysis
5. Performed ANOVA using Python 

### Some Visuals

<img width="867" height="478" alt="image" src="https://github.com/user-attachments/assets/29947939-d774-4e3a-8d5b-c057e0158f56" />

<img width="870" height="481" alt="image" src="https://github.com/user-attachments/assets/36343a0c-53aa-4c92-8548-8f0389b47a44" />

<img width="876" height="490" alt="image" src="https://github.com/user-attachments/assets/f477c32a-4b6a-46df-be73-24c6a5778e75" />

### Findings

The data revealed that the most profitable sales channel and store were **in-store** and **Store #284**, respectively. **Sales Team #18** sold the most products, while **Customer #12** bought the most. There were large differences in cost/selling price for the same Product ID, seeming to mean that the Product ID was not standard across the dataset, so any form of product analysis was avoided.






