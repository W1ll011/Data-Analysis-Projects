# Data Analysis Projects
A repository containing a few of my data analysis projects

[Yield Analysis using Synthetic Data](#project-1-yield-analysis-using-synthetic-data)

[Ship Fuel and CO2 Emission](#project-2-ship-fuel-and-co2-emission) 

[Analysis of US Regional Sales](#project-3-analysis-of-us-regional-sales)

## Project 1: Yield Analysis using Synthetic Data

The aim of this project was to perform yield analysis on synthetic data of a 300 mm wafer over 31 days of manufacturing. 

### Methodology

1. Prompted Chat GPT to write a Python script that would simulate a typical dataset from a semiconductor fab that yield engineers would use for visualization and analysis.
2. Made modfications to script 
3. Imported dataset into Power BI
4. Used Python to generate a wafer map to visualize failed die
5. Created a few Power BI measures of common yield metrics

### Some Visuals

<img width="730" height="395" alt="image" src="https://github.com/user-attachments/assets/09bfe6cd-7f3a-465f-a3ff-fc15df28a3bd" />

<img width="631" height="394" alt="image" src="https://github.com/user-attachments/assets/01dfc2fc-aa01-4e07-aedf-87c7e35ce281" />

<img width="761" height="516" alt="image" src="https://github.com/user-attachments/assets/3efd702c-e491-4d1a-91b9-1e9d04796d73" />

<img width="749" height="507" alt="image" src="https://github.com/user-attachments/assets/57a332e4-ffae-47fe-a90d-df1d318a1522" />

<img width="748" height="512" alt="image" src="https://github.com/user-attachments/assets/d40be9a2-a07c-4498-9664-97fba987fcd8" />

### Findings

The wafer had a die yield of **47.83%**, with **521,664 DPM**. By far, the most common cause of die failure was parametric: drain-source current (Idss) being larger than the set boundary of 1 nA. 



---
## Project 2: Ship Fuel and CO2 Emission
The primary aims for this project were to use the Kaggle dataset to check for correlation between fuel consumption and CO2 emission, check which ships produce most CO2, and any other of the given parameters that may affect CO2 emission. 

### Methodology
1. Imported a Kaggle dataset into Power BI
2. Used Power Query to change data types into their correct forms
3. Used a combination of Power BI charts and Python scripts to visualize data
4. Python was also used to perform t-tests

### Some Visuals
<img width="930" height="517" alt="image" src="https://github.com/user-attachments/assets/1f85107d-8000-4827-ab70-7e1ac52bc2da" />

<img width="933" height="515" alt="image" src="https://github.com/user-attachments/assets/4ca76169-a900-4917-9078-f24e4998c22c" />

<img width="796" height="467" alt="image" src="https://github.com/user-attachments/assets/a80de194-b2d5-4c21-99f2-2aea8da8913d" />

<img width="705" height="467" alt="image" src="https://github.com/user-attachments/assets/2f632691-f8e3-477b-a60b-09ead4e89e85" />

<img width="901" height="99" alt="image" src="https://github.com/user-attachments/assets/8ed167e1-e8ea-4615-83e2-372e16815bf1" /> 





### Findings

Using Python, a t-test was performed and found that there was not a signifcant difference **(p=0.357 > alpha=0.05)** between CO2 emission between the two fuel types: diesel and HFO. For this t-test the data was filtered for a specific ship type, and for distances greater than 100 miles. Correlation analysis revealed a strong positve correlation **(R=1)** between fuel and CO2 emission. Moreover, it was calculated that tanker ships accounted for 59.03% of CO2 emission, on average, amongst the four ship types. This made sense as they are the largest of the various ships. A line graph revealed that voyages during the first half of the year, on average, produced more CO2 than trips during the second half. Further analysis, though, showed that majority of the miles occured during the first half of the year, so that's most likely the reason. ANOVA also revealed **(p= 0.654 > alpha =0.05; eta^2 = 0.002)** that there was no significant difference for CO2 emission amongst the various weather conditions for tanker ships. 

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






