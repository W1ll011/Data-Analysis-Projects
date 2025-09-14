# Data Analysis Projects
A repository containing a few of my data analysis projects

## Project 1: Yield Analysis using Synethetic Data

The aim of this project was to perform yield analysis on synthetic data of a 300 mm wafer over 31 days of manufacturing. 

### Methodology

1. Prompted Chat GPT to write a Python script that would simulate a typical dataset from a semiconductor fab that yield engineers would use for visualization and analysis.
2. Made modfications to script 
3. Imported dataset into Power BI
4. Used Python to generate a wafer map to visualize failed die
5. Created a few Power BI measures of common yield metrics

### Some Visuals

<img width="730" height="395" alt="image" src="https://github.com/user-attachments/assets/09bfe6cd-7f3a-465f-a3ff-fc15df28a3bd" />

<img width="602" height="387" alt="image" src="https://github.com/user-attachments/assets/31b19fcc-e076-405c-851c-1e4572ee12d4" />

<img width="761" height="516" alt="image" src="https://github.com/user-attachments/assets/3efd702c-e491-4d1a-91b9-1e9d04796d73" />



### Findings

The wafer had a low die yield of **47.83%**, with **521,664 DPM**. 



---
## Project 2: Ship Fuel Efficiency
The primary aims for this project were to use the Kaggle dataset to check for correlation between fuel consumption and CO2 emission, check which ships produce most CO2, and any other parameters that may affect CO2 emission. 

### Methodology
1. Imported a Kaggle dataset into Power BI
2. Used Power Query to change data types into their correct forms
3. Used a combination of Power BI charts and Python scripts to visualize data
4. Python was also used to perform t-tests

### Some Visuals
<img width="930" height="517" alt="image" src="https://github.com/user-attachments/assets/1f85107d-8000-4827-ab70-7e1ac52bc2da" />

<img width="747" height="439" alt="image" src="https://github.com/user-attachments/assets/31b329b3-851d-42fa-8b6e-85511dcfc4d7" />

<img width="796" height="467" alt="image" src="https://github.com/user-attachments/assets/a80de194-b2d5-4c21-99f2-2aea8da8913d" />

<img width="705" height="467" alt="image" src="https://github.com/user-attachments/assets/2f632691-f8e3-477b-a60b-09ead4e89e85" />


### Findings

After analysis, it was determined that there is a strong positve correlation **(R=1)** between fuel and CO2 emission. Additionaly, tanker ships emitted the most CO2 on average amongst all ship types. Moreover, using a t-test, it was found that there wasn't a signifcant difference **(p=0.357 > 0.05)** between CO2 emission between the two fuel types: diesel and HFO. This meant that there was no need to separate the data by fuel type. 

---
## Project 3: Analysis of US Regional Sales (2018-2020) 

The aims of this project were to 1) build a dashboard that displays important sales metrics, 2) determine most profitable stores/sales channels etc., and 3) determine any factors affecting sales  

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






