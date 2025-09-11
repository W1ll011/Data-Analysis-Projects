# Data Analysis Projects
A repository containing a few of my data analysis projects

## Project 1: Yield Analysis using Synethetic Data

The aim of this project was to perform yield analysis on synthetic data of a 300 mm wafer over 31 days of manufacturing. 

### Methodology

1. Prompted Microsoft Co-pilot to write a Python script that will simulate a typical dataset from a semiconductor fab that yield engineers would use to for visualization and analysis.
2. Made modfications to script 
3. Imported dataset into Power BI
4. Used Python to generate a wafer map to visualize failed die
5. Created a few Power BI measures of common yield metrics

### Example Visuals

<img width="752" height="408" alt="image" src="https://github.com/user-attachments/assets/d1c81f4a-b5a6-4dbe-8c8e-697c5056a0de" />

<img width="710" height="260" alt="image" src="https://github.com/user-attachments/assets/86c864d8-b45c-4cea-be08-5e28a16ba87e" />



---
## Project 2: Ship Fuel Efficiency
The primary aims for this project were to use the Kaggle dataset to check for correlation between fuel consumption and CO2 emission, check which ships produce most CO2, and any other parameters that may affect CO2 emission. 

### Methodology
1. Imported a Kaggle dataset into Power BI
2. Used Power Query to change data types into their correct forms
3. Used a combination of Power BI charts and Python scripts to visualize data
4. Python was also used to perform t-tests

### Example Visuals
<img width="614" height="477" alt="image" src="https://github.com/user-attachments/assets/0ab73b4c-b934-4a16-abbd-3b5d80987cc7" />

<img width="414" height="204" alt="image" src="https://github.com/user-attachments/assets/e91450cb-3d27-4601-8049-87b6e5dcc852" />

<img width="391" height="202" alt="image" src="https://github.com/user-attachments/assets/f48015bd-ffe3-432e-be2d-8aee8db09f13" />

<img width="372" height="223" alt="image" src="https://github.com/user-attachments/assets/e5ddd32c-6de3-4c36-80ac-a7b5b39e8ddc" />

### Findings

After analysis, it was determined that there is a strong positve correlation (R=1) between fuel and CO2 emission. Additionaly, tanker ships emitted the most CO2 on average amongst all ship types. Moreover, using a t-test, it was found that there wasn't a signifcant difference (p=0.357 > 0.05) between CO2 emission between the two fuel types: diesel and HFO. This meant that there was no need to separate the data by fuel type. 

---
## Project 3: Analysis of US Regional Sales (2018-2020) 

The aims of this project were to 1) build a dashboard that displays important sales metrics, 2) determine most profitable stores/sales channels/sales teams etc., and 3) determine any factors affecting sales  

### Methodology

1. Imported a Kaggle dataset into Power BI
2. Transformed data in Power Query
3. Created necessary Power BI measures (total revenue/loss/profit etc.)
4. Created a combination of Python and Power BI charts for visualization and analysis
5. Performed ANOVA and correlation analysis using Python 

### Example visuals

<img width="867" height="478" alt="image" src="https://github.com/user-attachments/assets/29947939-d774-4e3a-8d5b-c057e0158f56" />

### Findings

The data revealed that the most profitable sales channel, and store were: in-store, store #284 respectively. Sales team #18 sold the most products, while customer #12 bought the most. There were large differences in cost/selling price for the same Product ID, seeming to mean that the Product ID was not standard across the dataset, so any form product analysis was avoided.   






