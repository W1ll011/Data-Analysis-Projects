# Data Analysis Projects
A repository containing a few of my data analysis projects

## Project List:

[Yield Analysis using Synthetic Wafer Sort Data](#project-1-yield-analysis-using-synthetic-wafer-sort-data)

[Ship Fuel and CO2 Emission](#project-2-ship-fuel-and-co2-emission) 

[Analysis of US Regional Sales](#project-3-analysis-of-us-regional-sales)

## Project 1: Yield Analysis using Synthetic Wafer Sort Data

The dataset for this project simulates wafer-sort data for the fabrication of power MOSFETs on a 300 mm wafer. Present in the dataset are key electrical measures & test results, environmental variables, die location co-ordinates, and more.  The aim of this project was to perform yield visualization and analysis to determine potential causes for die failures.

### Methodology

1. Prompted Chat GPT to write a Python script that would simulate a typical dataset from a semiconductor fab at the wafer-sort level for power MOSFETs 
2. Made continuous modifications to script to improve realism/accuracy
3. Imported dataset into Power BI
4. Created useful Power BI measures from data
5. Built a dashboard to display wafer map and key overall yield metrics
6. Analyzed data in Python and Power BI for potential source(s) of die failure

### Some Visuals

<img width="830" height="461" alt="image" src="https://github.com/user-attachments/assets/3360839c-76b3-49ee-9471-beeee6b60569" />

<img width="800" height="479" alt="image" src="https://github.com/user-attachments/assets/7c0ccc88-b402-4d70-9dca-703b63d741f1" />


### Findings and Analysis

The wafer had an overall die yield of 93.23% (482/517), with 67,698.26 DPM. Failures were classified as either primarily functional (physical) or parametric. 4.06% of failures were parametric, while only 2.71% were functional. Within the parametric category, the most common failure type was high drain-source leakage current (Idss). 

Given that the device being fabricated is a power MOSFET, the code was configured to flag any occurrence of Idss above 1e-6 A – a suitable limit for this type of device. For twenty-one (21) dies, an Idss failure was flagged as the first cause of failure. There were an additional 5 Idss failures that were accompanied by physical defects, and 2 more accompanied by both RON (on-resistance >200 mOhms) and phys (crack). 

The dies with solely Idss failures were all flagged to have likely occurred during the implantation step. Seeing that two different tools, I1 (8 dies) and I2 (13 dies), were used for implantation, a Welch’s t-test was performed using Python to help determine if one tool was more at fault than the other(?). A p-value of 0.807 (alpha 0.05) was calculated, indicating that there was not a statistically significant difference between the two tools.


There were seven (7) dies with both an Idss failure and a physical defect. Of those seven dies, 2 were flagged to have cracks, 2 had oxide defects, 3 were contaminated, and 1 had a scratch. Further investigation is needed to say with certainty, but such physical defects do point to why there would be an increase in leakage current. All these physical defects increase off-state current by either creating new generation sites or lowering the breakdown voltage. Therefore, the code was designed for physical defects to have a multiplicative effect on leakage current. 

Environmental and test hardware variables (ambient conditions, chemical lot, probe card) were constant for this single-wafer run; therefore, are not likely root causes for within-wafer variablity. 

---
## Project 2: Ship Fuel and CO2 Emission
The dataset for this project contains ship fuel consumption and CO2 emission information - among other parameters - for 4 ship types traveling across Nigerian waterways over a 1 year period. The primary aims for this project were to to check for correlation between fuel consumption and CO2 emission, check which ships emitt the most CO2, and whether any of the other given parameters affect the amount of CO2 emission. 

### Methodology
1. Imported a Kaggle dataset into Power BI
2. Used Power Query to change data types into their correct forms
3. Used a combination of Power BI charts and Python scripts to visualize data
4. Performed statistical tests like correlation, t-tests, and ANOVA

### Some Visuals
<img width="930" height="517" alt="image" src="https://github.com/user-attachments/assets/1f85107d-8000-4827-ab70-7e1ac52bc2da" /> <br>


<img width="894" height="504" alt="image" src="https://github.com/user-attachments/assets/f29f961d-c9c0-4924-b006-227d0e807141" /> <br>


<img width="880" height="478" alt="image" src="https://github.com/user-attachments/assets/33441c98-fc38-4927-87f1-2be799d10fa0" /> <br>


<img width="705" height="467" alt="image" src="https://github.com/user-attachments/assets/2f632691-f8e3-477b-a60b-09ead4e89e85" /> <br>


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

<img width="870" height="481" alt="image" src="https://github.com/user-attachments/assets/36343a0c-53aa-4c92-8548-8f0389b47a44" /> <br>


<img width="876" height="490" alt="image" src="https://github.com/user-attachments/assets/f477c32a-4b6a-46df-be73-24c6a5778e75" />

### Findings

The data revealed that the most profitable sales channel and store were **in-store** and **Store #284**, respectively. **Sales Team #18** sold the most products, while **Customer #12** bought the most. There were large differences in cost/selling price for the same Product ID, seeming to mean that the Product ID was not standard across the dataset, so any form of product analysis was avoided.






