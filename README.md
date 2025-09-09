# Data Analysis Projects
A repository containing a few of my data analysis projects

## Project 1: Ship Fuel Efficiency
The primary aims for this project were to use the Kaggle dataset to check for correlation between fuel consumption and CO2 emission, check which ships produce most CO2, and any other factors that may affect CO2 emission. 

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






