# project-5: Human Development Index Across Countries

------

### HDI Background:
 The Human Development Index (HDI) is a composite index of life expectancy, education (mean years of schooling completed and expected years of schooling upon entering the education system), and per capita income indicators, which is used to rank countries into four tiers of human development.HDI classifications are based on HDI fixed cutoff points, which are derived from the quartiles of distributions of the component indicators. The cutoff-points are:
- low human development (less than 0.550)
- medium human development (0.550–0.699)
- high human development (0.700–0.799)
- very high human development (0.800 or greater) 

The HDI was created to emphasize that people and their capabilities should be the ultimate criteria for assessing the development of a country, not economic growth alone (GDP). [source](https://hdr.undp.org/reports-and-publications/2020-human-development-report/data-readers-guide):

### Problem Statement:

**The goal of this project is to use modeling and explanatory data analysis to see what “soft” features can be used to predict a country's HDI-Index, and spotlight features that countries with high human development scores have in common (outside of GNI, education, and life expectancy).**

The goal is to not only predict a country's HDI index, but to also extract insights on how certain features impact a country's human development score, and whether countries can be clustered based on certain features that significantly impact their development. 

---

### Datasets

World Development Indicators (WDI) is the primary World Bank collection of development indicators, compiled from officially recognized international sources. It presents the most current and accurate global development data available, and includes national, regional and global estimates. The most recent update was July 2022.

The World Bank Indicators data can be accessed [here](https://data.worldbank.org/indicator).


### Data Dictionary

Data was pulled across the following categories (specifically excluding data tied to life expectancy, education levels, and gross national income):
- Agriculture and Rural Development
- Aid Effectiveness
- Climate Change 
- Energy and Mining
- Environment
- Finance
- Gender
- Health
- Infrastructure

Data for each of the features selected is from 2017. We selected data specifically from 2017 as this was the most recent and robust data across all features. 

|Feature| Description|
| :---:  | :--------------: |
| rural_pop_percent |  The percent of the population living rurally. |
| food_production_index |  The agricultural production index is prepared by the Food and Agriculture Organization of the United Nations (FAO). The FAO indices of agricultural production show the relative level of the aggregate volume of agricultural production for each year in comparison with the base period 2014-2016.  |
| ag_land_area |  The percent of total land area that is used for agricultural purposes (Agricultural land is defined as the land area that is either arable, under permanent crops, or under permanent pastures). |
| arable_land_percent |  The percent of total land area that is considered arable. Arable land includes land under temporary crops such as cereals, temporary meadows for mowing or for pasture, land under market or kitchen gardens, and land temporarily fallow. |
| net_migration |  The net migration rate is the difference between the number of immigrants and the number of emigrants throughout the year. |
| mat_mortality_ratio | Maternal mortality ratio(modeled estimate, per 100,000 live births). |
| under5_mortality_ratio   | Mortality rate for children under 5 (per 1,000 live births).  |
| tubercul_incidence |  Incidence of tuberculosis (per 100,000 people).   |
|  elec_access    |   The percent of the population with access to electricity.  |
|  ren_energy_percent   |  Renewable energy consumption as a percent of total energy consumption.  |
| co2_emissions | CO2 emissions in metric tons per capita. |
| pop_air_pollution  | The percent of the population exposed to air pollution exceeding WHO guideline values.  |
| foreign_dir_inv |  Net inflows of foreign direct investment in USD.  |
|  atm_access  |   Automated teller machines (ATMs) per 100,000 adults.  |
|  adol_fertility_rate |  Adolescent fertility rate (births per 1,000 women ages 15-19).  |
| fem_labor_part_rate | Percent of the total labor force that is female. |
| male_labor_part_rate   | Percent of the total labor force that is male.  |
| fertility_rate |  Total fertility rate (births per women).  |
|  dpt_immuniz_rate    |  DTP (Diphtheria, Tetanus, Pertussis) immunization rate for children ages 12-23 months.  |
|  undernourished_rate   |  Prevalence of undernourishment  as a percent of the total population).  |
| cell_subscriptions_per100 | Mobile cellular subscriptions per 100 people. |
| internet_per_mil   | Secure Internet servers per 1 million people. |
| military_exp |  Military expenditures as a percent of total GDP.   |
|  women_seats_percent    |  Proportion of seats held by women in national parliaments.  |
|  male_bus_start   |  Time required to start a business (in days) for males.  |
| female_bus_start| Time required to start a business (in days) for females.|
| patent_apps   | Total patent applications filed by residents.  |
| sci_articles |  Scientific and technical journal articles published.   |
| pop_density |   Population density (people per sq. km of land area).  |
|  HDI (Human Development Index)   |  The Human Development Index (HDI) is a statistic composite index of life expectancy, education (mean years of schooling completed and expected years of schooling upon entering the education system), and per capita income indicators, which is used to rank countries into four tiers of human development.  |

---

**Analysis**

Regression

Classification

Clustering? look at countries grouped based on HDI Index:

Human development classification
HDI classifications are based on HDI fixed cutoff points, which are derived from the quartiles of dis- tributions of the component indicators. The cutoff-points are HDI of less than 0.550 for low human development, 0.550–0.699 for medium human development, 0.700–0.799 for high human development and 0.800 or greater for very high human development.




---

### Findings and Recommendations

**Conclusion:** 



**Recommendations:**
