# GovtExpenditure-to-HealthIndicator-Labeller

Regional and Local governments have a limited budget to spread across multiple budget items. Assuming their goal is to improve the 'health equity' of their population, they must decide how to spread their money across different government budget items. This should usually be based on the affect of each budget item on a range of socioeconomic and health indicators.

Stochastic graph based methods can be used to estabilish which budget items affect which indicators. This would be reflected as edges between budget items and indicators. However due to limited data it is important to ensure that the initial topology of the graph is relatively 'correct', as this can prevent superflous relationships being established.

In this project we provide a NLP based approach to establishing the starting set of edges that exist in the graph; in simpler terms we estimate which government budget items and indicators are likely to be related.

## Setup
- Git Clone this repository
- Follow instructions in the Data section below to download the exact dataset used 

## How To Use
- Examples of how to key functions can be found in the example_scripts folder

## Data 
### Data Download
- Downaload the file at the following link: https://drive.google.com/file/d/14sIiCiT8ZPtvEI1DG8rGiMfZznxlYrt_/view?usp=sharing
- It is a tar.gz file so extract it into the repository and merge it with the /data folder in the repository
