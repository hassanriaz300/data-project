# Data Project 

<a target="_blank" href="https://datalumina.com/">
    <img src="https://img.shields.io/badge/Datalumina-Project%20Template-2856f7" alt="Datalumina Project" />
</a>

## Cookiecutter Data Science
This project template is a simplified version of the [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org) template, created to suit the needs of Datalumina and made available as a GitHub template.

## Adjusting .gitignore

Data files are kept private by gitignore
```plaintext
# exclude data from source control by default
# /data/
```

Typically, you want to exclude this folder if it contains either sensitive data that you do not want to add to version control or large files.

## Duplicating the .yaml virtual environment File
conda env create -f environment.yml to create a virtual environment for project to run




## Project Organization
Data folder content are kept hidden intentionally 
Model work can be done in future
Notebooks are for testing the working of data analysis code

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
|──webapp               <- Webapplication folder built in react detail to run the app can be found in readme in the folder

├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│   ├──edeka.ipynb      Trial and test for data analysis BART model is used here as it consume almost 14 hours so not include in webapp
│                       
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src                         <- Source code for this project
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    │    
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models not worked in this project          
    │   └── train.py            <- Code to train models not in the scope of this project existing models like BART is used in app code services
    │
    ├── plots.py                <- Code to create visualizations 
    │
    └── services                <- Service classes to connect with external platforms, tools, or APIs
        └── __init__.py
        └── api.py          FastAPI Endpoints for webapplication Ignore all othe files
        └── visualizationservice.py         Businesslogic for visualizations on web app
        └── cleaning.py                     Data Cleaning logic
        └── semanticservice.py              all-mpnet-base-v2 model business logic for semantic matching of review data against accusations 
--------
