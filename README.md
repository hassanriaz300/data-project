# Supermarket Review Analysis Web App

A full-stack review analysis project for cleaning, analyzing, classifying, and visualizing customer review data from supermarket reviews.

The project combines a **React frontend dashboard**, a **FastAPI backend**, and **Python-based review analysis modules** to explore customer complaints, semantic topics, business insights, and review patterns.

> Current status: The backend and frontend run locally. Core analysis modules are present, while some frontend pages and business logic integrations are still being improved.

---

## Project Overview

Customer reviews often contain valuable information about service quality, product issues, pricing concerns, staff behavior, hygiene, and customer satisfaction. This project aims to turn unstructured review text into useful insights by applying data cleaning, semantic analysis, prediction logic, and dashboard-based visualization.

The application is designed to support:

* Review data cleaning and preprocessing
* Complaint and topic analysis
* Semantic review understanding
* Multi-label review classification
* Visual exploration of review patterns
* Business insight generation through dashboards

---

## Tech Stack

### Frontend

* React
* React Router
* Material UI
* JavaScript
* CSS

### Backend

* FastAPI
* Uvicorn
* Python

### Data Analysis / ML

* Pandas
* scikit-learn
* Sentence Transformers
* Joblib
* YAML configuration
* WordCloud
* Matplotlib / visualization utilities

### Project Tools

* Git & GitHub
* VS Code
* Python virtual environment
* npm / Node.js

---

## Repository Structure

```text
data-project/
├── frontend/                 # React frontend application
│   ├── public/
│   ├── src/
│   │   ├── App.js
│   │   ├── HomePage.js
│   │   ├── CleaningPage.js
│   │   ├── PredictPage.js
│   │   ├── semanticpage.js
│   │   ├── VisualizationPage.js
│   │   └── DeepAnalyze.js
│   ├── package.json
│   └── package-lock.json
│
├── src/                      # FastAPI backend and analysis source code
│   ├── api.py                # Main FastAPI application
│   ├── services/
│   │   ├── analysis.py
│   │   ├── semanticservice.py
│   │   └── visualizationservice.py
│   └── modeling/
│
├── config/                   # YAML configuration files
│   ├── keywords.yml
│   └── labels.yml
│
├── notebooks/                # Experiments and modeling notebooks
│   └── models/
│       └── multilabel_model.joblib
│
├── reports/
│   └── figures/              # Output images and generated figures
│
├── data/                     # Dataset/static data files
├── backend/app/              # Reserved for future backend refactor
├── requirements.txt
└── README.md
```

---

## Features

### Implemented / Partially Implemented

* React dashboard interface
* Navigation between dashboard pages
* FastAPI backend startup
* Review cleaning workflow
* Semantic analysis service
* Multi-label model loading
* Visualization service
* YAML-based labels and keyword configuration
* Static data serving through FastAPI
* Local frontend and backend development setup

### Pages in the Frontend

* Home
* Clean
* Analysis
* Predict
* Semantic
* Visualize
* Deep Analysis

Some pages are currently placeholders or partially connected to backend logic. Further work is planned to complete the frontend-to-backend integration.

---

## Backend Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI backend:

```bash
uvicorn src.api:app
```

Backend API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Frontend Setup

Go to the frontend folder:

```bash
cd frontend
```

Install frontend dependencies:

```bash
npm install
```

Start the React development server:

```bash
npm start
```

Frontend app:

```text
http://localhost:3000
```

---

## Running the Full Application Locally

Use two terminals.

### Terminal 1: Backend

```bash
cd data-project
source .venv/bin/activate
uvicorn src.api:app
```

### Terminal 2: Frontend

```bash
cd data-project/frontend
npm start
```

Then open:

```text
Frontend: http://localhost:3000
Backend Docs: http://127.0.0.1:8000/docs
```

---

## Recent Fixes and Cleanup

The project structure was cleaned and updated to make the app easier to run and maintain.

Completed updates include:

* Moved YAML configuration files into the `config/` folder
* Moved generated output figures into `reports/figures/`
* Fixed a syntax issue in `semanticservice.py`
* Updated backend config file paths from `.yaml` to `.yml`
* Installed missing backend dependency for WordCloud
* Fixed FastAPI model path to load the existing trained model
* Temporarily disabled production React build mounting during backend development
* Fixed frontend routing issue for the Analysis page
* Verified that the FastAPI backend starts locally
* Verified that the React frontend starts locally

---

## Current Status

The project currently runs locally with:

* FastAPI backend working
* React frontend working
* Homepage and navigation available
* Model file loading from `notebooks/models/multilabel_model.joblib`
* Config files loading from `config/labels.yml` and `config/keywords.yml`

Some business logic and page integrations are still under development. The project should be considered an active full-stack data analysis project rather than a fully polished production application.

---

## Planned Improvements

Next development steps:

* Complete backend integration for all frontend pages
* Improve error handling in API endpoints
* Refactor backend code into a cleaner `backend/app/` structure
* Add clearer API documentation
* Improve dashboard design and user experience
* Add final screenshots to the README
* Add example input/output for prediction and semantic analysis
* Add business insights section based on review analysis results
* Add deployment instructions
* Add tests for backend services

---

## Example Use Cases

This project can be used to analyze customer review data for questions such as:

* What are the most common supermarket customer complaints?
* Which topics appear most often in negative reviews?
* Are complaints related to staff, pricing, hygiene, product quality, or availability?
* What patterns can be discovered through semantic similarity?
* How can customer feedback be converted into business insights?

---

## Project Purpose

This project was built as a practical portfolio project to demonstrate skills in:

* Full-stack application development
* Data cleaning and preprocessing
* Backend API development
* Review text analysis
* Semantic analysis
* Machine learning model integration
* Dashboard development
* GitHub project organization

---

## Author

**Hassan Riaz**

Master's student in Computer & Systems Engineering in Germany, focused on Data Engineering, Analytics, and applied AI.

GitHub: [hassanriaz300](https://github.com/hassanriaz300)
