# Supermarket Review Analysis Web App

A full-stack web application for analyzing supermarket customer reviews and extracting business insights from customer feedback.

The project focuses on preparing review datasets, mapping complaint categories, and visualizing patterns such as frequent accusations, semantic topics, store benchmarks, and grouped breakdowns.

## Overview

This application allows users to upload review datasets through a React frontend and process them using a FastAPI backend.

The backend handles data preparation, semantic mapping, and visualization logic. The frontend provides pages for uploading files, running analysis workflows, and viewing generated insights.

## Features

- Upload and prepare supermarket review datasets
- Clean and split review data into enriched outputs
- Generate semantic accusation mappings
- Create top accusation heatmaps
- Analyze accusations by review rank
- Generate semantic topic heatmaps
- View top accusation summaries
- Analyze top category trends
- Compare store benchmarks
- Create grouped breakdowns by fields such as city or source
- Run deep analysis for business-level insights

## Tech Stack

**Backend**
- Python
- FastAPI
- Pandas
- NLP and semantic analysis services
- Visualization services

**Frontend**
- React
- JavaScript
- Material UI
- Fetch API

**Configuration and Data**
- YAML configuration files
- Excel review datasets
- Local data folders for raw, processed, semantic, and visualization outputs

## Project Structure

```text
data-project/
├── config/
│   ├── keywords.yml
│   └── labels.yml
├── data/
│   ├── raw/
│   └── interim/
├── frontend/
│   └── src/
├── src/
│   ├── api.py
│   ├── dataset.py
│   ├── features.py
│   └── services/
│       ├── cleaning.py
│       ├── paths.py
│       ├── semanticservice.py
│       └── visualizationservice.py
└── README.md
```

## Active API Endpoints

The current FastAPI backend exposes these active endpoints:

```text
POST /prepare
POST /semantic/map
POST /visualize
POST /visualize/by-rank
POST /visualize/semantic
POST /visualize/top-accusations
POST /visualize/top1-category-trends
POST /visualize/store-benchmarks
POST /visualize/grouped-breakdown
POST /visualize/deep-analysis
```

Unused prediction and legacy analysis endpoints were removed during cleanup to keep the backend focused on the active application workflow.

## How to Run

### Backend

From the project root:

```bash
uvicorn src.api:app
```

Backend runs at:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

### Frontend

In another terminal:

```bash
cd frontend
npm start
```

Frontend runs at:

```text
http://localhost:3000
```

## Current Workflow

```text
Upload review dataset
        ↓
Prepare and clean reviews
        ↓
Generate enriched review files
        ↓
Run semantic accusation mapping
        ↓
Generate visualizations and benchmarks
        ↓
Explore deeper business insights
```

## Screenshots

Screenshots can be added under:

```text
docs/screenshots/
```

Example:

```markdown
![Dashboard Screenshot](docs/screenshots/dashboard-home.png)
```

## Refactoring Progress

This project is currently being cleaned and refactored step by step.

Completed cleanup:

- Organized `src/api.py` imports and sections
- Added reusable upload-saving helper
- Centralized data paths through `src/services/paths.py`
- Removed unused backend prediction endpoint
- Removed unused frontend prediction page
- Removed unused backend analysis endpoint
- Removed unused frontend build path from `api.py`

Planned improvements:

- Split `src/api.py` into route modules
- Separate large visualization logic into smaller services
- Improve documentation and screenshots
- Later evaluate moving backend code into `backend/app`

## Project Status

Current active development branch:

```text
review-and-refactor
```

The `main` branch is kept stable while refactoring is done safely in smaller commits.
