# Perplexify - Embedding & Perplexity Visualization

A React application that generates text embeddings, calculates perplexity, and visualizes semantic similarity using PCA projection.

## Features

- **Text Embeddings**: Uses [mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) for semantic embeddings
- **Perplexity Calculation**: Uses [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) to measure text predictability
- **PCA Visualization**: Projects high-dimensional embeddings to 2D for visualization
- **Color-coded Points**: Points are colored by perplexity score

## Prerequisites

- Node.js (v16+)
- Python 3.10+

## Getting Started

### 1. Install Frontend Dependencies

```bash
npm install
```

### 2. Set Up Python Backend

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install Python dependencies:

```bash
pip install -r src/backend/requirements.txt
```

### 3. Run the Application

**Option A: Run both frontend and backend together:**

```bash
npm run dev
```

**Option B: Run separately in two terminals:**

Terminal 1 (Backend):
```bash
npm run start:backend
```

Terminal 2 (Frontend):
```bash
npm start
```

The app will open at [http://localhost:3000](http://localhost:3000).  
The backend API runs at [http://localhost:3001](http://localhost:3001).

> **Note:** On first run, two models will be downloaded automatically:
> - Embedding model (~670MB)
> - SmolLM-135M for perplexity (~270MB)

## Available Scripts

- `npm start` - Runs the frontend only
- `npm run start:backend` - Runs the Python backend only
- `npm run dev` - Runs both frontend and backend concurrently
- `npm run build` - Builds the app for production
- `npm test` - Runs the test suite
