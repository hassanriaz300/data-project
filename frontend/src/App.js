//  This is the main entry point for the React application, setting up routing and navigation.
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';

import { AppBar, Toolbar, Typography, Button, Container, Box } from '@mui/material';

import HomePage from './HomePage';
import CleaningPage from './CleaningPage';
import AnalysisPage from './AnalysisPage';
import PredictPage from './PredictPage';
import SemanticPage from './semanticpage';
import VisualizationPage from "./VisualizationPage";

import DeepAnalyze from './DeepAnalyze';

export default function App() {
  return (
    <Router>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Supermarket Insights
          </Typography>
          <Button color="inherit" component={Link} to="/">
            Home
          </Button>
          <Button color="inherit" component={Link} to="/clean">
            Clean
          </Button>
          <Button color="inherit" component={Link} to="/analysis">
            Analysis
          </Button>
          <Button color="inherit" component={Link} to="/predict">
            Predict
          </Button>
          <Button color="inherit" component={Link} to="/semantic">
            Semantic
          </Button>
          <Button color="inherit" component={Link} to="/visualize">
            Visualize
          </Button>


        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Box>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/clean" element={<CleaningPage />} />
            <Route path="/analysis" element={<AnalysisPage />} />
            <Route path="/predict" element={<PredictPage />} />
            <Route path="/semantic" element={<SemanticPage />} />
            <Route path="/visualize" element={<VisualizationPage />} />
            <Route path="/deepanalyze" element={<DeepAnalyze />} />
          </Routes>
        </Box>
      </Container>
    </Router>
  );
}
