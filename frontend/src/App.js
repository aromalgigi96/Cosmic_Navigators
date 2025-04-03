import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './HomePage';
import SpaceExplorationPage from './SpaceExplorationPage';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/space-exploration" element={<SpaceExplorationPage />} />
      </Routes>
    </Router>
  );
}

export default App;
