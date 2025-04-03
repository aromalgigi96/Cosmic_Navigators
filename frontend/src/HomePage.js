import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Rocket, Star, Navigation, Globe2 } from 'lucide-react';
import ChatBot from './ChatBot';  // Import the ChatBot component

function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="page">
      {/* Hero Section */}
      <header className="header">
        <div className="header-background">
          <img
            src="https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&q=80"
            alt="Space background"
          />
        </div>
        <div className="header-content">
          <div className="header-title">
            <Rocket size={32} />
            Cosmic Navigators
          </div>
          <p className="header-description">
            Embark on a journey through the cosmos. We guide you through the wonders of space exploration and astronomical discovery.
          </p>
          <button
            className="button button-primary"
            onClick={() => navigate('/space-exploration')}
          >
            Start Exploring
          </button>
        </div>
      </header>

      {/* Features Section */}
      <section className="features">
        <div className="container features-grid">
          <FeatureCard
            icon={<Star className="feature-icon" size={40} />}
            title="Stellar Navigation"
            description="Advanced tools for celestial navigation and star mapping"
          />
          <FeatureCard
            icon={<Navigation className="feature-icon" size={40} />}
            title="Space Exploration"
            description="Real-time tracking of space missions and astronomical events"
          />
          <FeatureCard
            icon={<Globe2 className="feature-icon" size={40} />}
            title="Global Community"
            description="Connect with space enthusiasts from around the world"
          />
        </div>
      </section>

      {/* ChatBot Component (Floating Chatbot) */}
      <ChatBot />
    </div>
  );
}

function FeatureCard({ icon, title, description }) {
  return (
    <div className="feature-card">
      {icon}
      <div className="feature-title">{title}</div>
      <div className="feature-description">{description}</div>
    </div>
  );
}

export default HomePage;
