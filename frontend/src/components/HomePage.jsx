import React from 'react'
import { Music2, Heart, Sparkles, ArrowRight, Play, Music, Headphones, Star } from 'lucide-react'
import './HomePage.css'

function HomePage({ onGetStarted, onNavigate }) {
    const features = [
        {
            icon: Music2,
            title: 'Song-Based Recommendations',
            description: 'Find similar tracks based on your favorite songs using advanced audio features and listening patterns.'
        },
        {
            icon: Heart,
            title: 'Mood-Based Discovery',
            description: 'Explore music that matches your current mood - happy, sad, energetic, or chill vibes.'
        },
        {
            icon: Star,
            title: 'Hybrid Intelligence',
            description: 'Combine multiple songs with mood filters for personalized recommendations tailored to you.'
        }
    ]

    const stats = [
        { value: '170K+', label: 'Songs' },
        { value: '15+', label: 'Audio Features' },
        { value: 'ML', label: 'Powered' }
    ]

    return (
        <div className="home-page">
            {/* Hero Section */}
            <div className="hero-section">
                <div className="hero-content">
                    <div className="hero-badge">
                        <Sparkles className="badge-icon" size={18} />
                        <span>AI-Powered Music Discovery</span>
                    </div>
                    
                    <h1 className="hero-title">
                        Discover Your Next
                        <span className="gradient-text"> Favorite Song</span>
                    </h1>
                    
                    <p className="hero-description">
                        Experience intelligent music recommendations powered by machine learning.
                        Analyze audio features, moods, and listening patterns to find tracks you'll love.
                    </p>

                    <div className="hero-actions">
                        <button className="cta-button primary" onClick={onGetStarted}>
                            <span>Get Started</span>
                            <ArrowRight className="button-icon" size={20} />
                        </button>
                        <button className="cta-button secondary">
                            <Play className="button-icon" size={18} />
                            <span>Watch Demo</span>
                        </button>
                    </div>

                    {/* Stats */}
                    <div className="stats-bar">
                        {stats.map((stat, index) => (
                            <div key={index} className="stat-item">
                                <div className="stat-value">{stat.value}</div>
                                <div className="stat-label">{stat.label}</div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Decorative Elements */}
                <div className="hero-visual">
                    <div className="floating-card card-1">
                        <div className="music-wave">
                            <span></span><span></span><span></span><span></span><span></span>
                        </div>
                    </div>
                    <div className="floating-card card-2">
                        <div className="feature-list">
                            <span>😊</span><span>🎵</span><span>🎧</span><span>✨</span>
                        </div>
                    </div>
                    <div className="floating-card card-3">
                        <div className="progress-indicator">
                            <div className="progress-bar"></div>
                        </div>
                        <div className="song-info">
                            <div className="song-title">Analyzing...</div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Features Section */}
            <div className="features-section">
                <div className="section-header">
                    <h2 className="section-title">Powerful Features</h2>
                    <p className="section-subtitle">
                        Multiple ways to discover music tailored to your taste
                    </p>
                </div>

                <div className="features-grid">
                    {features.map((feature, index) => {
                        const IconComponent = feature.icon
                        return (
                            <div key={index} className="feature-card">
                                <div className="feature-icon">
                                    <IconComponent size={32} strokeWidth={2} />
                                </div>
                                <h3 className="feature-title">{feature.title}</h3>
                                <p className="feature-description">{feature.description}</p>
                                <div 
                                    className="feature-link" 
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        console.log('Learn more clicked, navigating to about');
                                        onNavigate && onNavigate('about');
                                    }}
                                >
                                    Learn more <ArrowRight size={16} className="inline" />
                                </div>
                            </div>
                        )
                    })}
                </div>
            </div>

            {/* How It Works */}
            <div className="how-it-works-section">
                <div className="section-header">
                    <h2 className="section-title">How It Works</h2>
                    <p className="section-subtitle">
                        Three simple steps to discover amazing music
                    </p>
                </div>

                <div className="steps-container">
                    <div className="step-card">
                        <div className="step-number">1</div>
                        <h3 className="step-title">Choose Your Input</h3>
                        <p className="step-description">
                            Search for a song, select a mood, or combine multiple tracks
                        </p>
                    </div>
                    
                    <div className="step-connector"></div>
                    
                    <div className="step-card">
                        <div className="step-number">2</div>
                        <h3 className="step-title">AI Analysis</h3>
                        <p className="step-description">
                            Our ML algorithm analyzes audio features and patterns
                        </p>
                    </div>
                    
                    <div className="step-connector"></div>
                    
                    <div className="step-card">
                        <div className="step-number">3</div>
                        <h3 className="step-title">Get Recommendations</h3>
                        <p className="step-description">
                            Receive personalized tracks with similarity scores
                        </p>
                    </div>
                </div>

                <div className="cta-section">
                    <button className="cta-button primary large" onClick={onGetStarted}>
                        <span>Start Exploring Now</span>
                        <ArrowRight className="button-icon" size={24} />
                    </button>
                </div>
            </div>
        </div>
    )
}

export default HomePage
