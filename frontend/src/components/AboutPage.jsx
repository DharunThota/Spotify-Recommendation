import React from 'react';
import { Music, Target, Shuffle, BarChart3, Zap, Heart, Music2, Sparkles, TrendingUp, Users, Database, Cpu } from 'lucide-react';
import './AboutPage.css';

const AboutPage = ({ onNavigate }) => {
    return (
        <div className="about-page">
            <div className="about-hero">
                <div className="about-hero-content">
                    <h1 className="about-hero-title">About Spotify Recommendation System</h1>
                    <p className="about-hero-subtitle">
                        Discover how AI and machine learning transform your music experience
                    </p>
                </div>
            </div>

            <div className="about-content">
                <section className="about-section">
                    <div className="section-icon-wrapper">
                        <Music className="section-icon-svg" size={48} />
                    </div>
                    <h2>Mood-Based Recommendations</h2>
                    <p>
                        Our advanced machine learning algorithms analyze audio features like energy, 
                        valence, danceability, and tempo to understand the emotional essence of songs. 
                        When you select a mood, we match it with songs that share similar acoustic 
                        characteristics, creating playlists that perfectly match your emotional state.
                    </p>
                    <div className="feature-highlights">
                        <div className="highlight-item">
                            <div className="highlight-icon-wrapper">
                                <Zap className="highlight-icon-svg" size={28} />
                            </div>
                            <div>
                                <strong>Energy Analysis</strong>
                                <p>Measure intensity and activity level</p>
                            </div>
                        </div>
                        <div className="highlight-item">
                            <div className="highlight-icon-wrapper">
                                <Heart className="highlight-icon-svg" size={28} />
                            </div>
                            <div>
                                <strong>Valence Detection</strong>
                                <p>Identify musical positivity and happiness</p>
                            </div>
                        </div>
                        <div className="highlight-item">
                            <div className="highlight-icon-wrapper">
                                <Music2 className="highlight-icon-svg" size={28} />
                            </div>
                            <div>
                                <strong>Acoustic Features</strong>
                                <p>Analyze tempo, key, and instrumentalness</p>
                            </div>
                        </div>
                    </div>
                </section>

                <section className="about-section">
                    <div className="section-icon-wrapper">
                        <Target className="section-icon-svg" size={48} />
                    </div>
                    <h2>Song-Based Recommendations</h2>
                    <p>
                        Using cosine similarity and content-based filtering, we find songs that are 
                        acoustically similar to your favorites. Our system analyzes multiple audio 
                        features simultaneously to identify tracks that share similar sonic characteristics, 
                        helping you discover new music that matches your taste.
                    </p>
                    <div className="tech-stack">
                        <div className="tech-item">
                            <Database size={18} />
                            <span>PySpark</span>
                        </div>
                        <div className="tech-item">
                            <Cpu size={18} />
                            <span>Machine Learning</span>
                        </div>
                        <div className="tech-item">
                            <TrendingUp size={18} />
                            <span>Content-Based Filtering</span>
                        </div>
                        <div className="tech-item">
                            <Sparkles size={18} />
                            <span>Cosine Similarity</span>
                        </div>
                    </div>
                </section>

                <section className="about-section">
                    <div className="section-icon-wrapper">
                        <Shuffle className="section-icon-svg" size={48} />
                    </div>
                    <h2>Hybrid Recommendations</h2>
                    <p>
                        Combine the power of both approaches! Our hybrid system merges song similarity 
                        with mood analysis to provide diverse yet cohesive recommendations. This ensures 
                        you get variety while maintaining the emotional tone you're looking for.
                    </p>
                </section>

                <section className="about-section">
                    <div className="section-icon-wrapper">
                        <BarChart3 className="section-icon-svg" size={48} />
                    </div>
                    <h2>Analytics Dashboard</h2>
                    <p>
                        Connect your Spotify account to unlock personalized insights powered by 
                        association rule mining and data visualization. Discover patterns in your 
                        listening habits, analyze your music personality, and see how your taste 
                        evolves over time.
                    </p>
                    <div className="analytics-features">
                        <div className="analytics-item">
                            <div className="analytics-icon">
                                <Users size={24} />
                            </div>
                            <h4>Top Artists & Tracks</h4>
                            <p>See your most-played music at a glance</p>
                        </div>
                        <div className="analytics-item">
                            <div className="analytics-icon">
                                <Sparkles size={24} />
                            </div>
                            <h4>Music Personality</h4>
                            <p>Visualize your unique audio preferences</p>
                        </div>
                        <div className="analytics-item">
                            <div className="analytics-icon">
                                <TrendingUp size={24} />
                            </div>
                            <h4>Listening Patterns</h4>
                            <p>Discover associations in your music taste</p>
                        </div>
                    </div>
                </section>

                <section className="about-section technology-section">
                    <h2>Technology Stack</h2>
                    <div className="tech-grid">
                        <div className="tech-card">
                            <h3>Frontend</h3>
                            <ul>
                                <li>React 18</li>
                                <li>React Router</li>
                                <li>Recharts</li>
                                <li>CSS3 Animations</li>
                            </ul>
                        </div>
                        <div className="tech-card">
                            <h3>Backend</h3>
                            <ul>
                                <li>Python FastAPI</li>
                                <li>PySpark</li>
                                <li>Pandas</li>
                                <li>Spotipy</li>
                            </ul>
                        </div>
                        <div className="tech-card">
                            <h3>Machine Learning</h3>
                            <ul>
                                <li>Content-Based Filtering</li>
                                <li>Cosine Similarity</li>
                                <li>Association Mining</li>
                                <li>Feature Engineering</li>
                            </ul>
                        </div>
                    </div>
                </section>

                <section className="about-section cta-section">
                    <h2>Ready to Discover Your Next Favorite Song?</h2>
                    <p>Start exploring personalized recommendations powered by AI</p>
                    <div className="cta-buttons">
                        <button 
                            className="cta-button primary"
                            onClick={() => onNavigate && onNavigate('app')}
                        >
                            Get Started
                        </button>
                        <button 
                            className="cta-button secondary"
                            onClick={() => onNavigate && onNavigate('analytics')}
                        >
                            View Analytics
                        </button>
                    </div>
                </section>
            </div>
        </div>
    );
};

export default AboutPage;
