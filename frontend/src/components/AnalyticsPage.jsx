import React, { useState, useEffect } from 'react'
import { BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
         PieChart, Pie, Cell, ResponsiveContainer, XAxis, YAxis, CartesianGrid, 
         Tooltip, Legend } from 'recharts'
import './AnalyticsPage.css'

function AnalyticsPage() {
    const [isAuthenticated, setIsAuthenticated] = useState(false)
    const [loading, setLoading] = useState(false)
    const [userData, setUserData] = useState(null)
    const [dashboard, setDashboard] = useState(null)
    const [activeTimeRange, setActiveTimeRange] = useState('medium_term')
    const [associations, setAssociations] = useState(null)

    useEffect(() => {
        // Listen for authentication success from popup
        const handleMessage = (event) => {
            if (event.data.type === 'spotify_auth_success') {
                setIsAuthenticated(true)
                setUserData({
                    user_id: event.data.user_id,
                    display_name: event.data.display_name
                })
                loadDashboard('medium_term')
            }
        }

        window.addEventListener('message', handleMessage)
        return () => window.removeEventListener('message', handleMessage)
    }, [])

    const handleSpotifyLogin = async () => {
        try {
            setLoading(true)
            const response = await fetch('http://localhost:8000/api/analytics/auth-url')
            const data = await response.json()
            
            // Open Spotify auth in popup
            const width = 600
            const height = 700
            const left = window.screen.width / 2 - width / 2
            const top = window.screen.height / 2 - height / 2
            
            window.open(
                data.auth_url,
                'Spotify Login',
                `width=${width},height=${height},left=${left},top=${top}`
            )
        } catch (error) {
            console.error('Error initiating Spotify auth:', error)
            alert('Failed to connect to Spotify. Please try again.')
        } finally {
            setLoading(false)
        }
    }

    const loadDashboard = async (timeRange) => {
        try {
            setLoading(true)
            const response = await fetch(
                `http://localhost:8000/api/analytics/dashboard?time_range=${timeRange}`
            )
            
            if (!response.ok) {
                throw new Error('Failed to load dashboard')
            }
            
            const data = await response.json()
            setDashboard(data)
            setActiveTimeRange(timeRange)
            
            // Load associations separately
            loadAssociations()
        } catch (error) {
            console.error('Error loading dashboard:', error)
            alert('Failed to load analytics. Please try again.')
        } finally {
            setLoading(false)
        }
    }

    const loadAssociations = async () => {
        try {
            const response = await fetch(
                'http://localhost:8000/api/analytics/associations?limit=50&min_support=0.15&min_confidence=0.6'
            )
            
            if (response.ok) {
                const data = await response.json()
                setAssociations(data)
            }
        } catch (error) {
            console.error('Error loading associations:', error)
        }
    }

    const timeRangeLabels = {
        'short_term': 'Last 4 Weeks',
        'medium_term': 'Last 6 Months',
        'long_term': 'All Time'
    }

    const COLORS = ['#1DB954', '#1ed760', '#169c46', '#117a37', '#0d5c29', '#1DB954', '#1ed760', '#169c46', '#117a37', '#0d5c29']
    
    // Prepare chart data
    const getAudioFeaturesChartData = () => {
        if (!dashboard?.listening_patterns?.audio_features) return []
        const features = dashboard.listening_patterns.audio_features
        return Object.entries(features)
            .filter(([_, value]) => value !== null && !['tempo', 'loudness'].includes(_))
            .map(([name, value]) => ({
                feature: name.charAt(0).toUpperCase() + name.slice(1),
                value: (value * 100).toFixed(0),
                fullMark: 100
            }))
    }

    const getTopArtistsChartData = () => {
        if (!dashboard?.listening_patterns?.top_artists) return []
        return Object.entries(dashboard.listening_patterns.top_artists)
            .slice(0, 10)
            .map(([artist, count]) => ({
                name: artist.length > 20 ? artist.substring(0, 20) + '...' : artist,
                plays: count
            }))
    }

    const getGenreDistribution = () => {
        if (!dashboard?.top_artists) return []
        const genreCounts = {}
        dashboard.top_artists.forEach(artist => {
            if (artist.genres) {
                artist.genres.forEach(genre => {
                    genreCounts[genre] = (genreCounts[genre] || 0) + 1
                })
            }
        })
        return Object.entries(genreCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 8)
            .map(([name, value]) => ({ name, value }))
    }

    if (!isAuthenticated) {
        return (
            <div className="analytics-page">
                <div className="auth-container">
                    <div className="auth-card">
                        <div className="auth-icon">📊</div>
                        <h1 className="auth-title">Your Music Analytics</h1>
                        <p className="auth-description">
                            Connect your Spotify account to unlock personalized insights about your listening habits,
                            discover patterns in your music taste, and get AI-powered recommendations.
                        </p>
                        
                        <div className="features-list">
                            <div className="feature-item">
                                <span className="feature-icon">🎵</span>
                                <span>Analyze your top tracks and artists</span>
                            </div>
                            <div className="feature-item">
                                <span className="feature-icon">📈</span>
                                <span>Visualize your music personality</span>
                            </div>
                            <div className="feature-item">
                                <span className="feature-icon">🔗</span>
                                <span>Discover listening patterns with AI</span>
                            </div>
                            <div className="feature-item">
                                <span className="feature-icon">✨</span>
                                <span>Get personalized recommendations</span>
                            </div>
                        </div>

                        <button
                            className="spotify-login-button"
                            onClick={handleSpotifyLogin}
                            disabled={loading}
                        >
                            {loading ? (
                                <>
                                    <span className="spinner-small"></span>
                                    <span>Connecting...</span>
                                </>
                            ) : (
                                <>
                                    <svg viewBox="0 0 24 24" fill="currentColor" width="24" height="24">
                                        <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
                                    </svg>
                                    <span>Connect with Spotify</span>
                                </>
                            )}
                        </button>

                        <p className="auth-disclaimer">
                            🔒 We only access your listening history and never modify your account
                        </p>
                    </div>
                </div>
            </div>
        )
    }

    if (loading && !dashboard) {
        return (
            <div className="analytics-page">
                <div className="loading-container">
                    <div className="spinner"></div>
                    <p>Loading your music analytics...</p>
                </div>
            </div>
        )
    }

    return (
        <div className="analytics-page">
            <div className="analytics-header">
                <div className="header-content">
                    <h1 className="page-title">
                        <span className="title-icon">📊</span>
                        Your Music Analytics
                    </h1>
                    {userData && (
                        <p className="welcome-text">Welcome back, {userData.display_name}!</p>
                    )}
                </div>

                <div className="time-range-selector">
                    {Object.entries(timeRangeLabels).map(([value, label]) => (
                        <button
                            key={value}
                            className={`time-range-button ${activeTimeRange === value ? 'active' : ''}`}
                            onClick={() => loadDashboard(value)}
                            disabled={loading}
                        >
                            {label}
                        </button>
                    ))}
                </div>
            </div>

            {dashboard && (
                <div className="analytics-content">
                    {/* Overview Stats */}
                    <div className="stats-grid">
                        <div className="stat-card">
                            <div className="stat-icon">🎵</div>
                            <div className="stat-content">
                                <div className="stat-value">{dashboard.listening_patterns?.total_tracks || 0}</div>
                                <div className="stat-label">Total Tracks</div>
                            </div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-icon">👤</div>
                            <div className="stat-content">
                                <div className="stat-value">{dashboard.listening_patterns?.unique_artists || 0}</div>
                                <div className="stat-label">Unique Artists</div>
                            </div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-icon">⭐</div>
                            <div className="stat-content">
                                <div className="stat-value">
                                    {dashboard.listening_patterns?.avg_popularity?.toFixed(0) || 0}
                                </div>
                                <div className="stat-label">Avg Popularity</div>
                            </div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-icon">⏱️</div>
                            <div className="stat-content">
                                <div className="stat-value">
                                    {dashboard.listening_patterns?.avg_duration_min?.toFixed(1) || 0}m
                                </div>
                                <div className="stat-label">Avg Track Length</div>
                            </div>
                        </div>
                    </div>

                    {/* Visualizations Row */}
                    <div className="charts-row">
                        {/* Music Personality Radar Chart */}
                        {dashboard.listening_patterns?.audio_features && dashboard.listening_patterns?.has_audio_features ? (
                            <div className="section-card chart-card">
                                <h2 className="section-title">
                                    <span className="title-icon">🎨</span>
                                    Your Music Personality
                                </h2>
                                <ResponsiveContainer width="100%" height={350}>
                                    <RadarChart data={getAudioFeaturesChartData()}>
                                        <PolarGrid stroke="#1DB954" strokeOpacity={0.2} />
                                        <PolarAngleAxis 
                                            dataKey="feature" 
                                            tick={{ fill: '#b3b3b3', fontSize: 12 }}
                                        />
                                        <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: '#b3b3b3' }} />
                                        <Radar 
                                            name="Audio Features" 
                                            dataKey="value" 
                                            stroke="#1DB954" 
                                            fill="#1DB954" 
                                            fillOpacity={0.6} 
                                        />
                                        <Tooltip 
                                            contentStyle={{ backgroundColor: '#282828', border: '1px solid #1DB954', borderRadius: '8px' }}
                                            labelStyle={{ color: '#fff' }}
                                        />
                                    </RadarChart>
                                </ResponsiveContainer>
                            </div>
                        ) : dashboard.listening_patterns && !dashboard.listening_patterns.has_audio_features ? (
                            <div className="section-card info-card">
                                <h2 className="section-title">
                                    <span className="title-icon">ℹ️</span>
                                    Audio Features Unavailable
                                </h2>
                                <p className="info-message">
                                    Audio feature analysis is currently unavailable due to Spotify API limitations. 
                                    You can still view your top tracks, artists, and listening patterns below!
                                </p>
                            </div>
                        ) : null}

                        {/* Genre Distribution Pie Chart */}
                        {getGenreDistribution().length > 0 && (
                            <div className="section-card chart-card">
                                <h2 className="section-title">
                                    <span className="title-icon">🎭</span>
                                    Genre Distribution
                                </h2>
                                <ResponsiveContainer width="100%" height={350}>
                                    <PieChart>
                                        <Pie
                                            data={getGenreDistribution()}
                                            cx="50%"
                                            cy="50%"
                                            labelLine={false}
                                            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                                            outerRadius={100}
                                            fill="#1DB954"
                                            dataKey="value"
                                        >
                                            {getGenreDistribution().map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                            ))}
                                        </Pie>
                                        <Tooltip 
                                            contentStyle={{ backgroundColor: '#282828', border: '1px solid #1DB954', borderRadius: '8px' }}
                                            labelStyle={{ color: '#fff' }}
                                        />
                                    </PieChart>
                                </ResponsiveContainer>
                            </div>
                        )}
                    </div>

                    {/* Top Artists Bar Chart */}
                    {getTopArtistsChartData().length > 0 && (
                        <div className="section-card full-width">
                            <h2 className="section-title">
                                <span className="title-icon">📊</span>
                                Top Artists Breakdown
                            </h2>
                            <ResponsiveContainer width="100%" height={400}>
                                <BarChart data={getTopArtistsChartData()} margin={{ top: 20, right: 30, left: 20, bottom: 80 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                    <XAxis 
                                        dataKey="name" 
                                        angle={-45} 
                                        textAnchor="end" 
                                        height={100}
                                        tick={{ fill: '#b3b3b3', fontSize: 12 }}
                                    />
                                    <YAxis tick={{ fill: '#b3b3b3' }} />
                                    <Tooltip 
                                        contentStyle={{ backgroundColor: '#282828', border: '1px solid #1DB954', borderRadius: '8px' }}
                                        labelStyle={{ color: '#fff' }}
                                        cursor={{ fill: 'rgba(29, 185, 84, 0.1)' }}
                                    />
                                    <Bar dataKey="plays" fill="#1DB954" radius={[8, 8, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    )}

                    {/* Top Artists */}
                    {dashboard.top_artists && dashboard.top_artists.length > 0 && (
                        <div className="section-card">
                            <h2 className="section-title">
                                <span className="title-icon">⭐</span>
                                Your Top Artists
                            </h2>
                            <div className="artists-grid">
                                {dashboard.top_artists.map((artist, index) => (
                                    <div key={artist.id || index} className="artist-card">
                                        <div className="artist-rank">#{index + 1}</div>
                                        <div className="artist-info">
                                            <div className="artist-name">{artist.name}</div>
                                            {artist.genres && artist.genres.length > 0 && (
                                                <div className="artist-genres">
                                                    {artist.genres.slice(0, 2).map((genre, i) => (
                                                        <span key={i} className="genre-tag">{genre}</span>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                        <div className="artist-popularity">
                                            {artist.popularity}
                                            <span className="popularity-label">popularity</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Top Tracks */}
                    {dashboard.top_tracks && dashboard.top_tracks.length > 0 && (
                        <div className="section-card">
                            <h2 className="section-title">
                                <span className="title-icon">🎧</span>
                                Your Top Tracks
                            </h2>
                            <div className="tracks-list">
                                {dashboard.top_tracks.map((track, index) => (
                                    <div key={track.id || index} className="track-item">
                                        <div className="track-rank">#{index + 1}</div>
                                        <div className="track-info">
                                            <div className="track-name">{track.name}</div>
                                            <div className="track-artist">{track.artist}</div>
                                        </div>
                                        <div className="track-meta">
                                            <span className="track-popularity">⭐ {track.popularity}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Association Rules */}
                    {associations && associations.artist_associations && (
                        <div className="section-card">
                            <h2 className="section-title">
                                <span className="title-icon">🔗</span>
                                Listening Pattern Insights
                            </h2>
                            <p className="section-description">
                                Discovered patterns in your music using association rule mining
                            </p>
                            
                            {associations.artist_associations.rules && (
                                <div className="associations-container">
                                    <h3 className="subsection-title">Artist Connections</h3>
                                    <div className="rules-list">
                                        {associations.artist_associations.rules.map((rule, index) => (
                                            <div key={index} className="rule-card">
                                                <div className="rule-content">
                                                    <div className="rule-description">{rule.description}</div>
                                                    <div className="rule-metrics">
                                                        <span className="metric">
                                                            Confidence: {(rule.confidence * 100).toFixed(1)}%
                                                        </span>
                                                        <span className="metric">
                                                            Lift: {rule.lift.toFixed(2)}x
                                                        </span>
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {associations.mood_patterns && associations.mood_patterns.rules && (
                                <div className="associations-container">
                                    <h3 className="subsection-title">Mood Patterns</h3>
                                    <div className="rules-list">
                                        {associations.mood_patterns.rules.slice(0, 5).map((rule, index) => (
                                            <div key={index} className="rule-card">
                                                <div className="rule-content">
                                                    <div className="rule-description">{rule.description}</div>
                                                    <div className="rule-metrics">
                                                        <span className="metric">
                                                            Confidence: {(rule.confidence * 100).toFixed(1)}%
                                                        </span>
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Recommendations */}
                    {dashboard.recommendations && dashboard.recommendations.length > 0 && (
                        <div className="section-card">
                            <h2 className="section-title">
                                <span className="title-icon">✨</span>
                                Personalized Insights
                            </h2>
                            <div className="recommendations-list">
                                {dashboard.recommendations.map((rec, index) => (
                                    <div key={index} className="recommendation-item">
                                        <span className="rec-icon">💡</span>
                                        <span className="rec-text">{rec}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}

export default AnalyticsPage
