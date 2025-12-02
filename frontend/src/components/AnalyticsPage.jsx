import React, { useState, useEffect } from 'react'
import {
    BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
    PieChart, Pie, Cell, ResponsiveContainer, XAxis, YAxis, CartesianGrid,
    Tooltip, Legend, LineChart, Line, Area, AreaChart
} from 'recharts'
import { Music, TrendingUp, Users, Clock, Trophy, Sparkles, Calendar, Target, Heart, Zap } from 'lucide-react'
import { getWrappedInsights } from '../services/api'
import './AnalyticsPage.css'

function AnalyticsPage() {
    const [isAuthenticated, setIsAuthenticated] = useState(false)
    const [loading, setLoading] = useState(false)
    const [userData, setUserData] = useState(null)
    const [dashboard, setDashboard] = useState(null)
    const [wrappedInsights, setWrappedInsights] = useState(null)
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
                loadWrappedInsights('medium_term')
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

    const loadWrappedInsights = async (timeRange) => {
        try {
            const data = await getWrappedInsights(timeRange)
            setWrappedInsights(data)
        } catch (error) {
            console.error('Error loading wrapped insights:', error)
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

    const handleTimeRangeChange = (timeRange) => {
        loadDashboard(timeRange)
        loadWrappedInsights(timeRange)
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
        if (!wrappedInsights?.top_artists_with_share) return []
        return wrappedInsights.top_artists_with_share.slice(0, 10).map(artist => ({
            name: artist.name.length > 20 ? artist.name.substring(0, 20) + '...' : artist.name,
            share: artist.share_percentage,
            count: artist.track_count
        }))
    }

    const getGenreDistribution = () => {
        if (!wrappedInsights?.genre_distribution) return []
        return wrappedInsights.genre_distribution
    }

    const getReleaseYearTrend = () => {
        if (!wrappedInsights?.release_trends?.distribution) return []
        return wrappedInsights.release_trends.distribution
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
                                        <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z" />
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
                            onClick={() => handleTimeRangeChange(value)}
                            disabled={loading}
                        >
                            {label}
                        </button>
                    ))}
                </div>
            </div>

            {dashboard && wrappedInsights && (
                <div className="analytics-content">
                    {/* Music Identity Card - Hero Section */}
                    {wrappedInsights.music_identity && (
                        <div className="identity-hero">
                            <div className="identity-card">
                                <div className="identity-badge">
                                    <Sparkles size={32} className="identity-icon" />
                                    <h2 className="identity-title">Your Music Identity</h2>
                                </div>
                                <div className="identity-type">
                                    {wrappedInsights.music_identity.primary_identity}
                                </div>
                                <p className="identity-description">
                                    {wrappedInsights.music_identity.description}
                                </p>
                                {wrappedInsights.music_identity.secondary_identity && (
                                    <p className="identity-secondary">
                                        with hints of <strong>{wrappedInsights.music_identity.secondary_identity}</strong>
                                    </p>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Wrapped Insights Grid */}
                    <div className="wrapped-grid">
                        {/* Artist Diversity Score */}
                        {wrappedInsights.artist_diversity && (
                            <div className="wrapped-card">
                                <div className="wrapped-header">
                                    <Users size={24} className="wrapped-icon" />
                                    <h3>Artist Diversity</h3>
                                </div>
                                <div className="score-circle">
                                    <svg className="score-svg" viewBox="0 0 160 160">
                                        <circle
                                            cx="80"
                                            cy="80"
                                            r="70"
                                            fill="none"
                                            stroke="#e0e0e0"
                                            strokeWidth="12"
                                        />
                                        <circle
                                            cx="80"
                                            cy="80"
                                            r="70"
                                            fill="none"
                                            stroke="#1DB954"
                                            strokeWidth="12"
                                            strokeDasharray={`${wrappedInsights.artist_diversity.score * 4.4} 440`}
                                            strokeLinecap="round"
                                            transform="rotate(-90 80 80)"
                                        />
                                        <text x="80" y="75" textAnchor="middle" className="score-value">
                                            {wrappedInsights.artist_diversity.score}
                                        </text>
                                        <text x="80" y="95" textAnchor="middle" className="score-label">
                                            / 100
                                        </text>
                                    </svg>
                                </div>
                                <div className="wrapped-classification">
                                    {wrappedInsights.artist_diversity.classification}
                                </div>
                                <p className="wrapped-message">
                                    {wrappedInsights.artist_diversity.message}
                                </p>
                                <div className="wrapped-stats">
                                    <span>{wrappedInsights.artist_diversity.unique_artists} unique artists</span>
                                    <span>across {wrappedInsights.artist_diversity.total_tracks} tracks</span>
                                </div>
                            </div>
                        )}

                        {/* Popularity Ranking */}
                        {wrappedInsights.popularity_ranking && (
                            <div className="wrapped-card">
                                <div className="wrapped-header">
                                    <TrendingUp size={24} className="wrapped-icon" />
                                    <h3>Popularity Profile</h3>
                                </div>
                                <div className="popularity-badge">
                                    <span className="popularity-emoji">{wrappedInsights.popularity_ranking.emoji}</span>
                                    <span className="popularity-score">{wrappedInsights.popularity_ranking.score}</span>
                                </div>
                                <div className="wrapped-classification">
                                    {wrappedInsights.popularity_ranking.classification} Listener
                                </div>
                                <p className="wrapped-message">
                                    {wrappedInsights.popularity_ranking.message}
                                </p>
                                <div className="popularity-bar">
                                    <div
                                        className="popularity-fill"
                                        style={{ width: `${wrappedInsights.popularity_ranking.score}%` }}
                                    ></div>
                                </div>
                            </div>
                        )}

                        {/* New vs Old Taste */}
                        {wrappedInsights.new_vs_old && !wrappedInsights.new_vs_old.error && (
                            <div className="wrapped-card">
                                <div className="wrapped-header">
                                    <Calendar size={24} className="wrapped-icon" />
                                    <h3>New vs Classic</h3>
                                </div>
                                <div className="taste-split">
                                    <div className="taste-section new">
                                        <div className="taste-percentage">{wrappedInsights.new_vs_old.recent_percentage}%</div>
                                        <div className="taste-label">Recent</div>
                                    </div>
                                    <div className="taste-divider"></div>
                                    <div className="taste-section old">
                                        <div className="taste-percentage">{100 - wrappedInsights.new_vs_old.recent_percentage}%</div>
                                        <div className="taste-label">Classic</div>
                                    </div>
                                </div>
                                <div className="wrapped-classification">
                                    {wrappedInsights.new_vs_old.classification}
                                </div>
                                <p className="wrapped-message">
                                    {wrappedInsights.new_vs_old.message}
                                </p>
                                <div className="wrapped-stats">
                                    <span>Avg song age: {wrappedInsights.new_vs_old.avg_age} years</span>
                                </div>
                            </div>
                        )}

                        {/* Playlist Personality */}
                        {wrappedInsights.playlist_personality && (
                            <div className="wrapped-card">
                                <div className="wrapped-header">
                                    <Heart size={24} className="wrapped-icon" />
                                    <h3>Playlist Personality</h3>
                                </div>
                                <div className="playlist-stats-grid">
                                    <div className="playlist-stat">
                                        <Clock size={20} />
                                        <div>
                                            <div className="stat-value">{wrappedInsights.playlist_personality.avg_duration}</div>
                                            <div className="stat-label">Avg Track Length</div>
                                        </div>
                                    </div>
                                    <div className="playlist-stat">
                                        <Music size={20} />
                                        <div>
                                            <div className="stat-value">{wrappedInsights.playlist_personality.total_tracks}</div>
                                            <div className="stat-label">Total Tracks</div>
                                        </div>
                                    </div>
                                </div>
                                <div className="top-artist-highlight">
                                    <Trophy size={20} />
                                    <div>
                                        <div className="highlight-label">Most Featured</div>
                                        <div className="highlight-value">
                                            {wrappedInsights.playlist_personality.most_common_artist}
                                        </div>
                                        <div className="highlight-count">
                                            {wrappedInsights.playlist_personality.most_common_artist_count} tracks
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Release Year Trend Chart */}
                    {wrappedInsights.release_trends && !wrappedInsights.release_trends.error && (
                        <div className="section-card full-width">
                            <h2 className="section-title">
                                <Calendar size={24} />
                                Release Year Trends
                            </h2>
                            <p className="section-description">
                                {wrappedInsights.release_trends.message}
                            </p>
                            <ResponsiveContainer width="100%" height={350}>
                                <AreaChart data={getReleaseYearTrend()} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                                    <defs>
                                        <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#1DB954" stopOpacity={0.8} />
                                            <stop offset="95%" stopColor="#1DB954" stopOpacity={0.1} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                                    <XAxis
                                        dataKey="year"
                                        tick={{ fill: '#666', fontSize: 12 }}
                                        label={{ value: 'Release Year', position: 'insideBottom', offset: -10 }}
                                    />
                                    <YAxis
                                        tick={{ fill: '#666' }}
                                        label={{ value: 'Track Count', angle: -90, position: 'insideLeft' }}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: 'white',
                                            border: '2px solid #1DB954',
                                            borderRadius: '12px',
                                            boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                                        }}
                                        labelStyle={{ color: '#181818', fontWeight: 'bold' }}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="count"
                                        stroke="#1DB954"
                                        strokeWidth={3}
                                        fillOpacity={1}
                                        fill="url(#colorCount)"
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    )}

                    {/* Top Artists with Share Percentage */}
                    {wrappedInsights.top_artists_with_share && (
                        <div className="section-card full-width">
                            <h2 className="section-title">
                                <Trophy size={24} />
                                Your Top Artists
                            </h2>
                            <p className="section-description">
                                Artists that dominate your listening
                            </p>
                            <ResponsiveContainer width="100%" height={400}>
                                <BarChart data={getTopArtistsChartData()} margin={{ top: 20, right: 30, left: 20, bottom: 80 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                                    <XAxis
                                        dataKey="name"
                                        angle={-45}
                                        textAnchor="end"
                                        height={100}
                                        tick={{ fill: '#666', fontSize: 12 }}
                                    />
                                    <YAxis
                                        tick={{ fill: '#666' }}
                                        label={{ value: 'Share %', angle: -90, position: 'insideLeft' }}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: 'white',
                                            border: '2px solid #1DB954',
                                            borderRadius: '12px',
                                            boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                                        }}
                                        labelStyle={{ color: '#181818', fontWeight: 'bold' }}
                                        formatter={(value, name) => {
                                            if (name === 'share') return [`${value}%`, 'Share']
                                            if (name === 'count') return [`${value} tracks`, 'Count']
                                            return [value, name]
                                        }}
                                    />
                                    <Bar dataKey="share" fill="#1DB954" radius={[8, 8, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    )}

                    {/* Time-Based Listening Patterns */}
                    {wrappedInsights.time_patterns && !wrappedInsights.time_patterns.error && (
                        <>
                            {/* Time Personality Card */}
                            <div className="section-card full-width">
                                <div className="time-personality-hero">
                                    <Clock size={48} className="time-icon" />
                                    <div>
                                        <h2 className="time-personality-title">
                                            {wrappedInsights.time_patterns.time_personality}
                                        </h2>
                                        <p className="time-personality-message">
                                            {wrappedInsights.time_patterns.time_message}
                                        </p>
                                    </div>
                                </div>
                            </div>

                            {/* Hourly Listening Pattern */}
                            <div className="section-card full-width">
                                <h2 className="section-title">
                                    <Clock size={24} />
                                    Listening by Hour
                                </h2>
                                <p className="section-description">
                                    When you tune in throughout the day
                                </p>
                                <ResponsiveContainer width="100%" height={300}>
                                    <BarChart data={wrappedInsights.time_patterns.hourly_data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                                        <XAxis
                                            dataKey="hour"
                                            tick={{ fill: '#666', fontSize: 12 }}
                                            label={{ value: 'Hour of Day', position: 'insideBottom', offset: -10 }}
                                        />
                                        <YAxis
                                            tick={{ fill: '#666' }}
                                            label={{ value: 'Plays', angle: -90, position: 'insideLeft' }}
                                        />
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: 'white',
                                                border: '2px solid #1DB954',
                                                borderRadius: '12px',
                                                boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                                            }}
                                            labelStyle={{ color: '#181818', fontWeight: 'bold' }}
                                            formatter={(value, name, props) => [
                                                `${value} plays`,
                                                props.payload.period
                                            ]}
                                        />
                                        <Bar dataKey="count" radius={[8, 8, 0, 0]}>
                                            {wrappedInsights.time_patterns.hourly_data.map((entry, index) => (
                                                <Cell
                                                    key={`cell-${index}`}
                                                    fill={entry.hour === wrappedInsights.time_patterns.peak_hour ? '#1DB954' : '#b3e5d1'}
                                                />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Daily Listening Pattern */}
                            <div className="section-card full-width">
                                <h2 className="section-title">
                                    <Calendar size={24} />
                                    Listening by Day of Week
                                </h2>
                                <p className="section-description">
                                    Your weekly listening rhythm
                                </p>
                                <ResponsiveContainer width="100%" height={300}>
                                    <BarChart data={wrappedInsights.time_patterns.daily_data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                                        <XAxis
                                            dataKey="day"
                                            tick={{ fill: '#666', fontSize: 12 }}
                                        />
                                        <YAxis
                                            tick={{ fill: '#666' }}
                                            label={{ value: 'Plays', angle: -90, position: 'insideLeft' }}
                                        />
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: 'white',
                                                border: '2px solid #1DB954',
                                                borderRadius: '12px',
                                                boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                                            }}
                                            labelStyle={{ color: '#181818', fontWeight: 'bold' }}
                                            formatter={(value) => [`${value} plays`, 'Count']}
                                        />
                                        <Bar dataKey="count" radius={[8, 8, 0, 0]}>
                                            {wrappedInsights.time_patterns.daily_data.map((entry, index) => (
                                                <Cell
                                                    key={`cell-${index}`}
                                                    fill={entry.day === wrappedInsights.time_patterns.peak_day ? '#1DB954' : '#b3e5d1'}
                                                />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </>
                    )}

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
                            <div className="section-card">
                                <h2 className="section-title">
                                    <Music size={24} />
                                    Genre Distribution
                                </h2>
                                <ResponsiveContainer width="100%" height={400}>
                                    <PieChart>
                                        <Pie
                                            data={getGenreDistribution()}
                                            cx="50%"
                                            cy="50%"
                                            labelLine={true}
                                            label={({ name, percentage }) => `${name}: ${percentage}%`}
                                            outerRadius={120}
                                            fill="#1DB954"
                                            dataKey="value"
                                        >
                                            {getGenreDistribution().map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                            ))}
                                        </Pie>
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: 'white',
                                                border: '2px solid #1DB954',
                                                borderRadius: '12px',
                                                boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                                            }}
                                            labelStyle={{ color: '#181818', fontWeight: 'bold' }}
                                        />
                                    </PieChart>
                                </ResponsiveContainer>
                            </div>
                        )}
                    </div>

                    {/* Top Artists Bar Chart
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
                    )} */}

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
