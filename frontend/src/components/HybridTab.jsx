import React, { useState } from 'react'
import SongSearch from './SongSearch'
import SongCard from './SongCard'
import { getHybridRecommendations } from '../services/api'
import './Tab.css'
import './HybridTab.css'

function HybridTab() {
    const [selectedSongs, setSelectedSongs] = useState([])
    const [selectedMood, setSelectedMood] = useState(null)
    const [recommendations, setRecommendations] = useState([])
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState(null)
    const [popularOnly, setPopularOnly] = useState(false)

    const moods = [
        { id: 'happy', label: 'Happy', emoji: '😊' },
        { id: 'chill', label: 'Chill', emoji: '😌' },
        { id: 'sad', label: 'Sad', emoji: '😢' },
        { id: 'energetic', label: 'Energetic', emoji: '⚡' }
    ]

    const handleSongSelect = (song) => {
        // Check if song already exists
        if (selectedSongs.find(s => s.id === song.id)) {
            return
        }
        setSelectedSongs([...selectedSongs, song])
    }

    const handleRemoveSong = (songId) => {
        setSelectedSongs(selectedSongs.filter(s => s.id !== songId))
    }

    const handleMoodToggle = (mood) => {
        setSelectedMood(selectedMood?.id === mood.id ? null : mood)
    }

    const handleGetRecommendations = async () => {
        if (selectedSongs.length === 0) {
            setError('Please select at least one song')
            return
        }

        setRecommendations([])
        setError(null)
        setIsLoading(true)

        try {
            const songIds = selectedSongs.map(s => s.id)
            const data = await getHybridRecommendations(
                songIds,
                selectedMood?.id || null,
                12
            )
            // Transform nested response structure to flat structure for SongCard
            let transformedRecs = data.recommendations.map(rec => ({
                ...rec.song,
                similarity_score: rec.score,
                explanation: rec.explanation?.explanation_text || rec.explanation?.text
            }))
            
            // Filter for popular songs if toggle is on
            if (popularOnly) {
                transformedRecs = transformedRecs.filter(song => 
                    song.popularity && song.popularity >= 50
                ).sort((a, b) => (b.popularity || 0) - (a.popularity || 0))
            }
            
            setRecommendations(transformedRecs)
        } catch (err) {
            console.error('Error fetching hybrid recommendations:', err)
            // Check if it's a 404 with no recommendations
            if (err.response?.status === 404 || err.response?.data?.detail?.includes('No recommendations')) {
                setRecommendations([])
                setError(null) // Don't show error, just empty state
            } else {
                setError('Failed to fetch recommendations. Please try again.')
            }
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <div className="tab-container">
            <div className="section">
                <h2 className="section-title">Add Songs</h2>
                <SongSearch 
                    onSongSelect={handleSongSelect}
                    placeholder="Search and add multiple songs..."
                />
            </div>

            {selectedSongs.length > 0 && (
                <div className="section">
                    <h2 className="section-title">
                        Selected Songs
                        <span className="count-badge">{selectedSongs.length}</span>
                    </h2>
                    <div className="selected-songs-grid">
                        {selectedSongs.map(song => (
                            <SongCard 
                                key={song.id}
                                song={song}
                                onRemove={() => handleRemoveSong(song.id)}
                            />
                        ))}
                    </div>
                </div>
            )}

            <div className="section">
                <div className="section-header-with-toggle">
                    <h2 className="section-title">Mood Filter (Optional)</h2>
                    <label className="toggle-switch">
                        <input
                            type="checkbox"
                            checked={popularOnly}
                            onChange={(e) => setPopularOnly(e.target.checked)}
                        />
                        <span className="toggle-slider"></span>
                        <span className="toggle-label">
                            <span className="toggle-icon">🔥</span>
                            Popular Only
                        </span>
                    </label>
                </div>
                <div className="mood-filter">
                    {moods.map(mood => (
                        <button
                            key={mood.id}
                            className={`mood-filter-button ${selectedMood?.id === mood.id ? 'active' : ''}`}
                            onClick={() => handleMoodToggle(mood)}
                        >
                            <span className="mood-emoji-small">{mood.emoji}</span>
                            {mood.label}
                        </button>
                    ))}
                </div>
            </div>

            <div className="section">
                <button 
                    className="get-recommendations-button"
                    onClick={handleGetRecommendations}
                    disabled={selectedSongs.length === 0 || isLoading}
                >
                    {isLoading ? 'Loading...' : 'Get Recommendations'}
                </button>
            </div>

            {error && (
                <div className="error-message">{error}</div>
            )}

            {isLoading && (
                <div className="loading-spinner">
                    <div className="spinner"></div>
                </div>
            )}

            {!isLoading && recommendations.length > 0 && (
                <div className="section">
                    <h2 className="section-title">
                        Hybrid Recommendations
                        <span className="count-badge">{recommendations.length}</span>
                        {popularOnly && (
                            <span className="popular-badge">
                                <span className="popular-badge-icon">🔥</span>
                                Popular
                            </span>
                        )}
                    </h2>
                    <div className="recommendations-grid">
                        {recommendations.map((rec, index) => (
                            <SongCard 
                                key={rec.id || index}
                                song={rec}
                                showExplanation={true}
                            />
                        ))}
                    </div>
                </div>
            )}

            {!isLoading && selectedSongs.length > 0 && recommendations.length === 0 && !error && (
                <div className="empty-state">
                    <h3>No Recommendations Found</h3>
                    <p>We couldn't find recommendations based on your selection. Try different songs or remove the mood filter.</p>
                </div>
            )}

            {!isLoading && selectedSongs.length === 0 && (
                <div className="empty-state">
                    <h3>🎭 Hybrid Recommendations</h3>
                    <p>Add multiple songs and optionally select a mood to get personalized recommendations</p>
                </div>
            )}
        </div>
    )
}

export default HybridTab
