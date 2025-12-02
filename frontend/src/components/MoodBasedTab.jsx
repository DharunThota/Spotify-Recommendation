import React, { useState } from 'react'
import SongCard from './SongCard'
import { getMoodRecommendations } from '../services/api'
import './Tab.css'
import './MoodBasedTab.css'

function MoodBasedTab() {
    const [selectedMood, setSelectedMood] = useState(null)
    const [recommendations, setRecommendations] = useState([])
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState(null)

    const moods = [
        { id: 'happy', label: 'Happy', emoji: '😊', color: '#FFD700' },
        { id: 'chill', label: 'Chill', emoji: '😌', color: '#87CEEB' },
        { id: 'sad', label: 'Sad', emoji: '😢', color: '#9370DB' },
        { id: 'energetic', label: 'Energetic', emoji: '⚡', color: '#FF6347' }
    ]

    const handleMoodSelect = async (mood) => {
        setSelectedMood(mood)
        setRecommendations([])
        setError(null)
        setIsLoading(true)

        try {
            const data = await getMoodRecommendations(mood.id, 12)
            // Transform nested response structure to flat structure for SongCard
            const transformedRecs = data.recommendations.map(rec => ({
                ...rec.song,
                similarity_score: rec.score,
                explanation: rec.explanation?.explanation_text || rec.explanation?.text
            }))
            setRecommendations(transformedRecs)
        } catch (err) {
            console.error('Error fetching mood recommendations:', err)
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
                <h2 className="section-title">Select Your Mood</h2>
                <div className="mood-grid">
                    {moods.map(mood => (
                        <button
                            key={mood.id}
                            className={`mood-button ${selectedMood?.id === mood.id ? 'active' : ''}`}
                            onClick={() => handleMoodSelect(mood)}
                            style={{
                                '--mood-color': mood.color
                            }}
                        >
                            <span className="mood-emoji">{mood.emoji}</span>
                            <span className="mood-label">{mood.label}</span>
                        </button>
                    ))}
                </div>
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
                        {selectedMood?.emoji} {selectedMood?.label} Songs
                        <span className="count-badge">{recommendations.length}</span>
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

            {!isLoading && selectedMood && recommendations.length === 0 && !error && (
                <div className="empty-state">
                    <h3>No Songs Found</h3>
                    <p>We couldn't find any {selectedMood.label.toLowerCase()} songs. Try a different mood.</p>
                </div>
            )}

            {!isLoading && !selectedMood && (
                <div className="empty-state">
                    <h3>😊 Choose Your Mood</h3>
                    <p>Select a mood above to discover songs that match your vibe</p>
                </div>
            )}
        </div>
    )
}

export default MoodBasedTab
