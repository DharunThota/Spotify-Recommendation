import React, { useState } from 'react'
import { Smile, Wind, CloudRain, Zap, Flame } from 'lucide-react'
import SongCard from './SongCard'
import SongModal from './SongModal'
import { getMoodRecommendations } from '../services/api'
import './Tab.css'
import './MoodBasedTab.css'

function MoodBasedTab() {
    const [selectedMood, setSelectedMood] = useState(null)
    const [recommendations, setRecommendations] = useState([])
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState(null)
    const [popularOnly, setPopularOnly] = useState(false)
    const [modalSong, setModalSong] = useState(null)
    const [displayCount, setDisplayCount] = useState(12)

    const moods = [
        { id: 'happy', label: 'Happy', icon: Smile, color: '#FFD700' },
        { id: 'chill', label: 'Chill', icon: Wind, color: '#87CEEB' },
        { id: 'sad', label: 'Sad', icon: CloudRain, color: '#9370DB' },
        { id: 'energetic', label: 'Energetic', icon: Zap, color: '#FF6347' }
    ]

    const handleMoodSelect = async (mood, isPopular = popularOnly) => {
        setSelectedMood(mood)
        setRecommendations([])
        setError(null)
        setIsLoading(true)
        setDisplayCount(12)

        try {
            const data = await getMoodRecommendations(mood.id, 50)
            // Transform nested response structure to flat structure for SongCard
            let transformedRecs = data.recommendations.map(rec => ({
                ...rec.song,
                similarity_score: rec.score,
                explanation: rec.explanation?.explanation
            }))
            
            // Filter for popular songs if toggle is on
            if (isPopular) {
                transformedRecs = transformedRecs.filter(song => 
                    song.popularity && song.popularity >= 50
                ).sort((a, b) => (b.popularity || 0) - (a.popularity || 0))
            }
            
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

    const handleTogglePopular = (checked) => {
        setPopularOnly(checked)
        if (selectedMood) {
            handleMoodSelect(selectedMood, checked)
        }
    }

    return (
        <div className="tab-container">
            <div className="section">
                <div className="section-header-with-toggle">
                    <h2 className="section-title">Select Your Mood</h2>
                    <label className="toggle-switch">
                        <input
                            type="checkbox"
                            checked={popularOnly}
                            onChange={(e) => handleTogglePopular(e.target.checked)}
                        />
                        <span className="toggle-slider"></span>
                        <span className="toggle-label">
                            Popular Only
                        </span>
                    </label>
                </div>
                <div className="mood-grid">
                    {moods.map(mood => {
                        const IconComponent = mood.icon
                        return (
                            <button
                                key={mood.id}
                                className={`mood-button ${selectedMood?.id === mood.id ? 'active' : ''}`}
                                onClick={() => handleMoodSelect(mood)}
                                style={{
                                    '--mood-color': mood.color
                                }}
                            >
                                <IconComponent className="mood-icon" size={40} strokeWidth={2} />
                                <span className="mood-label">{mood.label}</span>
                            </button>
                        )
                    })}
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
                        {selectedMood?.label} Songs
                        <span className="count-badge">{recommendations.length}</span>
                        {popularOnly && (
                            <span className="popular-badge">
                                <Flame size={14} />
                                Popular
                            </span>
                        )}
                    </h2>
                    <div className="recommendations-grid">
                        {recommendations.slice(0, displayCount).map((rec, index) => (
                            <SongCard 
                                key={rec.id || index}
                                song={rec}
                                onClick={setModalSong}
                            />
                        ))}
                    </div>
                    {displayCount < recommendations.length && (
                        <div className="load-more-container">
                            <button 
                                className="load-more-button"
                                onClick={() => setDisplayCount(prev => Math.min(prev + 12, recommendations.length))}
                            >
                                Load More ({recommendations.length - displayCount} remaining)
                            </button>
                        </div>
                    )}
                </div>
            )}

            {modalSong && (
                <SongModal song={modalSong} onClose={() => setModalSong(null)} />
            )}

            {!isLoading && selectedMood && recommendations.length === 0 && !error && (
                <div className="empty-state">
                    <h3>No Songs Found</h3>
                    <p>We couldn't find any {selectedMood.label.toLowerCase()} songs. Try a different mood.</p>
                </div>
            )}

            {!isLoading && !selectedMood && (
                <div className="empty-state">
                    <h3>Choose Your Mood</h3>
                    <p>Select a mood above to discover songs that match your vibe</p>
                </div>
            )}
        </div>
    )
}

export default MoodBasedTab
