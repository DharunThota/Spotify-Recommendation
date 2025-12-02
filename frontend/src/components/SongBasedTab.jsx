import React, { useState } from 'react'
import SongSearch from './SongSearch'
import SongCard from './SongCard'
import SongModal from './SongModal'
import { getSongRecommendations } from '../services/api'
import './Tab.css'

function SongBasedTab() {
    const [selectedSong, setSelectedSong] = useState(null)
    const [recommendations, setRecommendations] = useState([])
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState(null)
    const [modalSong, setModalSong] = useState(null)
    const [displayCount, setDisplayCount] = useState(12)

    const handleSongSelect = async (song) => {
        setSelectedSong(song)
        setRecommendations([])
        setError(null)
        setIsLoading(true)
        setDisplayCount(12)

        try {
            const data = await getSongRecommendations(song.id, 50)
            // Transform nested response structure to flat structure for SongCard
            const transformedRecs = data.recommendations.map(rec => ({
                ...rec.song,
                similarity_score: rec.score,
                explanation: rec.explanation?.explanation
            }))
            setRecommendations(transformedRecs)
        } catch (err) {
            console.error('Error fetching recommendations:', err)
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
                <h2 className="section-title">Search for a Song</h2>
                <SongSearch 
                    onSongSelect={handleSongSelect}
                    placeholder="Type a song name or artist..."
                />
            </div>

            {selectedSong && (
                <div className="section">
                    <h2 className="section-title">Selected Song</h2>
                    <SongCard song={selectedSong} />
                </div>
            )}

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
                        Recommended Songs
                        <span className="count-badge">{recommendations.length}</span>
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

            {!isLoading && selectedSong && recommendations.length === 0 && !error && (
                <div className="empty-state">
                    <h3>No Recommendations Found</h3>
                    <p>We couldn't find similar songs for this selection. Try choosing a different song.</p>
                </div>
            )}

            {!isLoading && !selectedSong && (
                <div className="empty-state">
                    <h3>Get Started</h3>
                    <p>Search and select a song to receive personalized recommendations</p>
                </div>
            )}
        </div>
    )
}

export default SongBasedTab
