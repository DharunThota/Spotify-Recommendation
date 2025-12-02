import React, { useState } from 'react'
import SongSearch from './SongSearch'
import SongCard from './SongCard'
import SongModal from './SongModal'
import { getSequenceAwareRecommendations } from '../services/api'
import './Tab.css'

function SequenceBasedTab() {
    const [selectedSong, setSelectedSong] = useState(null)
    const [recentContext, setRecentContext] = useState([])
    const [recommendations, setRecommendations] = useState([])
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState(null)
    const [modalSong, setModalSong] = useState(null)
    const [displayCount, setDisplayCount] = useState(12)
    const [sequenceWeight, setSequenceWeight] = useState(0.3)

    const handleSongSelect = async (song, contextOverride = null) => {
        setSelectedSong(song)
        setRecommendations([])
        setError(null)
        setIsLoading(true)
        setDisplayCount(12)

        try {
            // Use contextOverride if provided (for when context just updated), otherwise use current state
            const contextToSend = contextOverride !== null ? contextOverride : recentContext
            const data = await getSequenceAwareRecommendations(
                song.id, 
                contextToSend.map(s => s.id), 
                sequenceWeight,
                50
            )
            
            // Transform nested response structure to flat structure for SongCard
            const transformedRecs = data.recommendations.map(rec => ({
                ...rec.song,
                similarity_score: rec.score,
                explanation: rec.explanation?.explanation
            }))
            setRecommendations(transformedRecs)
        } catch (err) {
            console.error('Error fetching sequence recommendations:', err)
            if (err.response?.status === 404 || err.response?.data?.detail?.includes('No recommendations')) {
                setRecommendations([])
                setError(null)
            } else {
                setError('Failed to fetch recommendations. Please try again.')
            }
        } finally {
            setIsLoading(false)
        }
    }

    const addToContext = (song) => {
        if (!recentContext.find(s => s.id === song.id)) {
            setRecentContext(prev => [...prev, song].slice(-5)) // Keep last 5 songs
        }
    }

    const removeFromContext = (songId) => {
        setRecentContext(prev => prev.filter(s => s.id !== songId))
    }

    const clearContext = () => {
        setRecentContext([])
    }

    return (
        <div className="tab-container">
            <div className="section">
                <div className="section-header">
                    <h2 className="section-title">Sequence-Based Recommendations</h2>
                    <p className="section-description">
                        Get smart recommendations based on listening patterns. Add songs to your context to simulate a listening session.
                    </p>
                </div>
                
                {/* Sequence Weight Slider */}
                <div className="control-group">
                    <label htmlFor="sequence-weight">
                        Pattern Weight: <strong>{Math.round(sequenceWeight * 100)}%</strong>
                        <span className="help-text">
                            ({sequenceWeight < 0.5 ? 'More Audio Features' : sequenceWeight > 0.5 ? 'More Listening Patterns' : 'Balanced'})
                        </span>
                    </label>
                    <input
                        id="sequence-weight"
                        type="range"
                        min="0"
                        max="1"
                        step="0.1"
                        value={sequenceWeight}
                        onChange={(e) => setSequenceWeight(parseFloat(e.target.value))}
                        className="slider"
                    />
                    <div className="slider-labels">
                        <span>Audio Features</span>
                        <span>Listening Patterns</span>
                    </div>
                </div>
            </div>

            {/* Recent Context Section */}
            {recentContext.length > 0 && (
                <div className="section context-section">
                    <div className="section-header">
                        <h3 className="section-title">
                            Listening Context
                            <span className="count-badge">{recentContext.length}/5</span>
                        </h3>
                        <button onClick={clearContext} className="clear-button">Clear All</button>
                    </div>
                    <div className="context-list">
                        {recentContext.map((song, index) => (
                            <div key={song.id} className="context-item">
                                <span className="context-number">{index + 1}</span>
                                <div className="context-song-info">
                                    <div className="context-song-name">{song.name}</div>
                                    <div className="context-song-artist">{song.artists}</div>
                                </div>
                                <button 
                                    onClick={() => removeFromContext(song.id)}
                                    className="remove-button"
                                    aria-label="Remove from context"
                                >
                                    ×
                                </button>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Song Search */}
            <div className="section">
                <h2 className="section-title">Search for a Song</h2>
                <SongSearch 
                    onSongSelect={(song) => {
                        // Update context first, then pass the updated context to handleSongSelect
                        const updatedContext = recentContext.find(s => s.id === song.id) 
                            ? recentContext 
                            : [...recentContext, song].slice(-5)
                        setRecentContext(updatedContext)
                        handleSongSelect(song, updatedContext)
                    }}
                    placeholder="Type a song name or artist..."
                />
                {recentContext.length === 0 && (
                    <p className="help-text" style={{ marginTop: '10px' }}>
                        Start by searching for a song. It will be added to your listening context automatically.
                    </p>
                )}
            </div>

            {/* Selected Song */}
            {selectedSong && (
                <div className="section">
                    <h2 className="section-title">Current Song</h2>
                    <SongCard song={selectedSong} />
                </div>
            )}

            {/* Error */}
            {error && (
                <div className="error-message">{error}</div>
            )}

            {/* Loading */}
            {isLoading && (
                <div className="loading-spinner">
                    <div className="spinner"></div>
                </div>
            )}

            {/* Recommendations */}
            {!isLoading && recommendations.length > 0 && (
                <div className="section">
                    <h2 className="section-title">
                        Smart Recommendations
                        <span className="count-badge">{recommendations.length}</span>
                    </h2>
                    <p className="section-description">
                        Based on {recentContext.length > 0 ? `${recentContext.length} song${recentContext.length > 1 ? 's' : ''} in your context and ` : ''}
                        listening pattern analysis
                    </p>
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

            {/* Modal */}
            {modalSong && (
                <SongModal song={modalSong} onClose={() => setModalSong(null)} />
            )}

            {/* Empty States */}
            {!isLoading && selectedSong && recommendations.length === 0 && !error && (
                <div className="empty-state">
                    <h3>No Recommendations Found</h3>
                    <p>Try adding more songs to your listening context or adjusting the pattern weight.</p>
                </div>
            )}

            {!isLoading && !selectedSong && (
                <div className="empty-state">
                    <h3>Build Your Listening Session</h3>
                    <p>Search for songs to create a listening context. Our algorithm will learn from your patterns to suggest what to play next.</p>
                </div>
            )}
        </div>
    )
}

export default SequenceBasedTab
