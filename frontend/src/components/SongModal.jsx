import React, { useEffect } from 'react'
import { X, Calendar, TrendingUp, Music, Activity } from 'lucide-react'
import './SongModal.css'

function SongModal({ song, onClose }) {
    useEffect(() => {
        // Prevent body scroll when modal is open
        document.body.style.overflow = 'hidden'
        
        return () => {
            document.body.style.overflow = 'unset'
        }
    }, [])

    const handleBackdropClick = (e) => {
        if (e.target === e.currentTarget) {
            onClose()
        }
    }

    if (!song) return null

    return (
        <div className="modal-backdrop" onClick={handleBackdropClick}>
            <div className="modal-content">
                <button className="modal-close" onClick={onClose}>
                    <X size={24} />
                </button>

                <div className="modal-header">
                    <div className="modal-icon">
                        <Music size={32} />
                    </div>
                    <div className="modal-title-section">
                        <h2 className="modal-title">{song.name}</h2>
                        <p className="modal-artist">{song.artists}</p>
                    </div>
                </div>

                <div className="modal-body">
                    {/* Metadata Section */}
                    <div className="modal-section">
                        <h3 className="modal-section-title">Information</h3>
                        <div className="modal-metadata">
                            <div className="metadata-item">
                                <Calendar size={18} />
                                <div className="metadata-content">
                                    <span className="metadata-label">Year</span>
                                    <span className="metadata-value">{song.year}</span>
                                </div>
                            </div>
                            {song.popularity !== undefined && (
                                <div className="metadata-item">
                                    <TrendingUp size={18} />
                                    <div className="metadata-content">
                                        <span className="metadata-label">Popularity</span>
                                        <span className="metadata-value">{song.popularity}/100</span>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Similarity Score */}
                    {song.similarity_score !== undefined && (
                        <div className="modal-section">
                            <h3 className="modal-section-title">
                                <Activity size={18} />
                                Similarity Score
                            </h3>
                            <div className="modal-score">
                                <div className="modal-score-bar">
                                    <div 
                                        className="modal-score-fill" 
                                        style={{ width: `${song.similarity_score * 100}%` }}
                                    ></div>
                                </div>
                                <span className="modal-score-text">
                                    {(song.similarity_score * 100).toFixed(1)}% Match
                                </span>
                            </div>
                        </div>
                    )}

                    {/* Audio Features */}
                    {(song.valence !== undefined || song.energy !== undefined || song.danceability !== undefined) && (
                        <div className="modal-section">
                            <h3 className="modal-section-title">Audio Features</h3>
                            <div className="audio-features">
                                {song.valence !== undefined && (
                                    <div className="feature-item">
                                        <span className="feature-label">Valence</span>
                                        <div className="feature-bar">
                                            <div 
                                                className="feature-fill" 
                                                style={{ width: `${song.valence * 100}%` }}
                                            ></div>
                                        </div>
                                        <span className="feature-value">{(song.valence * 100).toFixed(0)}%</span>
                                    </div>
                                )}
                                {song.energy !== undefined && (
                                    <div className="feature-item">
                                        <span className="feature-label">Energy</span>
                                        <div className="feature-bar">
                                            <div 
                                                className="feature-fill" 
                                                style={{ width: `${song.energy * 100}%` }}
                                            ></div>
                                        </div>
                                        <span className="feature-value">{(song.energy * 100).toFixed(0)}%</span>
                                    </div>
                                )}
                                {song.danceability !== undefined && (
                                    <div className="feature-item">
                                        <span className="feature-label">Danceability</span>
                                        <div className="feature-bar">
                                            <div 
                                                className="feature-fill" 
                                                style={{ width: `${song.danceability * 100}%` }}
                                            ></div>
                                        </div>
                                        <span className="feature-value">{(song.danceability * 100).toFixed(0)}%</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Explanation */}
                    {song.explanation && (
                        <div className="modal-section">
                            <h3 className="modal-section-title">Why This Song?</h3>
                            <div className="modal-explanation">
                                <p>{song.explanation}</p>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

export default SongModal
