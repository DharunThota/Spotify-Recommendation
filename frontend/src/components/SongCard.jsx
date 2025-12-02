import React from 'react'
import { Calendar, TrendingUp, X } from 'lucide-react'
import './SongCard.css'

function SongCard({ song, showExplanation = false, onRemove = null, onClick = null }) {
    const handleClick = () => {
        if (onClick && !onRemove) {
            onClick(song)
        }
    }

    return (
        <div 
            className={`song-card ${onClick && !onRemove ? 'clickable' : ''}`}
            onClick={handleClick}
        >
            <div className="song-card-header">
                <div className="song-card-info">
                    <h3 className="song-card-title">{song.name}</h3>
                    <p className="song-card-artist">{song.artists}</p>
                    <div className="song-card-meta">
                        <span className="song-meta-item">
                            <Calendar size={14} />
                            <span className="meta-label">Year:</span> {song.year}
                        </span>
                        {song.popularity !== undefined && (
                            <span className="song-meta-item">
                                <TrendingUp size={14} />
                                <span className="meta-label">Popularity:</span> {song.popularity}
                            </span>
                        )}
                    </div>
                </div>
                {onRemove && (
                    <button 
                        className="remove-button" 
                        onClick={(e) => {
                            e.stopPropagation()
                            onRemove()
                        }}
                    >
                        <X size={16} />
                    </button>
                )}
            </div>
            
            {song.similarity_score !== undefined && (
                <div className="song-card-score">
                    <div className="score-bar">
                        <div 
                            className="score-fill" 
                            style={{ width: `${song.similarity_score * 100}%` }}
                        ></div>
                    </div>
                    <span className="score-text">
                        {(song.similarity_score * 100).toFixed(1)}% match
                    </span>
                </div>
            )}
        </div>
    )
}

export default SongCard
