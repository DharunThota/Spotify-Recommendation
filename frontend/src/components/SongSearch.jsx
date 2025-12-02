import React, { useState, useEffect, useRef } from 'react'
import { searchSongs } from '../services/api'
import './SongSearch.css'

function SongSearch({ onSongSelect, placeholder = "Search for a song..." }) {
    const [query, setQuery] = useState('')
    const [results, setResults] = useState([])
    const [isSearching, setIsSearching] = useState(false)
    const [showResults, setShowResults] = useState(false)
    const searchRef = useRef(null)
    const timeoutRef = useRef(null)

    useEffect(() => {
        // Close results when clicking outside
        const handleClickOutside = (event) => {
            if (searchRef.current && !searchRef.current.contains(event.target)) {
                setShowResults(false)
            }
        }

        document.addEventListener('mousedown', handleClickOutside)
        return () => document.removeEventListener('mousedown', handleClickOutside)
    }, [])

    const handleSearch = async (searchQuery) => {
        if (searchQuery.length < 2) {
            setResults([])
            setShowResults(false)
            return
        }

        setIsSearching(true)
        try {
            const data = await searchSongs(searchQuery, 10)
            setResults(data)
            setShowResults(true)
        } catch (error) {
            console.error('Search error:', error)
            setResults([])
        } finally {
            setIsSearching(false)
        }
    }

    const handleInputChange = (e) => {
        const value = e.target.value
        setQuery(value)

        // Debounce search
        if (timeoutRef.current) {
            clearTimeout(timeoutRef.current)
        }

        timeoutRef.current = setTimeout(() => {
            handleSearch(value)
        }, 300)
    }

    const handleSelectSong = (song) => {
        onSongSelect(song)
        setQuery('')
        setResults([])
        setShowResults(false)
    }

    return (
        <div className="song-search" ref={searchRef}>
            <div className="search-input-wrapper">
                <input
                    type="text"
                    className="search-input"
                    placeholder={placeholder}
                    value={query}
                    onChange={handleInputChange}
                    onFocus={() => results.length > 0 && setShowResults(true)}
                />
                {isSearching && <div className="search-loader"></div>}
            </div>

            {showResults && results.length > 0 && (
                <div className="search-results">
                    {results.map(song => (
                        <div
                            key={song.id}
                            className="search-result-item"
                            onClick={() => handleSelectSong(song)}
                        >
                            <div className="song-info">
                                <div className="song-name">{song.name}</div>
                                <div className="song-artist">{song.artists}</div>
                            </div>
                            <div className="song-year">{song.year}</div>
                        </div>
                    ))}
                </div>
            )}

            {showResults && results.length === 0 && query.length >= 2 && !isSearching && (
                <div className="search-results">
                    <div className="no-results">No songs found</div>
                </div>
            )}
        </div>
    )
}

export default SongSearch
