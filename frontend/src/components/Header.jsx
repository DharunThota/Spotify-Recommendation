import React from 'react'
import './Header.css'

function Header() {
    return (
        <header className="header">
            <div className="header-content">
                <h1 className="header-title">
                    Spotify Recommendation System
                </h1>
                <p className="header-subtitle">
                    Discover music tailored to your taste using advanced machine learning algorithms
                </p>
            </div>
        </header>
    )
}

export default Header
