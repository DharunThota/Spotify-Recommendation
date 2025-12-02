import React from 'react'
import './TabNavigation.css'

function TabNavigation({ activeTab, onTabChange }) {
    const tabs = [
        { id: 'song', label: 'Song-Based', icon: '🎵' },
        { id: 'mood', label: 'Mood-Based', icon: '😊' },
        { id: 'hybrid', label: 'Hybrid', icon: '🎭' },
        { id: 'sequence', label: 'Smart Patterns', icon: '🔮' }
    ]

    return (
        <div className="tab-navigation">
            {tabs.map(tab => (
                <button
                    key={tab.id}
                    className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
                    onClick={() => onTabChange(tab.id)}
                >
                    <span className="tab-icon">{tab.icon}</span>
                    <span className="tab-label">{tab.label}</span>
                </button>
            ))}
        </div>
    )
}

export default TabNavigation
