import React, { useState } from 'react'
import './App.css'
import Navigation from './components/Navigation'
import HomePage from './components/HomePage'
import Header from './components/Header'
import TabNavigation from './components/TabNavigation'
import SongBasedTab from './components/SongBasedTab'
import MoodBasedTab from './components/MoodBasedTab'
import HybridTab from './components/HybridTab'

function App() {
  const [currentPage, setCurrentPage] = useState('home')
  const [activeTab, setActiveTab] = useState('song')

  const handleGetStarted = () => {
    setCurrentPage('app')
  }

  return (
    <>
      <Navigation currentPage={currentPage} onNavigate={setCurrentPage} />
      
      <div className="app">
        {currentPage === 'home' ? (
          <HomePage onGetStarted={handleGetStarted} />
        ) : (
          <div className="container">
            <Header />
            <div className="discover-layout">
              <div className="discover-sidebar">
                <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />
              </div>
              
              <div className="discover-content">
                <div className="tab-content">
                  {activeTab === 'song' && <SongBasedTab />}
                  {activeTab === 'mood' && <MoodBasedTab />}
                  {activeTab === 'hybrid' && <HybridTab />}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  )
}

export default App
