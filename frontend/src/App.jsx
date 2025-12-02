import React, { useState, useEffect } from 'react'
import './App.css'
import Navigation from './components/Navigation'
import HomePage from './components/HomePage'
import AnalyticsPage from './components/AnalyticsPage'
import AboutPage from './components/AboutPage'
import Header from './components/Header'
import TabNavigation from './components/TabNavigation'
import SongBasedTab from './components/SongBasedTab'
import MoodBasedTab from './components/MoodBasedTab'
import HybridTab from './components/HybridTab'
import SequenceBasedTab from './components/SequenceBasedTab'

function App() {
  const [currentPage, setCurrentPage] = useState('home')
  const [activeTab, setActiveTab] = useState('song')

  const handleGetStarted = () => {
    setCurrentPage('app')
  }

  const handleNavigate = (page) => {
    console.log('Navigating to:', page)
    setCurrentPage(page)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  useEffect(() => {
    console.log('Current page changed to:', currentPage)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }, [currentPage])

  return (
    <>
      <Navigation currentPage={currentPage} onNavigate={handleNavigate} />
      
      <div className="app">
        {currentPage === 'home' ? (
          <HomePage onGetStarted={handleGetStarted} onNavigate={handleNavigate} />
        ) : currentPage === 'analytics' ? (
          <AnalyticsPage />
        ) : currentPage === 'about' ? (
          <AboutPage onNavigate={handleNavigate} />
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
                  {activeTab === 'sequence' && <SequenceBasedTab />}
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
