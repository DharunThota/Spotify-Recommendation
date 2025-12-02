import React, { useState } from 'react'
import './App.css'
import Header from './components/Header'
import TabNavigation from './components/TabNavigation'
import SongBasedTab from './components/SongBasedTab'
import MoodBasedTab from './components/MoodBasedTab'
import HybridTab from './components/HybridTab'

function App() {
  const [activeTab, setActiveTab] = useState('song')

  return (
    <div className="app">
      <Header />
      <div className="container">
        <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />
        
        <div className="tab-content">
          {activeTab === 'song' && <SongBasedTab />}
          {activeTab === 'mood' && <MoodBasedTab />}
          {activeTab === 'hybrid' && <HybridTab />}
        </div>
      </div>
    </div>
  )
}

export default App
