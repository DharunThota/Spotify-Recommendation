// Global state
let selectedSongId = null;
let selectedMood = null;
let hybridSelectedSongs = [];
let searchTimeout = null;

// API base URL
const API_BASE = '';

// Tab switching
function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active from all buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    
    // Activate button
    event.target.classList.add('active');
    
    // Clear previous results
    clearRecommendations();
}

// Search songs (for song-based)
function searchSongs(event) {
    const query = event.target.value.trim();
    
    if (query.length < 2) {
        hideSearchResults();
        return;
    }
    
    // Debounce search
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
        performSearch(query, 'search-results', selectSong);
    }, 300);
}

// Search songs (for hybrid)
function searchSongsHybrid(event) {
    const query = event.target.value.trim();
    
    if (query.length < 2) {
        document.getElementById('hybrid-search-results').classList.remove('show');
        return;
    }
    
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
        performSearch(query, 'hybrid-search-results', addSongToHybrid);
    }, 300);
}

// Perform search API call
async function performSearch(query, resultsElementId, onSelectCallback) {
    try {
        const response = await fetch(`${API_BASE}/api/search?query=${encodeURIComponent(query)}&limit=10`);
        const results = await response.json();
        
        displaySearchResults(results, resultsElementId, onSelectCallback);
    } catch (error) {
        console.error('Search error:', error);
    }
}

// Display search results
function displaySearchResults(results, elementId, onSelectCallback) {
    const resultsDiv = document.getElementById(elementId);
    
    if (results.length === 0) {
        resultsDiv.innerHTML = '<div class="search-result-item">No results found</div>';
        resultsDiv.classList.add('show');
        return;
    }
    
    // Clear previous content
    resultsDiv.innerHTML = '';
    
    // Create and add each result
    results.forEach(song => {
        const div = document.createElement('div');
        div.className = 'search-result-item';
        div.innerHTML = `
            <div class="song-name">${escapeHtml(song.name)}</div>
            <div class="song-artist">${escapeHtml(song.artists)}</div>
            <div class="song-meta">Year: ${song.year} | Popularity: ${song.popularity}</div>
        `;
        
        // Add click event listener
        div.addEventListener('click', () => onSelectCallback(song));
        
        resultsDiv.appendChild(div);
    });
    
    resultsDiv.classList.add('show');
}

// Hide search results
function hideSearchResults() {
    document.querySelectorAll('.search-results').forEach(div => {
        div.classList.remove('show');
    });
}

// Select song for song-based recommendations
function selectSong(song) {
    selectedSongId = song.id;
    
    document.getElementById('selected-song').innerHTML = `
        <div class="song-card">
            <div class="song-info">
                <h3>${escapeHtml(song.name)}</h3>
                <p>${escapeHtml(song.artists)} • ${song.year}</p>
            </div>
        </div>
    `;
    
    hideSearchResults();
    document.getElementById('song-search').value = '';
}

// Add song to hybrid selection
function addSongToHybrid(song) {
    // Check if already added
    if (hybridSelectedSongs.find(s => s.id === song.id)) {
        return;
    }
    
    hybridSelectedSongs.push(song);
    displayHybridSelectedSongs();
    
    document.getElementById('hybrid-search').value = '';
    document.getElementById('hybrid-search-results').classList.remove('show');
}

// Display hybrid selected songs
function displayHybridSelectedSongs() {
    const container = document.getElementById('hybrid-selected-songs');
    
    if (hybridSelectedSongs.length === 0) {
        container.innerHTML = '';
        return;
    }
    
    container.innerHTML = hybridSelectedSongs.map((song, index) => `
        <div class="song-card">
            <div class="song-info">
                <h3>${escapeHtml(song.name)}</h3>
                <p>${escapeHtml(song.artists)} • ${song.year}</p>
            </div>
            <button class="remove-btn" onclick="removeSongFromHybrid(${index})">✕</button>
        </div>
    `).join('');
}

// Remove song from hybrid selection
function removeSongFromHybrid(index) {
    hybridSelectedSongs.splice(index, 1);
    displayHybridSelectedSongs();
}

// Select mood
function selectMood(mood) {
    selectedMood = mood;
    
    document.querySelectorAll('.mood-btn').forEach(btn => {
        btn.classList.remove('selected');
    });
    
    event.target.classList.add('selected');
}

// Get song-based recommendations
async function getSongRecommendations() {
    if (!selectedSongId) {
        alert('Please select a song first');
        return;
    }
    
    const nRecs = parseInt(document.getElementById('song-n-recs').value);
    const diversity = parseFloat(document.getElementById('song-diversity').value);
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE}/api/recommend/song`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                song_id: selectedSongId,
                n_recommendations: nRecs,
                diversity_weight: diversity
            })
        });
        
        const data = await response.json();
        displayRecommendations(data, 'song-recommendations');
    } catch (error) {
        console.error('Recommendation error:', error);
        alert('Failed to get recommendations. Please try again.');
    } finally {
        hideLoading();
    }
}

// Get mood-based recommendations
async function getMoodRecommendations() {
    if (!selectedMood) {
        alert('Please select a mood first');
        return;
    }
    
    const nRecs = parseInt(document.getElementById('mood-n-recs').value);
    const includePopular = document.getElementById('mood-popular').checked;
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE}/api/recommend/mood`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mood: selectedMood,
                n_recommendations: nRecs,
                include_popular: includePopular
            })
        });
        
        const data = await response.json();
        displayRecommendations(data, 'mood-recommendations');
    } catch (error) {
        console.error('Recommendation error:', error);
        alert('Failed to get recommendations. Please try again.');
    } finally {
        hideLoading();
    }
}

// Get hybrid recommendations
async function getHybridRecommendations() {
    if (hybridSelectedSongs.length === 0) {
        alert('Please add at least one song');
        return;
    }
    
    const nRecs = parseInt(document.getElementById('hybrid-n-recs').value);
    const mood = document.getElementById('hybrid-mood').value || null;
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE}/api/recommend/hybrid`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                song_ids: hybridSelectedSongs.map(s => s.id),
                mood: mood,
                n_recommendations: nRecs
            })
        });
        
        const data = await response.json();
        displayRecommendations(data, 'hybrid-recommendations');
    } catch (error) {
        console.error('Recommendation error:', error);
        alert('Failed to get recommendations. Please try again.');
    } finally {
        hideLoading();
    }
}

// Display recommendations
function displayRecommendations(data, containerId) {
    const container = document.getElementById(containerId);
    
    if (!data.recommendations || data.recommendations.length === 0) {
        container.innerHTML = '<p>No recommendations found.</p>';
        return;
    }
    
    container.innerHTML = data.recommendations.map((rec, index) => {
        const song = rec.song;
        const explanation = rec.explanation;
        
        return `
            <div class="recommendation-item">
                <div class="rec-header">
                    <div class="rec-song-info">
                        <h3>${index + 1}. ${escapeHtml(song.name)}</h3>
                        <p>${escapeHtml(song.artists)} • ${song.year} • Popularity: ${song.popularity}</p>
                        <div class="feature-tags">
                            <span class="feature-tag">Valence: ${(song.valence * 100).toFixed(0)}%</span>
                            <span class="feature-tag">Energy: ${(song.energy * 100).toFixed(0)}%</span>
                            <span class="feature-tag">Danceability: ${(song.danceability * 100).toFixed(0)}%</span>
                        </div>
                    </div>
                    <div class="rec-score">${(rec.score * 100).toFixed(0)}%</div>
                </div>
                ${explanation ? `
                    <div class="explanation">
                        <div class="explanation-title">💡 Why this recommendation?</div>
                        <p>${explanation.explanation}</p>
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');
}

// Clear recommendations
function clearRecommendations() {
    document.getElementById('song-recommendations').innerHTML = '';
    document.getElementById('mood-recommendations').innerHTML = '';
    document.getElementById('hybrid-recommendations').innerHTML = '';
}

// Show loading
function showLoading() {
    document.getElementById('loading').classList.add('show');
}

// Hide loading
function hideLoading() {
    document.getElementById('loading').classList.remove('show');
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Close search results when clicking outside
document.addEventListener('click', (event) => {
    if (!event.target.closest('.search-box')) {
        hideSearchResults();
    }
});

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('Spotify Recommendation System loaded');
});
