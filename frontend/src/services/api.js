import axios from 'axios'

const API_BASE = 'http://localhost:8000'

const api = axios.create({
    baseURL: API_BASE,
    headers: {
        'Content-Type': 'application/json',
    },
})

export const searchSongs = async (query, limit = 10) => {
    const response = await api.get('/api/search', {
        params: { query, limit }
    })
    return response.data
}

export const getSongRecommendations = async (songId, numRecommendations = 10) => {
    const response = await api.post('/api/recommend/song', {
        song_id: songId,
        n_recommendations: numRecommendations
    })
    return response.data
}

export const getMoodRecommendations = async (mood, numRecommendations = 10) => {
    const response = await api.post('/api/recommend/mood', {
        mood,
        n_recommendations: numRecommendations
    })
    return response.data
}

export const getHybridRecommendations = async (songIds, mood = null, numRecommendations = 10) => {
    const response = await api.post('/api/recommend/hybrid', {
        song_ids: songIds,
        mood,
        n_recommendations: numRecommendations
    })
    return response.data
}

export const getSequenceAwareRecommendations = async (songId, recentContext = [], sequenceWeight = 0.3, numRecommendations = 10) => {
    const response = await api.post('/api/recommend/sequence-aware', {
        song_id: songId,
        recent_context: recentContext,
        sequence_weight: sequenceWeight,
        n_recommendations: numRecommendations
    })
    return response.data
}

// Analytics API calls
export const getWrappedInsights = async (timeRange = 'medium_term') => {
    const response = await api.get('/api/analytics/wrapped-insights', {
        params: { time_range: timeRange }
    })
    return response.data
}

export default api
