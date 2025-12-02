# Spotify Recommendation System - React Frontend

Modern React frontend for the Spotify Recommendation System.

## Features

- 🎵 **Song-Based Recommendations**: Search and select songs to get similar recommendations
- 😊 **Mood-Based Recommendations**: Choose your mood and discover matching songs
- 🎭 **Hybrid Recommendations**: Combine multiple songs with optional mood filtering
- ✨ **Modern UI**: Clean, responsive design with smooth animations
- 🚀 **Fast Performance**: Built with Vite for optimal development and production builds

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **Axios** - HTTP client for API calls
- **CSS3** - Styling with animations and gradients

## Getting Started

### Prerequisites

- Node.js 16+ and npm
- Backend server running on http://localhost:8000

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at http://localhost:3000

### Build for Production

```bash
npm run build
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/       # React components
│   │   ├── Header.jsx
│   │   ├── TabNavigation.jsx
│   │   ├── SongSearch.jsx
│   │   ├── SongCard.jsx
│   │   ├── SongBasedTab.jsx
│   │   ├── MoodBasedTab.jsx
│   │   └── HybridTab.jsx
│   ├── services/        # API service layer
│   │   └── api.js
│   ├── App.jsx          # Main app component
│   ├── App.css
│   ├── main.jsx         # Entry point
│   └── index.css        # Global styles
├── index.html
├── vite.config.js
└── package.json
```

## API Integration

The frontend communicates with the FastAPI backend through:
- `/api/search` - Search songs
- `/api/recommend/song` - Get song-based recommendations
- `/api/recommend/mood` - Get mood-based recommendations
- `/api/recommend/hybrid` - Get hybrid recommendations

## Development

- Hot module replacement enabled for instant updates
- Component-based architecture for maintainability
- Responsive design for mobile and desktop
- Error handling and loading states

## License

MIT
