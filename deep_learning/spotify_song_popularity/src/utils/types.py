from dataclasses import dataclass

@dataclass
class SongFeatures:
    danceability: float
    loudness: float
    key: str
    mode: str
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    duration_ms: float