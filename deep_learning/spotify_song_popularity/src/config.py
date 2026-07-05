class Config():
    DEVICE = 'cuda'
    # training
    EPOCHS = 20
    LEARNING_RATE = 0.001
    # data
    BATCH_SIZE = 128
    SUBSET_SIZE = 50000
    TARGET_COLUMN = "energy"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    DATA_PATH = "data/songs_With_attributes_and_lyrics.csv"
    CATEGORICAL_FEATURES = ["key", "mode"]
    NUMERICAL_FEATURES = [
        "danceability",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms"
    ]
    # model
    INPUT_SIZE = 51
    HIDDEN_SIZE = 32
    USE_RELU = True
    MODEL_PATH = "artifacts/energy_predictor.pth"
    ARTIFACT_DIR = "artifacts"
    SCALER_PATH = "artifacts/scaler.pkl"
    ENCODER_PATH = "artifacts/encoder.pkl"
