import torch

from src.config import Config

from src.data.preprocessing import process_data
from src.data.dataloaders import create_dataloaders

from src.models.energy_predictor import EnergyPredictor

from src.training.trainer import train_and_validate_model
from src.training.evaluator import evaluate_model

from src.utils.seed import set_seed
from src.utils.plotting import plot_losses

from src.utils.io import save_model, load_model, save_object

from src.inference.predictor import EnergyPredictorService
from src.utils.types import SongFeatures

def main() -> None:
    # --------------------------------------------------
    # Setup
    # --------------------------------------------------
    set_seed(Config.RANDOM_STATE)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    processed_data = process_data()
    data = processed_data.data
    train_loader, test_loader = create_dataloaders(
        data.X_train,
        data.X_test,
        data.y_train,
        data.y_test,
        batch_size=Config.BATCH_SIZE,
        subset_size=Config.SUBSET_SIZE,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = EnergyPredictor(
        input_size=Config.INPUT_SIZE,
        hidden_size=Config.HIDDEN_SIZE,
        use_relu=Config.USE_RELU,
    )
    print(model)

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    training_result = train_and_validate_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=Config.EPOCHS,
        lr=Config.LEARNING_RATE,
        device=device,
    )

    # --------------------------------------------------
    # Save Objects
    # --------------------------------------------------
    save_model(
        training_result.model,
        Config.MODEL_PATH,
    )
    save_object(
        processed_data.scaler,
        Config.SCALER_PATH,
    )
    save_object(
        processed_data.encoder,
        Config.ENCODER_PATH,
    )

    # --------------------------------------------------
    # Load Model
    # --------------------------------------------------
    loaded_model = load_model(
        EnergyPredictor(
            hidden_size=Config.HIDDEN_SIZE,
            use_relu=Config.USE_RELU,
        ),
        Config.MODEL_PATH,
        device,
    )
    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------
    evaluation = evaluate_model(
        # training_result.model,
        loaded_model,
        test_loader,
        device,
    )
    print("\n========== Final Evaluation ==========")
    print(f"MSE : {evaluation.mse:.6f}")
    print(f"MAE : {evaluation.mae:.6f}")
    print(f"R²  : {evaluation.r2:.4f}")

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plot_losses(
        training_result.train_losses,
        training_result.val_losses,
    )

    # --------------------------------------------------
    # User Sample Prediction
    # --------------------------------------------------
    predictor = EnergyPredictorService(
        training_result.model, 
        processed_data.scaler, 
        processed_data.encoder, 
        device
        )
    
    song = get_song_from_user()
    prediction = predictor.predict(song)
    print(f"\nPredicted Energy: {prediction:.3f}")

def get_song_from_user() -> SongFeatures:
        return SongFeatures(
            danceability = 0.81,
            loudness = -4.5,
            key = "C",
            mode = "Major",
            speechiness = 0.3150,
            acousticness = 0.00740,
            instrumentalness = 0.838000,
            liveness = 0.4790,
            valence = 0.000,
            tempo = 123.588,
            duration_ms = 79500.0
        )
        # return SongFeatures(
        #     danceability=float(input("Danceability: ")),
        #     loudness=float(input("Loudness: ")),
        #     key=input("Key: "),
        #     mode=input("Mode (Major/Minor): "),
        #     speechiness=float(input("Speechiness: ")),
        #     acousticness=float(input("Acousticness: ")),
        #     instrumentalness=float(input("Instrumentalness: ")),
        #     liveness=float(input("Liveness: ")),
        #     valence=float(input("Valence: ")),
        #     tempo=float(input("Tempo: ")),
        #     duration_ms=float(input("Duration(ms): "))
        # )

if __name__ == "__main__":
    main()
