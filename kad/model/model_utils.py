from kad.model import i_model, lstm_model, sarima_model, autoencoder_model, hmm_model
from kad.ts_analyzer import ts_analyzer


def name_2_model(model_name: str, model_selector: ts_analyzer.TsAnalyzer) -> i_model.IModel:
    if model_name == "LstmModel":
        return lstm_model.LstmModel()
    elif model_name == "HmmModel":
        return hmm_model.HmmModel()
    elif model_name == "SarimaModel":
        return sarima_model.SarimaModel(order=(0, 0, 0),
                                        seasonal_order=(
                                            1, 0, 1,
                                            model_selector.calculate_dominant_frequency()))
    elif model_name == "AutoEncoderModel":
        return autoencoder_model.AutoEncoderModel()
