from transformers import AutoModel, AutoFeatureExtractor

model_name = "microsoft/resnet-50"

model = AutoModel.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

model.save_pretrained("./my_local_model")
feature_extractor.save_pretrained("./my_local_model")

print("Model and feature extractor saved successfully!")
