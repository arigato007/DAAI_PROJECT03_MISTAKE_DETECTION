import argparse
import datetime
import os
import numpy as np
import torch
import torchvision.transforms as T
import concurrent.futures
import cv2
import logging
from perception_models_main.core.vision_encoder import pe
from tqdm import tqdm
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo
from PIL import Image
from natsort import natsorted

log_directory = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_file_path = os.path.join(log_directory, f"std.log")
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

logger = logging.getLogger(__name__)

# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for processing methods.")
    parser.add_argument("--backbone", type=str, default="Perception", help="Specify the method to be used.")
    return parser.parse_args()

# Feature Extraction
def extract_features(batched_image_data, feature_extractor, method):
    # Perception/CLIP returns (image_features, text_features, logit_scale). Use image_features.
    with torch.no_grad():
        image_features, text_features, logit_scale = feature_extractor(batched_image_data)
    return image_features.cpu().numpy()

# Video Processing
class ImageProcessor:
    def __init__(self, method, feature_extractor, image_transform):
        self.method = method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor
        self.image_transform = image_transform

    def process_video(self, video_file_path, output_features_path):
        """Decode an mp4 on-the-fly and extract features."""
        video_name = os.path.splitext(os.path.basename(video_file_path))[0]
        output_file_path = os.path.join(output_features_path, video_name)
        os.makedirs(output_features_path, exist_ok=True)

        batch_size = 600
        cap = cv2.VideoCapture(video_file_path)
        batch_images = []
        video_features = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            input_tensor = self.image_transform(img).to(self.device)
            batch_images.append(input_tensor)

            if len(batch_images) == batch_size:
                batch_tensor = torch.stack(batch_images)
                batch_features = extract_features(
                    batched_image_data=batch_tensor,
                    feature_extractor=self.feature_extractor,
                    method=self.method,
                )
                video_features.append(batch_features)
                batch_images = []

        if batch_images:
            batch_tensor = torch.stack(batch_images)
            batch_features = extract_features(
                batched_image_data=batch_tensor,
                feature_extractor=self.feature_extractor,
                method=self.method,
            )
            video_features.append(batch_features)

        cap.release()

        video_features = np.vstack(video_features)
        np.savez(f"{output_file_path}.npz", video_features)
        logger.info(f"Saved features for video {video_name} at {output_file_path}.npz")



def get_image_transformation(model_name):
    model_name = model_name.lower()
    if model_name == "perception":
        image_transform = T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return image_transform


def get_feature_extractor(name, device="cuda"):
    name = name.lower()
    if name == "perception":
        model_name = "PE-Core-B16-224"
        model = pe.CLIP.from_config(model_name, pretrained=True)
        model = model.to(device).eval()
        return model


# Main
def main():
    args = parse_arguments()
    method = args.backbone.lower()

    video_frames_directories_path = "/content/drive/MyDrive/Mistake_Detection/captain_cook_4d_gopro_resized"
    output_features_path = f"/content/drive/MyDrive/Mistake_Detection/features/video/perception"

    image_transform = get_image_transformation(method)
    feature_extractor = get_feature_extractor(method)

    processor = ImageProcessor(method, feature_extractor, image_transform)

    mp4_files = [f for f in os.listdir(video_frames_directories_path) if f.lower().endswith(".mp4")]
    
    for mp4_file in tqdm(mp4_files, desc="Processing mp4 videos"):
        processor.process_video(os.path.join(video_frames_directories_path, mp4_file), output_features_path)

if __name__ == "__main__":
    main()
