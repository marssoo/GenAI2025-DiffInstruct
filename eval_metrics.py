import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import polynomial_kernel


def preprocess_images(images):
    # make sure we have abatch
    if images.ndim == 3:  # Shape: (C, H, W)
        images = images.unsqueeze(0)

    # making sure 
    if images.ndim != 4:
        raise ValueError(f"Expected input tensor to have 4 dimensions (N, C, H, W), but got shape {images.shape}")

    # Convert grayscale to RGB by duplicating channels (yeah)
    if images.size(1) == 1:
        images = images.repeat(1, 3, 1, 1)

    # Resize to 299x299 (InceptionV3's input size)
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

    # Normalize to [-1, 1] (expected by InceptionV3)
    images = (images - 0.5) * 2

    return images


def extract_features(images, model):
    """Extract features from images using the InceptionV3 model."""
    images = preprocess_images(images)
    images = images.float()
    with torch.no_grad():
        features = model(images).cpu().numpy()
    return features

### Precision, Recall to assess quality
def compute_recall(real_features, fake_features, k=3):
    distances = pairwise_distances(fake_features, real_features, metric='euclidean')
    nearest_neighbors = np.argsort(distances, axis=1)[:, :k]
    recall = len(set(nearest_neighbors.flatten())) / len(real_features)
    return recall


def compute_precision(real_features, fake_features, k=3):
    distances = pairwise_distances(real_features, fake_features, metric='euclidean')
    nearest_neighbors = np.argsort(distances, axis=1)[:, :k]
    precision = len(set(nearest_neighbors.flatten())) / len(fake_features)
    return precision


def recall_score(real_images, fake_images, k=3, device='cuda'):
    real_images = real_images.to(torch.float32)
    fake_images = fake_images.to(torch.float32)
    # Load InceptionV3
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = torch.nn.Identity()  # Remove final classification layer
    inception.eval()

    # Extract features
    real_features = []
    fake_features = []
    for batch in real_images:
        real_features.append(extract_features(batch.to(device), inception))
    for batch in fake_images:
        fake_features.append(extract_features(batch.to(device), inception))

    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)

    # Compute Recall
    return compute_recall(real_features, fake_features, k=k)


def precision_recall(real_images, fake_images, k=3, device='cuda'):
    """
    compute Precision score for given real and fake images.
    """
    # Load pretrained InceptionV3 model
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = torch.nn.Identity()  # Remove final classification layer
    inception.eval()

    # Extract features for real and fake images
    real_features = []
    fake_features = []
    for batch in real_images:
        real_features.append(extract_features(batch.to(device), inception))
    for batch in fake_images:
        fake_features.append(extract_features(batch.to(device), inception))

    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)

    precision = compute_precision(real_features, fake_features, k=k)
    recall = compute_recall(real_features, fake_features, k=k)
    return precision, recall

###KID to assess spread
def compute_kid(real_features, fake_features, degree=3, coef0=1, gamma=None):
    if gamma is None:
        gamma = 1.0 / real_features.shape[1]

    real_kernel = polynomial_kernel(real_features, degree=degree, coef0=coef0, gamma=gamma)
    fake_kernel = polynomial_kernel(fake_features, degree=degree, coef0=coef0, gamma=gamma)
    cross_kernel = polynomial_kernel(real_features, fake_features, degree=degree, coef0=coef0, gamma=gamma)

    kid = np.mean(real_kernel) + np.mean(fake_kernel) - 2 * np.mean(cross_kernel)
    return kid

def kid_score(real_images, fake_images, device='cuda'):

    # Load pretrained InceptionV3 model
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = torch.nn.Identity()  # Remove final classification layer
    inception.eval()

    # Extract features for real and fake images
    real_features = []
    fake_features = []
    for batch in real_images:
        real_features.append(extract_features(batch.to(device), inception))
    for batch in fake_images:
        fake_features.append(extract_features(batch.to(device), inception))

    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)

    # Compute KID
    return compute_kid(real_features, fake_features)

if __name__ == "__main__":
    # Example usage
    real_images = torch.randn(10, 1, 28, 28)  # Replace with real images
    fake_images = torch.randn(10, 1, 28, 28)  # Replace with fake images

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    recall = recall_score([real_images], [fake_images], k=3, device=device)
    precision = precision_score([real_images], [fake_images], k=3, device=device)
    kid = kid_score([real_images], [fake_images], device=device)

    print(f"Recall Score: {recall}")
    print(f"Precision Score: {precision}")

