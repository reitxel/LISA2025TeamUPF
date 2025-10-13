import torch
import torchvision.transforms as T
from PIL import Image
from models.densenet import DenseNetQA
from utils.checkpoint import load_checkpoint

def infer(image_path, model_checkpoint_path, artefact_domain, device, input_size=224):
    model = DenseNetQA(num_classes=3).to(device)
    start_epoch, best_metric = load_checkpoint(model_checkpoint_path, model)
    model.eval()

    image = Image.open(image_path).convert('L')
    
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    class_map = {0: "Good", 1: "Moderate", 2: "Bad"}
    print(f"Image: {image_path}")
    print(f"Artefact Domain: {artefact_domain}")
    print(f"Predicted Quality: {class_map[predicted_class]} (Class {predicted_class})")
    print(f"Probabilities: {probabilities.cpu().numpy().flatten().tolist()}")
    return predicted_class, probabilities.cpu().numpy().flatten().tolist() 