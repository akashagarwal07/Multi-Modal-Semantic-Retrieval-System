import torch
import timm
from PIL import Image
from torchvision import transforms

class FigureClassifier:
    def __init__(self):
        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=2   # figure vs non-figure
        )

        # Load weights (download your trained weights here)
        self.model.load_state_dict(torch.load("chart_classifier.pth", map_location="cpu"))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def is_figure(self, img_path):
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x)
            pred = torch.argmax(logits, dim=1).item()

        return pred == 1   # class 1 = figure
