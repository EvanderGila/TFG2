import streamlit as st
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image
from src.models import CNN_3C  


# Configuración inicial
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align: center;'>Comparativa de Grad-CAMs</h1>", unsafe_allow_html=True)

# Clase GradCAM personalizada 
class GradCAM_Sigmoid:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(self._forward_hook)
                module.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output.squeeze()
        score.backward(retain_graph=True)

        pooled_gradients = self.gradients.mean(dim=(2, 3), keepdim=True)
        weighted_activations = pooled_gradients * self.activations
        cam = weighted_activations.sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=(64, 64), mode='bilinear', align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.squeeze()

# Carga de modelos (función)
@st.cache_resource
def load_model_CNN3C():
    model = CNN_3C(64, 128, 256, 64)
    model.load_state_dict(torch.load("model/model3C.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# Preprocesamiento (preprocess)
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4718, 0.4628, 0.4176], std=[0.2361, 0.2360, 0.2636])
])

# Carga de la imagen
uploaded_image = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
if not uploaded_image:
    st.stop()

image = Image.open(uploaded_image).convert("RGB")
#Preprocesado de la imagen
input_tensor = preprocess(image).unsqueeze(0)

# Carga de modelos (uno para cada tipo de Grad-CAM)
model_torchcam = load_model_CNN3C()
model_sigmoid = load_model_CNN3C()

# Inicializar Grad-CAMs antes del forward
cam_torch = GradCAM(model_torchcam, target_layer="conv3")
cam_sig = GradCAM_Sigmoid(model_sigmoid, target_layer_name="conv3")

# Forward después de hooks 
output_torchcam = model_torchcam(input_tensor)
output_sigmoid = model_sigmoid(input_tensor)  # ya incluido en cam_sig.__call__

# Calcular mapas
activation_map_torchcam = cam_torch(0, output_torchcam)
activation_map_sigmoid = cam_sig(input_tensor)

# Redimensionar imagen base
resized_img = transforms.Resize((64, 64))(image)

# Creación de columnas
col1, col2 = st.columns(2)

# Visualización
with col1:
    st.markdown("### 🔵 Grad-CAM (torchcam)")
    heatmap_torchcam = overlay_mask(resized_img, to_pil_image(activation_map_torchcam[0].detach(), mode='F'), alpha=0.5)
    st.image(heatmap_torchcam, width=700)

with col2:
    st.markdown("### 🟢 Grad-CAM personalizado (Sigmoid)")
    heatmap_sigmoid = overlay_mask(resized_img, to_pil_image(activation_map_sigmoid.detach(), mode='F'), alpha=0.5)
    st.image(heatmap_sigmoid, width=700)
