""" Este módulo contiene funciones para operar con Grad-CAM """
# Librerías externas
from torchcam.methods import GradCAM

# Inicializar Grad-CAM 
def initialize_gradcam(model, model_choice):
    # """Inicializa GradCAM para la capa deseada del modelo según el tipo"""
    if model_choice == "CNN_3C":
        target_layer = "conv3"  # Última capa convolucional del modelo 3C
    else:
        target_layer = "conv4"  # Última capa convolucional del modelo 4C

    return GradCAM(model, target_layer=target_layer)

# Limpiar los hooks 
def clear_gradcam_hooks(model):
    # """ Limpia los hooks registrados por GradCAM para evitar fugas de memoria"""
    for module in model.modules():
        if hasattr(module, 'registered_hooks'):
            for hook in module.registered_hooks:
                hook.remove()