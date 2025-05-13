import torch.nn as nn
import torch.nn.functional as F


class CNN_3C(nn.Module):
   
    def __init__(self, out_1, out_2, out_3, image_size):
        super(CNN_3C, self).__init__()
        
        self.out_1 = 64  #Salida 1, normalización y entrada 2
        self.out_2 = 128 # Salida 2, normalización y entrada 3
        self.out_3 = 256 # Salida 3 y normalización
        self.image_size = int(64/ (2 ** 3)) #Tamaño nuevo después de 3 poolings
        
        # Definir las capas de la CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= out_1, kernel_size=3, padding=1)  #3 to 64, kernel_size=3 == kernel_size=(3, 3), padding='same' daría como valor 1
        self.bn1 = nn.BatchNorm2d(out_1)  # Normaliza las salidas de conv1
        self.conv2 = nn.Conv2d(in_channels= out_1, out_channels= out_2, kernel_size=3, padding=1)  #64 to 128, kernel_size=3 == kernel_size=(3, 3), padding='same' daría como valor 1
        self.bn2 = nn.BatchNorm2d(out_2)  # Normaliza las salidas de conv2
        self.conv3 = nn.Conv2d(in_channels=out_2, out_channels= out_3, kernel_size=3, padding=1)  #128 to 256, kernel_size=3 == kernel_size=(3, 3), padding='same' daría como valor 1
        self.bn3 = nn.BatchNorm2d(out_3)  # Normaliza las salidas de conv3
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Kernel=2 indica que su ventana de pooling es de 2x2 y stride=2 indica que se moverá 2 píxeles evitando solapamiento
        
        
        #FULLY CONNECTED LAYERS
        self.fc1 = nn.Linear(out_3 * self.image_size * self.image_size, 512)   # (256 * 8 * 8) 256 es el output de la última capa, y 8 * 8 son las dimensiones de la imagen al hacer un maxpool 3 veces (64x64 -> 8x8)
        self.fc2 = nn.Linear(512, 1)  # 512 neuronas y solo una neurona final a la que aplicaremos la función sigmoid para que deje una probabilidad normalizada entre 0 y 1 
        
        
    def forward(self, x):
        # Aplicar las capas de convolución, activaciones(Relu) y pooling 
        # Por orden, se aplica la capa convolucional, se normalizan las activaciones de la capa, se aplica la función Relu y se hace un pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (batch_size, 64, 64) -> (batch_size, 64, 32, 32)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (batch_size, 128, 32, 32) -> (batch_size, 128, 16, 16)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # (batch_size, 256, 16, 16) -> (batch_size, 256, 8, 8)

        #FULLY CONNECTED LAYERS
 
        # Aplanar el tensor 
        x = x.view(-1, self.out_3 * self.image_size * self.image_size)  # (batch_size, 256 * 8 * 8)
        
        # Capas totalmente conectadas
        x = F.relu(self.fc1(x))  # (batch_size, 512)
        x = F.dropout(x, p=0.5, training=self.training) # 50% de desactivación aleatoria 
        x = self.fc2(x)

        return x    

class CNN_4C(nn.Module):
   
    def __init__(self, out_1, out_2, out_3, out_4, image_size):
        super(CNN_4C, self).__init__()
        
        
        self.out_1 = 64 #Salida 1, normalización y entrada 2
        self.out_2 = 128 # Salida 2, normalización y entrada 3
        self.out_3 = 256 # Salida 3, normalización y entrada 4
        self.out_4 = 512 # Salida 3 y normalización
        self.image_size = int(64/ (2 ** 4)) #Tamaño nuevo después de 4 poolings
       
        
        # Definir las capas de la CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= out_1, kernel_size=3, padding=1)  #3 to 64, kernel_size=3 == kernel_size=(3, 3), padding='same' daría como valor 1
        self.bn1 = nn.BatchNorm2d(out_1)  # Normaliza las salidas de conv1
        self.conv2 = nn.Conv2d(in_channels= out_1, out_channels= out_2, kernel_size=3, padding=1)  #64 to 128, kernel_size=3 == kernel_size=(3, 3), padding='same' daría como valor 1
        self.bn2 = nn.BatchNorm2d(out_2)  # Normaliza las salidas de conv2
        self.conv3 = nn.Conv2d(in_channels=out_2, out_channels= out_3, kernel_size=3, padding=1)  #128 to 256, kernel_size=3 == kernel_size=(3, 3), padding='same' daría como valor 1
        self.bn3 = nn.BatchNorm2d(out_3)  # Normaliza las salidas de conv3
        self.conv4 = nn.Conv2d(in_channels=out_3, out_channels= out_4, kernel_size=3, padding=1)  #256 to 512, kernel_size=3 == kernel_size=(3, 3), padding='same' daría como valor 1
        self.bn4 = nn.BatchNorm2d(out_4)  # Normaliza las salidas de conv4
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Kernel=2 indica que su ventana de pooling es de 2x2 y stride=2 indica que se moverá 2 píxeles evitando solapamiento
        
        
        #FULLY CONNECTED LAYERS
        self.fc1 = nn.Linear(out_4* self.image_size * self.image_size, 512)   # (512 * 4 * 4) 512 es el output de la última capa, y 4 * 4 son las dimensiones de la imagen al hacer un maxpool 4 veces (64x64 -> 4x4)
        self.fc2 = nn.Linear(512, 1)  # 512 neuronas y solo una neurona final a la que aplicaremos la función sigmoid para que deje una probabilidad normalizada entre 0 y 1 
        
        
    def forward(self, x):
        # Aplicar las capas de convolución, activaciones(Relu) y pooling 
        # Por orden, se aplica la capa convolucional, se normalizan las activaciones de la capa, se aplica la función Relu y se hace un pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (batch_size, 64, 64) -> (batch_size, 64, 32, 32)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (batch_size, 128, 32, 32) -> (batch_size, 128, 16, 16)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # (batch_size, 256, 16, 16) -> (batch_size, 256, 8, 8)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # (batch_size, 512, 8, 8) -> (batch_size, 512, 4, 4) 
        #FULLY CONNECTED LAYERS
        
        # Aplanar el tensor 
        x = x.view(-1, self.out_4* self.image_size * self.image_size)  # (batch_size, 512 * 4 * 4)
        
        # Capas totalmente conectadas
        x = F.relu(self.fc1(x))  # (batch_size, 512)
        x = F.dropout(x, p=0.5, training=self.training) # 50% de desactivación aleatoria 
        x = self.fc2(x)
       

        return x