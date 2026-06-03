# ESP32 Detector de Personas con Cámara + Servidor

Sistema de detección de movimiento con cámara ESP32 que envía fotos a un servidor web cuando detecta personas.

## Características

- Detección de movimiento con sensor **PIR**
- Captura de fotos en **ráfaga** (3 imágenes) al detectar movimiento
- Envío automático de imágenes en **Base64** a servidor
- **Keep-alive** cada 13 minutos (foto de "estado")
- Captura manual desde Serial (`t`)
- Flash LED durante la captura

## Hardware

- ESP32 con cámara OV2640 (AI Thinker o similar)
- Sensor PIR (pin 13)
- Conexión WiFi

## Configuración

1. Cambia las credenciales WiFi en el código
2. Asegúrate que tu servidor esté corriendo en `serverUrl`
3. Sube el código
4. Abre el Monitor Serial (115200 baudios)

## Uso

- Al detectar movimiento → envía **3 fotos** seguidas
- Cada 13 minutos → envía foto de keep-alive
- Escribe `t` en Serial → captura manual

## Notas de Seguridad

- Las credenciales WiFi deben ser cambiadas antes de subir a GitHub
- Usa contraseñas fuertes
- El servidor recibe imágenes en Base64 (verifica seguridad en tu backend)

## Próximas mejoras

- Configuración por Access Point + portal web
- Detección de personas con IA local (TensorFlow Lite)
- Notificaciones Telegram / Push

---
## 🚀 Cómo Crear y Conectar el Servidor

Este proyecto envía fotos capturadas por el ESP32 a un servidor web. A continuación te explico paso a paso cómo configurarlo.

### 1. Crear el Servidor (Recomendado: Render.com - Gratis)

#### Opción más fácil (Node.js + Express)

1. Crea un nuevo repositorio en GitHub con el nombre que prefieras para el caso `esp32-person-detector-server`
2. Crea los siguientes archivos:

**`index.js`**
```javascript
const express = require('express');
const cors = require('cors');
const app = express();

app.use(cors());
app.use(express.json({ limit: '10mb' })); // Para recibir Base64 grande

app.post('/detect', (req, res) => {
  const { image } = req.body;   // Imagen en Base64

  if (!image) {
    return res.status(400).json({ error: "No se recibió imagen" });
  }

  console.log("✅ Imagen recibida - Tamaño:", image.length, "caracteres");

  // Aquí puedes agregar lógica de IA (TensorFlow.js, detección de personas, etc.)
  // Por ahora solo confirmamos recepción
  res.json({
    status: "success",
    message: "Imagen recibida correctamente",
    timestamp: new Date().toISOString(),
    size: image.length
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Servidor corriendo en puerto ${PORT}`);
});
**Autor:** Jonathan Cabrera  
**Estado:** Funcional  
