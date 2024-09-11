<h1 align = "center">Reconocimiento de señales de tránsito y semaforos</h1>

# Instrucciones para clonar y ejecutar el proyecto de Visión Artificial

## Requisitos previos

Antes de comenzar, asegúrate de tener instaladas las siguientes herramientas en tu máquina:

### Docker:
- **Linux:** [Instalar Docker Engine en Linux](https://docs.docker.com/engine/install/#supported-platforms)
- **Windows:** [Instalar Docker Desktop en Windows](https://docs.docker.com/desktop/install/windows-install/)
- **macOS:** [Instalar Docker Desktop en macOS](https://docs.docker.com/desktop/install/mac-install/)

### Git:
- **Linux:** [Instalar git en Linux](https://git-scm.com/download/linux)
- **Windows:** [Instalar git en Windows](https://git-scm.com/download/win)
- **macOS:** [Instalar git en macOS](https://git-scm.com/download/mac)

### Otros requisitos:
- **Python 3.8+** (Opcional, solo si deseas ejecutar scripts Python de forma local).
- **Visual Studio Code** (Opcional, para conectar con el servidor remoto de Jupyter).

---

## 1. Clonar el repositorio del proyecto

Primero, clona el repositorio en tu máquina local usando Git:

```bash
git clone https://github.com/GGomezMorales/artificial_vision_project.git
cd artificial_vision_project
```

Este comando descargará todos los archivos del proyecto en tu directorio local y cambiará al directorio del proyecto.

---

## 2. Construir la imagen de Docker

Dentro del directorio del proyecto, ejecuta el siguiente script para construir la imagen Docker que contiene todas las dependencias necesarias para el proyecto:

```bash
./scripts/build
```

Este comando ejecuta un script que preparará el entorno Docker para ejecutar el proyecto.

---

## 3. Ejecutar el contenedor Docker

Una vez construida la imagen, puedes ejecutar el contenedor con el siguiente comando:

```bash
./scripts/run
```

Este script iniciará el contenedor Docker que contiene el entorno necesario para el proyecto. Dentro del contenedor, estarán disponibles todas las herramientas y dependencias necesarias.

---

## 4. Acceder al proyecto

Una vez que el contenedor esté en ejecución, tienes dos opciones para acceder al entorno del proyecto:

### 4.1. Conectar Visual Studio Code a Jupyter Lab (opcional)
Si prefieres trabajar desde Visual Studio Code, puedes suscribirte al servidor remoto de Jupyter utilizando la extensión de Jupyter para VS Code. Asegúrate de tener la extensión instalada, luego conéctate al servidor remoto de Jupyter en tu contenedor.

### 4.2. Acceder al entorno Jupyter Lab desde el navegador
Puedes acceder al entorno de Jupyter Lab desde tu navegador usando el siguiente enlace:

[http://127.0.0.1:8888/lab](http://127.0.0.1:8888/lab)

Este enlace abrirá el entorno interactivo de Jupyter Lab en tu navegador, desde el cual podrás ejecutar los notebooks del proyecto.

---

## Notas adicionales:

- **Ubicación del proyecto:** El archivo principal del proyecto se encuentra en `artificial_vision_project/main.ipynb`. Este archivo está ubicado dentro de un volumen montado en Docker, lo que asegura la persistencia de los datos generados y modificados dentro del contenedor. De esta manera, los datos del proyecto no se perderán cuando se detenga o reinicie el contenedor.

- **Entorno Conda:** Asegúrate de seleccionar el entorno Conda correcto para ejecutar el proyecto sin inconvenientes. El entorno adecuado es `conda env:artificial_vision_project`. Esto garantiza que todas las dependencias necesarias estén disponibles y configuradas correctamente para el correcto funcionamiento del proyecto.



Con esto, tendrás tu proyecto de visión artificial ejecutándose en un entorno controlado y reproducible.
