FROM python:3.12-slim

WORKDIR /app

# System deps for Pillow / pyarrow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libjpeg62-turbo libpng16-16 libtiff6 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU-only (200MB vs 2GB with CUDA)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install project
COPY pyproject.toml .
COPY src/ src/
COPY config/ config/
RUN pip install --no-cache-dir -e .

EXPOSE 8000 7860
CMD ["luki-api", "--host", "0.0.0.0", "--port", "8000"]
