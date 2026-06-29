# Etapa 1: imagem base
FROM python:3.11-slim

# Etapa 2: define diretório de trabalho
WORKDIR /app

# Etapa 3: bibliotecas de sistema exigidas por rasterio/GDAL, pyproj e shapely
RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Etapa 4: copia os arquivos de dependência
COPY requirements.txt .

# Etapa 5: instala dependências Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Etapa 6: copia o restante do código da aplicação
COPY . .

# Etapa 7: expõe a porta padrão do FastAPI/uvicorn
EXPOSE 8000

# Etapa 8: define variáveis de ambiente default
ENV ENV=production

# Etapa 9: comando para iniciar o servidor
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
