# Etapa 1: imagem base
FROM python:3.11-slim

# Etapa 2: define diretório de trabalho
WORKDIR /app

# Etapa 3: copia os arquivos de dependência
COPY requirements.txt .

# Etapa 4: instala dependências
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Etapa 5: copia o restante do código da aplicação
COPY . .

# Etapa 6: expõe a porta padrão do FastAPI/uvicorn
EXPOSE 8000

# Etapa 7: define variáveis de ambiente default
ENV ENV=production

# Etapa 8: comando para iniciar o servidor
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
