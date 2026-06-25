# Hexagonal FastAPI

API FastAPI com arquitetura hexagonal, endpoints de usuários e integração com imagens Sentinel via Microsoft Planetary Computer.

## Pré-requisitos

- **Python 3.11+** (testado com Python 3.13)
- **pip** (geralmente incluso na instalação do Python)
- **Docker Desktop** (opcional, recomendado para subir o PostgreSQL localmente)
- Acesso à internet (endpoints `/api/sentinel/*` consomem APIs externas)

## 1. Clonar e entrar no projeto

```powershell
git clone <url-do-repositorio>
cd python_hexagonal_fastapi
```

> Todos os comandos abaixo devem ser executados **na raiz do repositório** (onde estão `requirements.txt`, `app/` e `docker-compose.yml`).

## 2. Criar o ambiente virtual (venv)

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Windows (CMD)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Após ativar o venv, o prompt do terminal deve exibir algo como `(.venv)`.

## 3. Instalar dependências

Com o venv ativo:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

O projeto também utiliza bibliotecas de geoprocessamento e integração com o Planetary Computer que ainda não estão listadas em `requirements.txt`. Instale-as com:

```powershell
pip install asyncpg httpx shapely Pillow pystac pystac-client planetary-computer rasterio numpy pyproj
```

## 4. Configurar variáveis de ambiente

A aplicação carrega um arquivo `.env.{ENV}` com base na variável de ambiente `ENV`:

| Valor de `ENV` | Arquivo carregado   | Uso típico        |
|----------------|---------------------|-------------------|
| `development`  | `.env.development`  | padrão (fallback) |
| `test`         | `.env.test`         | desenvolvimento local |
| `staging`      | `.env.staging`      | homologação       |
| `production`   | `.env.production`   | produção          |

Arquivos disponíveis no repositório: `.env.test`, `.env.staging`, `.env.production`.

Para desenvolvimento local, use `ENV=test`:

### Windows (PowerShell)

```powershell
$env:ENV = "test"
```

### Windows (CMD)

```cmd
set ENV=test
```

### Linux / macOS

```bash
export ENV=test
```

O arquivo `.env.test` aponta para PostgreSQL em `localhost:5433`:

```
DATABASE_URL=postgresql+asyncpg://admin:admin@localhost:5433/curso_python
ENVIRONMENT=TEST
```

## 5. Subir o banco de dados (PostgreSQL)

Com Docker Desktop em execução, na raiz do projeto:

```powershell
docker compose up -d db
```

Isso sobe apenas o PostgreSQL com:

| Parâmetro | Valor          |
|-----------|----------------|
| Host      | `localhost`    |
| Porta     | `5433`         |
| Usuário   | `admin`        |
| Senha     | `admin`        |
| Database  | `curso_python` |

Para subir API + banco via Docker (sem venv):

```powershell
docker compose up --build
```

Nesse caso a API também sobe na porta `8000` com `ENV=test`.

## 6. Executar a aplicação localmente

Com o venv ativo, variável `ENV` definida e (se necessário) o PostgreSQL rodando:

```powershell
uvicorn app.main:app --reload
```

A API ficará disponível em:

- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc
- **Base URL:** http://127.0.0.1:8000

### Rotas principais

| Prefixo            | Descrição                          |
|--------------------|------------------------------------|
| `/api/users`       | Criação de usuários (requer API key) |
| `/api/sentinel/*`  | Consulta de imagens Sentinel       |

## 7. Validar que está funcionando

1. Acesse http://127.0.0.1:8000/docs — a página do Swagger deve carregar.
2. Teste um endpoint público, por exemplo `POST /api/sentinel/days-available-in-range`.
3. Para endpoints que exigem autenticação (`POST /api/users`), informe o header `x-api-key`.

## Resumo rápido (Windows PowerShell)

```powershell
cd python_hexagonal_fastapi

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install asyncpg httpx shapely Pillow pystac pystac-client planetary-computer rasterio numpy pyproj

$env:ENV = "test"
docker compose up -d db

uvicorn app.main:app --reload
```

## Solução de problemas

| Problema | Possível causa | Solução |
|----------|----------------|---------|
| Swagger: `Failed to load API definition` / `500` em `/openapi.json` | Instalação corrompida do Pydantic no venv (`No module named 'pydantic.root_model'`) | Pare o servidor, remova o venv antigo e recrie (passos 2 e 3). Use `.venv`, não `python_hexagonal_fastapi/` |
| `ModuleNotFoundError` ao iniciar | Dependências não instaladas ou venv inativo | Ative o venv e reinstale os pacotes (passos 2 e 3) |
| Erro de conexão com PostgreSQL | Banco não está rodando | Execute `docker compose up -d db` e aguarde o container ficar healthy |
| Porta `8000` em uso | Outra aplicação na mesma porta | Pare o processo ou use `--port 8001` no uvicorn |
| `Activate.ps1` bloqueado no PowerShell | Política de execução restritiva | Execute `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` ou use CMD |

## Estrutura do projeto

```
.
├── app/
│   ├── api/              # Rotas FastAPI
│   ├── application/      # Casos de uso e serviços
│   ├── core/             # Configuração e banco
│   ├── domain/           # Entidades e ports
│   └── infraestructure/  # Repositórios e dependências
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.test / .env.staging / .env.production
```
