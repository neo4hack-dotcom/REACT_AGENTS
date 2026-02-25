# AI Data Agents

Application multi-agents avec interface React et backend Python.

- Frontend: React + Vite
- Backend: FastAPI + LangChain + LangGraph
- Stockage: fichier `DB.json` (pas de SQLite)
- Authentification: désactivée (accès direct à l'application)

## Objectif du projet

AI Data Agents permet de:

- configurer un LLM (Ollama local ou endpoint HTTP compatible OpenAI)
- créer des agents spécialisés à partir de templates
- configurer des connexions de données
- discuter avec un agent unique ou un orchestrateur multi-agents
- suivre les étapes d'exécution en streaming (SSE)
- utiliser un manager renforcé (révision de plan à chaque itération, anti-doublons d'appels)

## Architecture

- `src/`: frontend React (pages Dashboard, Agents, Configuration, Chat)
- `backend/main.py`: API FastAPI et orchestration des routes métier
- `backend/agents.py`: exécution des agents + manager LangGraph
- `DB.json`: persistance des configurations et agents

## Liste des agents

Les agents disponibles dans l'interface (type + description):

| Type | Nom affiché | Description |
|---|---|---|
| `custom` | Custom Agent | Assistant généraliste configurable librement (rôle, objectifs, persona, prompt). |
| `manager` | Agent Manager (Orchestrator) | Orchestrateur multi-agents qui planifie, délègue et synthétise une réponse finale. |
| `sql_analyst` | SQL Analyst | Traduit des demandes en SQL ClickHouse optimisé et explique les résultats. |
| `clickhouse_table_manager` | ClickHouse Table Manager | Administre les tables ClickHouse avec garde-fous de sécurité. |
| `clickhouse_writer` | ClickHouse Writer | Crée des tables temporaires et écrit des données en imposant le préfixe `agent_`. |
| `clickhouse_specific` | ClickHouse Specific | Exécute des requêtes templates paramétrées (ex: `P1`, `P2`) remplies par le manager. |
| `unstructured_to_structured` | Unstructured to Structured | Extrait du JSON structuré depuis du texte non structuré selon un schéma. |
| `email_cleaner` | Email Cleaner | Résume des emails bruyants en sections actionnables et concises. |
| `file_assistant` | File Assistant | Répond à partir de fichiers locaux fournis (mode RAG léger sur fichiers). |
| `text_file_manager` | Text File Manager | Lit/écrit/ajoute du texte dans un dossier sandboxé. |
| `excel_manager` | Excel Manager | Lit et manipule des classeurs Excel en respectant l'intégrité des données. |
| `word_manager` | Word Manager | Lit/édite/génère des documents Word avec remplacement de contenu. |
| `elasticsearch_retriever` | Elasticsearch Retriever | Récupère des documents Elasticsearch et synthétise une réponse. |
| `rag_context` | RAG Context | Produit des réponses basées sur des chunks récupérés d'une base documentaire locale. |
| `rss_news` | RSS News | Agrège et filtre des flux RSS/Atom pour un briefing d'actualité pertinent. |
| `web_scraper` | Web Scraper | Extrait et synthétise du contenu web depuis des URLs de départ. |
| `web_navigator` | Web Navigator | Agent de navigation web multi-étapes (interactions web). |
| `knowledge_base_assistant` | Concierge Interne (Knowledge Base) | Assistant interne RAG avec citations et politique anti-hallucination. |
| `data_anomaly_hunter` | Chasseur d'Anomalies (Anomaly Hunter) | Interprète des anomalies statistiques et formule des alertes métier. |
| `text_to_sql_translator` | Traducteur Métier (Text-to-SQL) | Convertit des questions métier en SQL ClickHouse robuste avec auto-correction. |
| `data_profiler_cleaner` | Profilage et Nettoyage (Data Profiler) | Analyse la qualité des données et propose des scripts de nettoyage. |

## Lancer l'application en local sur Windows

### 1) Prérequis

- Windows 10/11
- [Node.js 20+](https://nodejs.org/)
- [Python 3.11+](https://www.python.org/downloads/windows/)
- Git (optionnel mais recommandé)
- Optionnel: Ollama si vous utilisez le provider local (`ollama`)

### 2) Quick Start (1 commande, recommandé)

Depuis la racine du projet:

```powershell
npm run quickstart
```

Cette commande:

- installe les dépendances frontend (`npm install`)
- crée `.venv` Python si nécessaire
- installe les dépendances backend (`backend/requirements.txt`)
- lance automatiquement le backend (`http://localhost:3000`) et l'UI (`http://localhost:5173`)

Pour arrêter: `Ctrl + C`.

### 3) Ouvrir un terminal dans le projet

PowerShell:

```powershell
cd C:\chemin\vers\CODEX_AGENTS
```

### 4) Créer et activer un environnement Python

PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Invite de commandes (CMD):

```bat
py -3.12 -m venv .venv
.\.venv\Scripts\activate.bat
```

Si PowerShell bloque l'activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 5) Installer les dépendances

```powershell
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
npm install
```

### 6) Démarrer l'application

```powershell
npm run dev
```

L'application sera disponible sur:

- Frontend: `http://localhost:5173`
- API backend: `http://localhost:3000`

### 7) Vérifier que tout fonctionne

```powershell
npm run lint
npm run lint:backend
npm run build
```

## Données et persistance

- Le stockage est effectué dans `DB.json` à la racine du projet.
- Ce fichier contient:
  - la configuration LLM
  - les connexions DB
  - les agents créés
- Sauvegarde simple: copier `DB.json`.

## Scripts utiles

- `npm run quickstart`: installation + lancement backend/frontend en une seule commande
- `npm run dev`: lance backend Python + frontend Vite
- `npm run dev:backend`: lance uniquement FastAPI
- `npm run dev:frontend`: lance uniquement Vite
- `npm run lint`: vérification TypeScript
- `npm run lint:backend`: compilation Python (sanity check)
- `npm run build`: build frontend

## Sécurité des dépendances npm

Le projet n'utilise plus l'ancien backend Node.js, donc les dépendances npm héritées (LangChain JS, Express, SQLite JS, etc.) ont été retirées pour réduire la surface d'attaque et éviter les vulnérabilités transitive inutiles.

Vérification recommandée:

```powershell
npm install
npm audit
```

Résultat actuel après nettoyage: `0 vulnerabilities` (audit npm local).
