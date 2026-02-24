import express from "express";
import { createServer as createViteServer } from "vite";
import Database from "better-sqlite3";
import { ChatOllama } from "@langchain/ollama";
import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, START, END, MemorySaver, Annotation } from "@langchain/langgraph";
import { BaseMessage, HumanMessage, AIMessage } from "@langchain/core/messages";
import jwt from "jsonwebtoken";
import bcrypt from "bcryptjs";

const JWT_SECRET = process.env.JWT_SECRET || "super-secret-key-change-in-prod";

const db = new Database("app.db");

// Initialize DB schema
db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS llm_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    provider TEXT NOT NULL,
    base_url TEXT NOT NULL,
    model_name TEXT NOT NULL,
    api_key TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
  );

  CREATE TABLE IF NOT EXISTS db_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    type TEXT NOT NULL,
    host TEXT NOT NULL,
    port INTEGER NOT NULL,
    username TEXT NOT NULL,
    password TEXT NOT NULL,
    database_name TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id)
  );

  CREATE TABLE IF NOT EXISTS agents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    name TEXT NOT NULL,
    role TEXT,
    objectives TEXT,
    persona TEXT,
    tools TEXT,
    memory_settings TEXT,
    system_prompt TEXT NOT NULL,
    db_config_id INTEGER,
    agent_type TEXT DEFAULT 'custom',
    config TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id),
    FOREIGN KEY(db_config_id) REFERENCES db_config(id)
  );
`);

try {
  db.exec("ALTER TABLE llm_config ADD COLUMN api_key TEXT;");
} catch (e) {
  // Column might already exist
}

// Try to add columns if they don't exist (for existing DBs)
try {
  db.exec("ALTER TABLE agents ADD COLUMN agent_type TEXT DEFAULT 'custom'");
  db.exec("ALTER TABLE agents ADD COLUMN config TEXT");
} catch (e) {
  // Columns likely already exist
}

try { db.exec("ALTER TABLE agents ADD COLUMN role TEXT"); } catch (e) {}
try { db.exec("ALTER TABLE agents ADD COLUMN objectives TEXT"); } catch (e) {}
try { db.exec("ALTER TABLE agents ADD COLUMN persona TEXT"); } catch (e) {}
try { db.exec("ALTER TABLE agents ADD COLUMN tools TEXT"); } catch (e) {}
try { db.exec("ALTER TABLE agents ADD COLUMN memory_settings TEXT"); } catch (e) {}

const app = express();
app.use(express.json());
const PORT = 3000;

// Auth Middleware
const authenticateToken = (req: any, res: any, next: any) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (token == null) return res.sendStatus(401);

  jwt.verify(token, JWT_SECRET, (err: any, user: any) => {
    if (err) return res.sendStatus(403);
    
    // Check if user actually exists in DB
    const dbUser = db.prepare("SELECT id FROM users WHERE id = ?").get(user.id);
    if (!dbUser) return res.sendStatus(401); // Force re-login if DB was wiped
    
    req.user = user;
    next();
  });
};

// --- Auth Routes ---
app.post("/api/auth/register", async (req, res) => {
  const { username, password } = req.body;
  try {
    const hashedPassword = await bcrypt.hash(password, 10);
    const stmt = db.prepare("INSERT INTO users (username, password) VALUES (?, ?)");
    const info = stmt.run(username, hashedPassword);
    
    // Insert default LLM config for new user
    db.prepare("INSERT INTO llm_config (user_id, provider, base_url, model_name, api_key) VALUES (?, ?, ?, ?, ?)").run(info.lastInsertRowid, 'ollama', 'http://localhost:11434', 'llama3', null);

    const token = jwt.sign({ id: info.lastInsertRowid, username }, JWT_SECRET);
    res.json({ token });
  } catch (error: any) {
    if (error.code === 'SQLITE_CONSTRAINT_UNIQUE') {
      res.status(400).json({ error: "Username already exists" });
    } else {
      res.status(500).json({ error: "Registration failed" });
    }
  }
});

app.post("/api/auth/login", async (req, res) => {
  const { username, password } = req.body;
  const user = db.prepare("SELECT * FROM users WHERE username = ?").get(username) as any;
  
  if (!user) {
    return res.status(400).json({ error: "Invalid credentials" });
  }

  const validPassword = await bcrypt.compare(password, user.password);
  if (!validPassword) {
    return res.status(400).json({ error: "Invalid credentials" });
  }

  const token = jwt.sign({ id: user.id, username: user.username }, JWT_SECRET);
  res.json({ token });
});

app.get("/api/auth/me", authenticateToken, (req: any, res) => {
  res.json({ user: req.user });
});

// --- API Routes (Protected) ---

// --- LLM Config ---
app.get("/api/config/llm", authenticateToken, (req: any, res) => {
  const config = db.prepare("SELECT * FROM llm_config WHERE user_id = ? ORDER BY id DESC LIMIT 1").get(req.user.id);
  res.json(config);
});

app.post("/api/config/llm", authenticateToken, (req: any, res) => {
  const { provider, base_url, model_name, api_key } = req.body;
  const stmt = db.prepare("INSERT INTO llm_config (user_id, provider, base_url, model_name, api_key) VALUES (?, ?, ?, ?, ?)");
  stmt.run(req.user.id, provider, base_url, model_name, api_key || null);
  res.json({ success: true });
});

app.post("/api/config/llm/test", authenticateToken, async (req: any, res: any) => {
  const { provider, base_url, api_key } = req.body;
  try {
    if (provider === 'ollama') {
      const response = await fetch(`${base_url.replace(/\/$/, '')}/api/tags`);
      if (!response.ok) throw new Error('Failed to connect to Ollama');
      const data = await response.json();
      const models = data.models.map((m: any) => m.name);
      res.json({ success: true, models });
    } else if (provider === 'http') {
      // Assuming OpenAI compatible endpoint for custom HTTP
      const headers: any = {};
      if (api_key) {
        headers['Authorization'] = `Bearer ${api_key}`;
      }
      const response = await fetch(`${base_url.replace(/\/$/, '')}/v1/models`, { headers });
      if (!response.ok) throw new Error('Failed to connect to HTTP LLM');
      const data = await response.json();
      const models = data.data ? data.data.map((m: any) => m.id) : [];
      res.json({ success: true, models });
    } else {
      res.status(400).json({ success: false, error: "Unsupported provider" });
    }
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// --- DB Config ---
app.get("/api/config/db", authenticateToken, (req: any, res) => {
  const configs = db.prepare("SELECT * FROM db_config WHERE user_id = ?").all(req.user.id);
  res.json(configs);
});

app.post("/api/config/db", authenticateToken, (req: any, res) => {
  try {
    const { type, host, port, username, password, database_name } = req.body;
    const stmt = db.prepare("INSERT INTO db_config (user_id, type, host, port, username, password, database_name) VALUES (?, ?, ?, ?, ?, ?, ?)");
    stmt.run(req.user.id, type || 'clickhouse', host || '', port || 0, username || '', password || '', database_name || '');
    res.json({ success: true });
  } catch (error: any) {
    console.error("Error creating db config:", error);
    res.status(500).json({ error: error.message });
  }
});

app.delete("/api/config/db/:id", authenticateToken, (req: any, res) => {
  const stmt = db.prepare("DELETE FROM db_config WHERE id = ? AND user_id = ?");
  stmt.run(req.params.id, req.user.id);
  res.json({ success: true });
});

app.post("/api/config/db/test", authenticateToken, (req: any, res) => {
  // Mock connection test
  const { type, host, port } = req.body;
  if (!host || !port) {
    return res.status(400).json({ success: false, error: "Host and port are required" });
  }
  // In a real app, you would attempt to connect using the specific driver
  setTimeout(() => {
    res.json({ success: true, message: `Successfully connected to ${type} at ${host}:${port}` });
  }, 1000);
});

// --- Agents ---
app.get("/api/agents", authenticateToken, (req: any, res) => {
  const agents = db.prepare("SELECT * FROM agents WHERE user_id = ?").all(req.user.id);
  res.json(agents);
});

app.post("/api/agents", authenticateToken, (req: any, res) => {
  try {
    const { name, role, objectives, persona, tools, memory_settings, system_prompt, db_config_id, agent_type, config } = req.body;
    const stmt = db.prepare("INSERT INTO agents (user_id, name, role, objectives, persona, tools, memory_settings, system_prompt, db_config_id, agent_type, config) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
    stmt.run(req.user.id, name, role, objectives, persona, tools, memory_settings, system_prompt || '', db_config_id || null, agent_type || 'custom', config ? JSON.stringify(config) : null);
    res.json({ success: true });
  } catch (error: any) {
    console.error("Error creating agent:", error);
    res.status(500).json({ error: error.message });
  }
});

app.delete("/api/agents/:id", authenticateToken, (req: any, res) => {
  const stmt = db.prepare("DELETE FROM agents WHERE id = ? AND user_id = ?");
  stmt.run(req.params.id, req.user.id);
  res.json({ success: true });
});

import { AgentExecutor, MultiAgentManager } from "./src/lib/agents.js";

// --- LangGraph Agent Execution ---

const GraphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
});

const memory = new MemorySaver();

app.post("/api/chat/:agentId", authenticateToken, async (req: any, res: any) => {
  const { agentId } = req.params;
  const { message, threadId } = req.body;

  try {
    const agent = db.prepare("SELECT * FROM agents WHERE id = ? AND user_id = ?").get(agentId, req.user.id) as any;
    if (!agent) {
      return res.status(404).json({ error: "Agent not found" });
    }

    const llmConfig = db.prepare("SELECT * FROM llm_config WHERE user_id = ? ORDER BY id DESC LIMIT 1").get(req.user.id) as any;
    if (!llmConfig) {
      return res.status(400).json({ error: "LLM configuration not found" });
    }

    let llm;
    if (llmConfig.provider === 'ollama') {
      llm = new ChatOllama({
        baseUrl: llmConfig.base_url,
        model: llmConfig.model_name,
      });
    } else if (llmConfig.provider === 'http') {
      llm = new ChatOpenAI({
        configuration: {
          baseURL: llmConfig.base_url,
        },
        modelName: llmConfig.model_name,
        openAIApiKey: llmConfig.api_key || "dummy-key",
      });
    } else {
      return res.status(400).json({ error: "Unsupported LLM provider" });
    }

    let parsedConfig = {};
    if (agent.config) {
      try {
        parsedConfig = JSON.parse(agent.config);
      } catch (e) {}
    }

    // Combine legacy fields into config for custom agents
    const fullConfig = {
      ...parsedConfig,
      role: agent.role,
      persona: agent.persona,
      objectives: agent.objectives,
      system_prompt: agent.system_prompt
    };

    const isStreaming = req.query.stream === 'true';

    if (isStreaming) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
    }

    if (agent.agent_type === 'manager') {
      const allAgents = db.prepare("SELECT * FROM agents WHERE user_id = ? AND agent_type != 'manager'").all(req.user.id);
      
      if (isStreaming) {
        const onStep = (step: any) => {
          res.write(`data: ${JSON.stringify({ type: 'step', step })}\n\n`);
        };
        const result = await MultiAgentManager.run_stream(message, allAgents, llm, fullConfig, onStep);
        res.write(`data: ${JSON.stringify({ type: 'result', response: result.answer, steps: result.steps })}\n\n`);
        res.end();
      } else {
        const result = await MultiAgentManager.run_stream(message, allAgents, llm, fullConfig);
        return res.json({
          response: result.answer,
          steps: result.steps,
          threadId: threadId || "default"
        });
      }
    } else {
      const result = await AgentExecutor.execute(agent.agent_type || 'custom', fullConfig, message, llm);
      if (isStreaming) {
        res.write(`data: ${JSON.stringify({ type: 'result', response: result.answer, details: result })}\n\n`);
        res.end();
      } else {
        return res.json({ 
          response: result.answer,
          details: result,
          threadId: threadId || "default"
        });
      }
    }

  } catch (error: any) {
    console.error("Agent execution error:", error);
    if (req.query.stream === 'true') {
      res.write(`data: ${JSON.stringify({ type: 'error', error: error.message || "Internal server error" })}\n\n`);
      res.end();
    } else {
      res.status(500).json({ error: error.message || "Internal server error" });
    }
  }
});

async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static("dist"));
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
