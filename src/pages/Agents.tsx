import React, { useState, useEffect } from 'react';
import { Bot, Plus, Trash2, Database, Settings } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const AGENT_TYPES = [
  { id: 'custom', name: 'Custom Agent' },
  { id: 'manager', name: 'Agent Manager (Orchestrator)' },
  { id: 'sql_analyst', name: 'SQL Analyst' },
  { id: 'clickhouse_table_manager', name: 'ClickHouse Table Manager' },
  { id: 'clickhouse_writer', name: 'ClickHouse Writer' },
  { id: 'unstructured_to_structured', name: 'Unstructured to Structured' },
  { id: 'email_cleaner', name: 'Email Cleaner' },
  { id: 'file_assistant', name: 'File Assistant' },
  { id: 'text_file_manager', name: 'Text File Manager' },
  { id: 'excel_manager', name: 'Excel Manager' },
  { id: 'word_manager', name: 'Word Manager' },
  { id: 'elasticsearch_retriever', name: 'Elasticsearch Retriever' },
  { id: 'rag_context', name: 'RAG Context' },
  { id: 'rss_news', name: 'RSS News' },
  { id: 'web_scraper', name: 'Web Scraper' },
  { id: 'web_navigator', name: 'Web Navigator' },
  { id: 'knowledge_base_assistant', name: 'Concierge Interne (Knowledge Base)' },
  { id: 'data_anomaly_hunter', name: 'Chasseur d\'Anomalies (Anomaly Hunter)' },
  { id: 'text_to_sql_translator', name: 'Traducteur Métier (Text-to-SQL)' },
  { id: 'data_profiler_cleaner', name: 'Profilage et Nettoyage (Data Profiler)' },
];

const AGENT_TEMPLATES: Record<string, any> = {
  custom: {
    config: {},
    role: "Polyvalent Assistant",
    objectives: "Help the user achieve their goals efficiently.",
    persona: "Helpful, concise, and professional.",
    system_prompt: "You are a helpful AI assistant. Answer the user's queries accurately and concisely."
  },
  manager: {
    config: { max_steps: 5, max_agent_calls: 10 },
    role: "Multi-Agent Orchestrator",
    objectives: "Analyze the user's request, select the best specialized agents, and synthesize their outputs into a final answer.",
    persona: "Logical, strategic, and decisive.",
    system_prompt: "You are the Multi-Agent Manager. Coordinate the available agents to solve the user's task."
  },
  sql_analyst: {
    config: { sql_use_case_mode: "llm_sql", sql_query_template: "SELECT * FROM {table} WHERE {conditions} LIMIT 100", sql_parameters: [] },
    role: "SQL Data Analyst",
    objectives: "Translate natural language questions into optimized, safe SQL queries and explain the results.",
    persona: "Analytical, precise, and data-driven."
  },
  clickhouse_table_manager: {
    config: { protect_existing_tables: true, allow_row_inserts: false, allow_row_updates: false, allow_row_deletes: false, max_statements: 5, preview_select_rows: 10, stop_on_error: true },
    role: "ClickHouse Database Administrator",
    objectives: "Safely manage ClickHouse tables, strictly enforcing security policies and preventing destructive operations.",
    persona: "Cautious, strict, and security-focused."
  },
  clickhouse_writer: {
    config: { enforce_agent_prefix: true, allow_inserts: true, max_rows_per_insert: 1000 },
    role: "ClickHouse Data Writer",
    objectives: "Create temporary tables and insert buffer data. MUST prefix table names with 'agent_'.",
    persona: "Diligent, secure, and precise."
  },
  unstructured_to_structured: {
    config: { output_schema: { type: "object", properties: { entities: { type: "array", items: { type: "string" } }, summary: { type: "string" } }, required: ["entities", "summary"] }, strict_json: true },
    role: "Data Extraction Specialist",
    objectives: "Extract structured JSON data from unstructured text based on a strict schema.",
    persona: "Meticulous, structured, and detail-oriented."
  },
  email_cleaner: {
    config: { max_bullets: 5, include_sections: "Action Items, Key Decisions, Summary" },
    role: "Email Triage Assistant",
    objectives: "Condense noisy email threads into clear, actionable bullet points and sections.",
    persona: "Efficient, organized, and clear."
  },
  file_assistant: {
    config: { folder_path: "/data/documents", file_extensions: [".txt", ".md", ".csv"], max_files: 10, max_file_size_kb: 1024, top_k: 3 },
    role: "Local File Assistant",
    objectives: "Answer questions accurately based ONLY on the contents of the provided local files.",
    persona: "Helpful, context-aware, and precise."
  },
  text_file_manager: {
    config: { folder_path: "/data/workspace", default_file_path: "output.txt", default_encoding: "utf-8", auto_create_folder: true, allow_overwrite: false, max_chars_read: 10000 },
    role: "Text File Operator",
    objectives: "Safely read, create, write, and append text files within a restricted sandbox directory.",
    persona: "Careful, systematic, and secure."
  },
  excel_manager: {
    config: { folder_path: "/data/spreadsheets", workbook_path: "data.xlsx", default_sheet: "Sheet1", auto_create_folder: true, auto_create_workbook: false, max_rows_read: 1000 },
    role: "Excel Spreadsheet Manager",
    objectives: "Read and manipulate Excel workbooks safely, ensuring valid cell references and data integrity.",
    persona: "Methodical, accurate, and data-focused."
  },
  word_manager: {
    config: { folder_path: "/data/documents", document_path: "report.docx", auto_create_folder: true, auto_create_document: false, allow_overwrite: false, max_paragraphs_read: 100 },
    role: "Word Document Editor",
    objectives: "Read, edit, and generate Word documents, replacing placeholders with concrete content.",
    persona: "Articulate, professional, and formatting-aware."
  },
  elasticsearch_retriever: {
    config: { base_url: "http://localhost:9200", index: "my-index", verify_ssl: true, top_k: 5, fields: "title,content,summary" },
    role: "Elasticsearch Search Expert",
    objectives: "Retrieve highly relevant documents from Elasticsearch and synthesize evidence-based answers.",
    persona: "Investigative, thorough, and objective."
  },
  rag_context: {
    config: { folder_path: "/data/knowledge_base", file_extensions: [".pdf", ".txt", ".md"], top_k_chunks: 5, chunk_size: 1000, chunk_overlap: 200, max_files: 50 },
    role: "RAG Knowledge Assistant",
    objectives: "Provide accurate answers by retrieving and citing the most relevant chunks from the local knowledge base.",
    persona: "Knowledgeable, academic, and citation-focused."
  },
  rss_news: {
    config: { feed_urls: "https://news.ycombinator.com/rss", interests: "technology, AI", exclude_keywords: "sports", top_k: 10, max_items_per_feed: 5, hours_lookback: 24, language_hint: "en", include_general_if_no_match: true },
    role: "News Briefing Analyst",
    objectives: "Fetch, filter, and score RSS/Atom feeds to provide a highly relevant and deduplicated news briefing.",
    persona: "Informed, objective, and concise."
  },
  web_scraper: {
    config: { start_urls: "https://example.com", include_urls_from_question: true, search_fallback: true, follow_links: false, same_domain_only: true, allowed_domains: "example.com", max_pages: 3, max_links_per_page: 10, max_chars_per_page: 10000, timeout_seconds: 15, region: "us-en", safe_search: true },
    role: "Web Scraping Specialist",
    objectives: "Extract and synthesize text from web pages, using search fallbacks when necessary.",
    persona: "Resourceful, fast, and analytical."
  },
  web_navigator: {
    config: { start_url: "https://google.com", headless: true, max_steps: 10, timeout_ms: 30000, capture_html_chars: 5000 },
    role: "Autonomous Web Navigator",
    objectives: "Navigate web pages, interact with elements, and extract information to accomplish complex tasks.",
    persona: "Action-oriented, persistent, and adaptable."
  },
  knowledge_base_assistant: {
    config: { vector_db: "qdrant", embedding_model: "text-embedding-004", chunk_size: 1000, use_hybrid_search: true, use_reranker: true, rbac_filter: true },
    role: "Concierge Interne (RAG)",
    objectives: "Answer questions using internal documents via Retrieval-Augmented Generation. Never hallucinate. Always cite sources.",
    persona: "Helpful, precise, and reliable.",
    system_prompt: "You are an internal knowledge base assistant. Use the provided retrieved chunks to answer the user's question. If the answer is not in the chunks, say you don't know. Always cite your sources."
  },
  data_anomaly_hunter: {
    config: { detection_method: "statistical", use_dynamic_thresholds: true, alert_channel: "slack", feedback_loop_enabled: true },
    role: "Data Anomaly Hunter",
    objectives: "Interpret statistical anomalies detected in the database and explain them in business terms.",
    persona: "Analytical, alert, and business-focused.",
    system_prompt: "You are a Data Anomaly Hunter. You receive statistical outliers and KPIs. Your job is to explain WHY this anomaly might have happened in clear business terms and format an alert message."
  },
  text_to_sql_translator: {
    config: { use_semantic_layer: true, max_retries: 3, generate_chart_config: true, use_golden_queries: true },
    role: "Text-to-SQL Translator",
    objectives: "Translate business questions into precise ClickHouse SQL using provided schema and Golden Queries. Auto-correct syntax errors.",
    persona: "Logical, precise, and expert in ClickHouse SQL.",
    system_prompt: "You are an expert ClickHouse SQL translator. Use the provided schema and Golden Queries to write the best SQL. If you receive a syntax error, analyze it and provide a corrected query."
  },
  data_profiler_cleaner: {
    config: { execution_engine: "polars", use_sampling: true, sample_size: 10000, generate_tests: true, require_human_validation: true },
    role: "Data Profiler & Cleaner",
    objectives: "Analyze data profiles (nulls, cardinality, min/max) to detect quality issues and generate cleaning scripts (Pandas/Polars/SQL).",
    persona: "Meticulous, quality-driven, and proactive.",
    system_prompt: "You are a Data Steward. Review the provided statistical profile of a dataset. Identify anomalies (e.g., negative ages, bad emails) and write a Python/SQL script to clean the data. Do not execute destructive actions directly."
  }
};

export default function Agents() {
  const [agents, setAgents] = useState<any[]>([]);
  const [dbConfigs, setDbConfigs] = useState<any[]>([]);
  const [isCreating, setIsCreating] = useState(false);
  const [newAgent, setNewAgent] = useState({ 
    name: '', 
    agent_type: 'custom',
    role: AGENT_TEMPLATES['custom'].role,
    objectives: AGENT_TEMPLATES['custom'].objectives,
    persona: AGENT_TEMPLATES['custom'].persona,
    tools: '',
    memory_settings: 'short_term',
    system_prompt: AGENT_TEMPLATES['custom'].system_prompt, 
    db_config_id: '',
    configStr: JSON.stringify(AGENT_TEMPLATES['custom'].config, null, 2)
  });
  const { token } = useAuth();

  useEffect(() => {
    if (token) {
      fetchAgents();
      fetchDbConfigs();
    }
  }, [token]);

  const handleAgentTypeChange = (type: string) => {
    const template = AGENT_TEMPLATES[type] || AGENT_TEMPLATES['custom'];
    setNewAgent({
      ...newAgent,
      agent_type: type,
      role: template.role || '',
      objectives: template.objectives || '',
      persona: template.persona || '',
      system_prompt: template.system_prompt || '',
      configStr: JSON.stringify(template.config || {}, null, 2)
    });
  };

  const fetchAgents = async () => {
    const res = await fetch('/api/agents', {
      headers: { Authorization: `Bearer ${token}` }
    });
    setAgents(await res.json());
  };

  const fetchDbConfigs = async () => {
    const res = await fetch('/api/config/db', {
      headers: { Authorization: `Bearer ${token}` }
    });
    setDbConfigs(await res.json());
  };

  const createAgent = async (e: React.FormEvent) => {
    e.preventDefault();
    let parsedConfig = {};
    try {
      parsedConfig = JSON.parse(newAgent.configStr);
    } catch (e) {
      alert("Invalid JSON in Advanced Config");
      return;
    }

    try {
      const res = await fetch('/api/agents', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          ...newAgent,
          config: parsedConfig,
          db_config_id: newAgent.db_config_id ? parseInt(newAgent.db_config_id) : null
        })
      });
      
      if (!res.ok) {
        const data = await res.json();
        alert(`Failed to create agent: ${data.error}`);
        return;
      }
      
      setIsCreating(false);
      setNewAgent({ name: '', agent_type: 'custom', role: '', objectives: '', persona: '', tools: '', memory_settings: 'short_term', system_prompt: '', db_config_id: '', configStr: '{}' });
      fetchAgents();
    } catch (e: any) {
      alert(`Error: ${e.message}`);
    }
  };

  const deleteAgent = async (id: number) => {
    await fetch(`/api/agents/${id}`, { 
      method: 'DELETE',
      headers: { Authorization: `Bearer ${token}` }
    });
    fetchAgents();
  };

  return (
    <div className="p-8 max-w-6xl mx-auto space-y-8">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white">AI Agents</h1>
        <button 
          onClick={() => setIsCreating(true)}
          className="flex items-center gap-2 bg-emerald-600 hover:bg-emerald-500 text-white px-5 py-2.5 rounded-xl font-medium transition-colors"
        >
          <Plus className="w-5 h-5" />
          New Agent
        </button>
      </div>

      {isCreating && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-6 mb-8">
          <h2 className="text-xl font-semibold text-white mb-6">Create New Agent</h2>
          <form onSubmit={createAgent} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-zinc-400 mb-2">Agent Name</label>
                <input 
                  required
                  type="text"
                  value={newAgent.name}
                  onChange={e => setNewAgent({...newAgent, name: e.target.value})}
                  placeholder="e.g., Sales Data Analyst"
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-2.5 text-white focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-zinc-400 mb-2">Agent Type</label>
                <select 
                  value={newAgent.agent_type}
                  onChange={e => handleAgentTypeChange(e.target.value)}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-2.5 text-white focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none"
                >
                  {AGENT_TYPES.map(type => (
                    <option key={type.id} value={type.id}>{type.name}</option>
                  ))}
                </select>
              </div>

              {newAgent.agent_type === 'custom' && (
                <>
                  <div>
                    <label className="block text-sm font-medium text-zinc-400 mb-2">Role</label>
                    <input 
                      type="text"
                      value={newAgent.role}
                      onChange={e => setNewAgent({...newAgent, role: e.target.value})}
                      placeholder="e.g., Data Analyst, Customer Support"
                      className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-2.5 text-white focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-zinc-400 mb-2">Objectives</label>
                    <input 
                      type="text"
                      value={newAgent.objectives}
                      onChange={e => setNewAgent({...newAgent, objectives: e.target.value})}
                      placeholder="e.g., Analyze sales trends, answer queries"
                      className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-2.5 text-white focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-zinc-400 mb-2">Persona</label>
                    <input 
                      type="text"
                      value={newAgent.persona}
                      onChange={e => setNewAgent({...newAgent, persona: e.target.value})}
                      placeholder="e.g., Professional, Friendly, Concise"
                      className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-2.5 text-white focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-zinc-400 mb-2">Tools Available</label>
                    <input 
                      type="text"
                      value={newAgent.tools}
                      onChange={e => setNewAgent({...newAgent, tools: e.target.value})}
                      placeholder="e.g., SQL Query, Web Search"
                      className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-2.5 text-white focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-zinc-400 mb-2">Memory Settings</label>
                    <select 
                      value={newAgent.memory_settings}
                      onChange={e => setNewAgent({...newAgent, memory_settings: e.target.value})}
                      className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-2.5 text-white focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none"
                    >
                      <option value="none">No Memory</option>
                      <option value="short_term">Short-term (Session)</option>
                      <option value="long_term">Long-term (Persistent)</option>
                    </select>
                  </div>
                </>
              )}
            </div>
            
            {newAgent.agent_type === 'custom' && (
              <div>
                <label className="block text-sm font-medium text-zinc-400 mb-2">System Prompt / Instructions</label>
                <textarea 
                  rows={4}
                  value={newAgent.system_prompt}
                  onChange={e => setNewAgent({...newAgent, system_prompt: e.target.value})}
                  placeholder="You are a helpful data analyst. You have access to a database..."
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-2.5 text-white focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none resize-none"
                />
              </div>
            )}

            {newAgent.agent_type !== 'custom' && (
              <div>
                <label className="block text-sm font-medium text-zinc-400 mb-2">Advanced Config (JSON)</label>
                <textarea 
                  rows={6}
                  value={newAgent.configStr}
                  onChange={e => setNewAgent({...newAgent, configStr: e.target.value})}
                  placeholder='{"folder_path": "/data", "max_files": 10}'
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-2.5 text-white focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none resize-none font-mono text-sm"
                />
                <p className="text-xs text-zinc-500 mt-2">Provide specific configuration for {newAgent.agent_type} in JSON format.</p>
              </div>
            )}
            <div>
              <label className="block text-sm font-medium text-zinc-400 mb-2">Connected Database (Optional)</label>
              <select 
                value={newAgent.db_config_id}
                onChange={e => setNewAgent({...newAgent, db_config_id: e.target.value})}
                className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-2.5 text-white focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none"
              >
                <option value="">None</option>
                {dbConfigs.map(db => (
                  <option key={db.id} value={db.id}>{db.database_name} ({db.type})</option>
                ))}
              </select>
            </div>
            <div className="flex justify-end gap-3 pt-4">
              <button 
                type="button"
                onClick={() => setIsCreating(false)}
                className="px-6 py-2.5 rounded-xl font-medium text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
              >
                Cancel
              </button>
              <button 
                type="submit"
                className="bg-emerald-600 hover:bg-emerald-500 text-white px-6 py-2.5 rounded-xl font-medium transition-colors"
              >
                Create Agent
              </button>
            </div>
          </form>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {agents.map(agent => (
          <div key={agent.id} className="bg-zinc-900 border border-zinc-800 rounded-2xl p-6 flex flex-col">
            <div className="flex items-start justify-between mb-4">
              <div className="p-3 bg-emerald-500/10 text-emerald-500 rounded-xl">
                <Bot className="w-6 h-6" />
              </div>
              <button 
                onClick={() => deleteAgent(agent.id)}
                className="p-2 text-zinc-500 hover:text-red-500 hover:bg-red-500/10 rounded-lg transition-colors"
              >
                <Trash2 className="w-5 h-5" />
              </button>
            </div>
            <h3 className="text-xl font-bold text-white mb-1">{agent.name}</h3>
            <p className="text-emerald-500 text-sm font-medium mb-3">
              {AGENT_TYPES.find(t => t.id === agent.agent_type)?.name || agent.role || 'Agent'}
            </p>
            <p className="text-zinc-400 text-sm line-clamp-3 mb-4 flex-1">
              {agent.agent_type === 'custom' ? agent.system_prompt : (agent.config ? JSON.stringify(JSON.parse(agent.config)).substring(0, 100) + '...' : 'No config')}
            </p>
            <div className="flex flex-wrap gap-2 mt-auto">
              {agent.db_config_id && (
                <div className="flex items-center gap-1.5 text-xs text-blue-400 bg-blue-500/10 px-2.5 py-1 rounded-lg w-fit">
                  <Database className="w-3 h-3" />
                  <span>DB Connected</span>
                </div>
              )}
              {agent.agent_type !== 'custom' && (
                <div className="flex items-center gap-1.5 text-xs text-purple-400 bg-purple-500/10 px-2.5 py-1 rounded-lg w-fit">
                  <Settings className="w-3 h-3" />
                  <span>Configured</span>
                </div>
              )}
            </div>
          </div>
        ))}
        
        {agents.length === 0 && !isCreating && (
          <div className="col-span-full text-center py-12 text-zinc-500 border border-dashed border-zinc-800 rounded-2xl">
            <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No agents configured yet.</p>
            <p className="text-sm mt-1">Click "New Agent" to get started.</p>
          </div>
        )}
      </div>
    </div>
  );
}
