import React, { useState, useEffect } from 'react';
import { Save, Trash2, Database, Cpu, Play, CheckCircle2 } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function Configuration() {
  const [llmConfig, setLlmConfig] = useState({ provider: 'ollama', base_url: 'http://localhost:11434', model_name: 'llama3', api_key: '' });
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [testingLlm, setTestingLlm] = useState(false);
  const [llmTestSuccess, setLlmTestSuccess] = useState<boolean | null>(null);

  const [dbConfigs, setDbConfigs] = useState<any[]>([]);
  const [newDb, setNewDb] = useState({ type: 'clickhouse', host: '', port: 8123, username: '', password: '', database_name: '' });
  const [testingDb, setTestingDb] = useState(false);
  const { token } = useAuth();

  useEffect(() => {
    if (token) fetchConfigs();
  }, [token]);

  const fetchConfigs = async () => {
    try {
      const headers = { Authorization: `Bearer ${token}` };
      const [llmRes, dbRes] = await Promise.all([
        fetch('/api/config/llm', { headers }),
        fetch('/api/config/db', { headers })
      ]);
      const llm = await llmRes.json();
      const dbs = await dbRes.json();
      if (llm) setLlmConfig(llm);
      if (dbs) setDbConfigs(dbs);
    } catch (e) {
      console.error(e);
    }
  };

  const saveLlmConfig = async () => {
    await fetch('/api/config/llm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify(llmConfig)
    });
    alert('LLM Configuration saved!');
  };

  const testLlmConnection = async () => {
    setTestingLlm(true);
    setLlmTestSuccess(null);
    try {
      const res = await fetch('/api/config/llm/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ provider: llmConfig.provider, base_url: llmConfig.base_url, api_key: llmConfig.api_key })
      });
      const data = await res.json();
      if (data.success) {
        setAvailableModels(data.models);
        setLlmTestSuccess(true);
        if (data.models.length > 0 && !data.models.includes(llmConfig.model_name)) {
          setLlmConfig({ ...llmConfig, model_name: data.models[0] });
        }
      } else {
        setLlmTestSuccess(false);
        alert(`Connection failed: ${data.error}`);
      }
    } catch (e) {
      setLlmTestSuccess(false);
      alert('Connection failed');
    } finally {
      setTestingLlm(false);
    }
  };

  const testDbConnection = async () => {
    setTestingDb(true);
    try {
      const res = await fetch('/api/config/db/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify(newDb)
      });
      const data = await res.json();
      if (data.success) {
        alert('Connection successful!');
      } else {
        alert(`Connection failed: ${data.error}`);
      }
    } catch (e) {
      alert('Connection failed');
    } finally {
      setTestingDb(false);
    }
  };

  const addDbConfig = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await fetch('/api/config/db', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify(newDb)
      });
      if (!res.ok) {
        const data = await res.json();
        alert(`Failed to add database connection: ${data.error}`);
        return;
      }
      setNewDb({ type: 'clickhouse', host: '', port: 8123, username: '', password: '', database_name: '' });
      fetchConfigs();
    } catch (e: any) {
      alert(`Error: ${e.message}`);
    }
  };

  const deleteDbConfig = async (id: number) => {
    await fetch(`/api/config/db/${id}`, { 
      method: 'DELETE',
      headers: { Authorization: `Bearer ${token}` }
    });
    fetchConfigs();
  };

  return (
    <div className="p-8 max-w-6xl mx-auto space-y-12">
      <h1 className="text-3xl font-bold text-gray-900 mb-8">Configuration</h1>

      {/* LLM Configuration */}
      <section className="bg-white border border-gray-200 shadow-sm rounded-3xl p-8">
        <div className="flex items-center gap-4 mb-8">
          <div className="p-3 bg-purple-100 text-purple-600 rounded-2xl">
            <Cpu className="w-6 h-6" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900">LLM Settings</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-500 mb-2">Provider</label>
            <select 
              value={llmConfig.provider}
              onChange={e => {
                setLlmConfig({...llmConfig, provider: e.target.value});
                setAvailableModels([]);
                setLlmTestSuccess(null);
              }}
              className="w-full bg-gray-50 border border-gray-200 rounded-2xl px-4 py-3 text-gray-900 focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none transition-all"
            >
              <option value="ollama">Ollama (Local)</option>
              <option value="http">Custom HTTP</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-500 mb-2">Base URL</label>
            <input 
              type="text"
              value={llmConfig.base_url}
              onChange={e => {
                setLlmConfig({...llmConfig, base_url: e.target.value});
                setAvailableModels([]);
                setLlmTestSuccess(null);
              }}
              className="w-full bg-gray-50 border border-gray-200 rounded-2xl px-4 py-3 text-gray-900 focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none transition-all"
            />
          </div>
          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-gray-500 mb-2">Model Name</label>
            {availableModels.length > 0 ? (
              <select
                value={llmConfig.model_name}
                onChange={e => setLlmConfig({...llmConfig, model_name: e.target.value})}
                className="w-full bg-gray-50 border border-gray-200 rounded-2xl px-4 py-3 text-gray-900 focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none transition-all"
              >
                {availableModels.map(model => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
            ) : (
              <input 
                type="text"
                value={llmConfig.model_name}
                onChange={e => setLlmConfig({...llmConfig, model_name: e.target.value})}
                placeholder="e.g., llama3 or gpt-4"
                className="w-full bg-gray-50 border border-gray-200 rounded-2xl px-4 py-3 text-gray-900 focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none transition-all"
              />
            )}
            {llmTestSuccess === true && (
              <p className="text-emerald-600 text-sm mt-2 flex items-center gap-1">
                <CheckCircle2 className="w-4 h-4" /> Connection successful. Select a model from the list.
              </p>
            )}
          </div>
          
          {llmConfig.provider === 'http' && (
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-500 mb-2">API Key (Optional)</label>
              <input 
                type="password"
                value={llmConfig.api_key || ''}
                onChange={e => setLlmConfig({...llmConfig, api_key: e.target.value})}
                placeholder="sk-..."
                className="w-full bg-gray-50 border border-gray-200 rounded-2xl px-4 py-3 text-gray-900 focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none transition-all"
              />
            </div>
          )}
        </div>
        <div className="mt-8 flex justify-end gap-4">
          <button 
            onClick={testLlmConnection}
            disabled={testingLlm || !llmConfig.base_url}
            className="flex items-center gap-2 bg-gray-100 hover:bg-gray-200 text-gray-700 px-6 py-3 rounded-full font-medium transition-colors disabled:opacity-50"
          >
            <Play className="w-5 h-5" />
            {testingLlm ? 'Testing...' : 'Test Connection'}
          </button>
          <button 
            onClick={saveLlmConfig}
            className="flex items-center gap-2 bg-purple-600 hover:bg-purple-500 text-white px-6 py-3 rounded-full font-medium transition-colors shadow-sm"
          >
            <Save className="w-5 h-5" />
            Save LLM Config
          </button>
        </div>
      </section>

      {/* Database Configuration */}
      <section className="bg-white border border-gray-200 shadow-sm rounded-3xl p-8">
        <div className="flex items-center gap-4 mb-8">
          <div className="p-3 bg-blue-100 text-blue-600 rounded-2xl">
            <Database className="w-6 h-6" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900">Database Connections</h2>
        </div>

        {/* Existing DBs */}
        <div className="space-y-4 mb-10">
          {dbConfigs.map(db => (
            <div key={db.id} className="flex items-center justify-between bg-gray-50 border border-gray-100 p-5 rounded-2xl hover:shadow-sm transition-shadow">
              <div>
                <div className="flex items-center gap-3">
                  <span className="font-bold text-gray-900">{db.database_name}</span>
                  <span className="text-xs px-3 py-1 bg-white border border-gray-200 rounded-full text-gray-600 uppercase tracking-wider font-semibold">{db.type}</span>
                </div>
                <p className="text-sm text-gray-500 mt-2 font-medium">{db.host}:{db.port} <span className="text-gray-400">(User: {db.username})</span></p>
              </div>
              <button 
                onClick={() => deleteDbConfig(db.id)}
                className="p-3 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-full transition-colors"
              >
                <Trash2 className="w-5 h-5" />
              </button>
            </div>
          ))}
          {dbConfigs.length === 0 && (
            <div className="text-center py-8 bg-gray-50 rounded-2xl border border-dashed border-gray-200">
              <p className="text-gray-500 font-medium">No database connections configured yet.</p>
            </div>
          )}
        </div>

        {/* Add New DB */}
        <div className="border-t border-gray-100 pt-8">
          <h3 className="text-xl font-bold text-gray-900 mb-6">Add New Connection</h3>
          <form onSubmit={addDbConfig} className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-500 mb-2">Type</label>
              <select 
                value={newDb.type}
                onChange={e => setNewDb({...newDb, type: e.target.value})}
                className="w-full bg-gray-50 border border-gray-200 rounded-2xl px-4 py-3 text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
              >
                <option value="clickhouse">ClickHouse</option>
                <option value="oracle">Oracle</option>
                <option value="elasticsearch">Elasticsearch</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-500 mb-2">Database Name</label>
              <input 
                required
                type="text"
                value={newDb.database_name}
                onChange={e => setNewDb({...newDb, database_name: e.target.value})}
                className="w-full bg-gray-50 border border-gray-200 rounded-2xl px-4 py-3 text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-500 mb-2">Host</label>
              <input 
                required
                type="text"
                value={newDb.host}
                onChange={e => setNewDb({...newDb, host: e.target.value})}
                className="w-full bg-gray-50 border border-gray-200 rounded-2xl px-4 py-3 text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-500 mb-2">Port</label>
              <input 
                required
                type="number"
                value={newDb.port}
                onChange={e => setNewDb({...newDb, port: parseInt(e.target.value)})}
                className="w-full bg-gray-50 border border-gray-200 rounded-2xl px-4 py-3 text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-500 mb-2">Username</label>
              <input 
                required
                type="text"
                value={newDb.username}
                onChange={e => setNewDb({...newDb, username: e.target.value})}
                className="w-full bg-gray-50 border border-gray-200 rounded-2xl px-4 py-3 text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-500 mb-2">Password</label>
              <input 
                required
                type="password"
                value={newDb.password}
                onChange={e => setNewDb({...newDb, password: e.target.value})}
                className="w-full bg-gray-50 border border-gray-200 rounded-2xl px-4 py-3 text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
              />
            </div>
            <div className="md:col-span-2 mt-6 flex justify-end gap-4">
              <button 
                type="button"
                onClick={testDbConnection}
                disabled={testingDb || !newDb.host || !newDb.port}
                className="flex items-center gap-2 bg-gray-100 hover:bg-gray-200 text-gray-700 px-6 py-3 rounded-full font-medium transition-colors disabled:opacity-50"
              >
                <Play className="w-5 h-5" />
                {testingDb ? 'Testing...' : 'Test Connection'}
              </button>
              <button 
                type="submit"
                className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white px-6 py-3 rounded-full font-medium transition-colors shadow-sm"
              >
                <Save className="w-5 h-5" />
                Add Connection
              </button>
            </div>
          </form>
        </div>
      </section>
    </div>
  );
}
