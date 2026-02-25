import React, { useEffect, useState } from 'react';
import { Bot, Database, Cpu } from 'lucide-react';

export default function Dashboard() {
  const [stats, setStats] = useState({ agents: 0, dbs: 0, llm: null as any });

  useEffect(() => {
    async function fetchStats() {
      try {
        const [agentsRes, dbsRes, llmRes] = await Promise.all([
          fetch('/api/agents'),
          fetch('/api/config/db'),
          fetch('/api/config/llm')
        ]);
        
        const agents = await agentsRes.json();
        const dbs = await dbsRes.json();
        const llm = await llmRes.json();
        
        setStats({
          agents: agents.length || 0,
          dbs: dbs.length || 0,
          llm: llm || null
        });
      } catch (e) {
        console.error('Failed to fetch stats', e);
      }
    }
    fetchStats();
  }, []);

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold text-gray-900 mb-8">Dashboard</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white border border-gray-200 shadow-sm rounded-3xl p-8 flex items-center gap-5">
          <div className="p-4 bg-emerald-100 text-emerald-600 rounded-full">
            <Bot className="w-8 h-8" />
          </div>
          <div>
            <p className="text-gray-500 text-sm font-medium">Active Agents</p>
            <p className="text-3xl font-bold text-gray-900">{stats.agents}</p>
          </div>
        </div>

        <div className="bg-white border border-gray-200 shadow-sm rounded-3xl p-8 flex items-center gap-5">
          <div className="p-4 bg-blue-100 text-blue-600 rounded-full">
            <Database className="w-8 h-8" />
          </div>
          <div>
            <p className="text-gray-500 text-sm font-medium">Database Connections</p>
            <p className="text-3xl font-bold text-gray-900">{stats.dbs}</p>
          </div>
        </div>

        <div className="bg-white border border-gray-200 shadow-sm rounded-3xl p-8 flex items-center gap-5">
          <div className="p-4 bg-purple-100 text-purple-600 rounded-full">
            <Cpu className="w-8 h-8" />
          </div>
          <div>
            <p className="text-gray-500 text-sm font-medium">Current LLM</p>
            <p className="text-xl font-bold text-gray-900 truncate max-w-[150px]">
              {stats.llm ? stats.llm.model_name : 'None'}
            </p>
            <p className="text-xs text-gray-400">{stats.llm ? stats.llm.provider : 'Not configured'}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
