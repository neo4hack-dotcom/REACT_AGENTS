import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Loader2 } from 'lucide-react';
import Markdown from 'react-markdown';

export default function Chat() {
  const [agents, setAgents] = useState<any[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<string>('');
  const [messages, setMessages] = useState<{role: 'user' | 'assistant', content: string, details?: any, steps?: any[]}[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchAgents();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const fetchAgents = async () => {
    const res = await fetch('/api/agents');
    const data = await res.json();
    setAgents(data);
    if (data.length > 0) {
      setSelectedAgentId(data[0].id.toString());
    }
  };

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !selectedAgentId) return;

    const userMsg = input.trim();
    setInput('');
    
    const newMessageIndex = messages.length + 1;
    setMessages(prev => [
      ...prev, 
      { role: 'user', content: userMsg },
      { role: 'assistant', content: '', steps: [] }
    ]);
    setIsLoading(true);

    try {
      const res = await fetch(`/api/chat/${selectedAgentId}?stream=true`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg, threadId: 'session-1' })
      });
      
      if (!res.ok) {
        const data = await res.json();
        setMessages(prev => {
          const newMsgs = [...prev];
          newMsgs[newMessageIndex] = { role: 'assistant', content: `Error: ${data.error}` };
          return newMsgs;
        });
        setIsLoading(false);
        return;
      }

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      
      if (reader) {
        let buffer = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          buffer += decoder.decode(value, { stream: true });
          const messages = buffer.split('\n\n');
          buffer = messages.pop() || ''; // Keep the last incomplete message in the buffer
          
          for (const msg of messages) {
            const lines = msg.split('\n').filter(line => line.trim() !== '');
            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const data = JSON.parse(line.slice(6));
                  
                  setMessages(prev => {
                    const newMsgs = [...prev];
                    const currentMsg = { ...newMsgs[newMessageIndex] };
                    
                    if (data.type === 'step') {
                      currentMsg.steps = [...(currentMsg.steps || []), data.step];
                    } else if (data.type === 'result') {
                      currentMsg.content = data.response;
                      currentMsg.details = data.details;
                    } else if (data.type === 'error') {
                      currentMsg.content = `Error: ${data.error}`;
                    }
                    
                    newMsgs[newMessageIndex] = currentMsg;
                    return newMsgs;
                  });
                } catch (e) {
                  console.error("Error parsing SSE data", e, line);
                }
              }
            }
          }
        }
      }
    } catch (e: any) {
      setMessages(prev => {
        const newMsgs = [...prev];
        newMsgs[newMessageIndex] = { role: 'assistant', content: `Error: ${e.message}` };
        return newMsgs;
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-zinc-950">
      {/* Header */}
      <div className="flex-none p-4 border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-md sticky top-0 z-10 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-emerald-500/10 text-emerald-500 rounded-lg">
            <Bot className="w-5 h-5" />
          </div>
          <h1 className="text-xl font-bold text-white">Agent Chat</h1>
        </div>
        
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium text-zinc-400">Select Agent:</label>
          <select 
            value={selectedAgentId}
            onChange={e => setSelectedAgentId(e.target.value)}
            className="bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-1.5 text-white focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none text-sm"
          >
            {agents.map(agent => (
              <option key={agent.id} value={agent.id}>{agent.name}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-zinc-500 space-y-4">
            <Bot className="w-16 h-16 opacity-20" />
            <p>Select an agent and start chatting.</p>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              {msg.role === 'assistant' && (
                <div className="w-8 h-8 rounded-full bg-emerald-500/20 text-emerald-500 flex items-center justify-center flex-shrink-0">
                  <Bot className="w-5 h-5" />
                </div>
              )}
              
              <div className={`max-w-[80%] rounded-2xl px-5 py-3.5 ${
                msg.role === 'user' 
                  ? 'bg-emerald-600 text-white rounded-tr-sm' 
                  : 'bg-zinc-800 text-zinc-200 rounded-tl-sm'
              }`}>
                <div className="prose prose-invert prose-emerald max-w-none text-sm leading-relaxed">
                  <Markdown>{msg.content || (msg.role === 'assistant' && msg.steps?.length ? '*Working...*' : '')}</Markdown>
                </div>
                {msg.steps && msg.steps.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-zinc-700">
                    <p className="text-xs font-semibold text-zinc-400 mb-2 uppercase tracking-wider">Execution Steps</p>
                    <ul className="space-y-2">
                      {msg.steps.map((step, i) => (
                        <li key={i} className="text-sm bg-zinc-900/50 p-3 rounded-lg border border-zinc-700/50">
                          <div className="flex items-center gap-2 mb-1">
                            <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                              step.status.includes('error') || step.status.includes('failed') ? 'bg-red-500/20 text-red-400' :
                              step.status.includes('completed') || step.status.includes('final') ? 'bg-emerald-500/20 text-emerald-400' :
                              'bg-blue-500/20 text-blue-400'
                            }`}>
                              {step.status}
                            </span>
                            {step.agent && <span className="text-zinc-300 font-medium">{step.agent}</span>}
                          </div>
                          {step.rationale && <p className="text-zinc-400 text-xs mt-1 italic">{step.rationale}</p>}
                          {step.plan && <p className="text-zinc-300 text-xs mt-1 border-l-2 border-zinc-600 pl-2">{step.plan}</p>}
                          {step.input && <p className="text-zinc-400 text-xs mt-1"><span className="font-semibold text-zinc-500">Input:</span> {step.input}</p>}
                          {step.sql && <pre className="mt-2 text-xs text-blue-400 bg-zinc-950 p-2 rounded overflow-x-auto border border-zinc-800">{step.sql}</pre>}
                          {step.result && step.result.answer && <p className="text-zinc-300 text-xs mt-2 bg-zinc-800/50 p-2 rounded"><span className="font-semibold text-zinc-500">Result:</span> {step.result.answer}</p>}
                          {step.error && <p className="text-red-400 text-xs mt-1">{step.error}</p>}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {msg.details && msg.details.sql && (
                  <div className="mt-4 pt-4 border-t border-zinc-700">
                    <p className="text-xs font-semibold text-zinc-400 mb-2 uppercase tracking-wider">Generated SQL</p>
                    <pre className="text-xs text-blue-400 bg-zinc-950 p-3 rounded-xl overflow-x-auto border border-zinc-800">
                      {msg.details.sql}
                    </pre>
                  </div>
                )}
              </div>

              {msg.role === 'user' && (
                <div className="w-8 h-8 rounded-full bg-zinc-800 text-zinc-400 flex items-center justify-center flex-shrink-0">
                  <User className="w-5 h-5" />
                </div>
              )}
            </div>
          ))
        )}
        {isLoading && (
          <div className="flex gap-4 justify-start">
            <div className="w-8 h-8 rounded-full bg-emerald-500/20 text-emerald-500 flex items-center justify-center flex-shrink-0">
              <Bot className="w-5 h-5" />
            </div>
            <div className="bg-zinc-800 text-zinc-200 rounded-2xl rounded-tl-sm px-5 py-3.5 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-emerald-500" />
              <span className="text-sm text-zinc-400">Agent is thinking...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="flex-none p-4 bg-zinc-900 border-t border-zinc-800">
        <form onSubmit={sendMessage} className="max-w-4xl mx-auto relative flex items-center">
          <input 
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask your agent something..."
            disabled={isLoading || !selectedAgentId}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-full pl-6 pr-14 py-4 text-white focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none disabled:opacity-50"
          />
          <button 
            type="submit"
            disabled={isLoading || !input.trim() || !selectedAgentId}
            className="absolute right-2 p-2.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-full transition-colors disabled:opacity-50 disabled:hover:bg-emerald-600"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
      </div>
    </div>
  );
}
