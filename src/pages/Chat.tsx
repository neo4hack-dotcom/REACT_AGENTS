import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Send, Bot, User, Loader2, ChevronDown, ChevronUp, Plus, MessageSquare } from 'lucide-react';
import Markdown from 'react-markdown';

type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
  details?: any;
  steps?: any[];
  showDetails?: boolean;
};

type ChatThreadSummary = {
  id: number;
  agent_id: number | null;
  title: string;
  updated_at?: string;
  message_count?: number;
  last_message_preview?: string;
};

type ChatThread = {
  id: number;
  agent_id: number | null;
  title: string;
  created_at?: string;
  updated_at?: string;
  messages?: Array<{
    role: string;
    content: string;
    details?: any;
  }>;
};

export default function Chat() {
  const [agents, setAgents] = useState<any[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<string>('');
  const [threads, setThreads] = useState<ChatThreadSummary[]>([]);
  const [selectedThreadId, setSelectedThreadId] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeAssistantMessageIndex, setActiveAssistantMessageIndex] = useState<number | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bootstrap();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const selectedThread = useMemo(
    () => threads.find(thread => thread.id === selectedThreadId) || null,
    [threads, selectedThreadId]
  );

  const readErrorMessage = async (res: Response): Promise<string> => {
    const contentType = res.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      try {
        const data = await res.json();
        if (data?.error) return String(data.error);
      } catch (_err) {
        // Fallback to text.
      }
    }

    try {
      const text = await res.text();
      if (text) return text;
    } catch (_err) {
      // No-op
    }

    return `${res.status} ${res.statusText}`.trim();
  };

  const normalizeRole = (role: string): 'user' | 'assistant' => {
    return role === 'user' ? 'user' : 'assistant';
  };

  const mapStoredMessages = (threadMessages: any[] | undefined): ChatMessage[] => {
    if (!Array.isArray(threadMessages)) return [];
    return threadMessages.map((msg: any) => {
      const details = msg?.details;
      const steps = Array.isArray(details?.steps) ? details.steps : undefined;
      return {
        role: normalizeRole(String(msg?.role || 'assistant')),
        content: String(msg?.content || ''),
        details,
        steps,
        showDetails: false,
      };
    });
  };

  const fetchAgents = async (): Promise<any[]> => {
    const res = await fetch('/api/agents');
    if (!res.ok) {
      throw new Error(await readErrorMessage(res));
    }
    const data = await res.json();
    setAgents(data);
    return data;
  };

  const fetchThreads = async (): Promise<ChatThreadSummary[]> => {
    const res = await fetch('/api/chat/threads');
    if (!res.ok) {
      throw new Error(await readErrorMessage(res));
    }
    const data = await res.json();
    setThreads(data);
    return data;
  };

  const fetchThreadDetail = async (threadId: number): Promise<ChatThread | null> => {
    const res = await fetch(`/api/chat/threads/${threadId}`);
    if (!res.ok) {
      return null;
    }
    return await res.json();
  };

  const bootstrap = async () => {
    try {
      const [agentData, threadData] = await Promise.all([fetchAgents(), fetchThreads()]);

      if (threadData.length > 0) {
        const first = threadData[0];
        setSelectedThreadId(first.id);
        if (first.agent_id) {
          setSelectedAgentId(String(first.agent_id));
        } else if (agentData.length > 0) {
          setSelectedAgentId(String(agentData[0].id));
        }

        const detail = await fetchThreadDetail(first.id);
        setMessages(mapStoredMessages(detail?.messages));
        return;
      }

      if (agentData.length > 0) {
        setSelectedAgentId(String(agentData[0].id));
      }
      setMessages([]);
    } catch (e: any) {
      console.error('Failed to initialize chat:', e);
      setThreads([]);
      setMessages([]);
    }
  };

  const refreshThreads = async (keepSelectionId?: number) => {
    try {
      const updatedThreads = await fetchThreads();
      const currentSelection = keepSelectionId ?? selectedThreadId;

      if (currentSelection && updatedThreads.some(thread => thread.id === currentSelection)) {
        return;
      }

      if (updatedThreads.length > 0) {
        const fallback = updatedThreads[0];
        setSelectedThreadId(fallback.id);
        if (fallback.agent_id) {
          setSelectedAgentId(String(fallback.agent_id));
        }
        const detail = await fetchThreadDetail(fallback.id);
        setMessages(mapStoredMessages(detail?.messages));
      } else {
        setSelectedThreadId(null);
        setMessages([]);
      }
    } catch (e) {
      console.error('Failed to refresh threads', e);
    }
  };

  const createThread = async (): Promise<ChatThreadSummary | null> => {
    const fallbackAgentId = agents.length > 0 ? String(agents[0].id) : '';
    const agentId = selectedAgentId || fallbackAgentId;

    if (!agentId) {
      alert('Create an agent first before starting a discussion.');
      return null;
    }

    const res = await fetch('/api/chat/threads', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        agent_id: parseInt(agentId, 10),
        title: 'Nouvelle discussion',
      }),
    });

    if (!res.ok) {
      throw new Error(await readErrorMessage(res));
    }

    const created = await res.json();
    const newThread: ChatThreadSummary = {
      id: created.id,
      agent_id: created.agent_id ?? parseInt(agentId, 10),
      title: created.title || `Discussion ${created.id}`,
      updated_at: created.updated_at,
      message_count: 0,
      last_message_preview: '',
    };

    setThreads(prev => [newThread, ...prev]);
    setSelectedThreadId(newThread.id);
    setSelectedAgentId(String(newThread.agent_id || agentId));
    setMessages([]);
    return newThread;
  };

  const startNewDiscussion = async () => {
    try {
      await createThread();
    } catch (e: any) {
      alert(`Failed to create discussion: ${e.message}`);
    }
  };

  const openThread = async (threadId: number) => {
    if (isLoading) return;
    setSelectedThreadId(threadId);

    const summary = threads.find(thread => thread.id === threadId);
    if (summary?.agent_id) {
      setSelectedAgentId(String(summary.agent_id));
    }

    try {
      const detail = await fetchThreadDetail(threadId);
      if (!detail) {
        alert('Discussion not found.');
        await refreshThreads();
        return;
      }
      if (detail.agent_id) {
        setSelectedAgentId(String(detail.agent_id));
      }
      setMessages(mapStoredMessages(detail.messages));
    } catch (e: any) {
      alert(`Failed to load discussion: ${e.message}`);
    }
  };

  const toggleDetails = (messageIndex: number) => {
    setMessages(prev => {
      const next = [...prev];
      if (!next[messageIndex]) return next;
      next[messageIndex] = {
        ...next[messageIndex],
        showDetails: !next[messageIndex].showDetails,
      };
      return next;
    });
  };

  const deriveFallbackAnswerFromSteps = (steps: any[] | undefined): string => {
    if (!Array.isArray(steps)) return '';
    for (let i = steps.length - 1; i >= 0; i -= 1) {
      const step = steps[i];
      const candidates = [step?.answer, step?.manager_summary, step?.message, step?.error];
      for (const candidate of candidates) {
        if (typeof candidate === 'string' && candidate.trim().length > 0) {
          return candidate.trim();
        }
      }
    }
    return '';
  };

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    const threadBoundAgentId = selectedThread?.agent_id ? String(selectedThread.agent_id) : '';
    const agentIdForChat = threadBoundAgentId || selectedAgentId;
    if (!input.trim() || !agentIdForChat) return;

    const userMsg = input.trim();
    setInput('');
    if (threadBoundAgentId && selectedAgentId !== threadBoundAgentId) {
      setSelectedAgentId(threadBoundAgentId);
    }

    let activeThreadId = selectedThreadId;
    if (!activeThreadId) {
      try {
        const createdThread = await createThread();
        if (!createdThread) return;
        activeThreadId = createdThread.id;
      } catch (createErr: any) {
        alert(`Failed to start a new discussion: ${createErr.message}`);
        return;
      }
    }

    const newMessageIndex = messages.length + 1;
    setMessages(prev => [
      ...prev,
      { role: 'user', content: userMsg },
      { role: 'assistant', content: '', steps: [], showDetails: false },
    ]);
    setActiveAssistantMessageIndex(newMessageIndex);
    setIsLoading(true);

    try {
      const res = await fetch(`/api/chat/${agentIdForChat}?stream=true`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg, threadId: activeThreadId }),
      });

      if (!res.ok) {
        const errorMessage = await readErrorMessage(res);
        setMessages(prev => {
          const next = [...prev];
          next[newMessageIndex] = { role: 'assistant', content: `Error: ${errorMessage}` };
          return next;
        });
        return;
      }

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      let responseThreadId = activeThreadId;

      if (reader) {
        let buffer = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const events = buffer.split('\n\n');
          buffer = events.pop() || '';

          for (const rawEvent of events) {
            const lines = rawEvent.split('\n').filter(line => line.trim() !== '');
            for (const line of lines) {
              if (!line.startsWith('data: ')) continue;

              try {
                const data = JSON.parse(line.slice(6));

                if (typeof data.threadId === 'number') {
                  responseThreadId = data.threadId;
                }

                setMessages(prev => {
                  const next = [...prev];
                  const current = { ...next[newMessageIndex] };

                  if (data.type === 'step') {
                    current.steps = [...(current.steps || []), data.step];
                  } else if (data.type === 'result') {
                    const fallbackFromSteps = deriveFallbackAnswerFromSteps(data.steps);
                    current.content = data.response || fallbackFromSteps || current.content || 'Le manager a terminé sans réponse finale explicite.';
                    current.details = data.details;
                    if (Array.isArray(data.steps)) {
                      current.steps = data.steps;
                    }
                  } else if (data.type === 'error') {
                    current.content = `Error: ${data.error}`;
                  }

                  next[newMessageIndex] = current;
                  return next;
                });
              } catch (parseErr) {
                console.error('Error parsing SSE data', parseErr, line);
              }
            }
          }
        }
      }

      if (responseThreadId !== selectedThreadId) {
        setSelectedThreadId(responseThreadId);
      }
      await refreshThreads(responseThreadId);
    } catch (e: any) {
      setMessages(prev => {
        const next = [...prev];
        next[newMessageIndex] = { role: 'assistant', content: `Error: ${e.message}` };
        return next;
      });
    } finally {
      setIsLoading(false);
      setActiveAssistantMessageIndex(null);
    }
  };

  return (
    <div className="h-full bg-zinc-950 flex">
      <aside className="w-80 border-r border-zinc-800 bg-zinc-900/80 flex flex-col">
        <div className="p-4 border-b border-zinc-800 space-y-3">
          <button
            onClick={startNewDiscussion}
            className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 px-4 py-2.5 text-sm font-medium text-white transition-colors"
          >
            <Plus className="w-4 h-4" />
            Nouvelle discussion
          </button>
          <div className="space-y-1">
            <label className="text-xs font-medium text-zinc-400">Agent for new discussions</label>
            <select
              value={selectedAgentId}
              onChange={e => setSelectedAgentId(e.target.value)}
              disabled={isLoading}
              className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-white focus:ring-2 focus:ring-emerald-500 focus:border-transparent outline-none disabled:opacity-50"
            >
              {agents.map(agent => (
                <option key={agent.id} value={agent.id}>{agent.name}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-2">
          {threads.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-zinc-500 text-sm px-4 text-center">
              <MessageSquare className="w-10 h-10 mb-3 opacity-40" />
              <p>No discussion yet.</p>
              <p className="mt-1">Create one to start chatting.</p>
            </div>
          ) : (
            <div className="space-y-1">
              {threads.map(thread => (
                <button
                  key={thread.id}
                  onClick={() => openThread(thread.id)}
                  disabled={isLoading}
                  className={`w-full text-left rounded-xl px-3 py-2.5 border transition-colors ${
                    selectedThreadId === thread.id
                      ? 'bg-emerald-500/10 border-emerald-500/30'
                      : 'bg-zinc-900/20 border-transparent hover:bg-zinc-800/70'
                  } disabled:opacity-50`}
                >
                  <p className="text-sm font-medium text-white truncate">{thread.title || `Discussion ${thread.id}`}</p>
                  <p className="text-xs text-zinc-400 mt-0.5 line-clamp-2">
                    {thread.last_message_preview || 'Empty conversation'}
                  </p>
                </button>
              ))}
            </div>
          )}
        </div>
      </aside>

      <section className="flex-1 flex flex-col">
        <div className="flex-none p-4 border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-md sticky top-0 z-10 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-emerald-500/10 text-emerald-500 rounded-lg">
              <Bot className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">Agent Chat</h1>
              <p className="text-xs text-zinc-400">
                {selectedThread ? selectedThread.title : 'No discussion selected'}
              </p>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-zinc-500 space-y-4">
              <Bot className="w-16 h-16 opacity-20" />
              <p>Select a discussion or start a new one.</p>
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
                  {msg.role === 'assistant' && msg.steps && msg.steps.length > 0 && (
                    <div className="mb-3 flex justify-end">
                      <button
                        type="button"
                        onClick={() => toggleDetails(idx)}
                        className="inline-flex items-center gap-1 rounded-md border border-zinc-700 bg-zinc-900/60 px-2 py-1 text-xs text-zinc-300 hover:text-white hover:border-zinc-500 transition-colors"
                      >
                        {msg.showDetails ? 'Hide details' : 'Détail'}
                        {msg.showDetails ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                      </button>
                    </div>
                  )}

                  <div className="prose prose-invert prose-emerald max-w-none text-sm leading-relaxed">
                    <Markdown>{msg.content || (msg.role === 'assistant' && msg.steps?.length ? '*Working...*' : '')}</Markdown>
                  </div>

                  {msg.steps && msg.steps.length > 0 && msg.showDetails && (
                    <div className="mt-4 pt-4 border-t border-zinc-700">
                      <p className="text-xs font-semibold text-zinc-400 mb-2 uppercase tracking-wider">Execution Steps (Live)</p>
                      <ul className="space-y-2">
                        {msg.steps.map((step, i) => {
                          const stepStatus = String(step.status || 'info');
                          return (
                            <li key={i} className="text-sm bg-zinc-900/50 p-3 rounded-lg border border-zinc-700/50">
                              <div className="flex items-center gap-2 mb-1">
                                <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                                  stepStatus.includes('error') || stepStatus.includes('failed') ? 'bg-red-500/20 text-red-400' :
                                  stepStatus.includes('completed') || stepStatus.includes('final') ? 'bg-emerald-500/20 text-emerald-400' :
                                  'bg-blue-500/20 text-blue-400'
                                }`}>
                                  {stepStatus}
                                </span>
                                {step.agent && <span className="text-zinc-300 font-medium">{step.agent}</span>}
                              </div>
                              {step.message && <p className="text-zinc-300 text-xs mt-1">{step.message}</p>}
                              {step.rationale && <p className="text-zinc-400 text-xs mt-1 italic">{step.rationale}</p>}
                              {step.plan && <p className="text-zinc-300 text-xs mt-1 border-l-2 border-zinc-600 pl-2">{step.plan}</p>}
                              {step.input && <p className="text-zinc-400 text-xs mt-1"><span className="font-semibold text-zinc-500">Input:</span> {step.input}</p>}
                              {step.sql && <pre className="mt-2 text-xs text-blue-400 bg-zinc-950 p-2 rounded overflow-x-auto border border-zinc-800">{step.sql}</pre>}
                              {step.result && step.result.answer && <p className="text-zinc-300 text-xs mt-2 bg-zinc-800/50 p-2 rounded"><span className="font-semibold text-zinc-500">Result:</span> {step.result.answer}</p>}
                              {step.error && <p className="text-red-400 text-xs mt-1">{step.error}</p>}
                            </li>
                          );
                        })}
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
                <button
                  type="button"
                  onClick={() => {
                    if (activeAssistantMessageIndex !== null) {
                      toggleDetails(activeAssistantMessageIndex);
                    }
                  }}
                  disabled={activeAssistantMessageIndex === null}
                  className="ml-2 inline-flex items-center gap-1 rounded-md border border-zinc-700 bg-zinc-900/60 px-2 py-1 text-xs text-zinc-300 hover:text-white hover:border-zinc-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  {activeAssistantMessageIndex !== null && messages[activeAssistantMessageIndex]?.showDetails ? 'Masquer détail' : 'Détail'}
                  {activeAssistantMessageIndex !== null && messages[activeAssistantMessageIndex]?.showDetails ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                </button>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="flex-none p-4 bg-zinc-900 border-t border-zinc-800">
          <form onSubmit={sendMessage} className="max-w-4xl mx-auto relative flex items-center">
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder={selectedThreadId ? 'Ask your agent something...' : 'Start by creating/selecting a discussion...'}
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
      </section>
    </div>
  );
}
