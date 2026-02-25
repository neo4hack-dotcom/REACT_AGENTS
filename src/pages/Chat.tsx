import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Send, Bot, User, Loader2, ChevronDown, ChevronUp, Plus, MessageSquare, Trash2, RotateCcw } from 'lucide-react';
import Markdown from 'react-markdown';

type ChatMessage = {
  local_id: string;
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
  const [activeAssistantMessageId, setActiveAssistantMessageId] = useState<string | null>(null);
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
  const activeAssistantMessage = useMemo(
    () => messages.find(message => message.local_id === activeAssistantMessageId) || null,
    [messages, activeAssistantMessageId]
  );

  const markdownComponents = useMemo(() => ({
    p: ({ children }: any) => <p className="mb-3 last:mb-0 leading-7">{children}</p>,
    a: ({ href, children }: any) => (
      <a
        href={href}
        target="_blank"
        rel="noreferrer"
        className="font-medium text-sky-700 underline decoration-sky-300 underline-offset-2 hover:text-sky-900"
      >
        {children}
      </a>
    ),
    pre: ({ children }: any) => (
      <pre className="my-3 overflow-x-auto rounded-xl border border-slate-300 bg-slate-950/95 px-3 py-3 text-xs text-slate-100 shadow-inner">
        {children}
      </pre>
    ),
    code: ({ inline, children }: any) => (
      inline
        ? <code className="rounded bg-slate-200 px-1.5 py-0.5 text-[0.85em] text-slate-900">{children}</code>
        : <code className="text-slate-100">{children}</code>
    ),
    blockquote: ({ children }: any) => (
      <blockquote className="my-3 border-l-4 border-slate-300 bg-slate-100 px-3 py-2 text-slate-700">
        {children}
      </blockquote>
    ),
    ul: ({ children }: any) => <ul className="my-3 list-disc space-y-1 pl-6">{children}</ul>,
    ol: ({ children }: any) => <ol className="my-3 list-decimal space-y-1 pl-6">{children}</ol>,
    table: ({ children }: any) => (
      <div className="my-3 overflow-x-auto rounded-xl border border-slate-300">
        <table className="w-full min-w-[360px] border-collapse text-sm">{children}</table>
      </div>
    ),
    th: ({ children }: any) => <th className="border-b border-slate-300 bg-slate-100 px-3 py-2 text-left font-semibold text-slate-700">{children}</th>,
    td: ({ children }: any) => <td className="border-b border-slate-200 px-3 py-2 align-top">{children}</td>,
    hr: () => <hr className="my-4 border-slate-300" />,
  }), []);

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

  const createLocalMessageId = (prefix: string): string => {
    return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  };

  const mapStoredMessages = (threadMessages: any[] | undefined): ChatMessage[] => {
    if (!Array.isArray(threadMessages)) return [];
    return threadMessages.map((msg: any, index: number) => {
      const details = msg?.details;
      const steps = Array.isArray(details?.steps) ? details.steps : undefined;
      return {
        local_id: String(msg?.local_id || `stored-${msg?.created_at || 'na'}-${index}`),
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

  const clearThread = async (threadId: number) => {
    if (isLoading) return;
    const thread = threads.find(item => item.id === threadId);
    const threadLabel = thread?.title || `Discussion ${threadId}`;
    const confirmed = window.confirm(`Clear all messages in "${threadLabel}"?`);
    if (!confirmed) return;

    const res = await fetch(`/api/chat/threads/${threadId}/clear`, { method: 'POST' });
    if (!res.ok) {
      throw new Error(await readErrorMessage(res));
    }

    setMessages(prev => (selectedThreadId === threadId ? [] : prev));
    setThreads(prev => prev.map(item => (
      item.id === threadId
        ? { ...item, message_count: 0, last_message_preview: '', updated_at: new Date().toISOString() }
        : item
    )));

    await refreshThreads(threadId);
    if (selectedThreadId === threadId) {
      await syncActiveThreadMessages(threadId);
    }
  };

  const deleteThread = async (threadId: number) => {
    if (isLoading) return;
    const thread = threads.find(item => item.id === threadId);
    const threadLabel = thread?.title || `Discussion ${threadId}`;
    const confirmed = window.confirm(`Delete "${threadLabel}" permanently?`);
    if (!confirmed) return;

    const res = await fetch(`/api/chat/threads/${threadId}`, { method: 'DELETE' });
    if (!res.ok) {
      throw new Error(await readErrorMessage(res));
    }

    const keepSelectionId = selectedThreadId && selectedThreadId !== threadId ? selectedThreadId : undefined;
    if (selectedThreadId === threadId) {
      setSelectedThreadId(null);
      setMessages([]);
    }
    setThreads(prev => prev.filter(item => item.id !== threadId));
    await refreshThreads(keepSelectionId);
  };

  const syncActiveThreadMessages = async (threadId: number | null) => {
    if (!threadId) return;
    try {
      const detail = await fetchThreadDetail(threadId);
      if (!detail) return;
      if (detail.agent_id) {
        setSelectedAgentId(String(detail.agent_id));
      }
      setMessages(mapStoredMessages(detail.messages));
    } catch (e) {
      console.error('Failed to sync thread messages', e);
    }
  };

  const formatThreadTimestamp = (value?: string): string => {
    if (!value) return '';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return '';
    return new Intl.DateTimeFormat('fr-FR', {
      day: '2-digit',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  };

  const toggleDetails = (messageId: string) => {
    setMessages(prev => {
      return prev.map(message => (
        message.local_id === messageId
          ? { ...message, showDetails: !message.showDetails }
          : message
      ));
    });
  };

  const summarizeLiveStep = (step: any): string => {
    const message = String(step?.message || '').trim();
    if (message) return message;
    const rationale = String(step?.rationale || '').trim();
    if (rationale) return rationale;
    const status = String(step?.status || 'thinking').replace(/_/g, ' ');
    return `Execution: ${status}`;
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

    const userMessageId = createLocalMessageId('user');
    const assistantMessageId = createLocalMessageId('assistant');
    setMessages(prev => [
      ...prev,
      { local_id: userMessageId, role: 'user', content: userMsg },
      { local_id: assistantMessageId, role: 'assistant', content: 'Preparing execution...', steps: [], showDetails: true },
    ]);
    setActiveAssistantMessageId(assistantMessageId);
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
          return prev.map(message => (
            message.local_id === assistantMessageId
              ? { ...message, content: `Error: ${errorMessage}` }
              : message
          ));
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
          const events = buffer.split(/\r?\n\r?\n/);
          buffer = events.pop() || '';

          for (const rawEvent of events) {
            const lines = rawEvent.split(/\r?\n/).filter(line => line.trim() !== '');
            for (const line of lines) {
              if (!line.startsWith('data:')) continue;

              try {
                const data = JSON.parse(line.slice(5).trimStart());

                if (typeof data.threadId === 'number') {
                  responseThreadId = data.threadId;
                }

                setMessages(prev => {
                  const next = [...prev];
                  const targetIndex = next.findIndex(message => message.local_id === assistantMessageId);
                  if (targetIndex < 0) return next;

                  const current = { ...next[targetIndex] };

                  if (data.type === 'step') {
                    current.steps = [...(current.steps || []), data.step];
                    current.showDetails = true;
                    if (!current.content || current.content.startsWith('Preparing execution') || current.content.startsWith('Execution:')) {
                      current.content = summarizeLiveStep(data.step);
                    }
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

                  next[targetIndex] = current;
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
      await syncActiveThreadMessages(responseThreadId);
    } catch (e: any) {
      setMessages(prev => {
        return prev.map(message => (
          message.local_id === assistantMessageId
            ? { ...message, content: `Error: ${e.message}` }
            : message
        ));
      });
    } finally {
      setIsLoading(false);
      setActiveAssistantMessageId(null);
    }
  };

  return (
    <div className="h-full bg-gradient-to-br from-slate-100 via-white to-sky-50 text-slate-900 flex">
      <aside className="w-80 border-r border-slate-200 bg-white/85 backdrop-blur-md flex flex-col shadow-sm">
        <div className="p-4 border-b border-slate-200 space-y-3">
          <button
            onClick={startNewDiscussion}
            className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-slate-900 hover:bg-slate-800 px-4 py-2.5 text-sm font-medium text-white transition-colors"
          >
            <Plus className="w-4 h-4" />
            Nouvelle discussion
          </button>
          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-500">Agent for new discussions</label>
            <select
              value={selectedAgentId}
              onChange={e => setSelectedAgentId(e.target.value)}
              disabled={isLoading}
              className="w-full bg-white border border-slate-300 rounded-lg px-3 py-2 text-sm text-slate-900 focus:ring-2 focus:ring-sky-500 focus:border-transparent outline-none disabled:opacity-50"
            >
              {agents.map(agent => (
                <option key={agent.id} value={agent.id}>{agent.name}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-2">
          {threads.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-500 text-sm px-4 text-center">
              <MessageSquare className="w-10 h-10 mb-3 opacity-40" />
              <p>No discussion yet.</p>
              <p className="mt-1">Create one to start chatting.</p>
            </div>
          ) : (
            <div className="space-y-1.5">
              {threads.map(thread => (
                <div
                  key={thread.id}
                  className={`flex items-stretch gap-1 rounded-xl border p-1 transition-colors ${
                    selectedThreadId === thread.id
                      ? 'border-sky-300 bg-sky-50'
                      : 'border-transparent bg-white hover:bg-slate-50'
                  }`}
                >
                  <button
                    onClick={() => openThread(thread.id)}
                    disabled={isLoading}
                    className="flex-1 text-left rounded-lg px-3 py-2.5 disabled:opacity-50"
                  >
                    <p className="text-sm font-semibold text-slate-900 truncate">{thread.title || `Discussion ${thread.id}`}</p>
                    <p className="text-xs text-slate-500 mt-0.5 line-clamp-2">
                      {thread.last_message_preview || 'Empty conversation'}
                    </p>
                    {thread.updated_at && (
                      <p className="mt-1 text-[11px] text-slate-400">{formatThreadTimestamp(thread.updated_at)}</p>
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={async () => {
                      try {
                        await deleteThread(thread.id);
                      } catch (e: any) {
                        alert(`Failed to delete discussion: ${e.message}`);
                      }
                    }}
                    disabled={isLoading}
                    className="self-center rounded-lg p-2 text-slate-400 hover:text-red-600 hover:bg-red-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                    title="Delete conversation"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </aside>

      <section className="flex-1 flex flex-col">
        <div className="flex-none px-5 py-4 border-b border-slate-200 bg-white/80 backdrop-blur-md sticky top-0 z-10 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-slate-100 text-slate-700 rounded-lg border border-slate-200">
              <Bot className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-900">Agent Chat</h1>
              <p className="text-xs text-slate-500">
                {selectedThread ? selectedThread.title : 'No discussion selected'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={!selectedThreadId || isLoading}
              onClick={async () => {
                if (!selectedThreadId) return;
                try {
                  await clearThread(selectedThreadId);
                } catch (e: any) {
                  alert(`Failed to clear discussion: ${e.message}`);
                }
              }}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              title="Clear messages in this discussion"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Clear
            </button>
            <button
              type="button"
              disabled={!selectedThreadId || isLoading}
              onClick={async () => {
                if (!selectedThreadId) return;
                try {
                  await deleteThread(selectedThreadId);
                } catch (e: any) {
                  alert(`Failed to delete discussion: ${e.message}`);
                }
              }}
              className="inline-flex items-center gap-1.5 rounded-lg border border-red-200 bg-red-50 px-3 py-1.5 text-xs font-medium text-red-700 hover:bg-red-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              title="Delete this conversation"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Delete
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6 bg-[radial-gradient(circle_at_20%_20%,rgba(125,211,252,0.15),transparent_45%),radial-gradient(circle_at_80%_5%,rgba(226,232,240,0.45),transparent_40%)]">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-500 space-y-4">
              <Bot className="w-16 h-16 opacity-25" />
              <p>Select a discussion or start a new one.</p>
            </div>
          ) : (
            messages.map(msg => (
              <div key={msg.local_id} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                {msg.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-full bg-sky-100 text-sky-700 border border-sky-200 flex items-center justify-center flex-shrink-0 shadow-sm">
                    <Bot className="w-5 h-5" />
                  </div>
                )}

                <div className={`max-w-[84%] rounded-2xl border px-5 py-3.5 shadow-sm ${
                  msg.role === 'user'
                    ? 'bg-white border-slate-300 rounded-tr-sm'
                    : 'bg-white/95 border-slate-200 rounded-tl-sm'
                }`}>
                  {msg.role === 'assistant' && msg.steps && msg.steps.length > 0 && (
                    <div className="mb-3 flex justify-end">
                      <button
                        type="button"
                        onClick={() => toggleDetails(msg.local_id)}
                        className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-slate-50 px-2 py-1 text-xs text-slate-600 hover:text-slate-900 hover:border-slate-400 transition-colors"
                      >
                        {msg.showDetails ? 'Hide details' : 'Détail'}
                        {msg.showDetails ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                      </button>
                    </div>
                  )}

                  <div className="prose prose-slate max-w-none text-[15px] leading-relaxed">
                    <Markdown components={markdownComponents}>
                      {msg.content || (msg.role === 'assistant' && msg.steps?.length ? '*Working...*' : '')}
                    </Markdown>
                  </div>

                  {msg.steps && msg.steps.length > 0 && msg.showDetails && (
                    <div className="mt-4 pt-4 border-t border-slate-200">
                      <p className="text-xs font-semibold text-slate-500 mb-2 uppercase tracking-wider">Execution Steps (Live)</p>
                      <ul className="space-y-2">
                        {msg.steps.map((step, i) => {
                          const stepStatus = String(step.status || 'info');
                          return (
                            <li key={i} className="text-sm bg-slate-50 p-3 rounded-lg border border-slate-200">
                              <div className="flex items-center gap-2 mb-1">
                                <span className={`px-2 py-0.5 rounded text-xs font-semibold ${
                                  stepStatus.includes('error') || stepStatus.includes('failed') ? 'bg-red-100 text-red-700' :
                                  stepStatus.includes('completed') || stepStatus.includes('final') ? 'bg-emerald-100 text-emerald-700' :
                                  'bg-sky-100 text-sky-700'
                                }`}>
                                  {stepStatus}
                                </span>
                                {step.agent && <span className="text-slate-800 font-medium">{step.agent}</span>}
                              </div>
                              {step.message && <p className="text-slate-700 text-xs mt-1">{step.message}</p>}
                              {step.rationale && <p className="text-slate-500 text-xs mt-1 italic">{step.rationale}</p>}
                              {step.plan && <p className="text-slate-700 text-xs mt-1 border-l-2 border-slate-300 pl-2">{step.plan}</p>}
                              {step.input && <p className="text-slate-500 text-xs mt-1"><span className="font-semibold text-slate-600">Input:</span> {step.input}</p>}
                              {step.sql && <pre className="mt-2 text-xs text-slate-100 bg-slate-900 p-2 rounded overflow-x-auto border border-slate-700">{step.sql}</pre>}
                              {step.result && step.result.answer && <p className="text-slate-700 text-xs mt-2 bg-white p-2 rounded border border-slate-200"><span className="font-semibold text-slate-600">Result:</span> {step.result.answer}</p>}
                              {step.error && <p className="text-red-600 text-xs mt-1">{step.error}</p>}
                            </li>
                          );
                        })}
                      </ul>
                    </div>
                  )}

                  {msg.details && msg.details.sql && (
                    <div className="mt-4 pt-4 border-t border-slate-200">
                      <p className="text-xs font-semibold text-slate-500 mb-2 uppercase tracking-wider">Generated SQL</p>
                      <pre className="text-xs text-slate-100 bg-slate-900 p-3 rounded-xl overflow-x-auto border border-slate-700">
                        {msg.details.sql}
                      </pre>
                    </div>
                  )}
                </div>

                {msg.role === 'user' && (
                  <div className="w-8 h-8 rounded-full bg-slate-100 text-slate-600 border border-slate-300 flex items-center justify-center flex-shrink-0 shadow-sm">
                    <User className="w-5 h-5" />
                  </div>
                )}
              </div>
            ))
          )}

          {isLoading && (
            <div className="flex gap-4 justify-start">
              <div className="w-8 h-8 rounded-full bg-sky-100 text-sky-700 border border-sky-200 flex items-center justify-center flex-shrink-0 shadow-sm">
                <Bot className="w-5 h-5" />
              </div>
              <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-sm px-5 py-3.5 flex items-center gap-2 shadow-sm">
                <Loader2 className="w-4 h-4 animate-spin text-sky-600" />
                <span className="text-sm text-slate-600">Agent is thinking...</span>
                <button
                  type="button"
                  onClick={() => {
                    if (activeAssistantMessageId) {
                      toggleDetails(activeAssistantMessageId);
                    }
                  }}
                  disabled={!activeAssistantMessageId}
                  className="ml-2 inline-flex items-center gap-1 rounded-md border border-slate-300 bg-slate-50 px-2 py-1 text-xs text-slate-600 hover:text-slate-900 hover:border-slate-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  {activeAssistantMessage?.showDetails ? 'Masquer détail' : 'Détail'}
                  {activeAssistantMessage?.showDetails ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                </button>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="flex-none p-4 bg-white/85 border-t border-slate-200 backdrop-blur-sm">
          <form onSubmit={sendMessage} className="max-w-4xl mx-auto relative flex items-center">
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder={selectedThreadId ? 'Ask your agent something...' : 'Start by creating/selecting a discussion...'}
              disabled={isLoading || !selectedAgentId}
              className="w-full bg-white border border-slate-300 rounded-full pl-6 pr-14 py-4 text-slate-900 placeholder:text-slate-400 focus:ring-2 focus:ring-sky-500 focus:border-transparent outline-none disabled:opacity-50 shadow-sm"
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim() || !selectedAgentId}
              className="absolute right-2 p-2.5 bg-slate-900 hover:bg-slate-800 text-white rounded-full transition-colors disabled:opacity-50 disabled:hover:bg-slate-900"
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
        </div>
      </section>
    </div>
  );
}
