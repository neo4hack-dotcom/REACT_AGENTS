import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation, Navigate } from 'react-router-dom';
import { LayoutDashboard, Settings, Bot, MessageSquare } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

// Pages
import Dashboard from './pages/Dashboard';
import Configuration from './pages/Configuration';
import Agents from './pages/Agents';
import Chat from './pages/Chat';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

function Sidebar() {
  const location = useLocation();
  
  const links = [
    { name: 'Dashboard', path: '/', icon: LayoutDashboard },
    { name: 'Agents', path: '/agents', icon: Bot },
    { name: 'Configuration', path: '/config', icon: Settings },
    { name: 'Chat', path: '/chat', icon: MessageSquare },
  ];

  return (
    <div className="w-64 bg-white text-gray-600 flex flex-col h-screen border-r border-gray-200 shadow-sm">
      <div className="p-6 text-xl font-bold text-gray-900 flex items-center gap-3">
        <div className="p-2 bg-emerald-100 text-emerald-600 rounded-2xl">
          <Bot className="w-6 h-6" />
        </div>
        <span>AI Data Agents</span>
      </div>
      <nav className="flex-1 px-4 space-y-2 mt-4">
        {links.map((link) => {
          const Icon = link.icon;
          const isActive = location.pathname === link.path || (link.path !== '/' && location.pathname.startsWith(link.path));
          return (
            <Link
              key={link.name}
              to={link.path}
              className={cn(
                "flex items-center gap-3 px-4 py-3 rounded-2xl transition-all font-medium",
                isActive 
                  ? "bg-emerald-50 text-emerald-600 shadow-sm" 
                  : "hover:bg-gray-50 hover:text-gray-900"
              )}
            >
              <Icon className="w-5 h-5" />
              {link.name}
            </Link>
          );
        })}
      </nav>
      <div className="p-4 border-t border-gray-100">
        <div className="px-4 py-3 bg-gray-50 rounded-2xl border border-gray-100">
          <span className="text-sm font-medium text-gray-700">Local mode</span>
        </div>
      </div>
    </div>
  );
}

function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen bg-gray-50 text-gray-900 font-sans">
      <Sidebar />
      <main className="flex-1 overflow-auto">
        {children}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<AppLayout><Dashboard /></AppLayout>} />
        <Route path="/agents" element={<AppLayout><Agents /></AppLayout>} />
        <Route path="/config" element={<AppLayout><Configuration /></AppLayout>} />
        <Route path="/chat" element={<AppLayout><Chat /></AppLayout>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}
