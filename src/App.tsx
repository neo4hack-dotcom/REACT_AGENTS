import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation, Navigate } from 'react-router-dom';
import { LayoutDashboard, Settings, Bot, MessageSquare, LogOut } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { AuthProvider, useAuth } from './context/AuthContext';

// Pages
import Dashboard from './pages/Dashboard';
import Configuration from './pages/Configuration';
import Agents from './pages/Agents';
import Chat from './pages/Chat';
import Login from './pages/Login';
import Register from './pages/Register';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

function Sidebar() {
  const location = useLocation();
  const { logout, user } = useAuth();
  
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
        <div className="flex items-center justify-between px-4 py-3 bg-gray-50 rounded-2xl border border-gray-100">
          <span className="text-sm font-medium text-gray-700 truncate max-w-[100px]">{user?.username}</span>
          <button onClick={logout} className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-xl transition-colors" title="Logout">
            <LogOut className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
}

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  
  if (loading) {
    return <div className="h-screen flex items-center justify-center bg-gray-50 text-emerald-500"><Bot className="w-12 h-12 animate-pulse" /></div>;
  }
  
  if (!user) {
    return <Navigate to="/login" />;
  }
  
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
    <AuthProvider>
      <Router>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
          <Route path="/agents" element={<ProtectedRoute><Agents /></ProtectedRoute>} />
          <Route path="/config" element={<ProtectedRoute><Configuration /></ProtectedRoute>} />
          <Route path="/chat" element={<ProtectedRoute><Chat /></ProtectedRoute>} />
        </Routes>
      </Router>
    </AuthProvider>
  );
}
