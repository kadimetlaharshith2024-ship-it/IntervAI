import React from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Compass, Home, LogIn } from 'lucide-react';
import { Button, ThemeToggle } from '../components/ui';

export default function NotFound() {
  const { isAuthenticated } = useAuth();

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-[#090B0F] text-slate-800 dark:text-slate-200 flex flex-col items-center justify-center p-6 text-center font-sans transition-colors relative">
      <div className="absolute top-6 right-6">
        <ThemeToggle variant="dropdown" />
      </div>

      <div className="w-16 h-16 rounded-2xl bg-white dark:bg-[#121620] border border-slate-200 dark:border-[#222838] flex items-center justify-center mb-6 shadow-xs text-indigo-600 dark:text-indigo-400">
        <Compass className="w-8 h-8" />
      </div>

      <span className="text-xs font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400 mb-2">
        Error 404
      </span>
      <h1 className="text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight">
        Page Not Found
      </h1>
      <p className="mt-3 text-sm text-slate-500 dark:text-slate-400 max-w-sm leading-relaxed">
        The route you requested does not exist or has been moved to another location.
      </p>

      <div className="mt-8 flex items-center gap-3">
        {isAuthenticated ? (
          <Button
            variant="primary"
            size="md"
            iconLeft={Home}
            onClick={() => (window.location.href = '/dashboard')}
          >
            Return to Dashboard
          </Button>
        ) : (
          <Button
            variant="primary"
            size="md"
            iconLeft={LogIn}
            onClick={() => (window.location.href = '/login')}
          >
            Go to Sign In
          </Button>
        )}
      </div>
    </div>
  );
}
