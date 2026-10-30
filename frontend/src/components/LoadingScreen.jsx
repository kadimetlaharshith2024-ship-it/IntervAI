import React from 'react';
import { Loader2 } from 'lucide-react';

export default function LoadingScreen({ message = 'Loading IntervAI...' }) {
  return (
    <div className="min-h-screen bg-slate-50 dark:bg-[#090B0F] text-slate-800 dark:text-slate-200 flex flex-col items-center justify-center p-4 transition-colors">
      <div className="relative flex items-center justify-center mb-4">
        <div className="w-12 h-12 rounded-full border-2 border-indigo-500/20 border-t-indigo-600 dark:border-t-indigo-500 animate-spin" />
        <Loader2 className="w-6 h-6 text-indigo-600 dark:text-indigo-400 absolute animate-pulse" />
      </div>
      <p className="text-sm font-semibold text-slate-500 dark:text-slate-400 tracking-wide">{message}</p>
    </div>
  );
}
