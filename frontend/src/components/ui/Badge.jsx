import React from 'react';

const VARIANTS = {
  indigo: {
    container: 'bg-indigo-50 text-indigo-700 border-indigo-200/80 dark:bg-indigo-500/10 dark:text-indigo-300 dark:border-indigo-500/20',
    dot: 'bg-indigo-500 dark:bg-indigo-400',
  },
  emerald: {
    container: 'bg-emerald-50 text-emerald-700 border-emerald-200/80 dark:bg-emerald-500/10 dark:text-emerald-300 dark:border-emerald-500/20',
    dot: 'bg-emerald-500 dark:bg-emerald-400',
  },
  amber: {
    container: 'bg-amber-50 text-amber-800 border-amber-200/80 dark:bg-amber-500/10 dark:text-amber-300 dark:border-amber-500/20',
    dot: 'bg-amber-500 dark:bg-amber-400',
  },
  rose: {
    container: 'bg-rose-50 text-rose-700 border-rose-200/80 dark:bg-rose-500/10 dark:text-rose-300 dark:border-rose-500/20',
    dot: 'bg-rose-500 dark:bg-rose-400',
  },
  sky: {
    container: 'bg-sky-50 text-sky-700 border-sky-200/80 dark:bg-sky-500/10 dark:text-sky-300 dark:border-sky-500/20',
    dot: 'bg-sky-500 dark:bg-sky-400',
  },
  purple: {
    container: 'bg-purple-50 text-purple-700 border-purple-200/80 dark:bg-purple-500/10 dark:text-purple-300 dark:border-purple-500/20',
    dot: 'bg-purple-500 dark:bg-purple-400',
  },
  neutral: {
    container: 'bg-slate-100 text-slate-700 border-slate-200 dark:bg-[#181D28] dark:text-slate-300 dark:border-[#283244]',
    dot: 'bg-slate-400 dark:bg-slate-500',
  },
};

export default function Badge({
  children,
  variant = 'neutral',
  size = 'md',
  dot = false,
  className = '',
  icon: Icon,
  ...props
}) {
  const config = VARIANTS[variant] || VARIANTS.neutral;
  const sizeClasses = size === 'sm' ? 'px-2 py-0.5 text-[11px] gap-1' : 'px-2.5 py-1 text-xs gap-1.5';

  return (
    <span
      className={`inline-flex items-center font-semibold rounded-full border tracking-tight ${sizeClasses} ${config.container} ${className}`}
      {...props}
    >
      {dot && <span className={`w-1.5 h-1.5 rounded-full ${config.dot}`} />}
      {Icon && <Icon className="w-3 h-3 flex-shrink-0" />}
      <span>{children}</span>
    </span>
  );
}
