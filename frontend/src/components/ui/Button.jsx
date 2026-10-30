import React from 'react';
import { Loader2 } from 'lucide-react';

const VARIANTS = {
  primary:
    'bg-indigo-600 hover:bg-indigo-500 text-white shadow-xs hover:shadow-sm focus-visible:ring-2 focus-visible:ring-indigo-500/50 active:bg-indigo-700 dark:bg-indigo-600 dark:hover:bg-indigo-500 border border-transparent',
  secondary:
    'bg-white dark:bg-[#161B26] hover:bg-slate-50 dark:hover:bg-[#1E2433] text-slate-800 dark:text-slate-100 border border-slate-200 dark:border-[#262D3D] shadow-xs active:bg-slate-100 dark:active:bg-[#232B3D] focus-visible:ring-2 focus-visible:ring-indigo-500/30',
  outline:
    'bg-transparent hover:bg-slate-100 dark:hover:bg-[#171B26] text-slate-700 dark:text-slate-300 border border-slate-300 dark:border-[#2A3347] active:bg-slate-200 dark:active:bg-[#1F2535] focus-visible:ring-2 focus-visible:ring-indigo-500/30',
  ghost:
    'bg-transparent hover:bg-slate-100 dark:hover:bg-[#171B26] text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 active:bg-slate-200 dark:active:bg-[#1F2535] border border-transparent focus-visible:ring-2 focus-visible:ring-indigo-500/30',
  danger:
    'bg-rose-600 hover:bg-rose-500 text-white shadow-xs focus-visible:ring-2 focus-visible:ring-rose-500/50 active:bg-rose-700 dark:bg-rose-600 dark:hover:bg-rose-500 border border-transparent',
  success:
    'bg-emerald-600 hover:bg-emerald-500 text-white shadow-xs focus-visible:ring-2 focus-visible:ring-emerald-500/50 active:bg-emerald-700 dark:bg-emerald-600 dark:hover:bg-emerald-500 border border-transparent',
};

const SIZES = {
  xs: 'px-2.5 py-1 text-xs gap-1.5 rounded-lg',
  sm: 'px-3 py-1.5 text-xs font-medium gap-1.5 rounded-xl',
  md: 'px-4 py-2 text-sm font-semibold gap-2 rounded-xl',
  lg: 'px-5 py-2.5 text-base font-semibold gap-2.5 rounded-xl',
};

export default function Button({
  children,
  variant = 'primary',
  size = 'md',
  isLoading = false,
  disabled = false,
  iconLeft: IconLeft,
  iconRight: IconRight,
  className = '',
  type = 'button',
  ...props
}) {
  const variantClass = VARIANTS[variant] || VARIANTS.primary;
  const sizeClass = SIZES[size] || SIZES.md;

  return (
    <button
      type={type}
      disabled={disabled || isLoading}
      className={`inline-flex items-center justify-center font-sans tracking-tight transition-all duration-150 select-none cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none focus:outline-none ${variantClass} ${sizeClass} ${className}`}
      {...props}
    >
      {isLoading ? (
        <>
          <Loader2 className="w-4 h-4 animate-spin flex-shrink-0" />
          <span>{children}</span>
        </>
      ) : (
        <>
          {IconLeft && <IconLeft className="w-4 h-4 flex-shrink-0" />}
          <span>{children}</span>
          {IconRight && <IconRight className="w-4 h-4 flex-shrink-0" />}
        </>
      )}
    </button>
  );
}
