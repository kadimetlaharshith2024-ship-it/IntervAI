import React, { forwardRef } from 'react';

export const Input = forwardRef(function Input(
  {
    label,
    error,
    helperText,
    iconLeft: IconLeft,
    iconRight: IconRight,
    className = '',
    id,
    disabled = false,
    ...props
  },
  ref
) {
  const inputId = id || (label ? label.toLowerCase().replace(/\s+/g, '-') : undefined);

  return (
    <div className="w-full space-y-1.5 text-left">
      {label && (
        <label
          htmlFor={inputId}
          className="block text-xs font-semibold tracking-tight text-slate-700 dark:text-slate-300"
        >
          {label}
        </label>
      )}

      <div className="relative flex items-center">
        {IconLeft && (
          <div className="absolute left-3.5 pointer-events-none text-slate-400 dark:text-slate-500">
            <IconLeft className="w-4 h-4" />
          </div>
        )}

        <input
          ref={ref}
          id={inputId}
          disabled={disabled}
          className={`w-full bg-white dark:bg-[#0C0E14] text-slate-900 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-500 text-sm rounded-xl border transition-all duration-150 py-2.5 px-3.5 focus:outline-none focus:ring-2 disabled:opacity-50 disabled:cursor-not-allowed ${
            IconLeft ? 'pl-10' : ''
          } ${IconRight ? 'pr-10' : ''} ${
            error
              ? 'border-rose-400 dark:border-rose-500/80 focus:border-rose-500 focus:ring-rose-500/20'
              : 'border-slate-300 dark:border-[#222838] focus:border-indigo-500 dark:focus:border-indigo-500 focus:ring-indigo-500/20'
          } ${className}`}
          {...props}
        />

        {IconRight && (
          <div className="absolute right-3.5 text-slate-400 dark:text-slate-500">
            {IconRight}
          </div>
        )}
      </div>

      {error ? (
        <p className="text-xs text-rose-600 dark:text-rose-400 font-medium">{error}</p>
      ) : helperText ? (
        <p className="text-xs text-slate-500 dark:text-slate-400">{helperText}</p>
      ) : null}
    </div>
  );
});

export default Input;
