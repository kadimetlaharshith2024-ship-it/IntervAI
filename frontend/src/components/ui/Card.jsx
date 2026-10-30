import React from 'react';

export function Card({ children, className = '', hover = false, ...props }) {
  return (
    <div
      className={`bg-white dark:bg-[#11141C] border border-slate-200/90 dark:border-[#202636] rounded-2xl shadow-xs transition-all duration-150 ${
        hover ? 'hover:border-slate-300 dark:hover:border-[#2D364B] hover:shadow-sm' : ''
      } ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardHeader({ children, className = '', ...props }) {
  return (
    <div className={`p-5 sm:p-6 border-b border-slate-100 dark:border-[#1C2230] ${className}`} {...props}>
      {children}
    </div>
  );
}

export function CardTitle({ children, className = '', as: Component = 'h3', ...props }) {
  return (
    <Component className={`text-base sm:text-lg font-bold tracking-tight text-slate-900 dark:text-slate-100 ${className}`} {...props}>
      {children}
    </Component>
  );
}

export function CardDescription({ children, className = '', ...props }) {
  return (
    <p className={`text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1 leading-relaxed ${className}`} {...props}>
      {children}
    </p>
  );
}

export function CardContent({ children, className = '', ...props }) {
  return (
    <div className={`p-5 sm:p-6 ${className}`} {...props}>
      {children}
    </div>
  );
}

export function CardFooter({ children, className = '', ...props }) {
  return (
    <div className={`p-5 sm:p-6 pt-0 border-t border-slate-100 dark:border-[#1C2230] mt-auto flex items-center ${className}`} {...props}>
      {children}
    </div>
  );
}

export default Card;
