import React, { useState, useEffect, useRef } from 'react';
import { Outlet, useNavigate, NavLink, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';
import {
  Sparkles,
  LogOut,
  LayoutDashboard,
  FileText,
  BrainCircuit,
  History,
  Menu,
  X,
  ChevronDown,
  PanelLeftClose,
  PanelLeft,
} from 'lucide-react';
import ThemeToggle from './ui/ThemeToggle';

const CURRENT_YEAR = new Date().getFullYear();
const SIDEBAR_STORAGE_KEY = 'intervai_sidebar_collapsed';

export default function AppLayout() {
  const { user, logout } = useAuth();
  const { theme, resolvedTheme, setTheme } = useTheme();
  const navigate = useNavigate();
  const location = useLocation();

  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [profileDropdownOpen, setProfileDropdownOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    try {
      return localStorage.getItem(SIDEBAR_STORAGE_KEY) === 'true';
    } catch {
      return false;
    }
  });

  const profileMenuRef = useRef(null);

  // Close menus on route change
  useEffect(() => {
    setMobileMenuOpen(false);
    setProfileDropdownOpen(false);
  }, [location.pathname]);

  // Persist sidebar collapsed state
  const toggleSidebar = () => {
    setSidebarCollapsed((prev) => {
      const next = !prev;
      try {
        localStorage.setItem(SIDEBAR_STORAGE_KEY, String(next));
      } catch {}
      return next;
    });
  };

  // Close profile dropdown on outside click
  useEffect(() => {
    function handleClickOutside(event) {
      if (profileMenuRef.current && !profileMenuRef.current.contains(event.target)) {
        setProfileDropdownOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleLogout = () => {
    logout();
    navigate('/login', { replace: true });
  };

  const getInitials = (name) => {
    if (!name) return 'U';
    return name
      .split(' ')
      .map((part) => part[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  const navLinks = [
    { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { to: '/resumes', label: 'Resumes', icon: FileText },
    { to: '/interview', label: 'Practice', icon: BrainCircuit },
    { to: '/history', label: 'History', icon: History },
  ];

  const getPageTitle = () => {
    const path = location.pathname;
    if (path.startsWith('/dashboard')) return 'Overview';
    if (path.startsWith('/resumes/')) return 'Resume Inspection';
    if (path.startsWith('/resumes')) return 'Resume Library';
    if (path.startsWith('/interview/')) return 'Interview Cockpit';
    if (path.startsWith('/interview')) return 'Practice Interview';
    if (path.startsWith('/history')) return 'Practice History';
    return 'Platform';
  };

  return (
    <div className="min-h-screen flex bg-slate-50 dark:bg-[#090B0F] text-slate-800 dark:text-slate-200 transition-colors duration-150">
      {/* =========================================================================
          DESKTOP SIDEBAR (FIXED VIEWPORT HEIGHT, STRUCTURED TOP-TO-BOTTOM)
         ========================================================================= */}
      <aside
        aria-label="Desktop sidebar navigation"
        className={`hidden md:flex flex-col sticky top-0 h-screen border-r border-slate-200/80 dark:border-[#1E2433] bg-white dark:bg-[#0D1017] transition-all duration-200 z-30 select-none flex-shrink-0 ${
          sidebarCollapsed ? 'w-18' : 'w-64'
        }`}
      >
        {/* 1. TOP: IntervAI Logo & Product Name */}
        <div className="h-14 flex items-center justify-between px-3.5 border-b border-slate-200/80 dark:border-[#1C2230] flex-shrink-0">
          <NavLink
            to="/dashboard"
            className="flex items-center gap-2.5 group focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 rounded-lg"
          >
            <div className="w-7 h-7 rounded-lg bg-indigo-600 text-white flex items-center justify-center shadow-xs flex-shrink-0">
              <Sparkles className="w-3.5 h-3.5" />
            </div>
            {!sidebarCollapsed && (
              <span className="text-sm font-bold tracking-tight text-slate-900 dark:text-white flex items-center gap-1.5">
                Interv<span className="text-indigo-600 dark:text-indigo-400">AI</span>
              </span>
            )}
          </NavLink>

          {/* Toggle Sidebar Width Button */}
          <button
            type="button"
            onClick={toggleSidebar}
            aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-[#161B26] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 cursor-pointer"
          >
            {sidebarCollapsed ? (
              <PanelLeft className="w-4 h-4" />
            ) : (
              <PanelLeftClose className="w-4 h-4" />
            )}
          </button>
        </div>

        {/* 2. IMMEDIATELY BELOW: User Profile Section with Dropdown */}
        {user && (
          <div
            className="p-2.5 border-b border-slate-200/80 dark:border-[#1C2230] flex-shrink-0 relative"
            ref={profileMenuRef}
          >
            {!sidebarCollapsed ? (
              <button
                type="button"
                onClick={() => setProfileDropdownOpen((prev) => !prev)}
                aria-expanded={profileDropdownOpen}
                aria-haspopup="true"
                className="w-full flex items-center gap-2.5 p-1.5 rounded-xl border border-slate-200/90 dark:border-[#222838] bg-slate-50/70 dark:bg-[#121620] hover:bg-slate-100 dark:hover:bg-[#171C28] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 cursor-pointer text-left"
              >
                <div className="w-7 h-7 rounded-lg bg-indigo-600 text-white font-bold text-xs flex items-center justify-center flex-shrink-0 shadow-2xs">
                  {getInitials(user.full_name)}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-semibold text-slate-900 dark:text-slate-100 truncate">
                    {user.full_name || 'Candidate'}
                  </p>
                  <p className="text-[10px] text-slate-500 dark:text-slate-400 truncate">
                    {user.email}
                  </p>
                </div>
                <ChevronDown
                  className={`w-3.5 h-3.5 text-slate-400 transition-transform duration-150 flex-shrink-0 ${
                    profileDropdownOpen ? 'rotate-180' : ''
                  }`}
                />
              </button>
            ) : (
              <div className="flex justify-center">
                <button
                  type="button"
                  onClick={() => setProfileDropdownOpen((prev) => !prev)}
                  title={`${user.full_name || 'Candidate'} (${user.email})`}
                  className="w-8 h-8 rounded-lg bg-indigo-600 text-white font-bold text-xs flex items-center justify-center shadow-2xs hover:opacity-90 transition-opacity focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 cursor-pointer"
                >
                  {getInitials(user.full_name)}
                </button>
              </div>
            )}

            {/* Profile Dropdown Menu */}
            {profileDropdownOpen && (
              <div
                className={`absolute bg-white dark:bg-[#141822] border border-slate-200 dark:border-[#222838] rounded-xl shadow-xl p-1.5 z-50 animate-fadeIn ${
                  sidebarCollapsed
                    ? 'top-0 left-full ml-2 w-52'
                    : 'top-full left-2.5 right-2.5 mt-1.5'
                }`}
              >
                <div className="px-2.5 py-1.5 border-b border-slate-100 dark:border-[#1E2533]">
                  <p className="text-xs font-bold text-slate-900 dark:text-slate-100 truncate">
                    {user.full_name}
                  </p>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400 truncate mt-0.5">
                    {user.email}
                  </p>
                </div>

                <div className="py-1">
                  <button
                    type="button"
                    onClick={() => {
                      navigate('/dashboard');
                      setProfileDropdownOpen(false);
                    }}
                    className="w-full px-2.5 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-[#1C2230] rounded-lg flex items-center gap-2 transition-colors cursor-pointer text-left"
                  >
                    <LayoutDashboard className="w-3.5 h-3.5 text-slate-400" />
                    <span>Dashboard</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      navigate('/resumes');
                      setProfileDropdownOpen(false);
                    }}
                    className="w-full px-2.5 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-[#1C2230] rounded-lg flex items-center gap-2 transition-colors cursor-pointer text-left"
                  >
                    <FileText className="w-3.5 h-3.5 text-slate-400" />
                    <span>Resumes</span>
                  </button>
                </div>

                <div className="pt-1 border-t border-slate-100 dark:border-[#1E2533]">
                  <button
                    type="button"
                    onClick={handleLogout}
                    className="w-full px-2.5 py-1.5 text-xs font-medium text-rose-600 dark:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-500/10 rounded-lg flex items-center gap-2 transition-colors cursor-pointer text-left"
                  >
                    <LogOut className="w-3.5 h-3.5" />
                    <span>Sign Out</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* 3. MIDDLE: Primary Navigation Items */}
        <nav className="flex-1 px-2.5 py-3 space-y-1 overflow-y-auto">
          {navLinks.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              title={sidebarCollapsed ? label : undefined}
              className={({ isActive }) => {
                const match = isActive || (to !== '/dashboard' && location.pathname.startsWith(to));
                return `flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs font-semibold tracking-tight transition-all duration-150 cursor-pointer ${
                  match
                    ? 'bg-indigo-50 dark:bg-indigo-500/12 text-indigo-700 dark:text-indigo-300 font-bold'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100/80 dark:hover:bg-[#161B26]'
                } ${sidebarCollapsed ? 'justify-center px-2' : ''}`;
              }}
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              {!sidebarCollapsed && <span>{label}</span>}
            </NavLink>
          ))}
        </nav>

        {/* 4. BOTTOM: Theme Selector & Utilities Separated by Subtle Divider */}
        <div className="p-3 border-t border-slate-200/80 dark:border-[#1C2230] flex-shrink-0 space-y-2">
          {!sidebarCollapsed ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between px-1">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">
                  Theme
                </span>
                <span className="text-[10px] font-mono text-slate-400 dark:text-slate-500">
                  IntervAI
                </span>
              </div>
              <ThemeToggle variant="segmented" className="w-full justify-between" />
            </div>
          ) : (
            <div className="flex justify-center">
              <ThemeToggle variant="dropdown" />
            </div>
          )}
        </div>
      </aside>

      {/* =========================================================================
          MAIN APPLICATION VIEWPORT
         ========================================================================= */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Sticky Top Header */}
        <header className="sticky top-0 z-40 h-14 border-b border-slate-200/80 dark:border-[#1E2433] bg-white/85 dark:bg-[#090B0F]/85 backdrop-blur-md flex items-center justify-between px-4 sm:px-6 lg:px-8 transition-colors">
          {/* Left: Mobile Brand & Desktop Breadcrumbs */}
          <div className="flex items-center gap-3 sm:gap-4">
            {/* Mobile Hamburger */}
            <button
              type="button"
              onClick={() => setMobileMenuOpen((prev) => !prev)}
              aria-label="Toggle navigation menu"
              className="md:hidden p-1.5 rounded-lg border border-slate-200 dark:border-[#222838] bg-white dark:bg-[#11141C] text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-[#181C26] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 cursor-pointer"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>

            {/* Mobile Brand */}
            <div className="flex md:hidden items-center gap-2">
              <div className="w-6 h-6 rounded-lg bg-indigo-600 text-white flex items-center justify-center shadow-xs">
                <Sparkles className="w-3 h-3" />
              </div>
              <span className="text-sm font-bold text-slate-900 dark:text-white">IntervAI</span>
            </div>

            {/* Desktop Breadcrumbs */}
            <div className="hidden md:flex items-center gap-2 text-xs font-medium text-slate-400 dark:text-slate-500">
              <span className="text-slate-500 dark:text-slate-400 font-semibold">Workspace</span>
              <span>/</span>
              <span className="text-slate-900 dark:text-slate-200 font-bold">{getPageTitle()}</span>
            </div>
          </div>

          {/* Right: Header Actions */}
          <div className="flex items-center gap-2.5">
            <ThemeToggle variant="dropdown" />

            <button
              type="button"
              onClick={() => navigate('/interview')}
              className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-2xs hover:shadow-xs transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 cursor-pointer"
            >
              <BrainCircuit className="w-3.5 h-3.5" />
              <span>Practice</span>
            </button>
          </div>
        </header>

        {/* Mobile Navigation Drawer (Profile Near Top, Navigation Middle, Theme Bottom) */}
        {mobileMenuOpen && (
          <div className="md:hidden border-b border-slate-200 dark:border-[#202737] bg-white dark:bg-[#0D1017] px-4 py-3 space-y-3 animate-fadeIn shadow-lg">
            {/* 1. Mobile Top: User Profile Section */}
            {user && (
              <div className="pb-3 border-b border-slate-100 dark:border-[#1E2533] flex items-center justify-between gap-3">
                <div className="flex items-center gap-2.5 min-w-0">
                  <div className="w-8 h-8 rounded-lg bg-indigo-600 text-white font-bold text-xs flex items-center justify-center flex-shrink-0">
                    {getInitials(user.full_name)}
                  </div>
                  <div className="min-w-0">
                    <p className="text-xs font-bold text-slate-900 dark:text-slate-100 truncate">
                      {user.full_name}
                    </p>
                    <p className="text-[11px] text-slate-500 dark:text-slate-400 truncate">
                      {user.email}
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={handleLogout}
                  className="text-xs font-semibold text-rose-600 dark:text-rose-400 hover:text-rose-700 p-1.5 rounded-lg hover:bg-rose-50 dark:hover:bg-rose-500/10 flex items-center gap-1.5 transition-colors cursor-pointer"
                >
                  <LogOut className="w-3.5 h-3.5" />
                  <span>Sign Out</span>
                </button>
              </div>
            )}

            {/* 2. Mobile Middle: Navigation Items */}
            <nav className="space-y-1">
              {navLinks.map(({ to, label, icon: Icon }) => (
                <NavLink
                  key={to}
                  to={to}
                  className={({ isActive }) =>
                    `px-3 py-2 rounded-xl text-xs font-semibold flex items-center gap-2.5 transition-colors ${
                      isActive || (to !== '/dashboard' && location.pathname.startsWith(to))
                        ? 'bg-indigo-50 dark:bg-indigo-500/15 text-indigo-700 dark:text-indigo-300 font-bold'
                        : 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-[#181D2A]'
                    }`
                  }
                >
                  <Icon className="w-4 h-4" />
                  <span>{label}</span>
                </NavLink>
              ))}
            </nav>

            {/* 3. Mobile Bottom: Theme Selector */}
            <div className="pt-2 border-t border-slate-100 dark:border-[#1E2533] space-y-1.5">
              <p className="px-1 text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">
                Theme
              </p>
              <ThemeToggle variant="segmented" className="w-full justify-between" />
            </div>
          </div>
        )}

        {/* Content Viewport */}
        <main className="flex-1 max-w-[1280px] w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <Outlet />
        </main>

        {/* Consistent Subtle Footer */}
        <footer className="border-t border-slate-200/80 dark:border-[#1C2230] bg-white dark:bg-[#0D1017] py-4 text-xs text-slate-500 dark:text-slate-400 transition-colors">
          <div className="max-w-[1280px] mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-2.5">
            <span className="font-medium text-[11px]">
              &copy; {CURRENT_YEAR} IntervAI. Resume-grounded adaptive technical mock interviews.
            </span>
            <div className="flex items-center gap-3">
              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 font-semibold text-[11px] border border-emerald-200/80 dark:border-emerald-500/20">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                Adaptive Engine Ready
              </span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}

