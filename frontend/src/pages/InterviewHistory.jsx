import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  History,
  Calendar,
  BrainCircuit,
  FileText,
  AlertCircle,
  RefreshCw,
  Search,
  CheckCircle2,
  ChevronRight,
  Clock
} from 'lucide-react';
import { listUserSessions } from '../api/interview';
import { Button, Badge, Card, SkeletonCard } from '../components/ui';

export default function InterviewHistory() {
  const navigate = useNavigate();
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterStatus, setFilterStatus] = useState('ALL');

  const fetchSessions = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listUserSessions();
      setSessions(Array.isArray(data) ? data : []);
    } catch (err) {
      setError(err.message || 'Failed to retrieve your past interview sessions.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSessions();
  }, []);

  const getScoreBadge = (score) => {
    if (score === null || score === undefined) {
      return (
        <Badge variant="neutral" size="sm">
          Unscored
        </Badge>
      );
    }
    const rounded = Math.round(score);
    if (rounded >= 80) {
      return (
        <Badge variant="emerald" size="sm">
          {rounded} / 100
        </Badge>
      );
    }
    if (rounded >= 65) {
      return (
        <Badge variant="indigo" size="sm">
          {rounded} / 100
        </Badge>
      );
    }
    if (rounded >= 50) {
      return (
        <Badge variant="amber" size="sm">
          {rounded} / 100
        </Badge>
      );
    }
    return (
      <Badge variant="rose" size="sm">
        {rounded} / 100
      </Badge>
    );
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case 'COMPLETED':
        return (
          <Badge variant="emerald" size="sm" icon={CheckCircle2}>
            Completed
          </Badge>
        );
      case 'IN_PROGRESS':
        return (
          <Badge variant="amber" size="sm" dot>
            In Progress
          </Badge>
        );
      case 'ENDED':
        return (
          <Badge variant="neutral" size="sm">
            Concluded
          </Badge>
        );
      default:
        return (
          <Badge variant="neutral" size="sm">
            {status || 'Unknown'}
          </Badge>
        );
    }
  };

  const filteredSessions = sessions.filter((s) => {
    const matchesSearch =
      (s.id && s.id.toLowerCase().includes(searchQuery.toLowerCase())) ||
      (s.resume_filename && s.resume_filename.toLowerCase().includes(searchQuery.toLowerCase()));

    const matchesStatus = filterStatus === 'ALL' || s.status === filterStatus;

    return matchesSearch && matchesStatus;
  });

  return (
    <div className="space-y-6 max-w-6xl mx-auto animate-fadeIn">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200/80 dark:border-[#1E2433] pb-4">
        <div className="space-y-1">
          <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
            Interview History
          </h1>
          <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400">
            Review past mock interview sessions, composite scorecards, and evaluated responses.
          </p>
        </div>

        <Button
          variant="primary"
          size="sm"
          iconLeft={BrainCircuit}
          onClick={() => navigate('/interview')}
        >
          Start Interview
        </Button>
      </div>

      {/* Controls & Search */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="w-4 h-4 text-slate-400 dark:text-slate-500 absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by session ID or resume name..."
            className="w-full pl-10 pr-4 py-2.5 bg-white dark:bg-[#0C0E14] border border-slate-300 dark:border-[#222838] rounded-xl text-xs sm:text-sm text-slate-900 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all shadow-2xs"
          />
        </div>

        <div className="flex items-center gap-2">
          {['ALL', 'COMPLETED', 'IN_PROGRESS'].map((st) => (
            <button
              key={st}
              onClick={() => setFilterStatus(st)}
              className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all cursor-pointer ${
                filterStatus === st
                  ? 'bg-indigo-50 dark:bg-indigo-500/15 text-indigo-700 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-500/30'
                  : 'bg-white dark:bg-[#11141C] text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 border border-slate-200 dark:border-[#202636]'
              }`}
            >
              {st === 'ALL' ? 'All Sessions' : st === 'COMPLETED' ? 'Completed' : 'In Progress'}
            </button>
          ))}

          <button
            type="button"
            onClick={fetchSessions}
            disabled={loading}
            title="Refresh sessions"
            className="p-2.5 rounded-xl bg-white dark:bg-[#11141C] border border-slate-200 dark:border-[#202636] text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 transition-colors shadow-2xs cursor-pointer disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="p-4 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-2xl text-xs text-rose-800 dark:text-rose-300 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-600 dark:text-rose-400 flex-shrink-0" />
            <span>{error}</span>
          </div>
          <button
            onClick={fetchSessions}
            className="font-bold underline hover:text-rose-950 dark:hover:text-rose-200 cursor-pointer"
          >
            Retry
          </button>
        </div>
      )}

      {/* Loading state skeleton */}
      {loading && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && filteredSessions.length === 0 && (
        <Card className="p-12 text-center space-y-4">
          <div className="w-12 h-12 rounded-2xl bg-indigo-50 dark:bg-indigo-500/15 border border-indigo-200 dark:border-indigo-500/30 text-indigo-600 dark:text-indigo-400 flex items-center justify-center mx-auto">
            <History className="w-6 h-6" />
          </div>
          <div className="space-y-1">
            <h3 className="text-base font-bold text-slate-900 dark:text-white">
              No interview sessions found
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 max-w-sm mx-auto leading-relaxed">
              {searchQuery || filterStatus !== 'ALL'
                ? 'No sessions matched your search filters. Try clearing your search.'
                : 'You have not completed any mock interviews yet. Start your first session to receive adaptive practice and personalized feedback.'}
            </p>
          </div>
          <div className="pt-2">
            <Button
              variant="primary"
              size="sm"
              iconLeft={BrainCircuit}
              onClick={() => navigate('/interview')}
            >
              Start an Interview
            </Button>
          </div>
        </Card>
      )}

      {/* Sessions Grid */}
      {!loading && !error && filteredSessions.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredSessions.map((session) => {
            const startedAtFormatted = session.started_at
              ? new Date(session.started_at).toLocaleDateString(undefined, {
                  year: 'numeric',
                  month: 'short',
                  day: 'numeric',
                  hour: '2-digit',
                  minute: '2-digit',
                })
              : 'Recent Session';

            const score = session.overall_score;

            return (
              <Card
                key={session.id}
                hover
                onClick={() => navigate(`/interview/${session.id}`)}
                className="group cursor-pointer p-5 flex flex-col justify-between space-y-4"
              >
                {/* Header: Date & Status */}
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
                    <Calendar className="w-3.5 h-3.5 text-slate-400" />
                    <span>{startedAtFormatted}</span>
                  </div>
                  {getStatusBadge(session.status)}
                </div>

                {/* Score & Difficulty */}
                <div className="flex items-center justify-between bg-slate-50 dark:bg-[#0E1118] border border-slate-200/80 dark:border-[#1E2535] rounded-xl p-3.5">
                  <div className="space-y-1">
                    <span className="text-[10px] uppercase font-bold text-slate-500 dark:text-slate-400 tracking-wider block">
                      Scorecard
                    </span>
                    <div>{getScoreBadge(score)}</div>
                  </div>

                  <div className="text-right space-y-1">
                    <span className="text-[10px] uppercase font-bold text-slate-500 dark:text-slate-400 tracking-wider block">
                      Starting Level
                    </span>
                    <div className="text-xs font-bold text-slate-900 dark:text-slate-100">
                      Level {session.current_difficulty || 3}
                    </div>
                  </div>
                </div>

                {/* Question count & Navigation CTA */}
                <div className="flex items-center justify-between pt-2 border-t border-slate-100 dark:border-[#1E2533] text-xs text-slate-500 dark:text-slate-400">
                  <span className="flex items-center gap-1.5">
                    <FileText className="w-3.5 h-3.5 text-slate-400" />
                    <span>
                      {session.completed_questions || session.total_questions || 0} Questions Evaluated
                    </span>
                  </span>

                  <span className="text-indigo-600 dark:text-indigo-400 font-bold flex items-center gap-1 group-hover:translate-x-0.5 transition-transform">
                    <span>
                      {session.status === 'COMPLETED' ? 'View Scorecard' : 'Resume Session'}
                    </span>
                    <ChevronRight className="w-3.5 h-3.5" />
                  </span>
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
