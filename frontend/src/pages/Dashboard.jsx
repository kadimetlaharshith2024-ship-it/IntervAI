import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import {
  Sparkles,
  FileText,
  BrainCircuit,
  Award,
  History,
  ShieldCheck,
  ArrowRight,
  PlusCircle,
  ExternalLink,
  Play,
  RotateCw,
  Clock
} from 'lucide-react';
import { listResumes } from '../api/resumes';
import { listUserSessions } from '../api/interview';
import { Button, Badge, Card, CardHeader, CardTitle, CardContent, Skeleton, SkeletonCard } from '../components/ui';

export default function Dashboard() {
  const { user } = useAuth();
  const navigate = useNavigate();

  const [resumes, setResumes] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [resumesRes, sessionsRes] = await Promise.allSettled([
        listResumes(),
        listUserSessions(),
      ]);

      if (resumesRes.status === 'fulfilled' && Array.isArray(resumesRes.value)) {
        setResumes(resumesRes.value);
      }
      if (sessionsRes.status === 'fulfilled' && Array.isArray(sessionsRes.value)) {
        setSessions(sessionsRes.value);
      }
    } catch {
      // Graceful fallback to empty state
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const completedSessions = sessions.filter(
    (s) => (s.status === 'COMPLETED' || s.status === 'ENDED') && s.overall_score !== null
  );
  const activeSession = sessions.find((s) => s.status === 'IN_PROGRESS');

  const averageScore =
    completedSessions.length > 0
      ? Math.round(
          completedSessions.reduce((acc, s) => acc + (s.overall_score || 0), 0) /
            completedSessions.length
        )
      : null;

  const totalQuestionsAnswered = sessions.reduce(
    (acc, s) => acc + (s.completed_questions || 0),
    0
  );

  const getReadinessTier = (avg) => {
    if (avg === null) {
      return { tier: 'Not Assessed Yet', variant: 'neutral' };
    }
    if (avg >= 85) return { tier: 'Strong Mastery (85+)', variant: 'emerald' };
    if (avg >= 70) return { tier: 'Proficient (70+)', variant: 'indigo' };
    if (avg >= 50) return { tier: 'Foundational (50+)', variant: 'amber' };
    return { tier: 'Needs Topic Review', variant: 'rose' };
  };

  const readiness = getReadinessTier(averageScore);

  if (loading) {
    return (
      <div className="space-y-6 animate-fadeIn">
        <div className="py-4 space-y-2">
          <Skeleton className="h-7 w-48" />
          <Skeleton className="h-4 w-96 max-w-full" />
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* =========================================================================
          PAGE HEADER (CLEAN, FOCUSED, LINEAR-INSPIRED)
         ========================================================================= */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-200/80 dark:border-[#1E2433]">
        <div className="space-y-1">
          <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
            Dashboard
          </h1>
          <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400">
            Resume-grounded adaptive technical mock interviews.
          </p>
        </div>

        <div className="flex items-center gap-2.5 flex-shrink-0">
          <Button
            variant="secondary"
            size="sm"
            iconLeft={FileText}
            onClick={() => navigate('/resumes')}
          >
            Resumes
          </Button>
          <Button
            variant="primary"
            size="sm"
            iconLeft={BrainCircuit}
            onClick={() => navigate('/interview')}
          >
            Start Interview
          </Button>
        </div>
      </div>

      {/* =========================================================================
          TELEMETRY STATS GRID (CLEAN 3-METRIC SUMMARY)
         ========================================================================= */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {/* Stat 1: Resumes */}
        <Card className="p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">
              Resumes Uploaded
            </span>
            <FileText className="w-4 h-4 text-slate-400 dark:text-slate-500" />
          </div>
          <div className="text-2xl font-bold text-slate-900 dark:text-white">
            {resumes.length}
          </div>
          <p className="text-[11px] text-slate-500 dark:text-slate-400">
            {resumes.length === 1 ? '1 active resume document' : `${resumes.length} active resume documents`}
          </p>
        </Card>

        {/* Stat 2: Interviews */}
        <Card className="p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">
              Interviews Completed
            </span>
            <BrainCircuit className="w-4 h-4 text-slate-400 dark:text-slate-500" />
          </div>
          <div className="text-2xl font-bold text-slate-900 dark:text-white">
            {completedSessions.length}
          </div>
          <p className="text-[11px] text-slate-500 dark:text-slate-400">
            {totalQuestionsAnswered} technical answers evaluated
          </p>
        </Card>

        {/* Stat 3: Average Score */}
        <Card className="p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">
              Average Performance
            </span>
            <Award className="w-4 h-4 text-slate-400 dark:text-slate-500" />
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="text-2xl font-bold text-slate-900 dark:text-white">
              {averageScore !== null ? averageScore : '—'}
            </span>
            {averageScore !== null && (
              <span className="text-xs text-slate-500 dark:text-slate-400 font-medium">/ 100</span>
            )}
          </div>
          <p className="text-[11px] text-slate-500 dark:text-slate-400">
            {completedSessions.length > 0 ? 'Across evaluated sessions' : 'Complete an interview to record'}
          </p>
        </Card>
      </div>

      {/* =========================================================================
          MAIN CONTENT COLUMNS
         ========================================================================= */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column (2 Cols): Active Session & Recent Interviews */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-base sm:text-lg font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <History className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
              <span>Recent Interview Sessions</span>
            </h2>
            <button
              type="button"
              onClick={() => navigate('/history')}
              className="text-xs font-bold text-indigo-600 dark:text-indigo-400 hover:text-indigo-500 flex items-center gap-1 transition-colors cursor-pointer"
            >
              <span>View All History</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Active In-Progress Highlight Banner */}
          {activeSession && (
            <div className="relative overflow-hidden rounded-2xl border border-indigo-200 dark:border-indigo-500/30 bg-indigo-50/70 dark:bg-indigo-500/10 p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shadow-xs">
              <div className="flex items-center gap-3.5">
                <div className="w-10 h-10 rounded-xl bg-indigo-600 text-white flex items-center justify-center flex-shrink-0 shadow-xs">
                  <Play className="w-4 h-4 fill-current ml-0.5" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-bold text-slate-900 dark:text-white">
                      Active Interview Session
                    </span>
                    <Badge variant="indigo" dot size="sm">
                      In Progress
                    </Badge>
                  </div>
                  <p className="text-xs text-indigo-700 dark:text-indigo-300 font-mono mt-0.5">
                    Session {activeSession.id.slice(0, 8)}...
                  </p>
                </div>
              </div>
              <Button
                variant="primary"
                size="sm"
                iconRight={ArrowRight}
                onClick={() => navigate(`/interview/${activeSession.id}`)}
              >
                Resume Interview
              </Button>
            </div>
          )}

          {/* Session Cards or Empty State */}
          {sessions.length === 0 ? (
            <Card className="p-8 text-center space-y-3">
              <div className="w-12 h-12 rounded-2xl bg-indigo-50 dark:bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 flex items-center justify-center mx-auto">
                <BrainCircuit className="w-6 h-6" />
              </div>
              <div className="space-y-1">
                <h3 className="text-sm font-bold text-slate-900 dark:text-white">
                  No interview sessions recorded yet
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400 max-w-sm mx-auto leading-relaxed">
                  Select one of your uploaded resumes to begin your first adaptive mock interview session.
                </p>
              </div>
              <div className="pt-2">
                <Button
                  variant="primary"
                  size="sm"
                  iconLeft={Play}
                  onClick={() => navigate('/interview')}
                >
                  Start Practicing
                </Button>
              </div>
            </Card>
          ) : (
            <div className="space-y-3">
              {sessions.slice(0, 4).map((session) => {
                const dateFormatted = session.started_at
                  ? new Date(session.started_at).toLocaleDateString(undefined, {
                      month: 'short',
                      day: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit',
                    })
                  : 'Recent Session';

                const isDone = session.status === 'COMPLETED' || session.status === 'ENDED';

                return (
                  <div
                    key={session.id}
                    onClick={() => navigate(`/interview/${session.id}`)}
                    className="cursor-pointer group rounded-2xl border border-slate-200/80 dark:border-[#202636] bg-white dark:bg-[#11141C] hover:bg-slate-50/80 dark:hover:bg-[#171C26] hover:border-slate-300 dark:hover:border-[#2C3549] p-4.5 shadow-2xs transition-all duration-150 flex items-center justify-between gap-4"
                  >
                    <div className="flex items-center gap-3.5 min-w-0">
                      <div className="w-10 h-10 rounded-xl bg-slate-100 dark:bg-[#181D2A] border border-slate-200 dark:border-[#252C3D] flex items-center justify-center flex-shrink-0 group-hover:bg-indigo-50 dark:group-hover:bg-indigo-500/15 group-hover:border-indigo-200 dark:group-hover:border-indigo-500/30 transition-colors">
                        <BrainCircuit className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
                      </div>
                      <div className="space-y-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-xs sm:text-sm font-bold text-slate-900 dark:text-slate-100 truncate">
                            Mock Interview #{session.id.slice(0, 8)}
                          </span>
                          <Badge variant={isDone ? 'emerald' : 'amber'} size="sm">
                            {session.status}
                          </Badge>
                        </div>
                        <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-xs text-slate-500 dark:text-slate-400">
                          <span className="flex items-center gap-1">
                            <Clock className="w-3 h-3 text-slate-400" />
                            <span>{dateFormatted}</span>
                          </span>
                          <span>•</span>
                          <span>Difficulty Level {session.current_difficulty || 3}</span>
                          <span>•</span>
                          <span>{session.completed_questions || 0} questions evaluated</span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-4 flex-shrink-0">
                      {session.overall_score !== null && session.overall_score !== undefined && (
                        <div className="text-right">
                          <span className="text-base sm:text-lg font-extrabold text-indigo-600 dark:text-indigo-400">
                            {Math.round(session.overall_score)}
                          </span>
                          <span className="text-[10px] text-slate-500 dark:text-slate-400 block -mt-1 font-semibold">
                            / 100
                          </span>
                        </div>
                      )}
                      <span className="text-slate-400 dark:text-slate-500 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                        <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Right Column (1 Col): Uploaded Resume Documents */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-base sm:text-lg font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <FileText className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
              <span>Resume Documents</span>
            </h2>
            <button
              type="button"
              onClick={() => navigate('/resumes')}
              className="text-xs font-bold text-indigo-600 dark:text-indigo-400 hover:text-indigo-500 flex items-center gap-1 transition-colors cursor-pointer"
            >
              <span>Manage</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <Card className="p-5 space-y-4">
            {resumes.length === 0 ? (
              <div className="text-center py-6 space-y-3">
                <div className="w-10 h-10 rounded-xl bg-slate-100 dark:bg-[#181D2A] text-slate-400 dark:text-slate-500 flex items-center justify-center mx-auto">
                  <FileText className="w-5 h-5" />
                </div>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  No resumes uploaded yet
                </p>
                <Button
                  variant="primary"
                  size="sm"
                  iconLeft={PlusCircle}
                  onClick={() => navigate('/resumes')}
                >
                  Upload Resume
                </Button>
              </div>
            ) : (
              <div className="space-y-3">
                {resumes.slice(0, 3).map((resume) => {
                  const detected = resume.detected_sections || {};
                  const detectedCount = Object.values(detected).filter(Boolean).length;

                  return (
                    <div
                      key={resume.id}
                      onClick={() => navigate(`/resumes/${resume.id}`)}
                      className="group cursor-pointer p-3.5 rounded-xl border border-slate-200/80 dark:border-[#202636] bg-slate-50/60 dark:bg-[#141822] hover:bg-slate-100/80 dark:hover:bg-[#1A202E] hover:border-indigo-200 dark:hover:border-indigo-500/30 transition-all flex items-center justify-between gap-3 shadow-2xs"
                    >
                      <div className="min-w-0 space-y-1">
                        <p className="text-xs font-bold text-slate-900 dark:text-slate-100 truncate group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                          {resume.filename}
                        </p>
                        <div className="flex items-center gap-2 text-[11px] text-slate-500 dark:text-slate-400">
                          <span>{detectedCount}/6 sections parsed</span>
                          <span>•</span>
                          <span className="text-emerald-600 dark:text-emerald-400 font-semibold">
                            Grounded
                          </span>
                        </div>
                      </div>
                      <ExternalLink className="w-3.5 h-3.5 text-slate-400 dark:text-slate-500 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 flex-shrink-0" />
                    </div>
                  );
                })}

                <Button
                  variant="outline"
                  size="sm"
                  iconLeft={PlusCircle}
                  onClick={() => navigate('/resumes')}
                  className="w-full justify-center"
                >
                  Upload Additional Resume
                </Button>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
