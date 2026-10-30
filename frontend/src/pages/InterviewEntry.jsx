import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import {
  FileText,
  Play,
  AlertCircle,
  Loader2,
  Sparkles,
  CheckCircle2,
  Sliders,
  ArrowRight,
  UploadCloud,
  Check,
  Clock,
  Layers,
  ShieldCheck
} from 'lucide-react';
import { listResumes } from '../api/resumes';
import { listUserSessions, startInterview } from '../api/interview';
import { Button, Badge, Card, CardHeader, CardTitle, CardContent, Skeleton } from '../components/ui';

const DIFFICULTY_LEVELS = [
  {
    level: 1,
    title: 'Level 1 • Fundamentals (Understand)',
    desc: 'Reasoning about core definitions, mechanisms, and architectural fundamentals explicitly supported by your resume.',
    badge: 'Fundamentals',
    badgeVariant: 'emerald',
  },
  {
    level: 2,
    title: 'Level 2 • Application (Apply)',
    desc: 'Apply theoretical principles and frameworks to concrete scenarios in your documented projects.',
    badge: 'Application',
    badgeVariant: 'sky',
  },
  {
    level: 3,
    title: 'Level 3 • Implementation (Build and Debug)',
    desc: 'In-depth debugging, algorithms, state management, edge cases, error handling, and component implementation.',
    badge: 'Implementation',
    badgeVariant: 'indigo',
  },
  {
    level: 4,
    title: 'Level 4 • Architecture (Design and Evaluate)',
    desc: 'System-level design, distributed patterns, data pipelines, caching, scalability bottlenecks, and trade-offs.',
    badge: 'Architecture',
    badgeVariant: 'amber',
  },
  {
    level: 5,
    title: 'Level 5 • Advanced Depth (Analyze and Defend)',
    desc: 'Defend technical decisions under high uncertainty, competing SLAs, production failure modes, and deep tradeoffs.',
    badge: 'Advanced Depth',
    badgeVariant: 'purple',
  },
];

export default function InterviewEntry() {
  const navigate = useNavigate();

  const [resumes, setResumes] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [pageError, setPageError] = useState(null);

  // Form selections
  const [selectedResumeId, setSelectedResumeId] = useState('');
  const [selectedDifficulty, setSelectedDifficulty] = useState(3);
  const [isStarting, setIsStarting] = useState(false);
  const [startError, setStartError] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    setPageError(null);
    try {
      const [resumesData, sessionsData] = await Promise.all([
        listResumes(),
        listUserSessions().catch(() => []),
      ]);
      setResumes(resumesData || []);
      setSessions(sessionsData || []);

      if (resumesData && resumesData.length > 0) {
        setSelectedResumeId(resumesData[0].id);
      }
    } catch (err) {
      setPageError(err.message || 'Failed to load your resumes.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const activeSession = sessions.find((s) => s.status === 'IN_PROGRESS');
  const selectedResume = resumes.find((r) => r.id === selectedResumeId);
  const selectedDifficultyObj = DIFFICULTY_LEVELS.find((d) => d.level === selectedDifficulty);

  const handleStart = async (e) => {
    e.preventDefault();
    if (!selectedResumeId) {
      setStartError('Please select a resume before beginning the interview.');
      return;
    }

    setIsStarting(true);
    setStartError(null);

    try {
      const session = await startInterview({
        resume_id: selectedResumeId,
        initial_difficulty: selectedDifficulty,
      });
      navigate(`/interview/${session.id}`);
    } catch (err) {
      setStartError(err.message || 'Unable to start interview session. Please try again.');
      setIsStarting(false);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto animate-fadeIn">
      {/* Top Header */}
      <div className="border-b border-slate-200/80 dark:border-[#1E2433] pb-4 space-y-1">
        <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
          Technical Interview Setup
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400">
          Select a verified resume and baseline difficulty to begin your adaptive interview session.
        </p>
      </div>

      {/* Active Session Alert Banner (if exists) */}
      {activeSession && (
        <div className="p-5 rounded-2xl bg-indigo-50/80 dark:bg-indigo-500/10 border border-indigo-200 dark:border-indigo-500/30 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shadow-xs">
          <div className="flex items-start gap-3.5">
            <div className="w-10 h-10 rounded-xl bg-indigo-100 dark:bg-indigo-900/60 text-indigo-600 dark:text-indigo-400 flex items-center justify-center flex-shrink-0 mt-0.5">
              <Play className="w-4 h-4 fill-current ml-0.5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-bold text-slate-900 dark:text-white">
                  You have an active interview session
                </span>
                <Badge variant="emerald" size="sm" dot>
                  In Progress
                </Badge>
              </div>
              <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                Difficulty Level {activeSession.current_difficulty || 3} • {activeSession.completed_questions || 0} questions evaluated.
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

      {/* Loading state */}
      {loading && (
        <div className="py-20 text-center space-y-4">
          <Loader2 className="w-8 h-8 text-indigo-600 dark:text-indigo-400 animate-spin mx-auto" />
          <p className="text-xs font-semibold text-slate-500 dark:text-slate-400">
            Loading your resumes and setup preferences...
          </p>
        </div>
      )}

      {/* Page Error Banner */}
      {pageError && (
        <div className="p-4 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-2xl text-xs text-rose-800 dark:text-rose-300 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <AlertCircle className="w-4 h-4 text-rose-600 dark:text-rose-400 flex-shrink-0" />
            <span>{pageError}</span>
          </div>
          <button
            type="button"
            onClick={fetchData}
            className="font-bold underline hover:text-rose-950 dark:hover:text-rose-200 transition-colors cursor-pointer"
          >
            Retry
          </button>
        </div>
      )}

      {/* Empty Resumes State */}
      {!loading && !pageError && resumes.length === 0 && (
        <Card className="py-16 px-6 text-center space-y-4">
          <div className="w-12 h-12 rounded-2xl bg-indigo-50 dark:bg-indigo-500/10 border border-indigo-200 dark:border-indigo-500/20 text-indigo-600 dark:text-indigo-400 flex items-center justify-center mx-auto">
            <FileText className="w-6 h-6" />
          </div>
          <div className="space-y-1">
            <h3 className="text-base font-bold text-slate-900 dark:text-white">
              No Resume Uploaded Yet
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 max-w-sm mx-auto leading-relaxed">
              Interview questions are customized to your specific background and projects. Upload a resume to begin practicing.
            </p>
          </div>
          <div className="pt-2">
            <Button
              variant="primary"
              size="sm"
              iconLeft={UploadCloud}
              onClick={() => navigate('/resumes')}
            >
              Upload Resume
            </Button>
          </div>
        </Card>
      )}

      {/* Interview Configuration Form */}
      {!loading && !pageError && resumes.length > 0 && (
        <form onSubmit={handleStart} className="space-y-8">
          {/* Step 1: Choose Resume */}
          <Card className="p-6 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <h2 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
                  <span className="w-6 h-6 rounded-full bg-indigo-50 dark:bg-indigo-500/15 text-indigo-700 dark:text-indigo-300 flex items-center justify-center text-xs font-bold border border-indigo-200 dark:border-indigo-500/30">
                    1
                  </span>
                  <span>Select Resume</span>
                </h2>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  Questions will be based on the technical competencies extracted from this document.
                </p>
              </div>
              <Link
                to="/resumes"
                className="text-xs text-indigo-600 dark:text-indigo-400 hover:text-indigo-500 font-bold inline-flex items-center gap-1 transition-colors"
              >
                <span>Upload New Resume</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
              {resumes.map((resume) => {
                const isSelected = selectedResumeId === resume.id;
                const detected = resume.detected_sections || {};
                const detectedCount = Object.values(detected).filter(Boolean).length;

                return (
                  <div
                    key={resume.id}
                    onClick={() => setSelectedResumeId(resume.id)}
                    className={`p-4 rounded-xl border cursor-pointer transition-all flex items-start justify-between gap-3 ${
                      isSelected
                        ? 'bg-indigo-50/60 dark:bg-indigo-500/12 border-indigo-600 dark:border-indigo-500 ring-2 ring-indigo-500/20 shadow-xs'
                        : 'bg-white dark:bg-[#11141C] border-slate-200 dark:border-[#202636] hover:border-slate-300 dark:hover:border-[#2C3549] hover:bg-slate-50/60 dark:hover:bg-[#161B26]'
                    }`}
                  >
                    <div className="flex items-start gap-3 truncate">
                      <div
                        className={`w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 mt-0.5 ${
                          isSelected
                            ? 'bg-indigo-600 text-white'
                            : 'bg-slate-100 dark:bg-[#181D2A] text-slate-500 dark:text-slate-400'
                        }`}
                      >
                        <FileText className="w-4 h-4" />
                      </div>
                      <div className="truncate space-y-1">
                        <p className="text-xs font-bold text-slate-900 dark:text-slate-100 truncate">
                          {resume.filename}
                        </p>
                        <p className="text-[11px] text-slate-500 dark:text-slate-400 flex items-center gap-2">
                          <span>{detectedCount}/6 sections parsed</span>
                          <span>•</span>
                          <span>{resume.character_count?.toLocaleString()} chars</span>
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center pt-1">
                      <div
                        className={`w-5 h-5 rounded-full border flex items-center justify-center transition-colors ${
                          isSelected
                            ? 'border-indigo-600 bg-indigo-600 text-white'
                            : 'border-slate-300 dark:border-[#2E374D]'
                        }`}
                      >
                        {isSelected && <Check className="w-3.5 h-3.5 stroke-[3]" />}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>

          {/* Step 2: Choose Difficulty */}
          <Card className="p-6 space-y-4">
            <div>
              <h2 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-indigo-50 dark:bg-indigo-500/15 text-indigo-700 dark:text-indigo-300 flex items-center justify-center text-xs font-bold border border-indigo-200 dark:border-indigo-500/30">
                  2
                </span>
                <span>Select Starting Difficulty</span>
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                Choose your target baseline. The adaptive engine raises or lowers difficulty dynamically based on answer quality.
              </p>
            </div>

            <div className="space-y-2.5 pt-1">
              {DIFFICULTY_LEVELS.map((item) => {
                const isSelected = selectedDifficulty === item.level;
                return (
                  <div
                    key={item.level}
                    onClick={() => setSelectedDifficulty(item.level)}
                    className={`p-4 rounded-xl border cursor-pointer transition-all flex items-start justify-between gap-3 ${
                      isSelected
                        ? 'bg-indigo-50/60 dark:bg-indigo-500/12 border-indigo-600 dark:border-indigo-500 ring-2 ring-indigo-500/20 shadow-xs'
                        : 'bg-white dark:bg-[#11141C] border-slate-200 dark:border-[#202636] hover:border-slate-300 dark:hover:border-[#2C3549] hover:bg-slate-50/60 dark:hover:bg-[#161B26]'
                    }`}
                  >
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold text-slate-900 dark:text-slate-100">
                          {item.title}
                        </span>
                        <Badge variant={item.badgeVariant} size="sm">
                          {item.badge}
                        </Badge>
                      </div>
                      <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                        {item.desc}
                      </p>
                    </div>

                    <div className="flex items-center pt-1 flex-shrink-0">
                      <div
                        className={`w-5 h-5 rounded-full border flex items-center justify-center transition-colors ${
                          isSelected
                            ? 'border-indigo-600 bg-indigo-600 text-white'
                            : 'border-slate-300 dark:border-[#2E374D]'
                        }`}
                      >
                        {isSelected && <Check className="w-3.5 h-3.5 stroke-[3]" />}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>

          {/* Step 3: Summary and Launch */}
          <Card className="p-6 space-y-4">
            <h2 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-indigo-50 dark:bg-indigo-500/15 text-indigo-700 dark:text-indigo-300 flex items-center justify-center text-xs font-bold border border-indigo-200 dark:border-indigo-500/30">
                3
              </span>
              <span>Interview Configuration Summary</span>
            </h2>

            <div className="p-4 rounded-xl bg-slate-50 dark:bg-[#0E1118] border border-slate-200/80 dark:border-[#1E2535] text-xs space-y-2.5">
              <div className="flex justify-between items-center text-slate-600 dark:text-slate-400">
                <span>Selected Resume:</span>
                <span className="font-bold text-slate-900 dark:text-slate-100">
                  {selectedResume?.filename || 'None selected'}
                </span>
              </div>
              <div className="flex justify-between items-center text-slate-600 dark:text-slate-400">
                <span>Starting Level:</span>
                <span className="font-bold text-slate-900 dark:text-slate-100">
                  {selectedDifficultyObj?.title}
                </span>
              </div>
              <div className="flex justify-between items-center text-slate-600 dark:text-slate-400">
                <span>Format:</span>
                <span className="font-medium text-slate-900 dark:text-slate-100">
                  Adaptive technical questions with multi-dimensional rubric scoring
                </span>
              </div>
            </div>

            {startError && (
              <div className="p-4 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-2xl text-xs text-rose-800 dark:text-rose-300 flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-rose-600 dark:text-rose-400 flex-shrink-0" />
                <span>{startError}</span>
              </div>
            )}

            <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-2">
              <span className="text-xs text-slate-500 dark:text-slate-400">
                Your answers and evaluations are automatically persisted to your account.
              </span>

              <Button
                type="submit"
                variant="primary"
                size="md"
                isLoading={isStarting}
                disabled={isStarting || !selectedResumeId}
                iconLeft={Play}
                className="w-full sm:w-auto"
              >
                Launch Interview Session
              </Button>
            </div>
          </Card>
        </form>
      )}
    </div>
  );
}
