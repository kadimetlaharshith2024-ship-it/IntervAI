import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  FileText,
  UploadCloud,
  Clock,
  Sparkles,
  ArrowRight,
  AlertCircle,
  Loader2,
  CheckCircle2,
  RefreshCw,
  Play
} from 'lucide-react';
import { listResumes } from '../api/resumes';
import ResumeUploadModal from '../components/ResumeUploadModal';
import { Button, Badge, Card, SkeletonCard } from '../components/ui';

export default function Resumes() {
  const [resumes, setResumes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isUploadOpen, setIsUploadOpen] = useState(false);

  const navigate = useNavigate();

  const fetchResumes = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listResumes();
      setResumes(data);
    } catch (err) {
      setError(err.message || 'Failed to load resumes. Please verify your connection.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchResumes();
  }, []);

  const formatDate = (dateStr) => {
    if (!dateStr) return 'Recently';
    return new Date(dateStr).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="space-y-6 animate-fadeIn max-w-6xl mx-auto">
      {/* Top Header & Actions */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200/80 dark:border-[#1E2433] pb-4">
        <div className="space-y-1">
          <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
            Resumes
          </h1>
          <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400">
            Upload and manage resumes used to calibrate technical interview questions.
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            type="button"
            onClick={fetchResumes}
            disabled={loading}
            aria-label="Refresh resumes list"
            title="Refresh list"
            className="p-2.5 rounded-xl border border-slate-200 dark:border-[#202636] bg-white dark:bg-[#11141C] text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-50 dark:hover:bg-[#181D2A] transition-colors shadow-2xs disabled:opacity-50 cursor-pointer"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>

          <Button
            variant="primary"
            size="md"
            iconLeft={UploadCloud}
            onClick={() => setIsUploadOpen(true)}
          >
            Upload Resume
          </Button>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="p-4 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-2xl text-xs text-rose-800 dark:text-rose-300 flex items-start justify-between gap-3 shadow-xs">
          <div className="flex items-center gap-2.5">
            <AlertCircle className="w-4 h-4 text-rose-600 dark:text-rose-400 flex-shrink-0" />
            <span>{error}</span>
          </div>
          <button
            type="button"
            onClick={fetchResumes}
            className="underline hover:text-rose-950 dark:hover:text-rose-200 transition-colors flex-shrink-0 font-bold cursor-pointer"
          >
            Retry
          </button>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <SkeletonCard />
          <SkeletonCard />
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && resumes.length === 0 && (
        <Card className="py-16 px-6 text-center max-w-2xl mx-auto space-y-4">
          <div className="w-14 h-14 rounded-2xl bg-indigo-50 dark:bg-indigo-500/15 border border-indigo-200/80 dark:border-indigo-500/30 text-indigo-600 dark:text-indigo-400 flex items-center justify-center mx-auto shadow-xs">
            <FileText className="w-7 h-7" />
          </div>
          <div className="space-y-1">
            <h3 className="text-base font-bold text-slate-900 dark:text-white">
              No Resumes Uploaded Yet
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 max-w-sm mx-auto leading-relaxed">
              Upload your technical PDF resume to extract skills, resolve contextual knowledge graphs, and index claims for adaptive mock interviews.
            </p>
          </div>
          <div className="pt-2">
            <Button
              variant="primary"
              size="md"
              iconLeft={UploadCloud}
              onClick={() => setIsUploadOpen(true)}
            >
              Upload Your First Resume
            </Button>
          </div>
        </Card>
      )}

      {/* Resumes Grid */}
      {!loading && !error && resumes.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 px-1">
            <span>
              Showing{' '}
              <strong className="text-slate-900 dark:text-white font-mono">
                {resumes.length}
              </strong>{' '}
              {resumes.length === 1 ? 'resume document' : 'resume documents'}
            </span>
            <span className="text-[11px]">Indexed for adaptive interviews</span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {resumes.map((resume, index) => {
              const detected = resume.detected_sections || {};
              const detectedCount = Object.values(detected).filter(Boolean).length;
              const isLatest = index === 0;

              return (
                <Card
                  key={resume.id}
                  hover
                  className="group relative p-6 transition-all duration-200 flex flex-col justify-between"
                >
                  <div className="space-y-4">
                    {/* Top Row: Icon, Filename, Metadata */}
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex items-start gap-3.5 min-w-0">
                        <div className="w-10 h-10 rounded-xl bg-indigo-50 dark:bg-indigo-500/15 border border-indigo-200/80 dark:border-indigo-500/30 text-indigo-600 dark:text-indigo-400 flex items-center justify-center flex-shrink-0 group-hover:scale-105 transition-transform">
                          <FileText className="w-5 h-5" />
                        </div>
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <h2 className="text-sm sm:text-base font-bold text-slate-900 dark:text-white group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors truncate">
                              {resume.filename}
                            </h2>
                            {isLatest && (
                              <Badge variant="emerald" size="sm">
                                Latest
                              </Badge>
                            )}
                          </div>
                          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500 dark:text-slate-400 mt-1">
                            <span className="flex items-center gap-1">
                              <Clock className="w-3.5 h-3.5 text-slate-400" />
                              <span>{formatDate(resume.created_at)}</span>
                            </span>
                            <span>•</span>
                            <span className="font-mono text-[11px]">
                              ID: {resume.id.slice(0, 8)}
                            </span>
                            <span>•</span>
                            <span className="font-mono">
                              {resume.character_count?.toLocaleString()} chars
                            </span>
                          </div>
                        </div>
                      </div>

                      <Badge variant="neutral" size="sm">
                        {detectedCount}/6 Sections
                      </Badge>
                    </div>

                    {/* Detected Sections Pills */}
                    <div className="space-y-1.5 pt-1">
                      <span className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider block">
                        Parsed Sections
                      </span>
                      <div className="flex flex-wrap gap-1.5">
                        {Object.entries(detected).map(([secName, isDetected]) => (
                          <span
                            key={secName}
                            className={`text-[11px] font-semibold px-2.5 py-0.5 rounded-lg border capitalize ${
                              isDetected
                                ? 'bg-indigo-50 dark:bg-indigo-500/10 text-indigo-700 dark:text-indigo-300 border-indigo-200/80 dark:border-indigo-500/20'
                                : 'bg-slate-100 dark:bg-[#181D2A] text-slate-400 dark:text-slate-500 border-slate-200 dark:border-[#262D3D] line-through opacity-70'
                            }`}
                          >
                            {secName}
                          </span>
                        ))}
                      </div>
                    </div>

                    {/* Sans-serif Text Preview Snippet */}
                    {resume.raw_text_preview && (
                      <div className="bg-slate-50 dark:bg-[#0D1017] border border-slate-200/80 dark:border-[#1E2535] rounded-xl p-3 text-xs text-slate-600 dark:text-slate-400 line-clamp-3 leading-relaxed font-sans">
                        {resume.raw_text_preview.replace(/\s+/g, ' ').trim()}
                      </div>
                    )}
                  </div>

                  {/* Actions Bar */}
                  <div className="mt-5 pt-4 border-t border-slate-100 dark:border-[#1E2533] flex flex-wrap items-center justify-between gap-3">
                    <span className="text-xs text-slate-500 dark:text-slate-400 flex items-center gap-1.5">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                      <span>Ready for Interview</span>
                    </span>

                    <div className="flex items-center gap-2">
                      <Button
                        variant="secondary"
                        size="xs"
                        iconRight={ArrowRight}
                        onClick={() => navigate(`/resumes/${resume.id}`)}
                      >
                        Inspect Details
                      </Button>

                      <Button
                        variant="primary"
                        size="xs"
                        iconLeft={Play}
                        onClick={() => navigate(`/interview?resume=${resume.id}`)}
                      >
                        Practice
                      </Button>
                    </div>
                  </div>
                </Card>
              );
            })}
          </div>
        </div>
      )}

      {/* Upload Modal */}
      <ResumeUploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onUploadSuccess={() => {
          fetchResumes();
        }}
        onViewResume={(id) => {
          navigate(`/resumes/${id}`);
        }}
      />
    </div>
  );
}
