import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import {
  BrainCircuit,
  FileText,
  Send,
  Loader2,
  AlertCircle,
  CheckCircle2,
  AlertTriangle,
  RotateCw,
  LogOut,
  Sparkles,
  ArrowRight,
  ShieldAlert,
  Sliders,
  CornerDownLeft,
  Layers,
  ShieldCheck,
  ChevronDown,
  ChevronUp,
  Code2
} from 'lucide-react';
import {
  getInterviewState,
  getNextQuestion,
  submitAnswer,
  endInterview
} from '../api/interview';
import InterviewScorecard from '../components/InterviewScorecard';
import { Button, Badge, Card, CardHeader, CardTitle, CardContent, Modal, Toast } from '../components/ui';

const RUBRIC_LABELS = {
  technical_accuracy: 'Technical Accuracy',
  conceptual_understanding: 'Conceptual Understanding',
  completeness: 'Completeness',
  depth_and_specificity: 'Depth & Specificity',
  practical_reasoning: 'Practical Reasoning',
  evidence_alignment: 'Resume Evidence Alignment',
};

const CANONICAL_DIFFICULTIES = {
  1: { name: 'Fundamentals (Understand)', variant: 'emerald' },
  2: { name: 'Application (Apply)', variant: 'sky' },
  3: { name: 'Implementation (Build and Debug)', variant: 'indigo' },
  4: { name: 'Architecture (Design and Evaluate)', variant: 'amber' },
  5: { name: 'Advanced Depth (Analyze and Defend)', variant: 'purple' },
};

// Structured error classification helper
function categorizeError(err) {
  const msg = (err?.message || err?.error_code || err?.code || String(err) || '').toLowerCase();
  if (msg.includes('network') || msg.includes('failed to fetch') || msg.includes('econnrefused')) {
    return {
      type: 'NETWORK',
      title: 'Network Connectivity Failure',
      description: 'Unable to reach the interview service. Please verify your connection.',
      recoverable: true,
    };
  }
  if (msg.includes('timeout') || msg.includes('504') || msg.includes('timed out') || msg.includes('aborted')) {
    return {
      type: 'TIMEOUT',
      title: 'Backend Generation Timeout',
      description: 'The adaptive engine took longer than expected to formulate a verified question. Your session history is intact.',
      recoverable: true,
    };
  }
  if (msg.includes('401') || msg.includes('403') || msg.includes('unauthorized') || msg.includes('forbidden')) {
    return {
      type: 'AUTH',
      title: 'Authentication Required',
      description: 'Your session credentials expired or are unauthorized. Please sign in again.',
      recoverable: false,
    };
  }
  if (msg.includes('difficulty') || msg.includes('calibration') || msg.includes('mismatch')) {
    return {
      type: 'DIFFICULTY_MISMATCH',
      title: 'Difficulty Calibration Notice',
      description: 'The generated question did not meet the reasoning demand required for this difficulty level. Retrying will recalibrate the generation prompt.',
      recoverable: true,
    };
  }
  if (msg.includes('insufficient_evidence') || msg.includes('evidence unavailable') || msg.includes('no candidate evidence')) {
    return {
      type: 'INSUFFICIENT_EVIDENCE',
      title: 'Insufficient Resume Evidence',
      description: 'The candidate resume does not contain sufficient verifiable technical evidence for this topic. Retrying will explore adjacent topics.',
      recoverable: true,
    };
  }
  if (msg.includes('duplicate') || msg.includes('semantically duplicated') || msg.includes('redundancy_rejection')) {
    return {
      type: 'DUPLICATE',
      title: 'Semantic Duplicate Filtered',
      description: 'Candidate questions were too similar to earlier turns. Retrying will target alternative technical dimensions.',
      recoverable: true,
    };
  }
  if (msg.includes('grounding') || msg.includes('not grounded') || msg.includes('no eligible')) {
    return {
      type: 'GROUNDING',
      title: 'Grounding Verification Notice',
      description: 'Could not formulate a verified question with strict resume provenance. Retrying will seek supported evidence.',
      recoverable: true,
    };
  }
  return {
    type: 'SERVER',
    title: 'Question Generation Notice',
    description: err?.message || 'An unexpected error occurred while generating the next question.',
    recoverable: true,
  };
}

export default function InterviewSession() {
  const { sessionId } = useParams();
  const navigate = useNavigate();

  // Core Session State
  const [sessionState, setSessionState] = useState(null);
  const [loadingState, setLoadingState] = useState(true);
  const [sessionError, setSessionError] = useState(null);
  const [isCompleted, setIsCompleted] = useState(false);

  // Active Question State
  const [activeQuestion, setActiveQuestion] = useState(null);
  const [isGeneratingQuestion, setIsGeneratingQuestion] = useState(false);
  const [questionError, setQuestionError] = useState(null);
  const [evidenceDrawerOpen, setEvidenceDrawerOpen] = useState(false);

  // Answer & Evaluation State
  const [candidateAnswer, setCandidateAnswer] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [evaluationResult, setEvaluationResult] = useState(null);
  const [evaluationError, setEvaluationError] = useState(null);

  // End Interview Modal State
  const [showEndModal, setShowEndModal] = useState(false);
  const [isEnding, setIsEnding] = useState(false);

  const answerTextareaRef = useRef(null);

  // Group repeated knowledge gaps by topic
  const groupedKnowledgeGaps = useMemo(() => {
    if (!sessionState?.knowledge_gaps?.length) return [];
    const map = new Map();
    sessionState.knowledge_gaps.forEach((gap) => {
      const key = gap.topic || gap.category || 'General Technical';
      if (!map.has(key)) {
        map.set(key, {
          topic: key,
          count: 0,
          highestSeverity: 'LOW',
          severities: new Set(),
          weaknesses: [],
          items: [],
        });
      }
      const entry = map.get(key);
      entry.count += 1;
      if (gap.severity) {
        entry.severities.add(gap.severity);
        if (gap.severity === 'HIGH') entry.highestSeverity = 'HIGH';
        else if (gap.severity === 'MODERATE' && entry.highestSeverity !== 'HIGH') entry.highestSeverity = 'MODERATE';
      }
      if (Array.isArray(gap.weaknesses)) {
        gap.weaknesses.forEach((w) => {
          if (w && !entry.weaknesses.includes(w)) entry.weaknesses.push(w);
        });
      } else if (gap.weaknesses && !entry.weaknesses.includes(gap.weaknesses)) {
        entry.weaknesses.push(gap.weaknesses);
      }
      entry.items.push(gap);
    });
    return Array.from(map.values());
  }, [sessionState?.knowledge_gaps]);

  const getDraftKey = (qId) => `intervai_draft_${sessionId}_${qId}`;

  // Request next adaptive question with deterministic idempotency key per turn
  const fetchNextQuestion = async () => {
    if (isGeneratingQuestion) return;
    setIsGeneratingQuestion(true);
    setQuestionError(null);
    setEvaluationError(null);
    setEvidenceDrawerOpen(false);

    const turnIndex = sessionState?.asked_questions?.length || 0;
    const idempotencyKey = `session_${sessionId}_turn_${turnIndex}`;

    try {
      const res = await getNextQuestion(sessionId, { request_idempotency_key: idempotencyKey });

      if (
        res.status === 'SESSION_ENDED' ||
        res.status === 'BUDGET_EXHAUSTED' ||
        res.status === 'ALL_TOPICS_COVERED'
      ) {
        setIsCompleted(true);
        return;
      }

      if (res.status === 'GENERATED' && res.question_text) {
        const nextQ = {
          question_id: res.question_id,
          id: res.question_id,
          question_text: res.question_text,
          category: res.category || res.selection?.category || 'Technical',
          difficulty: res.difficulty || res.selection?.difficulty || 3,
          assessed_difficulty: res.assessed_difficulty || res.difficulty || 3,
          difficulty_rationale: res.difficulty_rationale,
          target_project: res.target_project || res.selection?.target_project,
          target_entity: res.target_entity || res.selection?.target_entity || res.selection?.target_topic,
          evidence_ids: res.evidence_ids || [],
          evidence_text: res.evidence_text,
          grounding_status: res.grounding_status || 'VERIFIED',
          selection: res.selection,
          is_follow_up: res.selection?.is_follow_up || false,
          follow_up_reason: res.selection?.follow_up_reason,
        };

        setActiveQuestion(nextQ);

        // Check for unsent draft
        const savedDraft = sessionStorage.getItem(getDraftKey(res.question_id));
        setCandidateAnswer(savedDraft || '');
        setEvaluationResult(null);

        // Refresh telemetry in background
        getInterviewState(sessionId).then((s) => setSessionState(s)).catch(() => {});
      } else {
        setQuestionError(categorizeError({
          message: res.message || 'Adaptive question selection did not return a valid prompt.',
          error_code: res.error_code,
        }));
      }
    } catch (err) {
      setQuestionError(categorizeError(err));
    } finally {
      setIsGeneratingQuestion(false);
    }
  };

  // Load interview session state
  const loadSession = async () => {
    setLoadingState(true);
    setSessionError(null);
    try {
      const state = await getInterviewState(sessionId);
      setSessionState(state);

      if (state.status === 'COMPLETED' || state.status === 'ENDED') {
        setIsCompleted(true);
        setLoadingState(false);
        return;
      }

      const askedQuestions = state.asked_questions || [];
      if (askedQuestions.length === 0) {
        fetchNextQuestion();
      } else {
        const lastQ = askedQuestions[askedQuestions.length - 1];
        if (lastQ.status === 'UNANSWERED' || lastQ.status === 'PENDING') {
          setActiveQuestion({
            question_id: lastQ.question_id,
            id: lastQ.question_id,
            question_text: lastQ.question_text,
            category: lastQ.category,
            difficulty: lastQ.difficulty,
            target_project: lastQ.target_project,
            target_entity: lastQ.target_entity,
            is_follow_up: lastQ.is_follow_up,
          });
          const savedDraft = sessionStorage.getItem(getDraftKey(lastQ.question_id));
          setCandidateAnswer(savedDraft || '');
          setEvaluationResult(null);
        } else if (lastQ.status === 'EVALUATED') {
          setActiveQuestion({
            question_id: lastQ.question_id,
            id: lastQ.question_id,
            question_text: lastQ.question_text,
            category: lastQ.category,
            difficulty: lastQ.difficulty,
            assessed_difficulty: lastQ.assessed_difficulty || lastQ.difficulty,
            difficulty_rationale: lastQ.difficulty_rationale,
            target_project: lastQ.target_project,
            target_entity: lastQ.target_entity,
            evidence_ids: lastQ.evidence_ids || [],
            evidence_text: lastQ.evidence_text,
            grounding_status: lastQ.grounding_status || 'VERIFIED',
            is_follow_up: lastQ.is_follow_up,
          });
          setEvaluationResult({
            overall_score: lastQ.latest_score,
            feedback: {
              summary: lastQ.latest_feedback_summary || 'Evaluation completed.',
            },
          });
        }
      }
    } catch (err) {
      setSessionError(err.message || 'Failed to connect to active interview session.');
    } finally {
      setLoadingState(false);
    }
  };

  useEffect(() => {
    if (sessionId) {
      loadSession();
    }
  }, [sessionId]);

  // Handle draft persistence
  const handleAnswerChange = (e) => {
    const val = e.target.value;
    setCandidateAnswer(val);
    const qId = activeQuestion?.question_id || activeQuestion?.id;
    if (qId) {
      if (val.trim()) {
        sessionStorage.setItem(getDraftKey(qId), val);
      } else {
        sessionStorage.removeItem(getDraftKey(qId));
      }
    }
  };

  // Submit candidate answer
  const handleSubmitAnswer = async (e) => {
    if (e) e.preventDefault();
    if (isSubmitting) return;
    const cleanAnswer = candidateAnswer.trim();

    if (!cleanAnswer) {
      setEvaluationError('Please enter your technical explanation before submitting.');
      return;
    }

    if (!activeQuestion?.question_id && !activeQuestion?.id) {
      setEvaluationError('No active question found.');
      return;
    }

    setIsSubmitting(true);
    setEvaluationError(null);

    const qId = activeQuestion.question_id || activeQuestion.id;

    try {
      const result = await submitAnswer({
        question_id: qId,
        session_id: sessionId,
        candidate_answer: cleanAnswer,
      });

      sessionStorage.removeItem(getDraftKey(qId));
      setEvaluationResult(result);

      const updatedState = await getInterviewState(sessionId);
      setSessionState(updatedState);
    } catch (err) {
      setEvaluationError(err.message || 'Answer evaluation failed. Please check your network and retry.');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Keyboard shortcut Ctrl+Enter / Cmd+Enter
  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      if (!isSubmitting && !evaluationResult && candidateAnswer.trim()) {
        handleSubmitAnswer();
      }
    }
  };

  // End Interview manually
  const handleConfirmEnd = async () => {
    if (isEnding) return;
    setIsEnding(true);
    try {
      await endInterview(sessionId, 'USER_COMPLETED');
      setShowEndModal(false);
      setIsCompleted(true);
    } catch (err) {
      alert(`Unable to conclude interview: ${err.message}`);
    } finally {
      setIsEnding(false);
    }
  };

  // If completed, render the final scorecard view
  if (isCompleted) {
    return (
      <InterviewScorecard
        sessionId={sessionId}
        onRestart={() => navigate('/interview')}
      />
    );
  }

  // Initial loading state
  if (loadingState) {
    return (
      <div className="py-24 text-center space-y-4 max-w-md mx-auto animate-fadeIn">
        <Loader2 className="w-10 h-10 text-indigo-600 dark:text-indigo-400 animate-spin mx-auto" />
        <div className="space-y-1">
          <p className="text-sm font-bold text-slate-900 dark:text-white">
            Connecting to Live Interview Session...
          </p>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Restoring session telemetry, resume grounding, and candidate state.
          </p>
        </div>
      </div>
    );
  }

  // Session Connection Error
  if (sessionError) {
    return (
      <Card className="p-8 text-center space-y-4 max-w-md mx-auto my-12 border-rose-200 dark:border-rose-500/30">
        <AlertCircle className="w-8 h-8 text-rose-500 mx-auto" />
        <div>
          <h3 className="text-base font-bold text-slate-900 dark:text-white">
            Session Inaccessible
          </h3>
          <p className="text-xs text-rose-600 dark:text-rose-400 mt-1">{sessionError}</p>
        </div>
        <div className="flex items-center justify-center gap-3 pt-2">
          <Button variant="outline" size="sm" onClick={() => navigate('/interview')}>
            Back to Practice
          </Button>
          <Button variant="primary" size="sm" iconLeft={RotateCw} onClick={loadSession}>
            Retry Connection
          </Button>
        </div>
      </Card>
    );
  }

  const askedList = sessionState?.asked_questions || [];
  const completedCount = sessionState?.progress?.completed_questions || 0;
  const currentQuestionNumber = askedList.length > 0 ? askedList.length : 1;
  const currentDifficulty = activeQuestion?.difficulty || sessionState?.current_difficulty || 3;
  const difficultyInfo = CANONICAL_DIFFICULTIES[currentDifficulty] || CANONICAL_DIFFICULTIES[3];

  return (
    <div className="space-y-6 animate-fadeIn max-w-6xl mx-auto pb-16">
      {/* =========================================================================
          COCKPIT TELEMETRY BAR
         ========================================================================= */}
      <div className="sticky top-18 z-30 rounded-2xl border border-slate-200/90 dark:border-[#202636] bg-white/95 dark:bg-[#11141C]/95 backdrop-blur-md p-4 sm:p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shadow-sm transition-colors">
        <div className="flex items-center gap-3.5">
          <div className="w-10 h-10 rounded-xl bg-indigo-50 dark:bg-indigo-500/15 border border-indigo-200/80 dark:border-indigo-500/30 text-indigo-600 dark:text-indigo-400 flex items-center justify-center flex-shrink-0 shadow-xs">
            <BrainCircuit className="w-5 h-5" />
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-extrabold text-slate-900 dark:text-white uppercase tracking-wider">
                Turn #{currentQuestionNumber}
              </span>
              <Badge variant="emerald" size="sm" dot>
                LIVE SESSION
              </Badge>
              <Badge variant={difficultyInfo.variant} size="sm">
                Level {currentDifficulty} • {difficultyInfo.name.split(' ')[0]}
              </Badge>
              <span className="text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-slate-100 dark:bg-[#181D2A] text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-[#262D3D]">
                {completedCount} Evaluated
              </span>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 font-sans">
              Resume-Grounded Adaptive Cockpit • Session <span className="font-mono text-slate-700 dark:text-slate-300 font-semibold">{sessionId.slice(0, 8)}...</span>
            </p>
          </div>
        </div>

        {/* Cockpit Actions */}
        <div className="flex items-center gap-2.5 self-end sm:self-auto">
          <Button
            variant="danger"
            size="sm"
            iconLeft={LogOut}
            onClick={() => setShowEndModal(true)}
          >
            End Interview
          </Button>
        </div>
      </div>

      {/* =========================================================================
          WORKSPACE GRID
         ========================================================================= */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column (2 Cols): Question & Answer Editor */}
        <div className="lg:col-span-2 space-y-6">
          {/* Question Generation Loading State */}
          {isGeneratingQuestion && (
            <Card className="p-8 text-center space-y-3 border-indigo-200 dark:border-indigo-500/30 animate-pulseGlow">
              <Loader2 className="w-8 h-8 text-indigo-600 dark:text-indigo-400 animate-spin mx-auto" />
              <div className="space-y-1">
                <p className="text-sm font-bold text-slate-900 dark:text-white">
                  Synthesizing next adaptive question...
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-400 max-w-md mx-auto leading-relaxed">
                  Calibrating difficulty against your previous rubric performance and verifying supporting claims on your resume.
                </p>
              </div>
            </Card>
          )}

          {/* Structured Question Error Banner */}
          {questionError && (
            <div className="p-4 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-2xl text-xs text-rose-800 dark:text-rose-300 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 shadow-xs">
              <div className="flex items-start gap-2.5">
                <AlertCircle className="w-4 h-4 text-rose-600 dark:text-rose-400 flex-shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold block text-rose-900 dark:text-rose-200">
                    {questionError.title}
                  </span>
                  <span className="text-rose-700 dark:text-rose-300 leading-relaxed">
                    {questionError.description}
                  </span>
                </div>
              </div>
              {questionError.recoverable && (
                <Button
                  variant="outline"
                  size="xs"
                  iconLeft={RotateCw}
                  onClick={fetchNextQuestion}
                  disabled={isGeneratingQuestion}
                  className="self-end sm:self-auto flex-shrink-0 border-rose-300 dark:border-rose-500/50 text-rose-800 dark:text-rose-300"
                >
                  Retry Question
                </Button>
              )}
            </div>
          )}

          {/* Active Question Container */}
          {activeQuestion && !isGeneratingQuestion && (
            <Card className="p-6 sm:p-7 space-y-4">
              {/* Question Metadata Tags */}
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant={difficultyInfo.variant} size="md">
                  Level {activeQuestion.difficulty}: {difficultyInfo.name}
                </Badge>

                <Badge variant="emerald" size="md" icon={ShieldCheck}>
                  Resume-Grounded
                </Badge>

                {activeQuestion.target_entity && (
                  <Badge variant="neutral" size="md">
                    Topic: {activeQuestion.target_entity}
                  </Badge>
                )}

                {activeQuestion.target_project && (
                  <Badge variant="indigo" size="md">
                    Project: {activeQuestion.target_project}
                  </Badge>
                )}

                {activeQuestion.is_follow_up && (
                  <Badge variant="amber" size="md" icon={Sparkles}>
                    Adaptive Follow-Up
                  </Badge>
                )}
              </div>

              {/* Follow-up Context Callout */}
              {activeQuestion.follow_up_reason && (
                <div className="p-3.5 rounded-xl bg-amber-50/80 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/30 text-xs text-amber-900 dark:text-amber-200 leading-relaxed">
                  <span className="font-bold text-amber-900 dark:text-amber-300">
                    Adaptive Follow-Up Focus:{' '}
                  </span>
                  {activeQuestion.follow_up_reason}
                </div>
              )}

              {/* Main Question Heading */}
              <div className="pt-1">
                <h2 className="text-lg sm:text-xl font-bold text-slate-900 dark:text-white leading-relaxed tracking-tight">
                  {activeQuestion.question_text}
                </h2>
              </div>

              {/* Collapsible Verified Resume Evidence Drawer */}
              {activeQuestion.evidence_text && (
                <div className="border border-slate-200/80 dark:border-[#222838] rounded-xl overflow-hidden bg-slate-50/70 dark:bg-[#0D1017] transition-colors">
                  <button
                    type="button"
                    onClick={() => setEvidenceDrawerOpen((prev) => !prev)}
                    className="w-full px-4 py-2.5 flex items-center justify-between text-xs font-bold text-slate-700 dark:text-slate-300 hover:bg-slate-100/70 dark:hover:bg-[#151923] transition-colors cursor-pointer"
                  >
                    <span className="flex items-center gap-2">
                      <ShieldCheck className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                      <span>Verified Resume Evidence Provenance</span>
                      {activeQuestion.evidence_ids?.length > 0 && (
                        <span className="text-[10px] font-mono font-normal text-slate-500 dark:text-slate-400">
                          ({activeQuestion.evidence_ids.join(', ')})
                        </span>
                      )}
                    </span>
                    {evidenceDrawerOpen ? (
                      <ChevronUp className="w-4 h-4 text-slate-400" />
                    ) : (
                      <ChevronDown className="w-4 h-4 text-slate-400" />
                    )}
                  </button>

                  {evidenceDrawerOpen && (
                    <div className="p-4 border-t border-slate-200/80 dark:border-[#222838] space-y-2 text-xs animate-fadeIn">
                      <p className="font-sans italic text-slate-600 dark:text-slate-400 bg-white dark:bg-[#121620] p-3 rounded-lg border border-slate-200/60 dark:border-[#202737] leading-relaxed">
                        "{activeQuestion.evidence_text}"
                      </p>
                      {activeQuestion.difficulty_rationale && (
                        <p className="text-[11px] text-slate-500 dark:text-slate-400 pt-1">
                          <strong className="text-slate-700 dark:text-slate-300">
                            Difficulty Calibration Rationale:{' '}
                          </strong>
                          {activeQuestion.difficulty_rationale}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )}
            </Card>
          )}

          {/* =========================================================================
              ANSWER EDITOR (SHOWN WHILE ANSWERING)
             ========================================================================= */}
          {activeQuestion && !evaluationResult && !isGeneratingQuestion && (
            <Card className="p-6 sm:p-7 space-y-4">
              <form onSubmit={handleSubmitAnswer} className="space-y-4">
                <div className="flex items-center justify-between">
                  <label
                    htmlFor="candidate-answer-textarea"
                    className="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider flex items-center gap-1.5"
                  >
                    <Code2 className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
                    <span>Your Technical Response</span>
                  </label>
                  <span className="text-xs font-mono text-slate-500 dark:text-slate-400">
                    {candidateAnswer.length} / 4000 characters
                  </span>
                </div>

                <div className="relative">
                  <textarea
                    id="candidate-answer-textarea"
                    ref={answerTextareaRef}
                    value={candidateAnswer}
                    onChange={handleAnswerChange}
                    onKeyDown={handleKeyDown}
                    disabled={isSubmitting}
                    placeholder="Provide your in-depth technical explanation here. Discuss specific architectural decisions, edge cases, error recovery, algorithms, concurrency, and trade-offs..."
                    rows={9}
                    maxLength={4000}
                    className="w-full bg-slate-50/50 dark:bg-[#0B0D13] border border-slate-300 dark:border-[#222838] rounded-xl p-4 text-xs sm:text-sm text-slate-900 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-600 font-mono leading-relaxed resize-y focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 dark:focus:border-indigo-500 disabled:opacity-50 transition-all shadow-inner"
                  />
                </div>

                {evaluationError && (
                  <div className="p-3.5 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-xl text-xs text-rose-800 dark:text-rose-300 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 text-rose-600 dark:text-rose-400 flex-shrink-0" />
                    <span>{evaluationError}</span>
                  </div>
                )}

                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pt-2 border-t border-slate-100 dark:border-[#1E2535]">
                  <span className="text-xs text-slate-500 dark:text-slate-400 flex items-center gap-1.5">
                    <CornerDownLeft className="w-3.5 h-3.5 text-slate-400" />
                    <span>
                      Press{' '}
                      <kbd className="px-1.5 py-0.5 rounded bg-slate-100 dark:bg-[#1A202D] border border-slate-200 dark:border-[#283244] text-slate-700 dark:text-slate-300 font-mono text-[10px]">
                        Ctrl+Enter
                      </kbd>{' '}
                      to submit
                    </span>
                  </span>

                  <Button
                    type="submit"
                    variant="primary"
                    size="md"
                    isLoading={isSubmitting}
                    disabled={isSubmitting || !candidateAnswer.trim()}
                    iconLeft={Send}
                    className="w-full sm:w-auto"
                  >
                    Submit Answer for Evaluation
                  </Button>
                </div>
              </form>
            </Card>
          )}

          {/* =========================================================================
              EVALUATION FEEDBACK PANEL (AFTER EVALUATION)
             ========================================================================= */}
          {evaluationResult && (
            <Card className="p-6 sm:p-7 space-y-6 animate-fadeIn">
              {/* Feedback Header */}
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-100 dark:border-[#1E2533] pb-4">
                <div>
                  <Badge variant="emerald" size="sm" icon={CheckCircle2} className="mb-1">
                    Answer Evaluated
                  </Badge>
                  <h3 className="text-base sm:text-lg font-bold text-slate-900 dark:text-white">
                    Rubric Evaluation & Candidate Feedback
                  </h3>
                </div>

                {/* Score Pill */}
                {evaluationResult.overall_score !== null && evaluationResult.overall_score !== undefined && (
                  <div className="px-4 py-2.5 rounded-xl bg-indigo-50 dark:bg-indigo-500/15 border border-indigo-200 dark:border-indigo-500/30 text-center flex-shrink-0">
                    <span className="text-2xl font-black text-indigo-600 dark:text-indigo-400">
                      {Math.round(evaluationResult.overall_score)}
                    </span>
                    <span className="text-xs text-slate-500 dark:text-slate-400 font-normal ml-0.5">
                      /100
                    </span>
                    <div className="text-[10px] text-slate-500 dark:text-slate-400 uppercase tracking-wider font-bold">
                      Turn Score
                    </div>
                  </div>
                )}
              </div>

              {/* Synthesized Executive Summary */}
              {evaluationResult.feedback?.summary && (
                <div className="space-y-1.5">
                  <span className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                    Executive Evaluation Summary
                  </span>
                  <p className="text-xs sm:text-sm text-slate-800 dark:text-slate-200 leading-relaxed bg-slate-50/80 dark:bg-[#0D1017] p-4 rounded-xl border border-slate-200/80 dark:border-[#1E2535]">
                    {evaluationResult.feedback.summary}
                  </p>
                </div>
              )}

              {/* Strengths & Weaknesses Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Strengths */}
                {evaluationResult.feedback?.strengths?.length > 0 && (
                  <div className="p-4 rounded-xl bg-emerald-50/80 dark:bg-emerald-500/10 border border-emerald-200 dark:border-emerald-500/30 space-y-2">
                    <span className="text-xs font-bold text-emerald-800 dark:text-emerald-300 uppercase tracking-wider flex items-center gap-1.5">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
                      <span>Demonstrated Strengths</span>
                    </span>
                    <ul className="list-disc list-inside space-y-1.5 text-xs text-emerald-900 dark:text-emerald-200">
                      {evaluationResult.feedback.strengths.map((str, idx) => (
                        <li key={idx} className="leading-relaxed">
                          {str}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Weaknesses */}
                {evaluationResult.feedback?.weaknesses?.length > 0 && (
                  <div className="p-4 rounded-xl bg-amber-50/80 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/30 space-y-2">
                    <span className="text-xs font-bold text-amber-800 dark:text-amber-300 uppercase tracking-wider flex items-center gap-1.5">
                      <AlertTriangle className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400 flex-shrink-0" />
                      <span>Identified Weaknesses & Gaps</span>
                    </span>
                    <ul className="list-disc list-inside space-y-1.5 text-xs text-amber-900 dark:text-amber-200">
                      {evaluationResult.feedback.weaknesses.map((wk, idx) => (
                        <li key={idx} className="leading-relaxed">
                          {wk}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              {/* Missing Concepts */}
              {evaluationResult.feedback?.missing_concepts?.length > 0 && (
                <div className="space-y-2">
                  <span className="text-xs font-bold text-indigo-700 dark:text-indigo-400 uppercase tracking-wider">
                    Expected Technical Concepts Omitted
                  </span>
                  <div className="flex flex-wrap gap-2">
                    {evaluationResult.feedback.missing_concepts.map((concept, idx) => (
                      <Badge key={idx} variant="indigo" size="sm">
                        {concept}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Rubric Scores Breakdown */}
              {evaluationResult.scores && (
                <div className="space-y-3 pt-2 border-t border-slate-100 dark:border-[#1E2535]">
                  <span className="text-xs font-bold text-slate-900 dark:text-white uppercase tracking-wider block">
                    Rubric Criteria Breakdown
                  </span>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {Object.entries(RUBRIC_LABELS).map(([key, label]) => {
                      const scoreVal = evaluationResult.scores[key];
                      const val = scoreVal !== undefined && scoreVal !== null ? Math.round(scoreVal) : null;
                      return (
                        <div
                          key={key}
                          className="p-3.5 rounded-xl bg-slate-50/80 dark:bg-[#0E1118] border border-slate-200/80 dark:border-[#1E2535] space-y-2"
                        >
                          <div className="flex justify-between text-xs">
                            <span className="text-slate-600 dark:text-slate-400 font-medium">
                              {label}
                            </span>
                            <span className="font-bold text-slate-900 dark:text-slate-100 font-mono">
                              {val !== null ? `${val}/100` : '—'}
                            </span>
                          </div>
                          <div className="w-full h-1.5 bg-slate-200 dark:bg-[#202738] rounded-full overflow-hidden">
                            <div
                              className="h-full bg-indigo-600 dark:bg-indigo-500 rounded-full transition-all duration-300"
                              style={{ width: `${val || 0}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Next Question CTA */}
              <div className="pt-4 border-t border-slate-100 dark:border-[#1E2535] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  Ready for the next round? The adaptive engine adjusts difficulty based on your verified performance.
                </span>

                <Button
                  variant="primary"
                  size="md"
                  iconRight={ArrowRight}
                  onClick={fetchNextQuestion}
                  disabled={isGeneratingQuestion}
                  isLoading={isGeneratingQuestion}
                >
                  Next Adaptive Question
                </Button>
              </div>
            </Card>
          )}
        </div>

        {/* Right Column (1 Col): Session Telemetry & Knowledge State Sidebar */}
        <div className="space-y-6">
          {/* Progress & Difficulty Card */}
          <Card className="p-5 space-y-4">
            <h3 className="text-xs font-bold text-slate-900 dark:text-white uppercase tracking-wider flex items-center gap-2 border-b border-slate-100 dark:border-[#1E2533] pb-3">
              <Sliders className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" />
              <span>Adaptive Telemetry</span>
            </h3>

            <div className="space-y-3 text-xs">
              <div className="flex justify-between items-center">
                <span className="text-slate-600 dark:text-slate-400">Session Status</span>
                <span className="font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                  {sessionState?.status || 'IN_PROGRESS'}
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-slate-600 dark:text-slate-400">Current Difficulty</span>
                <span className="font-bold text-slate-900 dark:text-slate-100">
                  L{currentDifficulty}: {difficultyInfo.name.split(' ')[0]}
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-slate-600 dark:text-slate-400">Questions Evaluated</span>
                <span className="font-bold text-slate-900 dark:text-slate-100 font-mono">
                  {sessionState?.progress?.completed_questions || 0}
                </span>
              </div>

              {sessionState?.progress?.overall_score !== null && sessionState?.progress?.overall_score !== undefined && (
                <div className="flex justify-between items-center">
                  <span className="text-slate-600 dark:text-slate-400">Running Composite Score</span>
                  <span className="font-bold text-indigo-600 dark:text-indigo-400 font-mono">
                    {Math.round(sessionState.progress.overall_score)}/100
                  </span>
                </div>
              )}
            </div>
          </Card>

          {/* Tested Skills Tags */}
          {sessionState?.tested_skills?.length > 0 && (
            <Card className="p-5 space-y-3">
              <h3 className="text-xs font-bold text-slate-900 dark:text-white uppercase tracking-wider flex items-center gap-2 border-b border-slate-100 dark:border-[#1E2533] pb-2.5">
                <Layers className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" />
                <span>Tested Technical Skills</span>
              </h3>

              <div className="flex flex-wrap gap-1.5">
                {sessionState.tested_skills.map((skill, idx) => (
                  <span
                    key={idx}
                    className="px-2.5 py-1 rounded-md bg-slate-100 dark:bg-[#181D2A] border border-slate-200 dark:border-[#262D3D] text-slate-700 dark:text-slate-300 text-xs font-medium"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </Card>
          )}

          {/* Active Knowledge Gaps */}
          {groupedKnowledgeGaps.length > 0 && (
            <Card className="p-5 space-y-3">
              <h3 className="text-xs font-bold text-amber-800 dark:text-amber-400 uppercase tracking-wider flex items-center gap-2 border-b border-slate-100 dark:border-[#1E2533] pb-2.5">
                <AlertTriangle className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400" />
                <span>Identified Knowledge Gaps ({groupedKnowledgeGaps.length})</span>
              </h3>

              <div className="space-y-2.5 max-h-56 overflow-y-auto pr-1">
                {groupedKnowledgeGaps.map((group, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-xl bg-amber-50/80 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/30 text-xs space-y-1.5"
                  >
                    <div className="flex items-center justify-between font-bold text-slate-900 dark:text-slate-100">
                      <div className="flex items-center gap-1.5">
                        <span>{group.topic}</span>
                        {group.count > 1 && (
                          <span className="text-[10px] font-extrabold text-amber-800 dark:text-amber-300 bg-amber-200/80 dark:bg-amber-500/20 px-1.5 py-0.2 rounded-md">
                            × {group.count}
                          </span>
                        )}
                      </div>
                      <Badge variant="amber" size="sm">
                        {group.highestSeverity}
                      </Badge>
                    </div>

                    {group.weaknesses.length > 0 && (
                      <ul className="text-slate-600 dark:text-slate-400 text-[11px] list-disc list-inside space-y-1 pt-0.5">
                        {group.weaknesses.map((w, wIdx) => (
                          <li key={wIdx} className="leading-relaxed">
                            {w}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Practice Tips */}
          <Card className="p-5 text-xs text-slate-600 dark:text-slate-400 space-y-2.5 bg-slate-50/60 dark:bg-[#0E1118]">
            <span className="font-bold text-slate-900 dark:text-slate-200 uppercase tracking-wider text-[11px] block">
              Interview Practice Guidelines
            </span>
            <ul className="space-y-1.5 list-disc list-inside text-[11px] leading-relaxed">
              <li>Address concrete trade-offs, not just high-level definitions.</li>
              <li>Cite concrete architectural patterns from your verified projects.</li>
              <li>Explain error handling, concurrency, and failure recovery.</li>
            </ul>
          </Card>
        </div>
      </div>

      {/* End Interview Confirmation Modal */}
      <Modal
        isOpen={showEndModal}
        onClose={() => setShowEndModal(false)}
        title="Conclude Interview Session?"
        description="Ending the interview will finalize all evaluated responses, compute your 6-dimensional rubric scorecard, and generate your candidate knowledge gap report."
      >
        <div className="space-y-4">
          <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-[#0D1017] border border-slate-200 dark:border-[#202636] text-xs text-slate-600 dark:text-slate-400">
            Questions Answered in this Round:{' '}
            <span className="font-bold text-slate-900 dark:text-slate-100 font-mono">
              {sessionState?.progress?.completed_questions || 0}
            </span>
          </div>

          <div className="flex items-center justify-end gap-3 pt-2">
            <Button
              variant="outline"
              size="sm"
              disabled={isEnding}
              onClick={() => setShowEndModal(false)}
            >
              Continue Interview
            </Button>

            <Button
              variant="danger"
              size="sm"
              isLoading={isEnding}
              disabled={isEnding}
              iconLeft={LogOut}
              onClick={handleConfirmEnd}
            >
              Conclude & View Scorecard
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
