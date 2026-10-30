import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  Award,
  CheckCircle2,
  AlertTriangle,
  RotateCw,
  LayoutDashboard,
  BrainCircuit,
  Compass,
  ChevronDown,
  ChevronUp,
  Loader2,
  AlertCircle,
  FileText
} from 'lucide-react';
import {
  getSessionScorecard,
  getSessionKnowledgeState,
  getSessionKnowledgeGaps
} from '../api/interview';
import { Button, Badge, Card, CardHeader, CardTitle, CardContent } from './ui';

function getScoreTier(score) {
  if (score === null || score === undefined) {
    return { label: 'Incomplete', variant: 'neutral' };
  }
  if (score >= 85) return { label: 'Exceptional (85+)', variant: 'emerald' };
  if (score >= 70) return { label: 'Strong Mastery (70+)', variant: 'indigo' };
  if (score >= 55) return { label: 'Proficient (55+)', variant: 'amber' };
  return { label: 'Needs Development (<55)', variant: 'rose' };
}

export default function InterviewScorecard({ sessionId, onRestart }) {
  const [scorecard, setScorecard] = useState(null);
  const [knowledgeState, setKnowledgeState] = useState(null);
  const [knowledgeGaps, setKnowledgeGaps] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedAnswers, setExpandedAnswers] = useState({});

  const fetchScorecardData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [scorecardData, ksData, gapsData] = await Promise.all([
        getSessionScorecard(sessionId),
        getSessionKnowledgeState(sessionId).catch(() => null),
        getSessionKnowledgeGaps(sessionId).catch(() => []),
      ]);

      setScorecard(scorecardData);
      setKnowledgeState(ksData);
      setKnowledgeGaps(gapsData || []);
    } catch (err) {
      setError(err.message || 'Failed to load session scorecard telemetry.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (sessionId) {
      fetchScorecardData();
    }
  }, [sessionId]);

  const toggleAnswerExpand = (index) => {
    setExpandedAnswers((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  if (loading) {
    return (
      <div className="py-24 text-center space-y-4 animate-fadeIn">
        <Loader2 className="w-10 h-10 text-indigo-600 dark:text-indigo-400 animate-spin mx-auto" />
        <div className="space-y-1">
          <p className="text-sm font-bold text-slate-900 dark:text-white">
            Synthesizing Adaptive Scorecard...
          </p>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Aggregating multi-dimensional rubric evaluations and updating candidate knowledge state.
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Card className="p-8 border-rose-200 dark:border-rose-500/30 text-center space-y-4 max-w-xl mx-auto my-12 animate-fadeIn">
        <AlertCircle className="w-8 h-8 text-rose-500 mx-auto" />
        <div>
          <h3 className="text-base font-bold text-slate-900 dark:text-white">
            Scorecard Synthesis Notice
          </h3>
          <p className="text-xs text-rose-600 dark:text-rose-400 mt-1">{error}</p>
        </div>
        <div className="pt-2">
          <Button variant="outline" size="sm" iconLeft={RotateCw} onClick={fetchScorecardData}>
            Retry Loading Scorecard
          </Button>
        </div>
      </Card>
    );
  }

  if (!scorecard) return null;

  const overallScore =
    scorecard.overall_score !== null && scorecard.overall_score !== undefined
      ? Math.round(scorecard.overall_score)
      : null;

  const tier = getScoreTier(overallScore);

  const answers = scorecard.answers || [];
  const evaluatedAnswers = answers.filter((a) => a.scores);

  const rubricDimensions = [
    { key: 'technical_accuracy', label: 'Technical Accuracy' },
    { key: 'conceptual_understanding', label: 'Conceptual Understanding' },
    { key: 'completeness', label: 'Completeness' },
    { key: 'depth_and_specificity', label: 'Depth & Specificity' },
    { key: 'practical_reasoning', label: 'Practical Reasoning' },
    { key: 'evidence_alignment', label: 'Resume Evidence Alignment' },
  ];

  const dimensionAverages = rubricDimensions.map(({ key, label }) => {
    if (evaluatedAnswers.length === 0) return { key, label, avg: null };
    const total = evaluatedAnswers.reduce((sum, ans) => sum + (ans.scores[key] || 0), 0);
    return { key, label, avg: Math.round(total / evaluatedAnswers.length) };
  });

  return (
    <div className="space-y-8 animate-fadeIn max-w-5xl mx-auto pb-16">
      {/* Top Banner */}
      <Card className="p-6 sm:p-8 overflow-hidden relative">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2">
              <Badge variant="emerald" size="sm" icon={CheckCircle2}>
                Interview Concluded • {scorecard.status}
              </Badge>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white">
              Adaptive Technical Interview Scorecard
            </h1>
            <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400 max-w-2xl leading-relaxed">
              Performance breakdown synthesized from real-time question responses, grounded claim verifications, and multi-dimensional rubric evaluations.
            </p>
          </div>

          {/* Overall Score Highlight */}
          <div className="flex items-center gap-4 bg-slate-50 dark:bg-[#0E1118] border border-slate-200/80 dark:border-[#1E2535] p-4.5 rounded-2xl flex-shrink-0">
            <div className="text-center">
              <div className="text-3xl sm:text-4xl font-black tracking-tight text-slate-900 dark:text-white font-mono">
                {overallScore !== null ? overallScore : '—'}
                <span className="text-xs text-slate-500 dark:text-slate-400 font-normal ml-0.5">/100</span>
              </div>
              <div className="mt-1.5">
                <Badge variant={tier.variant} size="sm">
                  {tier.label}
                </Badge>
              </div>
            </div>
            <div className="border-l border-slate-200 dark:border-[#232B3D] pl-4 space-y-1 text-xs text-slate-600 dark:text-slate-400">
              <div>
                Evaluated:{' '}
                <span className="font-bold text-slate-900 dark:text-slate-100 font-mono">
                  {scorecard.completed_questions}
                </span>
              </div>
              <div>
                Total Turns:{' '}
                <span className="font-bold text-slate-900 dark:text-slate-100 font-mono">
                  {scorecard.total_questions}
                </span>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Grid: Rubric Dimensions Breakdown + Knowledge Gaps */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Rubric Dimensions */}
        <Card className="p-6 space-y-5">
          <div className="flex items-center justify-between border-b border-slate-100 dark:border-[#1E2533] pb-3">
            <h2 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Award className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
              <span>6-Dimensional Rubric Performance</span>
            </h2>
            <span className="text-[11px] text-slate-500 dark:text-slate-400 font-medium font-mono">
              Deterministic 0–100 Scale
            </span>
          </div>

          <div className="space-y-3.5">
            {dimensionAverages.map(({ key, label, avg }) => (
              <div key={key} className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-600 dark:text-slate-400 font-medium">{label}</span>
                  <span className="font-bold text-slate-900 dark:text-slate-100 font-mono">
                    {avg !== null ? `${avg}/100` : '—'}
                  </span>
                </div>
                <div className="w-full h-2 bg-slate-100 dark:bg-[#1D2333] rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      avg === null
                        ? 'bg-slate-300 w-0'
                        : avg >= 80
                        ? 'bg-emerald-500'
                        : avg >= 60
                        ? 'bg-indigo-600 dark:bg-indigo-500'
                        : avg >= 40
                        ? 'bg-amber-500'
                        : 'bg-rose-500'
                    }`}
                    style={{ width: `${avg || 0}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Knowledge Gaps & Priority Focus */}
        <Card className="p-6 space-y-5">
          <div className="flex items-center justify-between border-b border-slate-100 dark:border-[#1E2533] pb-3">
            <h2 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Compass className="w-4 h-4 text-amber-600 dark:text-amber-400" />
              <span>Identified Knowledge Gaps & Priorities</span>
            </h2>
            <Badge variant="amber" size="sm">
              {knowledgeGaps.length} Gaps Flagged
            </Badge>
          </div>

          {knowledgeGaps.length === 0 ? (
            <div className="py-10 text-center space-y-2">
              <CheckCircle2 className="w-8 h-8 text-emerald-600 dark:text-emerald-400 mx-auto" />
              <p className="text-xs font-bold text-slate-900 dark:text-white">
                No Critical Knowledge Gaps Detected
              </p>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 max-w-xs mx-auto leading-relaxed">
                Candidate answers demonstrated solid conceptual foundation and architectural accuracy across tested areas.
              </p>
            </div>
          ) : (
            <div className="space-y-3 max-h-80 overflow-y-auto pr-1">
              {knowledgeGaps.map((gap, idx) => {
                const isHigh = gap.severity === 'HIGH';
                return (
                  <div
                    key={gap.gap_id || idx}
                    className="p-3.5 rounded-xl bg-amber-50/70 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/30 text-xs space-y-1.5"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-bold text-slate-900 dark:text-slate-100 flex items-center gap-1.5">
                        <AlertTriangle
                          className={`w-3.5 h-3.5 flex-shrink-0 ${
                            isHigh ? 'text-rose-500' : 'text-amber-600 dark:text-amber-400'
                          }`}
                        />
                        <span>{gap.topic}</span>
                      </span>
                      <Badge variant={isHigh ? 'rose' : 'amber'} size="sm">
                        {gap.severity} PRIORITY
                      </Badge>
                    </div>
                    <p className="text-slate-600 dark:text-slate-400 text-[11px] leading-relaxed">
                      {gap.description}
                    </p>
                    {gap.evidence_or_feedback && (
                      <p className="text-slate-500 dark:text-slate-400 text-[11px] italic border-l-2 border-amber-400 dark:border-amber-500 pl-2 mt-1">
                        "{gap.evidence_or_feedback}"
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </Card>
      </div>

      {/* Candidate Knowledge State Summary */}
      {knowledgeState && knowledgeState.summary && (
        <Card className="p-6 space-y-4">
          <h2 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2 border-b border-slate-100 dark:border-[#1E2533] pb-3">
            <BrainCircuit className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
            <span>Demonstrated Topic Competency & Mastery Profile</span>
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Strengths */}
            <div className="p-4 rounded-xl bg-emerald-50/80 dark:bg-emerald-500/10 border border-emerald-200 dark:border-emerald-500/30 space-y-2">
              <span className="text-xs font-bold text-emerald-800 dark:text-emerald-300 uppercase tracking-wider flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                <span>Demonstrated Strengths (Mastery &ge; 75%)</span>
              </span>
              {knowledgeState.summary.demonstrated_strengths?.length > 0 ? (
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {knowledgeState.summary.demonstrated_strengths.map((str, idx) => (
                    <Badge key={idx} variant="emerald" size="sm">
                      {str}
                    </Badge>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  No topics met the &ge; 75% threshold in this session.
                </p>
              )}
            </div>

            {/* Weaknesses */}
            <div className="p-4 rounded-xl bg-rose-50/80 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 space-y-2">
              <span className="text-xs font-bold text-rose-800 dark:text-rose-300 uppercase tracking-wider flex items-center gap-1.5">
                <AlertTriangle className="w-3.5 h-3.5 text-rose-500" />
                <span>Demonstrated Weaknesses (Mastery &lt; 50%)</span>
              </span>
              {knowledgeState.summary.demonstrated_weaknesses?.length > 0 ? (
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {knowledgeState.summary.demonstrated_weaknesses.map((wk, idx) => (
                    <Badge key={idx} variant="rose" size="sm">
                      {wk}
                    </Badge>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  No weak topics fell below 50% threshold.
                </p>
              )}
            </div>
          </div>
        </Card>
      )}

      {/* Question-by-Question Evaluation Review */}
      {answers.length > 0 && (
        <Card className="p-6 space-y-4">
          <h2 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2 border-b border-slate-100 dark:border-[#1E2533] pb-3">
            <FileText className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
            <span>Turn-by-Turn Question & Evaluation Breakdown ({answers.length})</span>
          </h2>

          <div className="space-y-3">
            {answers.map((ans, idx) => {
              const isExpanded = !!expandedAnswers[idx];
              const score =
                ans.overall_score !== null && ans.overall_score !== undefined
                  ? Math.round(ans.overall_score)
                  : null;
              const feedback = ans.feedback || {};

              return (
                <div
                  key={ans.answer_id || idx}
                  className="rounded-xl border border-slate-200/80 dark:border-[#202636] bg-slate-50/50 dark:bg-[#11141C] overflow-hidden transition-all shadow-2xs"
                >
                  <div
                    onClick={() => toggleAnswerExpand(idx)}
                    className="p-4 cursor-pointer hover:bg-slate-100/60 dark:hover:bg-[#161B26] flex items-center justify-between gap-3 transition-colors"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-7 h-7 rounded-lg bg-indigo-50 dark:bg-indigo-500/15 border border-indigo-200 dark:border-indigo-500/30 text-indigo-600 dark:text-indigo-400 flex items-center justify-center font-bold text-xs flex-shrink-0 font-mono">
                        Q{idx + 1}
                      </div>
                      <div className="text-xs">
                        <span className="font-bold text-slate-900 dark:text-slate-100">
                          {ans.candidate_answer
                            ? `${ans.candidate_answer.slice(0, 80)}...`
                            : 'Answer submission'}
                        </span>
                        <div className="flex items-center gap-2 text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                          <span>
                            Status:{' '}
                            <span className="text-slate-800 dark:text-slate-200 font-semibold">
                              {ans.status}
                            </span>
                          </span>
                          {ans.attempt_number > 1 && (
                            <span>• Attempt #{ans.attempt_number}</span>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      {score !== null && (
                        <span className="px-2.5 py-1 rounded-lg bg-indigo-50 dark:bg-indigo-500/15 border border-indigo-200 dark:border-indigo-500/30 text-indigo-700 dark:text-indigo-300 font-bold text-xs font-mono">
                          {score}/100
                        </span>
                      )}
                      {isExpanded ? (
                        <ChevronUp className="w-4 h-4 text-slate-400" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-slate-400" />
                      )}
                    </div>
                  </div>

                  {isExpanded && (
                    <div className="p-4 border-t border-slate-200/80 dark:border-[#202636] bg-white dark:bg-[#0E1118] space-y-4 text-xs animate-fadeIn">
                      {/* Full Answer Text */}
                      {ans.candidate_answer && (
                        <div className="space-y-1">
                          <span className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider block">
                            Submitted Candidate Answer
                          </span>
                          <div className="p-3 bg-slate-50 dark:bg-[#11141C] rounded-xl border border-slate-200/80 dark:border-[#202636] text-slate-800 dark:text-slate-200 whitespace-pre-wrap font-mono text-[11px] leading-relaxed shadow-inner">
                            {ans.candidate_answer}
                          </div>
                        </div>
                      )}

                      {/* Evaluator Summary */}
                      {feedback.summary && (
                        <div className="space-y-1">
                          <span className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider block">
                            Evaluator Feedback Synthesis
                          </span>
                          <p className="text-slate-800 dark:text-slate-200 leading-relaxed bg-slate-50 dark:bg-[#11141C] p-3 rounded-xl border border-slate-200/80 dark:border-[#202636]">
                            {feedback.summary}
                          </p>
                        </div>
                      )}

                      {/* Strengths & Weaknesses */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
                        {feedback.strengths?.length > 0 && (
                          <div className="space-y-1.5 p-3 rounded-xl bg-emerald-50/80 dark:bg-emerald-500/10 border border-emerald-200 dark:border-emerald-500/30">
                            <span className="text-[11px] font-bold text-emerald-800 dark:text-emerald-300 uppercase tracking-wider">
                              Demonstrated Strengths
                            </span>
                            <ul className="list-disc list-inside space-y-1 text-emerald-900 dark:text-emerald-200 text-[11px]">
                              {feedback.strengths.map((str, sIdx) => (
                                <li key={sIdx}>{str}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {feedback.weaknesses?.length > 0 && (
                          <div className="space-y-1.5 p-3 rounded-xl bg-amber-50/80 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/30">
                            <span className="text-[11px] font-bold text-amber-800 dark:text-amber-300 uppercase tracking-wider">
                              Identified Weaknesses & Gaps
                            </span>
                            <ul className="list-disc list-inside space-y-1 text-amber-900 dark:text-amber-200 text-[11px]">
                              {feedback.weaknesses.map((wk, wIdx) => (
                                <li key={wIdx}>{wk}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>

                      {/* Missing Concepts */}
                      {feedback.missing_concepts?.length > 0 && (
                        <div className="space-y-1.5">
                          <span className="text-[11px] font-bold text-indigo-700 dark:text-indigo-400 uppercase tracking-wider">
                            Expected Concepts Omitted
                          </span>
                          <div className="flex flex-wrap gap-1.5">
                            {feedback.missing_concepts.map((concept, cIdx) => (
                              <Badge key={cIdx} variant="indigo" size="sm">
                                {concept}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {/* Navigation Actions */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t border-slate-200/80 dark:border-[#1E2535]">
        <Button
          variant="outline"
          size="md"
          iconLeft={LayoutDashboard}
          onClick={() => (window.location.href = '/dashboard')}
        >
          Return to Dashboard
        </Button>

        <Button
          variant="primary"
          size="md"
          iconLeft={RotateCw}
          onClick={onRestart || (() => (window.location.href = '/interview'))}
        >
          Practice Another Interview
        </Button>
      </div>
    </div>
  );
}
