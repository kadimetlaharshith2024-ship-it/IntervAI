import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  FileText,
  ArrowLeft,
  Database,
  Share2,
  FileCheck,
  CheckCircle2,
  AlertCircle,
  Loader2,
  RotateCcw,
  Sparkles,
  Clock,
  ChevronRight,
  Filter,
  FileCode,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  Play
} from 'lucide-react';
import {
  getResume,
  getEntities,
  extractEntities,
  getRelationships,
  extractRelationships,
  getClaims,
  indexResume
} from '../api/resumes';
import { Button, Badge, Card, SkeletonCard } from '../components/ui';

export default function ResumeDetail() {
  const { resumeId } = useParams();

  // Core Data
  const [resume, setResume] = useState(null);
  const [entitiesData, setEntitiesData] = useState({ total_extracted: 0, entities: [] });
  const [relationshipsData, setRelationshipsData] = useState({ total_relationships: 0, relationships: [] });
  const [claimsData, setClaimsData] = useState([]);

  // Loading & Error states
  const [initialLoading, setInitialLoading] = useState(true);
  const [pageError, setPageError] = useState(null);

  // Active Tab: 'entities' | 'relationships' | 'claims' | 'raw_text'
  const [activeTab, setActiveTab] = useState('entities');

  // Filter for entities category
  const [selectedCategory, setSelectedCategory] = useState('ALL');

  // Stage execution loading states
  const [runningStage, setRunningStage] = useState(null);
  const [stageError, setStageError] = useState(null);

  // Raw text preview state
  const [isTextExpanded, setIsTextExpanded] = useState(false);
  const [copiedText, setCopiedText] = useState(false);

  // Fetch all resume assets
  const fetchAllData = useCallback(async () => {
    setInitialLoading(true);
    setPageError(null);
    try {
      const resumeRes = await getResume(resumeId);
      setResume(resumeRes);

      try {
        const entRes = await getEntities(resumeId);
        setEntitiesData(entRes || { total_extracted: 0, entities: [] });
      } catch (err) {
        console.warn('Could not fetch entities:', err);
      }

      try {
        const relRes = await getRelationships(resumeId, 'advanced');
        setRelationshipsData(relRes || { total_relationships: 0, relationships: [] });
      } catch (err) {
        console.warn('Could not fetch relationships:', err);
      }

      try {
        const claimsRes = await getClaims(resumeId);
        setClaimsData(claimsRes || []);
      } catch (err) {
        console.warn('Could not fetch claims:', err);
      }
    } catch (err) {
      setPageError(err.message || 'Failed to load resume details.');
    } finally {
      setInitialLoading(false);
    }
  }, [resumeId]);

  useEffect(() => {
    if (resumeId) {
      fetchAllData();
    }
  }, [resumeId, fetchAllData]);

  // Stage triggers
  const handleExtractEntities = async () => {
    setRunningStage('entities');
    setStageError(null);
    try {
      const res = await extractEntities(resumeId);
      setEntitiesData(res);
    } catch (err) {
      setStageError(`Entity extraction failed: ${err.message}`);
    } finally {
      setRunningStage(null);
    }
  };

  const handleExtractRelationships = async () => {
    setRunningStage('relationships');
    setStageError(null);
    try {
      const res = await extractRelationships(resumeId, 'advanced');
      setRelationshipsData(res);
    } catch (err) {
      setStageError(`Relationship extraction failed: ${err.message}`);
    } finally {
      setRunningStage(null);
    }
  };

  const handleIndexRAG = async () => {
    setRunningStage('rag');
    setStageError(null);
    try {
      await indexResume(resumeId);
      const updatedClaims = await getClaims(resumeId);
      setClaimsData(updatedClaims || []);
    } catch (err) {
      setStageError(`Indexing failed: ${err.message}`);
    } finally {
      setRunningStage(null);
    }
  };

  const handleRunFullPipeline = async () => {
    setRunningStage('full');
    setStageError(null);
    try {
      const entRes = await extractEntities(resumeId);
      setEntitiesData(entRes);

      const relRes = await extractRelationships(resumeId, 'advanced');
      setRelationshipsData(relRes);

      await indexResume(resumeId);
      const updatedClaims = await getClaims(resumeId);
      setClaimsData(updatedClaims || []);
    } catch (err) {
      setStageError(`Analysis execution stopped: ${err.message}`);
    } finally {
      setRunningStage(null);
    }
  };

  const handleCopyRawText = () => {
    if (resume?.raw_text_preview) {
      navigator.clipboard.writeText(resume.raw_text_preview);
      setCopiedText(true);
      setTimeout(() => setCopiedText(false), 2000);
    }
  };

  // Categories list
  const categories = useMemo(() => {
    const set = new Set();
    entitiesData.entities?.forEach((e) => {
      if (e.category) set.add(e.category);
    });
    return ['ALL', ...Array.from(set)];
  }, [entitiesData.entities]);

  // Filtered entities
  const filteredEntities = useMemo(() => {
    if (selectedCategory === 'ALL') return entitiesData.entities || [];
    return (entitiesData.entities || []).filter((e) => e.category === selectedCategory);
  }, [entitiesData.entities, selectedCategory]);

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

  if (initialLoading) {
    return (
      <div className="py-24 text-center space-y-4 animate-fadeIn">
        <Loader2 className="w-8 h-8 text-indigo-600 dark:text-indigo-400 animate-spin mx-auto" />
        <p className="text-xs font-semibold text-slate-500 dark:text-slate-400">
          Loading resume analysis and extraction data...
        </p>
      </div>
    );
  }

  if (pageError) {
    return (
      <div className="max-w-2xl mx-auto py-12 space-y-4 animate-fadeIn">
        <Link
          to="/resumes"
          className="inline-flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-900 dark:hover:text-slate-100 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Resumes</span>
        </Link>
        <Card className="p-6 border-rose-200 dark:border-rose-500/30 text-xs space-y-3">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-rose-500 flex-shrink-0" />
            <span className="font-bold text-sm text-slate-900 dark:text-white">
              Failed to retrieve resume
            </span>
          </div>
          <p className="text-rose-600 dark:text-rose-400">{pageError}</p>
          <Button variant="outline" size="xs" iconLeft={RotateCcw} onClick={fetchAllData}>
            Try Again
          </Button>
        </Card>
      </div>
    );
  }

  const hasEntities = (entitiesData.entities?.length || 0) > 0;
  const hasRelationships = (relationshipsData.relationships?.length || 0) > 0;
  const hasClaims = claimsData.length > 0;
  const isFullyProcessed = hasEntities && hasRelationships && hasClaims;

  return (
    <div className="space-y-8 max-w-6xl mx-auto animate-fadeIn">
      {/* Navigation & Header */}
      <div className="space-y-4">
        <Link
          to="/resumes"
          className="inline-flex items-center gap-1.5 text-xs font-bold text-slate-500 dark:text-slate-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to All Resumes</span>
        </Link>

        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-slate-200/80 dark:border-[#1E2433] pb-6">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Badge variant="emerald" size="sm">
                Parsed Document
              </Badge>
              <span className="text-xs text-slate-400 dark:text-slate-500 font-mono">
                ID: {resume?.id?.slice(0, 8)}...
              </span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white flex items-center gap-3">
              <FileText className="w-7 h-7 text-indigo-600 dark:text-indigo-400 flex-shrink-0" />
              <span>{resume?.filename}</span>
            </h1>
            <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500 dark:text-slate-400">
              <span className="flex items-center gap-1">
                <Clock className="w-3.5 h-3.5 text-slate-400" />
                Uploaded {formatDate(resume?.created_at)}
              </span>
              <span>•</span>
              <span className="font-mono">
                {resume?.character_count?.toLocaleString()} characters parsed
              </span>
              <span>•</span>
              <span className="font-semibold text-slate-700 dark:text-slate-300">
                {Object.values(resume?.detected_sections || {}).filter(Boolean).length}/6 sections identified
              </span>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="flex items-center gap-2.5 self-start lg:self-auto">
            <Button
              variant="secondary"
              size="sm"
              iconRight={ArrowLeft}
              onClick={() => (window.location.href = `/interview?resume=${resume?.id}`)}
              className="rotate-180"
            >
              Practice with this Resume
            </Button>

            {!isFullyProcessed && (
              <Button
                variant="primary"
                size="sm"
                isLoading={runningStage === 'full'}
                disabled={runningStage !== null}
                iconLeft={Sparkles}
                onClick={handleRunFullPipeline}
              >
                Complete Analysis
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Stage-level Error Banner */}
      {stageError && (
        <div className="p-4 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-2xl text-xs text-rose-800 dark:text-rose-300 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-600 dark:text-rose-400 flex-shrink-0" />
            <span>{stageError}</span>
          </div>
          <button
            type="button"
            onClick={() => setStageError(null)}
            className="text-xs text-rose-700 dark:text-rose-300 hover:text-rose-900 font-bold cursor-pointer"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Pipeline Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Stage 1: Entities */}
        <Card className="p-5 flex flex-col justify-between space-y-4">
          <div>
            <div className="flex items-center justify-between gap-2 mb-2">
              <span className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                Skills & Technologies
              </span>
              {hasEntities ? (
                <Badge variant="emerald" size="sm" icon={CheckCircle2}>
                  Ready
                </Badge>
              ) : (
                <Badge variant="amber" size="sm">
                  Pending
                </Badge>
              )}
            </div>
            <h2 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Database className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
              <span>Technical Skills</span>
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-relaxed">
              {hasEntities
                ? `${entitiesData.total_extracted} verified skills, languages, and tools identified.`
                : 'Extract technical keywords and tools from your document.'}
            </p>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 dark:border-[#1E2533] flex items-center justify-between">
            <span className="text-[11px] text-slate-500 dark:text-slate-400 font-medium">
              {hasEntities ? `${entitiesData.entities.length} skills listed` : 'Not extracted yet'}
            </span>
            <button
              type="button"
              disabled={runningStage !== null}
              onClick={handleExtractEntities}
              className="text-xs text-indigo-600 dark:text-indigo-400 hover:text-indigo-500 font-bold inline-flex items-center gap-1 disabled:opacity-40 cursor-pointer"
            >
              {runningStage === 'entities' ? (
                <>
                  <Loader2 className="w-3 h-3 animate-spin" />
                  <span>Extracting...</span>
                </>
              ) : (
                <>
                  <span>{hasEntities ? 'Re-extract' : 'Run Extraction'}</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                </>
              )}
            </button>
          </div>
        </Card>

        {/* Stage 2: Relationships */}
        <Card className="p-5 flex flex-col justify-between space-y-4">
          <div>
            <div className="flex items-center justify-between gap-2 mb-2">
              <span className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                Project Connections
              </span>
              {hasRelationships ? (
                <Badge variant="emerald" size="sm" icon={CheckCircle2}>
                  Ready
                </Badge>
              ) : (
                <Badge variant="amber" size="sm">
                  Pending
                </Badge>
              )}
            </div>
            <h2 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Share2 className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
              <span>Connected Concepts</span>
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-relaxed">
              {hasRelationships
                ? `${relationshipsData.total_relationships} technology-to-project relationships mapped.`
                : 'Connect your tools and frameworks to specific projects.'}
            </p>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 dark:border-[#1E2533] flex items-center justify-between">
            <span className="text-[11px] text-slate-500 dark:text-slate-400 font-medium">
              {hasRelationships ? `${relationshipsData.relationships.length} connections` : 'Not mapped yet'}
            </span>
            <button
              type="button"
              disabled={runningStage !== null}
              onClick={handleExtractRelationships}
              className="text-xs text-indigo-600 dark:text-indigo-400 hover:text-indigo-500 font-bold inline-flex items-center gap-1 disabled:opacity-40 cursor-pointer"
            >
              {runningStage === 'relationships' ? (
                <>
                  <Loader2 className="w-3 h-3 animate-spin" />
                  <span>Mapping...</span>
                </>
              ) : (
                <>
                  <span>{hasRelationships ? 'Re-map' : 'Map Connections'}</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                </>
              )}
            </button>
          </div>
        </Card>

        {/* Stage 3: Grounded Claims */}
        <Card className="p-5 flex flex-col justify-between space-y-4">
          <div>
            <div className="flex items-center justify-between gap-2 mb-2">
              <span className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                Interview Grounding
              </span>
              {hasClaims ? (
                <Badge variant="emerald" size="sm" icon={CheckCircle2}>
                  Indexed
                </Badge>
              ) : (
                <Badge variant="amber" size="sm">
                  Pending
                </Badge>
              )}
            </div>
            <h2 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <FileCheck className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
              <span>Verified Claims</span>
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-relaxed">
              {hasClaims
                ? `${claimsData.length} verified statements ready for question generation.`
                : 'Index specific claims and accomplishments from your resume.'}
            </p>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 dark:border-[#1E2533] flex items-center justify-between">
            <span className="text-[11px] text-slate-500 dark:text-slate-400 font-medium">
              {hasClaims ? `${claimsData.length} claims verified` : 'Not indexed yet'}
            </span>
            <button
              type="button"
              disabled={runningStage !== null}
              onClick={handleIndexRAG}
              className="text-xs text-indigo-600 dark:text-indigo-400 hover:text-indigo-500 font-bold inline-flex items-center gap-1 disabled:opacity-40 cursor-pointer"
            >
              {runningStage === 'rag' ? (
                <>
                  <Loader2 className="w-3 h-3 animate-spin" />
                  <span>Indexing...</span>
                </>
              ) : (
                <>
                  <span>{hasClaims ? 'Re-index' : 'Index Claims'}</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                </>
              )}
            </button>
          </div>
        </Card>
      </div>

      {/* Tabs Navigation */}
      <div className="border-b border-slate-200/80 dark:border-[#1E2433] flex gap-2 overflow-x-auto">
        <button
          type="button"
          onClick={() => setActiveTab('entities')}
          className={`pb-3 px-3 text-xs sm:text-sm font-bold flex items-center gap-2 border-b-2 transition-colors whitespace-nowrap cursor-pointer ${
            activeTab === 'entities'
              ? 'border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
          }`}
        >
          <Database className="w-4 h-4" />
          <span>Skills & Technologies ({entitiesData.entities?.length || 0})</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('relationships')}
          className={`pb-3 px-3 text-xs sm:text-sm font-bold flex items-center gap-2 border-b-2 transition-colors whitespace-nowrap cursor-pointer ${
            activeTab === 'relationships'
              ? 'border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
          }`}
        >
          <Share2 className="w-4 h-4" />
          <span>Connected Concepts ({relationshipsData.relationships?.length || 0})</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('claims')}
          className={`pb-3 px-3 text-xs sm:text-sm font-bold flex items-center gap-2 border-b-2 transition-colors whitespace-nowrap cursor-pointer ${
            activeTab === 'claims'
              ? 'border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
          }`}
        >
          <FileCheck className="w-4 h-4" />
          <span>Verified Claims ({claimsData.length})</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('raw_text')}
          className={`pb-3 px-3 text-xs sm:text-sm font-bold flex items-center gap-2 border-b-2 transition-colors whitespace-nowrap cursor-pointer ${
            activeTab === 'raw_text'
              ? 'border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
          }`}
        >
          <FileCode className="w-4 h-4" />
          <span>Document & Sections</span>
        </button>
      </div>

      {/* Tab 1: Entities Content */}
      {activeTab === 'entities' && (
        <div className="space-y-6">
          {!hasEntities ? (
            <Card className="py-14 text-center p-6 space-y-3">
              <Database className="w-10 h-10 text-slate-400 mx-auto" />
              <h3 className="text-base font-bold text-slate-900 dark:text-white">
                Skills Have Not Been Extracted Yet
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400 max-w-sm mx-auto leading-relaxed">
                Run skill extraction to discover programming languages, frameworks, databases, and architectural concepts in this resume.
              </p>
              <div className="pt-2">
                <Button
                  variant="primary"
                  size="sm"
                  disabled={runningStage !== null}
                  iconLeft={Sparkles}
                  onClick={handleExtractEntities}
                >
                  Extract Skills Now
                </Button>
              </div>
            </Card>
          ) : (
            <div className="space-y-4">
              {/* Category Filters */}
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs text-slate-500 dark:text-slate-400 mr-1 flex items-center gap-1 font-bold">
                  <Filter className="w-3.5 h-3.5" /> Filter Category:
                </span>
                {categories.map((cat) => (
                  <button
                    key={cat}
                    type="button"
                    onClick={() => setSelectedCategory(cat)}
                    className={`text-xs px-3 py-1.5 rounded-xl border transition-all font-bold cursor-pointer ${
                      selectedCategory === cat
                        ? 'bg-indigo-600 text-white border-indigo-600 shadow-2xs'
                        : 'bg-white dark:bg-[#11141C] text-slate-600 dark:text-slate-400 border-slate-200 dark:border-[#202636] hover:bg-slate-50 dark:hover:bg-[#181D2A]'
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>

              {/* Entity Cards Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {filteredEntities.map((entity, idx) => {
                  const confPct = Math.round(entity.confidence * 100);

                  let badgeVariant = 'neutral';
                  if (entity.category === 'Language') badgeVariant = 'sky';
                  else if (entity.category === 'Framework') badgeVariant = 'indigo';
                  else if (entity.category === 'Database') badgeVariant = 'emerald';
                  else if (entity.category === 'Tool/DevOps') badgeVariant = 'amber';
                  else if (entity.category === 'Core Concept') badgeVariant = 'purple';

                  return (
                    <Card
                      key={`${entity.name}-${idx}`}
                      className="p-3.5 space-y-2.5"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100 truncate">
                          {entity.name}
                        </h4>
                        <Badge variant={badgeVariant} size="sm">
                          {entity.category}
                        </Badge>
                      </div>

                      {/* Confidence Bar */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-[10px] text-slate-500 dark:text-slate-400">
                          <span>Confidence</span>
                          <span className="font-bold text-slate-900 dark:text-slate-100 font-mono">
                            {confPct}%
                          </span>
                        </div>
                        <div className="w-full h-1.5 bg-slate-100 dark:bg-[#1E2535] rounded-full overflow-hidden">
                          <div
                            className="h-full bg-indigo-600 dark:bg-indigo-500 rounded-full"
                            style={{ width: `${confPct}%` }}
                          />
                        </div>
                      </div>
                    </Card>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tab 2: Relationships Content */}
      {activeTab === 'relationships' && (
        <div className="space-y-4">
          {!hasRelationships ? (
            <Card className="py-14 text-center p-6 space-y-3">
              <Share2 className="w-10 h-10 text-slate-400 mx-auto" />
              <h3 className="text-base font-bold text-slate-900 dark:text-white">
                Connections Have Not Been Mapped Yet
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400 max-w-sm mx-auto leading-relaxed">
                Discover semantic relationships between your projects, languages, and tools to prepare grounded questions.
              </p>
              <div className="pt-2">
                <Button
                  variant="primary"
                  size="sm"
                  disabled={runningStage !== null}
                  iconLeft={Sparkles}
                  onClick={handleExtractRelationships}
                >
                  Map Connections Now
                </Button>
              </div>
            </Card>
          ) : (
            <div className="space-y-3">
              <div className="text-xs text-slate-500 dark:text-slate-400 flex items-center justify-between">
                <span>
                  Showing{' '}
                  <strong className="text-slate-900 dark:text-white font-mono">
                    {relationshipsData.relationships.length}
                  </strong>{' '}
                  mapped project-to-skill connections
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {relationshipsData.relationships.map((rel, idx) => (
                  <Card
                    key={idx}
                    className="p-4 space-y-3"
                  >
                    {/* Triplet Header: Source -> Type -> Target */}
                    <div className="flex flex-wrap items-center gap-1.5 text-xs">
                      <span className="px-2.5 py-1 rounded-md bg-slate-100 dark:bg-[#181D2A] font-bold text-slate-900 dark:text-slate-100 border border-slate-200 dark:border-[#262D3D]">
                        {rel.source_entity}
                      </span>
                      <Badge variant="indigo" size="sm">
                        {rel.relation_type}
                      </Badge>
                      <span className="px-2.5 py-1 rounded-md bg-slate-100 dark:bg-[#181D2A] font-bold text-slate-900 dark:text-slate-100 border border-slate-200 dark:border-[#262D3D]">
                        {rel.target_entity}
                      </span>
                    </div>

                    {/* Context metadata */}
                    <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-500 dark:text-slate-400">
                      {rel.section && (
                        <span className="px-2 py-0.5 rounded bg-slate-50 dark:bg-[#0D1017] border border-slate-200/80 dark:border-[#202737]">
                          Section: {rel.section}
                        </span>
                      )}
                      {rel.context_entity && (
                        <span className="px-2 py-0.5 rounded bg-slate-50 dark:bg-[#0D1017] border border-slate-200/80 dark:border-[#202737]">
                          Context: {rel.context_entity}
                        </span>
                      )}
                      <Badge variant="emerald" size="sm">
                        Confidence: {Math.round(rel.confidence * 100)}%
                      </Badge>
                    </div>

                    {/* Grounded Evidence Text */}
                    {rel.evidence_text && (
                      <div className="bg-slate-50 dark:bg-[#0D1017] border border-slate-200/80 dark:border-[#202737] rounded-lg p-2.5 text-xs text-slate-600 dark:text-slate-400 italic border-l-2 border-l-indigo-600 dark:border-l-indigo-400 leading-relaxed font-sans">
                        "{rel.evidence_text}"
                      </div>
                    )}
                  </Card>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tab 3: RAG Grounded Claims Content */}
      {activeTab === 'claims' && (
        <div className="space-y-4">
          {!hasClaims ? (
            <Card className="py-14 text-center p-6 space-y-3">
              <FileCheck className="w-10 h-10 text-slate-400 mx-auto" />
              <h3 className="text-base font-bold text-slate-900 dark:text-white">
                Claims Have Not Been Indexed Yet
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400 max-w-sm mx-auto leading-relaxed">
                Index resume statements and accomplishments to ground interview questions directly in your verified project experience.
              </p>
              <div className="pt-2">
                <Button
                  variant="primary"
                  size="sm"
                  disabled={runningStage !== null}
                  iconLeft={Sparkles}
                  onClick={handleIndexRAG}
                >
                  Index Claims Now
                </Button>
              </div>
            </Card>
          ) : (
            <div className="space-y-3">
              <div className="text-xs text-slate-500 dark:text-slate-400">
                Showing{' '}
                <strong className="text-slate-900 dark:text-white font-mono">
                  {claimsData.length}
                </strong>{' '}
                verified resume claims prepared for interview questions
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {claimsData.map((claim) => (
                  <Card
                    key={claim.claim_id}
                    className="p-4 space-y-3"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 dark:bg-[#181D2A] border border-slate-200 dark:border-[#262D3D] text-slate-500 dark:text-slate-400">
                        {claim.claim_id?.slice(0, 10)}...
                      </span>
                      <Badge variant="emerald" size="sm">
                        {claim.status || 'Verified'}
                      </Badge>
                    </div>

                    <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100 leading-relaxed">
                      {claim.claim_text}
                    </h4>

                    {/* Metadata tags */}
                    <div className="flex flex-wrap gap-1.5 text-[10px] text-slate-500 dark:text-slate-400">
                      {claim.section && (
                        <span className="px-2 py-0.5 rounded bg-slate-50 dark:bg-[#0D1017] border border-slate-200/80 dark:border-[#202737]">
                          {claim.section}
                        </span>
                      )}
                      {claim.project && (
                        <Badge variant="indigo" size="sm">
                          Project: {claim.project}
                        </Badge>
                      )}
                      {claim.target_entity && (
                        <Badge variant="sky" size="sm">
                          {claim.target_entity}
                        </Badge>
                      )}
                      <span className="px-2 py-0.5 rounded bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-200 dark:border-emerald-500/20 text-emerald-700 dark:text-emerald-300 font-semibold font-mono">
                        Conf: {Math.round(claim.confidence * 100)}%
                      </span>
                    </div>

                    {/* Evidence quote */}
                    {claim.evidence_text && (
                      <div className="bg-slate-50 dark:bg-[#0D1017] border border-slate-200/80 dark:border-[#202737] rounded-lg p-2.5 text-xs text-slate-600 dark:text-slate-400 italic border-l-2 border-l-emerald-500 leading-relaxed font-sans">
                        "{claim.evidence_text}"
                      </div>
                    )}
                  </Card>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tab 4: Raw Text & Sections Content */}
      {activeTab === 'raw_text' && (
        <div className="space-y-6">
          {/* Detected Sections Summary */}
          <Card className="p-5 space-y-3">
            <h3 className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
              Identified Document Sections
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2.5">
              {Object.entries(resume?.detected_sections || {}).map(([sec, isDetected]) => {
                let badgeText = isDetected ? 'Detected' : 'Not Found';
                let subtext = null;

                if (sec === 'skills' && isDetected) {
                  if (entitiesData.entities?.length > 0) {
                    badgeText = 'Detected';
                    subtext = `${entitiesData.entities.length} skills`;
                  } else {
                    badgeText = 'Detected';
                    subtext = 'Ready to extract';
                  }
                } else if (sec === 'projects' && isDetected) {
                  const projCount = relationshipsData.relationships?.filter(
                    (r) => r.source_entity === 'Candidate' && r.relation_type === 'DEVELOPED'
                  ).length || 0;
                  if (projCount > 0) {
                    subtext = `${projCount} mapped`;
                  }
                }

                return (
                  <div
                    key={sec}
                    className={`p-3 rounded-xl border text-center transition-all ${
                      isDetected
                        ? 'bg-emerald-50/70 dark:bg-emerald-500/10 border-emerald-200 dark:border-emerald-500/20 text-emerald-800 dark:text-emerald-300'
                        : 'bg-slate-50 dark:bg-[#141822] border-slate-200 dark:border-[#202636] text-slate-400 opacity-60'
                    }`}
                  >
                    <p className="text-xs font-bold capitalize">{sec}</p>
                    <p className="text-[10px] mt-0.5 font-semibold">
                      {badgeText}
                    </p>
                    {subtext && (
                      <p className="text-[9px] text-emerald-600 dark:text-emerald-400 font-medium">
                        {subtext}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          </Card>

          {/* Clean Readable Text Preview */}
          <Card className="p-6 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 dark:border-[#1E2533] pb-4">
              <div>
                <h3 className="text-sm font-bold text-slate-900 dark:text-white">
                  Resume Text Preview
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                  Clean parsed document text extracted from your uploaded file ({resume?.character_count?.toLocaleString()} characters).
                </p>
              </div>

              <div className="flex items-center gap-2 self-start sm:self-auto">
                <Button
                  variant="outline"
                  size="xs"
                  iconLeft={copiedText ? Check : Copy}
                  onClick={handleCopyRawText}
                >
                  {copiedText ? 'Copied' : 'Copy Text'}
                </Button>

                <Button
                  variant="secondary"
                  size="xs"
                  iconRight={isTextExpanded ? ChevronUp : ChevronDown}
                  onClick={() => setIsTextExpanded(!isTextExpanded)}
                >
                  {isTextExpanded ? 'Collapse' : 'Expand All'}
                </Button>
              </div>
            </div>

            {/* Typography */}
            <div
              className={`text-sm text-slate-700 dark:text-slate-300 leading-relaxed whitespace-pre-wrap font-sans transition-all ${
                isTextExpanded ? 'max-h-none' : 'max-h-80 overflow-hidden relative'
              }`}
            >
              {resume?.raw_text_preview || 'No text preview available for this document.'}

              {!isTextExpanded && (
                <div className="absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-white dark:from-[#11141C] via-white/80 dark:via-[#11141C]/80 to-transparent flex items-end justify-center pb-2">
                  <button
                    type="button"
                    onClick={() => setIsTextExpanded(true)}
                    className="text-xs font-bold text-indigo-600 dark:text-indigo-400 bg-white dark:bg-[#161B26] px-3.5 py-1 rounded-full shadow-xs border border-slate-200 dark:border-[#222838] hover:bg-slate-50 dark:hover:bg-[#1E2535] cursor-pointer"
                  >
                    Click to view full text
                  </button>
                </div>
              )}
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
