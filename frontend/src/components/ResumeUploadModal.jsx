import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  FileText,
  X,
  CheckCircle2,
  AlertCircle,
  Loader2,
  ArrowRight,
  RotateCcw,
  Sparkles,
  Database,
  Share2,
  FileCheck,
  ShieldCheck
} from 'lucide-react';
import {
  uploadResume,
  extractEntities,
  extractRelationships,
  indexResume
} from '../api/resumes';
import { Button, Badge } from './ui';

const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10MB limit

const PIPELINE_STAGES = [
  {
    id: 'upload',
    label: 'Upload & Parse PDF',
    description: 'Segment document sections and extract clean text structure',
    icon: UploadCloud,
  },
  {
    id: 'entities',
    label: 'Extract Skills & Taxonomy',
    description: 'Detect languages, frameworks, cloud platforms, and engineering tools',
    icon: Database,
  },
  {
    id: 'relationships',
    label: 'Analyze Project Relationships',
    description: 'Link projects to technologies and architectural patterns',
    icon: Share2,
  },
  {
    id: 'rag',
    label: 'Index Grounded Claims',
    description: 'Prepare verifiable claims for adaptive interview question generation',
    icon: FileCheck,
  },
];

export default function ResumeUploadModal({ isOpen, onClose, onUploadSuccess, onViewResume }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileError, setFileError] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadPercent, setUploadPercent] = useState(0);

  const [resumeId, setResumeId] = useState(null);

  const [stageStates, setStageStates] = useState({
    upload: { status: 'idle', result: null, error: null },
    entities: { status: 'idle', result: null, error: null },
    relationships: { status: 'idle', result: null, error: null },
    rag: { status: 'idle', result: null, error: null },
  });

  const fileInputRef = useRef(null);

  if (!isOpen) return null;

  const resetModalState = () => {
    setSelectedFile(null);
    setFileError('');
    setIsProcessing(false);
    setUploadPercent(0);
    setResumeId(null);
    setStageStates({
      upload: { status: 'idle', result: null, error: null },
      entities: { status: 'idle', result: null, error: null },
      relationships: { status: 'idle', result: null, error: null },
      rag: { status: 'idle', result: null, error: null },
    });
  };

  const handleClose = () => {
    if (isProcessing) return;
    resetModalState();
    onClose();
  };

  const validateAndSetFile = (file) => {
    setFileError('');
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf') && file.type !== 'application/pdf') {
      setFileError('Invalid file type. Please select a PDF file (.pdf).');
      return;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      setFileError('File size exceeds the 10MB limit. Please select a smaller PDF.');
      return;
    }

    if (file.size === 0) {
      setFileError('The selected file is empty.');
      return;
    }

    setSelectedFile(file);
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    validateAndSetFile(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    validateAndSetFile(file);
  };

  const runPipeline = async (startStage = 'upload') => {
    if (!selectedFile && startStage === 'upload') return;
    setIsProcessing(true);

    let currentResumeId = resumeId;

    try {
      // Stage 1: Upload
      if (startStage === 'upload') {
        setStageStates((prev) => ({
          ...prev,
          upload: { status: 'running', result: null, error: null },
        }));

        try {
          const uploadRes = await uploadResume(selectedFile, (percent) => {
            setUploadPercent(percent);
          });
          currentResumeId = uploadRes.id;
          setResumeId(currentResumeId);

          setStageStates((prev) => ({
            ...prev,
            upload: {
              status: 'completed',
              result: `${uploadRes.character_count?.toLocaleString()} characters parsed`,
              error: null,
            },
          }));
        } catch (err) {
          setStageStates((prev) => ({
            ...prev,
            upload: { status: 'failed', result: null, error: err.message },
          }));
          setIsProcessing(false);
          return;
        }
      }

      // Stage 2: Entity Extraction
      if (startStage === 'upload' || startStage === 'entities') {
        setStageStates((prev) => ({
          ...prev,
          entities: { status: 'running', result: null, error: null },
        }));

        try {
          const entitiesRes = await extractEntities(currentResumeId);
          setStageStates((prev) => ({
            ...prev,
            entities: {
              status: 'completed',
              result: `${entitiesRes.total_extracted} technical entities extracted`,
              error: null,
            },
          }));
        } catch (err) {
          setStageStates((prev) => ({
            ...prev,
            entities: { status: 'failed', result: null, error: err.message },
          }));
          setIsProcessing(false);
          return;
        }
      }

      // Stage 3: Relationship Extraction
      if (startStage === 'upload' || startStage === 'entities' || startStage === 'relationships') {
        setStageStates((prev) => ({
          ...prev,
          relationships: { status: 'running', result: null, error: null },
        }));

        try {
          const relsRes = await extractRelationships(currentResumeId, 'advanced');
          setStageStates((prev) => ({
            ...prev,
            relationships: {
              status: 'completed',
              result: `${relsRes.total_relationships} relationships verified`,
              error: null,
            },
          }));
        } catch (err) {
          setStageStates((prev) => ({
            ...prev,
            relationships: { status: 'failed', result: null, error: err.message },
          }));
          setIsProcessing(false);
          return;
        }
      }

      // Stage 4: RAG Indexing & Claims
      if (
        startStage === 'upload' ||
        startStage === 'entities' ||
        startStage === 'relationships' ||
        startStage === 'rag'
      ) {
        setStageStates((prev) => ({
          ...prev,
          rag: { status: 'running', result: null, error: null },
        }));

        try {
          const ragRes = await indexResume(currentResumeId);
          setStageStates((prev) => ({
            ...prev,
            rag: {
              status: 'completed',
              result: `${ragRes.total_chunks} chunks indexed, ${ragRes.total_claims} verified claims`,
              error: null,
            },
          }));

          if (onUploadSuccess) {
            onUploadSuccess(currentResumeId);
          }
        } catch (err) {
          setStageStates((prev) => ({
            ...prev,
            rag: { status: 'failed', result: null, error: err.message },
          }));
          setIsProcessing(false);
          return;
        }
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const isPipelineComplete =
    stageStates.upload.status === 'completed' &&
    stageStates.entities.status === 'completed' &&
    stageStates.relationships.status === 'completed' &&
    stageStates.rag.status === 'completed';

  const formatFileSize = (bytes) => {
    if (!bytes) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="upload-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 dark:bg-black/75 backdrop-blur-xs animate-fadeIn"
    >
      <div className="bg-white dark:bg-[#121620] border border-slate-200 dark:border-[#222838] rounded-3xl max-w-xl w-full shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Modal Header */}
        <div className="px-6 py-5 border-b border-slate-100 dark:border-[#1E2533] flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-indigo-50 dark:bg-indigo-500/15 text-indigo-600 dark:text-indigo-400 flex items-center justify-center flex-shrink-0">
              <UploadCloud className="w-5 h-5" />
            </div>
            <div>
              <h2 id="upload-modal-title" className="text-base font-bold text-slate-900 dark:text-white">
                Upload & Process Resume
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Extract technical skills, analyze project claims, and prepare mock interviews
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={handleClose}
            disabled={isProcessing}
            aria-label="Close upload modal"
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-[#1C2230] transition-colors disabled:opacity-40 cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-5">
          {/* Privacy Notice */}
          {!isProcessing && !resumeId && (
            <div className="flex items-center gap-2.5 p-3 rounded-xl bg-emerald-50/80 dark:bg-emerald-500/10 border border-emerald-200/80 dark:border-emerald-500/20 text-emerald-900 dark:text-emerald-200 text-xs">
              <ShieldCheck className="w-4 h-4 flex-shrink-0 text-emerald-600 dark:text-emerald-400" />
              <span>
                <strong>Privacy Sanitization Protected:</strong> Contact details (email, phone, LinkedIn) and student IDs are automatically sanitized upon upload to guarantee zero PII leakage while preserving all technical skills and projects.
              </span>
            </div>
          )}

          {/* File Picker */}
          {!isProcessing && !resumeId && (
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all duration-150 ${
                isDragging
                  ? 'border-indigo-600 bg-indigo-50/60 dark:bg-indigo-500/15'
                  : 'border-slate-300 dark:border-[#283244] hover:border-indigo-500 dark:hover:border-indigo-400 bg-slate-50/60 dark:bg-[#0D1017]'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,application/pdf"
                onChange={handleFileChange}
                className="hidden"
                id="resume-pdf-input"
              />
              <UploadCloud className="w-10 h-10 mx-auto text-indigo-600 dark:text-indigo-400 mb-2" />
              <p className="text-sm font-bold text-slate-900 dark:text-white">
                Click to browse or drag and drop your PDF resume
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                Standard PDF format • Maximum file size: 10 MB
              </p>
            </div>
          )}

          {/* File Error */}
          {fileError && (
            <div className="p-3 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30 rounded-xl text-xs text-rose-800 dark:text-rose-300 flex items-center gap-2">
              <AlertCircle className="w-4 h-4 flex-shrink-0 text-rose-600 dark:text-rose-400" />
              <span>{fileError}</span>
            </div>
          )}

          {/* Selected File Card */}
          {selectedFile && (
            <div className="bg-slate-50 dark:bg-[#0D1017] border border-slate-200/80 dark:border-[#202737] rounded-xl p-4 flex items-center justify-between">
              <div className="flex items-center gap-3 truncate">
                <div className="w-9 h-9 rounded-lg bg-indigo-50 dark:bg-indigo-500/15 text-indigo-600 dark:text-indigo-400 flex items-center justify-center flex-shrink-0">
                  <FileText className="w-5 h-5" />
                </div>
                <div className="truncate">
                  <p className="text-sm font-bold text-slate-900 dark:text-white truncate">
                    {selectedFile.name}
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400 font-mono">
                    {formatFileSize(selectedFile.size)}
                  </p>
                </div>
              </div>

              {!isProcessing && !resumeId && (
                <button
                  type="button"
                  onClick={() => setSelectedFile(null)}
                  className="text-xs text-slate-500 hover:text-rose-600 dark:hover:text-rose-400 font-bold transition-colors p-1 cursor-pointer"
                >
                  Change
                </button>
              )}
            </div>
          )}

          {/* Pipeline Execution Stages */}
          {(isProcessing || resumeId) && (
            <div className="space-y-3 pt-1">
              <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
                <span className="font-bold text-slate-900 dark:text-white uppercase tracking-wider">
                  Analysis Pipeline Stages
                </span>
                {isProcessing && (
                  <span className="flex items-center gap-1.5 text-indigo-600 dark:text-indigo-400 font-bold">
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    Processing resume...
                  </span>
                )}
              </div>

              <div className="space-y-2">
                {PIPELINE_STAGES.map((stage) => {
                  const state = stageStates[stage.id];
                  const StageIcon = stage.icon;

                  let statusIcon = (
                    <div className="w-5 h-5 rounded-full border border-slate-300 dark:border-[#2E374D] bg-white dark:bg-[#151923] flex items-center justify-center text-[10px] text-slate-400">
                      •
                    </div>
                  );
                  let borderColor = 'border-slate-200/80 dark:border-[#202737]';
                  let bgColor = 'bg-slate-50/60 dark:bg-[#0D1017]';

                  if (state.status === 'running') {
                    statusIcon = (
                      <Loader2 className="w-4 h-4 text-indigo-600 dark:text-indigo-400 animate-spin flex-shrink-0" />
                    );
                    borderColor = 'border-indigo-300 dark:border-indigo-500/40';
                    bgColor = 'bg-indigo-50/60 dark:bg-indigo-500/10';
                  } else if (state.status === 'completed') {
                    statusIcon = (
                      <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
                    );
                    borderColor = 'border-emerald-200 dark:border-emerald-500/30';
                    bgColor = 'bg-emerald-50/60 dark:bg-emerald-500/10';
                  } else if (state.status === 'failed') {
                    statusIcon = (
                      <AlertCircle className="w-4 h-4 text-rose-600 dark:text-rose-400 flex-shrink-0" />
                    );
                    borderColor = 'border-rose-200 dark:border-rose-500/30';
                    bgColor = 'bg-rose-50/70 dark:bg-rose-500/10';
                  }

                  return (
                    <div
                      key={stage.id}
                      className={`border ${borderColor} ${bgColor} rounded-xl p-3 flex items-start justify-between gap-3 transition-colors`}
                    >
                      <div className="flex items-start gap-3">
                        <div className="mt-0.5">{statusIcon}</div>
                        <div>
                          <div className="flex items-center gap-2">
                            <StageIcon className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" />
                            <span className="text-xs font-bold text-slate-900 dark:text-white">
                              {stage.label}
                            </span>
                            {stage.id === 'upload' && state.status === 'running' && uploadPercent > 0 && (
                              <span className="text-[10px] font-mono text-indigo-600 dark:text-indigo-400">
                                ({uploadPercent}%)
                              </span>
                            )}
                            {state.status === 'completed' && (
                              <Badge variant="emerald" size="sm">
                                Complete
                              </Badge>
                            )}
                            {state.status === 'failed' && (
                              <Badge variant="rose" size="sm">
                                Failed
                              </Badge>
                            )}
                          </div>
                          <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                            {stage.description}
                          </p>

                          {state.result && (
                            <p className="text-xs text-emerald-700 dark:text-emerald-300 font-semibold mt-1">
                              ✓ {state.result}
                            </p>
                          )}

                          {state.error && (
                            <p className="text-xs text-rose-700 dark:text-rose-300 font-semibold mt-1">
                              ✗ {state.error}
                            </p>
                          )}
                        </div>
                      </div>

                      {state.status === 'failed' && !isProcessing && (
                        <button
                          type="button"
                          onClick={() => runPipeline(stage.id)}
                          className="px-2.5 py-1 rounded-lg border border-rose-300 dark:border-rose-500/40 bg-rose-100 dark:bg-rose-500/20 text-rose-800 dark:text-rose-300 hover:bg-rose-200 text-xs font-semibold flex items-center gap-1 transition-colors flex-shrink-0 cursor-pointer"
                        >
                          <RotateCcw className="w-3 h-3" />
                          <span>Retry Stage</span>
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-4 border-t border-slate-100 dark:border-[#1E2533] bg-slate-50 dark:bg-[#0D1017] flex items-center justify-between gap-3">
          <Button
            variant="outline"
            size="sm"
            disabled={isProcessing}
            onClick={handleClose}
          >
            {isPipelineComplete ? 'Close' : 'Cancel'}
          </Button>

          {!resumeId && (
            <Button
              variant="primary"
              size="sm"
              disabled={!selectedFile || isProcessing || !!fileError}
              isLoading={isProcessing}
              iconLeft={Sparkles}
              onClick={() => runPipeline('upload')}
            >
              Process Resume
            </Button>
          )}

          {isPipelineComplete && (
            <Button
              variant="success"
              size="sm"
              iconRight={ArrowRight}
              onClick={() => {
                handleClose();
                if (onViewResume && resumeId) {
                  onViewResume(resumeId);
                }
              }}
            >
              View Extracted Details
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
