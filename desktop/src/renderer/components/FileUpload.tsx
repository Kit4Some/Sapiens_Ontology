/**
 * File Upload Component for SDDI Ingestion
 *
 * Supports drag & drop and file selection for:
 * - Plain text (.txt)
 * - Markdown (.md)
 * - CSV (.csv)
 * - PDF (.pdf)
 * - JSON (.json)
 * - JSON-LD (.jsonld)
 * - XML (.xml, .rdf, .owl)
 * - YAML (.yaml, .yml)
 * - HTML (.html, .htm)
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import {
  Upload,
  X,
  FileText,
  FileType,
  Table,
  File,
  CheckCircle,
  AlertCircle,
  Loader2,
  Trash2,
  FileJson,
  FileCode,
  Globe,
  Activity,
} from 'lucide-react';
import { clsx } from 'clsx';
import { apiClient, IngestionStreamEvent } from '../api/client';
import type { IngestionStatusResponse } from '../types';
import { ProcessingDashboard } from './ProcessingDashboard';

// Processing step type for dashboard
interface ProcessingStep {
  id: string;
  name: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
  details?: Record<string, unknown>;
  startTime?: number;
  endTime?: number;
}

const SUPPORTED_EXTENSIONS = [
  // Text formats
  '.txt', '.text', '.md', '.markdown',
  // Data formats
  '.csv', '.pdf',
  // Structured data
  '.json', '.jsonld', '.json-ld',
  // XML/RDF
  '.xml', '.rdf', '.owl',
  // YAML
  '.yaml', '.yml',
  // HTML
  '.html', '.htm', '.xhtml',
];

interface FileWithPreview extends File {
  preview?: string;
}

function getFileIcon(filename: string) {
  const ext = filename.split('.').pop()?.toLowerCase();
  switch (ext) {
    case 'pdf':
      return FileType;
    case 'csv':
      return Table;
    case 'md':
    case 'markdown':
      return FileText;
    case 'json':
    case 'jsonld':
    case 'json-ld':
      return FileJson;
    case 'xml':
    case 'rdf':
    case 'owl':
    case 'yaml':
    case 'yml':
      return FileCode;
    case 'html':
    case 'htm':
    case 'xhtml':
      return Globe;
    default:
      return File;
  }
}

// Initial processing steps
const INITIAL_STEPS: ProcessingStep[] = [
  { id: 'started', name: 'Initializing', status: 'pending', progress: 0, message: 'Waiting to start...' },
  { id: 'ingest', name: 'Validating Files', status: 'pending', progress: 0, message: 'Validating documents...' },
  { id: 'chunk', name: 'Chunking', status: 'pending', progress: 0, message: 'Splitting into chunks...' },
  { id: 'extract_entities', name: 'Entity Extraction', status: 'pending', progress: 0, message: 'Extracting entities...' },
  { id: 'extract_relations', name: 'Relation Extraction', status: 'pending', progress: 0, message: 'Finding relationships...' },
  { id: 'embed', name: 'Embedding', status: 'pending', progress: 0, message: 'Generating embeddings...' },
  { id: 'load', name: 'Loading to Neo4j', status: 'pending', progress: 0, message: 'Saving to database...' },
];

export function FileUpload() {
  const [files, setFiles] = useState<FileWithPreview[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<IngestionStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Dashboard state
  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>(INITIAL_STEPS);
  const [currentStep, setCurrentStep] = useState<string>('');
  const [stats, setStats] = useState({
    entitiesCount: 0,
    relationsCount: 0,
    chunksCount: 0,
  });

  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamingRef = useRef<boolean>(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Stream job progress via SSE
  useEffect(() => {
    if (!jobId || streamingRef.current) return;

    streamingRef.current = true;
    setIsProcessing(true);
    abortControllerRef.current = new AbortController();

    const streamProgress = async () => {
      try {
        for await (const event of apiClient.streamIngestionProgress(
          jobId,
          abortControllerRef.current?.signal
        )) {
          handleStreamEvent(event);

          if (event.event === 'done' || event.event === 'error') {
            break;
          }
        }
      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          setError(err instanceof Error ? err.message : 'Failed to get status');
        }
      } finally {
        streamingRef.current = false;
        setIsProcessing(false);
      }
    };

    streamProgress();

    return () => {
      abortControllerRef.current?.abort();
    };
  }, [jobId]);

  // Handle stream events
  const handleStreamEvent = useCallback((event: IngestionStreamEvent) => {
    const { data } = event;

    if (event.event === 'status' || event.event === 'progress') {
      const stepId = data.step || 'started';
      const progress = data.progress ?? 0;

      setCurrentStep(stepId);

      // Update step status
      setProcessingSteps((prev) =>
        prev.map((step) => {
          if (step.id === stepId) {
            return {
              ...step,
              status: 'processing',
              progress,
              message: data.message || step.message,
              details: data.details,
              startTime: step.startTime || Date.now(),
            };
          }
          // Mark previous steps as completed
          const stepIndex = prev.findIndex((s) => s.id === stepId);
          const currentIndex = prev.findIndex((s) => s.id === step.id);
          if (currentIndex < stepIndex && step.status !== 'completed') {
            return {
              ...step,
              status: 'completed',
              progress: 1,
              endTime: Date.now(),
            };
          }
          return step;
        })
      );

      // Update stats - extract from root level or details fallback
      const details = (data.details || {}) as Record<string, unknown>;
      const entitiesCount = (data.entities_created ?? details.entities_so_far ?? stats.entitiesCount) as number;
      const relationsCount = (data.relations_created ?? details.relations_so_far ?? stats.relationsCount) as number;
      const chunksCount = (data.chunks_created ?? details.chunks_processed ?? stats.chunksCount) as number;

      setStats({
        entitiesCount,
        relationsCount,
        chunksCount,
      });

      // Update job status for legacy progress display
      setJobStatus((prev) => ({
        ...prev!,
        progress,
        message: data.message || '',
        entities_created: entitiesCount,
        relations_created: relationsCount,
        chunks_created: chunksCount,
      }));
    }

    if (event.event === 'done') {
      // Mark all steps as completed
      setProcessingSteps((prev) =>
        prev.map((step) => ({
          ...step,
          status: 'completed',
          progress: 1,
          endTime: step.endTime || Date.now(),
        }))
      );
      setJobStatus((prev) => prev ? { ...prev, status: 'completed' } : null);
    }

    if (event.event === 'error') {
      setError(data.message || 'Unknown error');
      setProcessingSteps((prev) =>
        prev.map((step) =>
          step.id === currentStep
            ? { ...step, status: 'failed' }
            : step
        )
      );
      setJobStatus((prev) => prev ? { ...prev, status: 'failed' } : null);
    }
  }, [currentStep, stats]);

  const validateFile = useCallback((file: File): string | null => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!SUPPORTED_EXTENSIONS.includes(ext)) {
      return `Unsupported file type: ${ext}`;
    }
    // 1GB max file size
    if (file.size > 1024 * 1024 * 1024) {
      return 'File too large (max 1GB)';
    }
    return null;
  }, []);

  const handleFiles = useCallback(
    (newFiles: FileList | File[]) => {
      const validFiles: FileWithPreview[] = [];
      const errors: string[] = [];

      Array.from(newFiles).forEach((file) => {
        const error = validateFile(file);
        if (error) {
          errors.push(`${file.name}: ${error}`);
        } else if (!files.some((f) => f.name === file.name && f.size === file.size)) {
          validFiles.push(file);
        }
      });

      if (errors.length > 0) {
        setError(errors.join('\n'));
      }

      if (validFiles.length > 0) {
        setFiles((prev) => [...prev, ...validFiles]);
        setError(null);
      }
    },
    [files, validateFile]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files) {
        handleFiles(e.target.files);
      }
    },
    [handleFiles]
  );

  const removeFile = useCallback((index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const clearAll = useCallback(() => {
    // Abort any ongoing streaming
    abortControllerRef.current?.abort();
    streamingRef.current = false;

    setFiles([]);
    setJobId(null);
    setJobStatus(null);
    setError(null);
    setUploadProgress(0);
    setIsProcessing(false);
    setProcessingSteps(INITIAL_STEPS);
    setCurrentStep('');
    setStats({ entitiesCount: 0, relationsCount: 0, chunksCount: 0 });
  }, []);

  const handleUpload = useCallback(async () => {
    if (files.length === 0) return;

    setIsUploading(true);
    setError(null);
    setUploadProgress(0);

    try {
      const response = await apiClient.uploadFiles(files, (progress) => {
        setUploadProgress(progress * 0.5); // Upload is 50% of total progress
      });

      setJobId(response.job_id);
      setUploadProgress(0.5);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  }, [files]);

  const getStatusColor = () => {
    if (!jobStatus) return 'text-dark-400';
    switch (jobStatus.status) {
      case 'completed':
        return 'text-green-400';
      case 'failed':
        return 'text-red-400';
      case 'processing':
        return 'text-blue-400';
      default:
        return 'text-yellow-400';
    }
  };

  const totalProgress = jobStatus
    ? 0.5 + jobStatus.progress * 0.5
    : uploadProgress;

  return (
    <div className="space-y-4">
      {/* Drop Zone */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => fileInputRef.current?.click()}
        className={clsx(
          'relative border-2 border-dashed rounded-xl p-8 transition-all cursor-pointer',
          'flex flex-col items-center justify-center gap-3',
          isDragging
            ? 'border-primary-400 bg-primary-500/10'
            : 'border-dark-600 hover:border-dark-500 hover:bg-dark-800/50',
          (isUploading || jobStatus?.status === 'processing') && 'pointer-events-none opacity-50'
        )}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={SUPPORTED_EXTENSIONS.join(',')}
          onChange={handleFileInput}
          className="hidden"
        />

        <div className="p-3 bg-dark-700 rounded-full">
          <Upload className="w-6 h-6 text-dark-400" />
        </div>

        <div className="text-center">
          <p className="text-dark-200 font-medium">
            Drop files here or click to browse
          </p>
          <p className="text-sm text-dark-400 mt-1">
            TXT, MD, CSV, PDF, JSON, JSON-LD, XML, YAML, HTML (max 1GB)
          </p>
        </div>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="bg-dark-800 rounded-xl p-4 space-y-2">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-dark-200">
              {files.length} file(s) selected
            </span>
            <button
              onClick={clearAll}
              className="text-xs text-dark-400 hover:text-dark-200 flex items-center gap-1"
            >
              <Trash2 className="w-3 h-3" />
              Clear all
            </button>
          </div>

          <div className="max-h-48 overflow-y-auto space-y-2">
            {files.map((file, index) => {
              const FileIcon = getFileIcon(file.name);
              return (
                <div
                  key={`${file.name}-${index}`}
                  className="flex items-center justify-between p-2 bg-dark-700 rounded-lg"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <FileIcon className="w-4 h-4 text-dark-400 flex-shrink-0" />
                    <span className="text-sm text-dark-200 truncate">
                      {file.name}
                    </span>
                    <span className="text-xs text-dark-500">
                      ({(file.size / 1024).toFixed(1)} KB)
                    </span>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      removeFile(index);
                    }}
                    className="p-1 text-dark-400 hover:text-red-400"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Processing Dashboard */}
      {(isUploading || isProcessing || jobStatus) && (
        <ProcessingDashboard
          type="ingestion"
          isActive={isUploading || isProcessing}
          steps={processingSteps}
          currentStep={currentStep}
          overallProgress={totalProgress}
          stats={stats}
          error={error || undefined}
        />
      )}

      {/* Completed/Failed Status */}
      {jobStatus?.status === 'completed' && (
        <div className="flex items-center gap-2 p-3 bg-green-500/10 rounded-lg">
          <CheckCircle className="w-5 h-5 text-green-400" />
          <span className="text-sm text-green-400">
            Ingestion completed successfully! {stats.entitiesCount} entities, {stats.relationsCount} relations, {stats.chunksCount} chunks created.
          </span>
        </div>
      )}

      {jobStatus?.status === 'failed' && (
        <div className="p-3 bg-red-500/10 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <span className="text-sm text-red-400">Ingestion failed</span>
          </div>
          {jobStatus.errors && jobStatus.errors.length > 0 && (
            <ul className="text-xs text-red-300 list-disc list-inside">
              {jobStatus.errors.map((err, i) => (
                <li key={i}>{err}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Error Display */}
      {error && !jobStatus && (
        <div className="p-3 bg-red-900/20 border border-red-800 rounded-lg">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {/* Upload Button */}
      <button
        onClick={handleUpload}
        disabled={files.length === 0 || isUploading || jobStatus?.status === 'processing'}
        className={clsx(
          'w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl font-medium transition-colors',
          files.length > 0 && !isUploading && jobStatus?.status !== 'processing'
            ? 'bg-primary-600 hover:bg-primary-500 text-white'
            : 'bg-dark-700 text-dark-500 cursor-not-allowed'
        )}
      >
        {isUploading || jobStatus?.status === 'processing' ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Processing...
          </>
        ) : (
          <>
            <Upload className="w-5 h-5" />
            Upload & Process
          </>
        )}
      </button>

      {/* New Upload Button (after completion) */}
      {(jobStatus?.status === 'completed' || jobStatus?.status === 'failed') && (
        <button
          onClick={clearAll}
          className="w-full px-4 py-2 text-sm text-dark-300 hover:text-dark-100 transition-colors"
        >
          Upload more files
        </button>
      )}
    </div>
  );
}

export default FileUpload;
