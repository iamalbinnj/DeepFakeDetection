"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { useDropzone } from "react-dropzone";
import { uploadVideo } from "@/api";
import { PredictResponse } from "@/type/index.p";
import { VideoValidator } from "@/utils/VideoValidator";

// Initialize validator with custom config
const videoValidator = new VideoValidator({
  maxSizeInMB: 500,
  maxDurationInSeconds: 600,
  allowedFormats: ["mp4", "mov", "webm", "avi"],
  minWidth: 320,
  minHeight: 240,
});

export default function VideoUploader() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [validating, setValidating] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // react-dropzone setup
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const f = acceptedFiles[0];
    if (!f) return;

    setError(null);
    setValidating(true);

    try {
      // Validate the video file
      const validationResult = await videoValidator.validate(f);

      if (!validationResult.valid) {
        setError(validationResult.error || "Invalid video file.");
        setFile(null);
        setValidating(false);
        return;
      }

      setFile(f);
    } catch (err: any) {
      console.error("Validation error:", err);
      setError("Failed to validate video file.");
      setFile(null);
    } finally {
      setValidating(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    multiple: false,
    noClick: true,
    accept: { "video/*": [] },
    disabled: validating || loading,
  });

  // generate preview whenever file changes
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // upload and call predict
  async function handleAnalyze() {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const data = await uploadVideo(file);
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err?.message ?? "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  const isProcessing = validating || loading;

  return (
    <div className="max-w-3xl mx-auto p-6 bg-white rounded-lg shadow">
      <div
        {...getRootProps()}
        className={
          "border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition " +
          (isProcessing
            ? "border-gray-300 bg-gray-100 opacity-60 cursor-not-allowed"
            : isDragActive
              ? "border-blue-400 bg-blue-50"
              : "border-gray-300 bg-gray-50")
        }
      >
        <input {...getInputProps()} />
        <p className="text-gray-700">
          {validating
            ? "Validating video..."
            : isDragActive
              ? "Drop the file here ..."
              : "Drag & drop your video here"}
        </p>
        <p className="text-sm text-gray-500 mt-2">
          Or{" "}
          <button
            type="button"
            onClick={open}
            disabled={isProcessing}
            className={
              isProcessing
                ? "text-gray-400 cursor-not-allowed"
                : "text-blue-600 underline hover:text-blue-800"
            }
          >
            choose a file
          </button>
        </p>
        <p className="text-xs text-gray-400 mt-2">
          Accepted: mp4, mov, webm, avi — max 500MB, 10 minutes
        </p>
      </div>

      {previewUrl && (
        <div className="mt-6">
          <video
            ref={videoRef}
            src={previewUrl}
            controls
            className="w-full max-h-[400px] rounded-md bg-black"
          />
        </div>
      )}

      <div className="mt-4 flex gap-3">
        <button
          type="button"
          onClick={() => {
            setFile(null);
            setResult(null);
            setError(null);
            open();
          }}
          disabled={isProcessing}
          className={
            "px-4 py-2 rounded " +
            (isProcessing
              ? "bg-gray-200 text-gray-400 cursor-not-allowed"
              : "bg-gray-200 hover:bg-gray-300")
          }
        >
          Choose another file
        </button>

        <button
          type="button"
          onClick={handleAnalyze}
          disabled={!file || isProcessing}
          className={
            "ml-auto px-4 py-2 rounded text-white " +
            (file && !isProcessing
              ? "bg-blue-600 hover:bg-blue-700"
              : "bg-gray-400 cursor-not-allowed")
          }
        >
          {loading ? "Analyzing…" : "Analyze Video"}
        </button>
      </div>

      {validating && (
        <div className="mt-4 flex items-center justify-center gap-2 text-blue-600">
          <div className="w-5 h-5 border-2 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
          <span className="text-sm">Validating video...</span>
        </div>
      )}

      {loading && (
        <div className="mt-4 flex justify-center">
          <div className="w-12 h-12 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin" />
        </div>
      )}

      {error && (
        <div className="mt-4 p-3 rounded bg-red-50 text-red-700">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className="mt-4 p-4 rounded bg-gray-50 border">
          <h3 className="font-semibold mb-2">Analysis Result</h3>
          <pre className="whitespace-pre-wrap text-sm text-gray-800">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
