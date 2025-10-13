"use client";

import VideoUploader from "@/components/VideoUploader";
import ServerStatus from "@/components/ServerStatus";

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-3xl mx-auto space-y-8">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Deepfake Detector
          </h1>
          <p className="text-gray-600">
            Upload a video to check if it’s AI-generated or authentic.
          </p>
        </div>

        <ServerStatus />

        <div className="bg-white p-6 rounded-lg shadow border border-gray-200">
          <VideoUploader />
        </div>
      </div>
    </main>
  );
}
