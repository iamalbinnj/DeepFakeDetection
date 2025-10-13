"use client";
import React, { useEffect, useState } from "react";
import { getServerStatus } from "@/api/index";

type Status = {
  isOnline: boolean;
  modelVersion: string;
  lastChecked: string;
};

export default function ServerStatus() {
  const [status, setStatus] = useState<Status>({
    isOnline: false,
    modelVersion: "Unknown",
    lastChecked: "",
  });
  const [loading, setLoading] = useState(false);

  const checkServerStatus = async () => {
    setLoading(true);
    const s = await getServerStatus();
    setStatus(s);
    setLoading(false);
  };

  useEffect(() => {
    checkServerStatus();

    const interval = setInterval(() => checkServerStatus(), 30000);
    return () => clearInterval(interval);
  }, []);

  const formatTime = (iso: string) => {
    if (!iso) return "–";
    const date = new Date(iso);
    return date.toLocaleTimeString();
  };

  return (
    <div className="mt-6 p-4 bg-white rounded-lg shadow border border-gray-200">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-700">Server:</span>
          {loading ? (
            <span className="text-xs text-gray-500">Checking...</span>
          ) : (
            <>
              <div
                className={`w-2.5 h-2.5 rounded-full ${
                  status.isOnline ? "bg-green-500 animate-pulse" : "bg-red-500"
                }`}
              />
              <span
                className={`text-sm font-semibold ${
                  status.isOnline ? "text-green-600" : "text-red-600"
                }`}
              >
                {status.isOnline ? "Online" : "Offline"}
              </span>
            </>
          )}
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-700">Model:</span>
          <span className="text-sm font-mono text-gray-900 bg-gray-100 px-2 py-0.5 rounded">
            {status.modelVersion}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">
            Last checked: {formatTime(status.lastChecked)}
          </span>
          <button
            onClick={checkServerStatus}
            disabled={loading}
            className="text-xs text-blue-600 hover:text-blue-800 underline disabled:text-gray-400 disabled:no-underline cursor-pointer"
            title="Refresh status"
          >
            Refresh
          </button>
        </div>
      </div>
    </div>
  );
}
