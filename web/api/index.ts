import { PredictResponse } from "@/type/index.p";

const API_URL = process.env.NEXT_PUBLIC_API_URL;
const API_VERSION = process.env.NEXT_PUBLIC_API_VERSION;
console.log(API_VERSION);


export async function uploadVideo(file: File): Promise<PredictResponse> {
  const form = new FormData();
  form.append("video", file);

  const res = await fetch(`${API_URL}/api/${API_VERSION}/predict`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Server error: ${res.status} ${text}`);
  }

  try {
    const data: PredictResponse = await res.json();
    return data;
  } catch {
    throw new Error("Invalid JSON response from server");
  }
}

export async function getServerStatus() {
  try {
    const response = await fetch(`${API_URL}/api/${API_VERSION}/health`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
      next: { revalidate: 60 },
      cache: "no-store",
    });

    if (response.ok) {
      const data = await response.json();

      return {
        isOnline: true,
        modelVersion: data.modelVersion ?? "Unknown",
        lastChecked: new Date().toISOString(),
      };
    }

    return {
      isOnline: false,
      modelVersion: "Unknown",
      lastChecked: new Date().toISOString(),
    };
  } catch (error) {
    return {
      isOnline: false,
      modelVersion: "Unknown",
      lastChecked: new Date().toISOString(),
    };
  }
}