export interface VideoValidationResult {
  valid: boolean;
  error?: string;
}

export interface VideoValidationConfig {
  maxSizeInMB?: number;
  maxDurationInSeconds?: number;
  allowedFormats?: string[];
  minWidth?: number;
  minHeight?: number;
  maxWidth?: number;
  maxHeight?: number;
}

const DEFAULT_CONFIG: VideoValidationConfig = {
  maxSizeInMB: 500,
  maxDurationInSeconds: 600, // 10 minutes
  allowedFormats: ["mp4", "mov", "webm", "avi", "mkv"],
  minWidth: 320,
  minHeight: 240,
  maxWidth: 7680, // 8K
  maxHeight: 4320,
};

export class VideoValidator {
  private config: VideoValidationConfig;

  constructor(config: Partial<VideoValidationConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  validateFileBasics(file: File): VideoValidationResult {
    // Check if it's a video file
    if (!file.type.startsWith("video/")) {
      return {
        valid: false,
        error: "Please select a valid video file.",
      };
    }

    // Check file extension
    const extension = file.name.split(".").pop()?.toLowerCase();
    if (
      extension &&
      this.config.allowedFormats &&
      !this.config.allowedFormats.includes(extension)
    ) {
      return {
        valid: false,
        error: `Unsupported format. Allowed: ${this.config.allowedFormats.join(", ")}`,
      };
    }

    // Check file size
    const maxSizeInBytes = (this.config.maxSizeInMB || 500) * 1024 * 1024;
    if (file.size > maxSizeInBytes) {
      return {
        valid: false,
        error: `File size exceeds ${this.config.maxSizeInMB}MB limit.`,
      };
    }

    return { valid: true };
  }

  async validateVideoProperties(file: File): Promise<VideoValidationResult> {
    return new Promise((resolve) => {
      const video = document.createElement("video");
      const url = URL.createObjectURL(file);

      video.preload = "metadata";

      video.onloadedmetadata = () => {
        URL.revokeObjectURL(url);

        // Check duration
        if (
          this.config.maxDurationInSeconds &&
          video.duration > this.config.maxDurationInSeconds
        ) {
          resolve({
            valid: false,
            error: `Video duration exceeds ${this.config.maxDurationInSeconds}s limit.`,
          });
          return;
        }

        // Check resolution
        const width = video.videoWidth;
        const height = video.videoHeight;

        if (this.config.minWidth && width < this.config.minWidth) {
          resolve({
            valid: false,
            error: `Video width (${width}px) is below minimum ${this.config.minWidth}px.`,
          });
          return;
        }

        if (this.config.minHeight && height < this.config.minHeight) {
          resolve({
            valid: false,
            error: `Video height (${height}px) is below minimum ${this.config.minHeight}px.`,
          });
          return;
        }

        if (this.config.maxWidth && width > this.config.maxWidth) {
          resolve({
            valid: false,
            error: `Video width (${width}px) exceeds maximum ${this.config.maxWidth}px.`,
          });
          return;
        }

        if (this.config.maxHeight && height > this.config.maxHeight) {
          resolve({
            valid: false,
            error: `Video height (${height}px) exceeds maximum ${this.config.maxHeight}px.`,
          });
          return;
        }

        resolve({ valid: true });
      };

      video.onerror = () => {
        URL.revokeObjectURL(url);
        resolve({
          valid: false,
          error: "Failed to load video metadata. The file may be corrupted.",
        });
      };

      video.src = url;
    });
  }

  async validate(file: File): Promise<VideoValidationResult> {
    // First check basics
    const basicsResult = this.validateFileBasics(file);
    if (!basicsResult.valid) {
      return basicsResult;
    }

    // Then check video properties
    return this.validateVideoProperties(file);
  }
}
