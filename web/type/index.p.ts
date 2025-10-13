export type PredictResponse = {
  Result?: string;
  Confidence?: number | string;
  [k: string]: any;
  message: string;
};
