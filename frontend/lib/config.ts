// Fallback only. The sidebar prefers the real model name from the backend /health.
export const MODEL_LABEL = process.env.NEXT_PUBLIC_MODEL_LABEL ?? "Model not configured";

// INTEGRATION(EVOLUS): link to the Evolus workspace so judges can see the created task.
export const EVOLUS_URL = process.env.NEXT_PUBLIC_EVOLUS_URL ?? "";

export const IS_MOCK = process.env.NEXT_PUBLIC_USE_MOCK === "true";

// Same limit as MAX_UPLOAD_MB in the backend
export const MAX_UPLOAD_MB = Number(process.env.NEXT_PUBLIC_MAX_UPLOAD_MB ?? 10);