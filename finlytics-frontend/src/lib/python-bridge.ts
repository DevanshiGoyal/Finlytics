import "server-only";

import { existsSync } from "node:fs";
import { resolve, join } from "node:path";
import { spawn, spawnSync } from "node:child_process";

const FRONTEND_ROOT = process.cwd();
const PROJECT_ROOT = resolve(FRONTEND_ROOT, "..");
const BRIDGE_SCRIPT = join(FRONTEND_ROOT, "scripts", "model_bridge.py");
const REQUIRED_IMPORT_PROBE = "import joblib, numpy, pandas, sklearn, xgboost";

let cachedPythonExecutable: string | null = null;

function canRunBridgeWith(pythonExecutable: string): boolean {
  try {
    const probe = spawnSync(pythonExecutable, ["-c", REQUIRED_IMPORT_PROBE], {
      cwd: PROJECT_ROOT,
      env: process.env,
      stdio: "pipe"
    });
    return probe.status === 0;
  } catch {
    return false;
  }
}

function detectPythonExecutable(): string {
  if (cachedPythonExecutable) {
    return cachedPythonExecutable;
  }

  const fromEnv = process.env.FINLYTICS_PYTHON;
  const candidates = [
    fromEnv,
    join(PROJECT_ROOT, ".venv312", "Scripts", "python.exe"),
    join(PROJECT_ROOT, ".venv", "Scripts", "python.exe"),
    join(PROJECT_ROOT, ".venv312", "bin", "python"),
    join(PROJECT_ROOT, ".venv", "bin", "python"),
    "python"
  ].filter(Boolean) as string[];

  for (const candidate of candidates) {
    if (candidate !== "python" && !existsSync(candidate)) {
      continue;
    }

    if (canRunBridgeWith(candidate)) {
      cachedPythonExecutable = candidate;
      return candidate;
    }
  }

  for (const candidate of candidates) {
    if (candidate === "python" || existsSync(candidate)) {
      cachedPythonExecutable = candidate;
      return candidate;
    }
  }

  cachedPythonExecutable = "python";
  return "python";
}

export async function callPythonBridge<T>(operation: string, payload: unknown): Promise<T> {
  const python = detectPythonExecutable();

  return new Promise<T>((resolvePromise, rejectPromise) => {
    const proc = spawn(python, [BRIDGE_SCRIPT, operation], {
      cwd: PROJECT_ROOT,
      env: process.env,
      stdio: ["pipe", "pipe", "pipe"]
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });

    proc.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });

    proc.on("error", (error) => {
      rejectPromise(new Error(`Failed to start Python bridge: ${error.message}`));
    });

    proc.on("close", (code) => {
      if (code !== 0) {
        rejectPromise(
          new Error(
            `Python bridge failed using '${python}' (code ${code}). stderr: ${stderr || "<empty>"}`
          )
        );
        return;
      }

      try {
        const parsed = JSON.parse(stdout || "{}");
        resolvePromise(parsed as T);
      } catch {
        rejectPromise(
          new Error(
            `Python bridge returned invalid JSON. stdout: ${stdout || "<empty>"}; stderr: ${stderr || "<empty>"}`
          )
        );
      }
    });

    proc.stdin.write(JSON.stringify(payload ?? {}));
    proc.stdin.end();
  });
}
