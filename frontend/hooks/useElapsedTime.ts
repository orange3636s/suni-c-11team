"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type UseElapsedTimeOptions = {
  running: boolean;
  resetKey?: string | number | null;
};

export type UseElapsedTimeResult = {
  elapsedSeconds: number;
  formattedElapsed: string;
  reset: () => void;
};

export function formatElapsedTime(totalSeconds: number): string {
  const safeSeconds = Math.max(0, Math.floor(totalSeconds));
  const minutes = Math.floor(safeSeconds / 60);
  const seconds = safeSeconds % 60;

  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

export default function useElapsedTime({
  running,
  resetKey = null,
}: UseElapsedTimeOptions): UseElapsedTimeResult {
  const [elapsed, setElapsed] = useState({
    resetKey,
    seconds: 0,
  });
  const startedAtRef = useRef<number | null>(null);

  const reset = useCallback(() => {
    startedAtRef.current = running ? Date.now() : null;
    setElapsed({ resetKey, seconds: 0 });
  }, [resetKey, running]);

  useEffect(() => {
    startedAtRef.current = null;

    if (!running) return;

    startedAtRef.current = Date.now();
    const intervalId = window.setInterval(() => {
      if (startedAtRef.current === null) return;
      setElapsed({
        resetKey,
        seconds: Math.floor((Date.now() - startedAtRef.current) / 1000),
      });
    }, 1000);

    return () => {
      window.clearInterval(intervalId);
      startedAtRef.current = null;
    };
  }, [running, resetKey]);

  const elapsedSeconds =
    running && elapsed.resetKey === resetKey ? elapsed.seconds : 0;

  return {
    elapsedSeconds,
    formattedElapsed: formatElapsedTime(elapsedSeconds),
    reset,
  };
}
