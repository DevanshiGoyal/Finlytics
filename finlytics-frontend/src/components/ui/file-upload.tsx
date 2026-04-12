"use client";

import { UploadCloud } from "lucide-react";
import { useRef } from "react";

import { Button } from "@/components/ui/button";

interface FileUploadProps {
  accept?: string;
  onFileSelect: (file: File) => void;
  title?: string;
}

export function FileUpload({
  accept = ".csv",
  onFileSelect,
  title = "Upload CSV"
}: FileUploadProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);

  return (
    <div className="rounded-2xl border border-dashed border-white/25 bg-slate-900/40 p-4">
      <div className="flex flex-col items-center justify-center gap-2 text-center">
        <UploadCloud className="h-7 w-7 text-cyan-300" />
        <p className="text-sm font-medium text-slate-200">{title}</p>
        <p className="text-xs text-slate-400">Drop file or browse from your device</p>
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          accept={accept}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) {
              onFileSelect(file);
            }
          }}
        />
        <Button variant="secondary" size="sm" onClick={() => inputRef.current?.click()}>
          Choose File
        </Button>
      </div>
    </div>
  );
}
