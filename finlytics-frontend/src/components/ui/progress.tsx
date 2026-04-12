import { cn } from "@/utils/cn";

interface ProgressProps {
  value: number;
  className?: string;
}

export function Progress({ value, className }: ProgressProps) {
  const width = Math.max(0, Math.min(100, value));

  return (
    <div className={cn("h-3 w-full rounded-full bg-white/10", className)}>
      <div
        className="h-full rounded-full bg-gradient-to-r from-blue-500 via-cyan-400 to-emerald-400 transition-all duration-500"
        style={{ width: `${width}%` }}
      />
    </div>
  );
}
