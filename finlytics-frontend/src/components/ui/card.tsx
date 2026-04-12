import { cn } from "@/utils/cn";
import { type HTMLAttributes } from "react";

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-white/10 bg-slate-900/50 p-5 backdrop-blur-xl shadow-glass",
        className
      )}
      {...props}
    />
  );
}

export function CardTitle({ className, ...props }: HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={cn("text-sm font-medium text-slate-300 tracking-wide uppercase", className)}
      {...props}
    />
  );
}

export function CardValue({ className, ...props }: HTMLAttributes<HTMLParagraphElement>) {
  return <p className={cn("text-3xl font-bold text-white", className)} {...props} />;
}
