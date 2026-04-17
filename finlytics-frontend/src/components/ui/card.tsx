import { cn } from "@/utils/cn";
import { type HTMLAttributes } from "react";

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-white/12 bg-gradient-to-br from-slate-900/78 via-slate-900/66 to-slate-950/74 p-5 shadow-premium-soft backdrop-blur-xl transition duration-200 hover:-translate-y-0.5 hover:shadow-premium-lg",
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
