import * as React from "react";

import { cn } from "@/utils/cn";

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(({ className, ...props }, ref) => {
  return (
    <input
      ref={ref}
      className={cn(
        "h-10 w-full rounded-xl border border-white/20 bg-gradient-to-r from-slate-950/80 to-slate-900/75 px-3 text-sm text-slate-100 placeholder:text-slate-500 shadow-inner-soft transition duration-200 focus-visible:border-cyan-300/45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/45",
        className
      )}
      {...props}
    />
  );
});

Input.displayName = "Input";

export { Input };
