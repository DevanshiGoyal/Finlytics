import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/utils/cn";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold uppercase tracking-wide",
  {
    variants: {
      variant: {
        neutral: "border-white/15 text-slate-300 bg-slate-900/70",
        success: "border-emerald-400/40 text-emerald-300 bg-emerald-500/10",
        warning: "border-amber-400/40 text-amber-300 bg-amber-500/10",
        danger: "border-red-400/40 text-red-300 bg-red-500/10",
        info: "border-cyan-400/40 text-cyan-300 bg-cyan-500/10"
      }
    },
    defaultVariants: {
      variant: "neutral"
    }
  }
);

type BadgeProps = React.HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>;

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}
