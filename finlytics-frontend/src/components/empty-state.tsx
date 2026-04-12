import { Inbox } from "lucide-react";

export function EmptyState({
  title = "Nothing to show",
  message = "Try adjusting filters or upload new data to populate this panel."
}: {
  title?: string;
  message?: string;
}) {
  return (
    <div className="rounded-2xl border border-dashed border-white/20 bg-slate-900/40 p-10 text-center">
      <Inbox className="mx-auto h-8 w-8 text-slate-500" />
      <h4 className="mt-3 text-sm font-semibold text-slate-200">{title}</h4>
      <p className="mt-1 text-xs text-slate-400">{message}</p>
    </div>
  );
}
