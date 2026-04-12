import { cn } from "@/utils/cn";
import type { ReactNode } from "react";

interface Column<T> {
  key: keyof T;
  header: string;
  render?: (value: T[keyof T], row: T) => ReactNode;
}

interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  emptyText?: string;
}

export function DataTable<T extends Record<string, unknown>>({
  columns,
  data,
  emptyText = "No data available"
}: DataTableProps<T>) {
  if (!data.length) {
    return (
      <div className="rounded-2xl border border-dashed border-white/20 bg-slate-900/30 p-8 text-center text-sm text-slate-400">
        {emptyText}
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-2xl border border-white/10 bg-slate-950/40">
      <table className="w-full border-collapse text-left text-sm">
        <thead>
          <tr className="border-b border-white/10 bg-slate-900/60 text-slate-300">
            {columns.map((column) => (
              <th key={String(column.key)} className="px-4 py-3 font-semibold">
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, rowIndex) => (
            <tr
              key={rowIndex}
              className={cn(
                "border-b border-white/5 text-slate-200",
                rowIndex % 2 === 0 ? "bg-slate-950/30" : "bg-slate-900/20"
              )}
            >
              {columns.map((column) => (
                <td key={String(column.key)} className="px-4 py-3">
                  {column.render ? column.render(row[column.key], row) : String(row[column.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
