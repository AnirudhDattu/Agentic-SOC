import { ArrowDownRight, ArrowRight, ArrowUpRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { SkeletonBlock } from "./panel";

export function KpiTile({
  label,
  value,
  delta,
  accent = "default",
  loading,
  hint,
}: {
  label: string;
  value: number;
  delta?: number | null;
  accent?: "default" | "high" | "medium" | "low" | "primary";
  loading?: boolean;
  hint?: string;
}) {
  const accentClass = {
    default: "text-foreground",
    primary: "text-primary",
    high: "text-sev-high",
    medium: "text-sev-medium",
    low: "text-sev-low",
  }[accent];

  const dir = delta === null || delta === undefined ? null : delta > 0 ? "up" : delta < 0 ? "down" : "flat";
  const Icon = dir === "up" ? ArrowUpRight : dir === "down" ? ArrowDownRight : ArrowRight;

  return (
    <div className="panel flex flex-col justify-between gap-2 px-3 py-2.5">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-[10px] font-medium tracking-[0.14em] text-muted-foreground uppercase">
          {label}
        </span>
        {dir && (
          <span
            className={cn(
              "inline-flex items-center gap-0.5 font-mono text-[10px] tabular",
              dir === "up" ? "text-sev-high" : dir === "down" ? "text-sev-low" : "text-muted-foreground",
            )}
            title={hint}
          >
            <Icon className="size-3" aria-hidden />
            {Math.abs(delta ?? 0)}%
          </span>
        )}
      </div>
      {loading ? (
        <SkeletonBlock className="h-8 w-20" />
      ) : (
        <span className={cn("font-mono text-3xl leading-none font-semibold tabular", accentClass)}>
          {value.toLocaleString("en-US")}
        </span>
      )}
      <span className="text-[11px] text-muted-foreground">{hint ?? "last 24h"}</span>
    </div>
  );
}
