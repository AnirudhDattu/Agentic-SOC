import { cn } from "@/lib/utils";
import { isSeverity, severityMeta } from "@/lib/soc";

export function SeverityBadge({ severity, className }: { severity: string; className?: string }) {
  const meta = isSeverity(severity) ? severityMeta[severity] : null;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-sm px-1.5 py-0.5 font-mono text-[10px] font-semibold tracking-wider uppercase",
        meta ? `${meta.bg} ${meta.text}` : "bg-muted text-muted-foreground",
        className,
      )}
    >
      <span className={cn("size-1.5 rounded-full", meta ? meta.dot : "bg-muted-foreground")} />
      {meta ? meta.label : severity}
    </span>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const attack = status === "Attack";
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-sm border px-1.5 py-0.5 font-mono text-[10px] font-semibold tracking-wider uppercase",
        attack
          ? "border-sev-high/40 text-sev-high"
          : "border-border-strong text-muted-foreground",
      )}
    >
      {status}
    </span>
  );
}

export function WorkflowBadge({ status }: { status: string }) {
  const map: Record<string, string> = {
    New: "border-info/50 text-info",
    Investigating: "border-sev-medium/50 text-sev-medium",
    Resolved: "border-sev-low/50 text-sev-low",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-sm border px-1.5 py-0.5 font-mono text-[10px] tracking-wider uppercase",
        map[status] ?? "border-border-strong text-muted-foreground",
      )}
    >
      {status}
    </span>
  );
}
