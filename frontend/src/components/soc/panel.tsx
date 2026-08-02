import { AlertTriangle, Inbox } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className,
  bodyClassName,
}: {
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section className={cn("panel flex min-w-0 flex-col overflow-hidden", className)}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-3 border-b border-border px-3 py-2">
          <div className="min-w-0">
            {title && (
              <h2 className="font-mono text-[11px] font-semibold tracking-[0.14em] text-muted-foreground uppercase">
                {title}
              </h2>
            )}
            {subtitle && <p className="truncate text-xs text-muted-foreground">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
        </header>
      )}
      <div className={cn("min-w-0 flex-1 p-3", bodyClassName)}>{children}</div>
    </section>
  );
}

export function EmptyState({ message, hint }: { message: string; hint?: string }) {
  return (
    <div className="flex h-full min-h-40 flex-col items-center justify-center gap-2 px-4 py-8 text-center">
      <Inbox className="size-5 text-muted-foreground" aria-hidden />
      <p className="text-sm font-medium">{message}</p>
      {hint && <p className="max-w-xs text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div
      role="alert"
      className="flex h-full min-h-40 flex-col items-center justify-center gap-2 px-4 py-8 text-center"
    >
      <AlertTriangle className="size-5 text-sev-high" aria-hidden />
      <p className="text-sm font-medium">Could not load data</p>
      <p className="max-w-sm font-mono text-xs text-muted-foreground">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-1 rounded-sm border border-border-strong px-2.5 py-1 text-xs transition-colors hover:bg-secondary"
        >
          Retry
        </button>
      )}
    </div>
  );
}

export function SkeletonRows({ rows = 6, height = 28 }: { rows?: number; height?: number }) {
  return (
    <div className="flex flex-col gap-1.5" aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }).map((_, i) => (
        <div
          key={i}
          className="animate-pulse rounded-sm bg-muted"
          style={{ height, opacity: 1 - i * 0.08 }}
        />
      ))}
    </div>
  );
}

export function SkeletonBlock({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-sm bg-muted", className)} aria-busy="true" />;
}
